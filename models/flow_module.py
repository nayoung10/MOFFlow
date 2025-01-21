from typing import Any
import torch
import time
import os
import random
import json
import numpy as np
import pandas as pd
import logging
import torch.distributed as dist
from pytorch_lightning import LightningModule
from models.flow_model import FlowModel
from models import utils as mu
from data.interpolant import Interpolant
from data import utils as du
from data import so3_utils
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.cif import CifWriter
from pymatgen.analysis.structure_matcher import StructureMatcher


class FlowModule(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._interpolant_cfg = cfg.interpolant

        # Set-up vector field prediction model
        self.model = FlowModel(cfg.model)

        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant)

        # self.results_df = pd.DataFrame(columns=['batch_id', 'num_bb', 'rms_dist', 'time'])
        self.results = []
        self.save_hyperparameters()

        self._checkpoint_dir = None
        self._inference_dir = None

        # Setup structure matcher
        self.matcher = StructureMatcher(
            stol=cfg.matcher.stol, 
            angle_tol=cfg.matcher.angle_tol, 
            ltol=cfg.matcher.ltol
        )
    
    @property
    def checkpoint_dir(self):
        if self._checkpoint_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    checkpoint_dir = [self._exp_cfg.checkpointer.dirpath]
                else:
                    checkpoint_dir = [None]
                dist.broadcast_object_list(checkpoint_dir, src=0)
                checkpoint_dir = checkpoint_dir[0]
            else:
                checkpoint_dir = self._exp_cfg.checkpointer.dirpath
            self._checkpoint_dir = checkpoint_dir
            os.makedirs(self._checkpoint_dir, exist_ok=True)
        return self._checkpoint_dir

    @property
    def inference_dir(self):
        if self._inference_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    inference_dir = [self._exp_cfg.inference_dir]
                else:
                    inference_dir = [None]
                dist.broadcast_object_list(inference_dir, src=0)
                inference_dir = inference_dir[0]
            else:
                inference_dir = self._exp_cfg.inference_dir
            self._inference_dir = inference_dir
            os.makedirs(self._inference_dir, exist_ok=True)
        return self._inference_dir

    def _log_scalar(
        self,
        key,
        value,
        on_step=True,
        on_epoch=False,
        prog_bar=True,
        batch_size=None,
        sync_dist=False,
        rank_zero_only=True
    ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )

    def configure_optimizers(self):
        optimizer =  torch.optim.AdamW(
            params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )
        if not self._exp_cfg.use_lr_scheduler:
            return optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **self._exp_cfg.lr_scheduler
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'valid/loss'
        }

    def model_step(self, noisy_batch: Any):
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['res_mask'] * noisy_batch['diffuse_mask']
        if torch.any(torch.sum(loss_mask, dim=-1) < 1):
            raise ValueError('Empty batch encountered')
        num_batch, num_bb = loss_mask.shape

        # Ground truth labels
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        gt_lattice_1 = noisy_batch['lattice_1']
        rotmats_t = noisy_batch['rotmats_t']
        gt_rot_vf = so3_utils.calc_rot_vf(
            rotmats_t, gt_rotmats_1.type(torch.float32))
        if torch.any(torch.isnan(gt_rot_vf)):
            raise ValueError('NaN encountered in gt_rot_vf')
        gt_bb_atoms = noisy_batch['gt_coords']

        # Timestep used for normalization
        r3_t = noisy_batch['r3_t']
        so3_t = noisy_batch['so3_t']
        l_t= noisy_batch['l_t']
        r3_norm_scale = 1 - torch.min(
            r3_t[..., None], torch.tensor(training_cfg.t_normalize_clip))
        so3_norm_scale = 1 - torch.min(
            so3_t[..., None], torch.tensor(training_cfg.t_normalize_clip)
        )
        l_norm_scale = 1 - torch.min(
            l_t, torch.tensor(training_cfg.t_normalize_clip)
        )

        # Model output predictions
        model_output = self.model(noisy_batch)
        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
        if torch.any(torch.isnan(pred_rots_vf)):
            raise ValueError('NaN encountered in pred_rots_vf')
        pred_lattice_1 = model_output['pred_lattice']

        # Translation VF loss
        loss_denom = torch.sum(loss_mask, dim=-1) * 3

        trans_error = (gt_trans_1 - pred_trans_1) / r3_norm_scale * training_cfg.trans_scale
        trans_loss = training_cfg.translation_loss_weight * torch.sum(
            trans_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom
        trans_loss = torch.clamp(trans_loss, max=5)

        # Rotation VF loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / so3_norm_scale
        rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(
            rots_vf_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        se3_vf_loss = trans_loss + rots_vf_loss
        if torch.any(torch.isnan(se3_vf_loss)):
            raise ValueError('NaN encountered in se3_vf_loss')

        # Cell VF loss
        cell_error = (gt_lattice_1 - pred_lattice_1) / l_norm_scale
        cell_error[:, :3] *= training_cfg.cell_scale
        cell_error[:, 3:] *= torch.pi / 180.0
        cell_loss = training_cfg.cell_loss_weight * torch.mean(
            cell_error ** 2, dim=-1
        )

        return {
            'trans_loss': trans_loss,
            'rots_vf_loss': rots_vf_loss,
            'se3_vf_loss': se3_vf_loss,
            'cell_loss': cell_loss
        }

    def on_train_start(self):
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()

    def training_step(self, batch: Any, stage: int):
        step_start_time = time.time()
        self.interpolant.set_device(batch['res_mask'].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)
        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                model_sc = self.model(noisy_batch)
                noisy_batch['trans_sc'] = (
                    model_sc['pred_trans'] * noisy_batch['diffuse_mask'][..., None]
                    + noisy_batch['trans_1'] * (1 - noisy_batch['diffuse_mask'][..., None])
                )
        batch_losses = self.model_step(noisy_batch)
        num_batch = batch_losses['trans_loss'].shape[0]
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
        }
        for k,v in total_losses.items():
            self._log_scalar(
                f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # Losses to track. Stratified across t.
        so3_t = torch.squeeze(noisy_batch['so3_t'])
        self._log_scalar(
            "train/so3_t",
            np.mean(du.to_numpy(so3_t)),
            prog_bar=False, batch_size=num_batch)
        r3_t = torch.squeeze(noisy_batch['r3_t'])
        self._log_scalar(
            "train/r3_t",
            np.mean(du.to_numpy(r3_t)),
            prog_bar=False, batch_size=num_batch)
        l_t = torch.squeeze(noisy_batch['l_t'])
        self._log_scalar(
            "train/l_t",
            np.mean(du.to_numpy(l_t)),
            prog_bar=False, batch_size=num_batch)
        for loss_name, loss_dict in batch_losses.items():
            if loss_name == 'rots_vf_loss':
                batch_t = so3_t
            elif loss_name == 'trans_loss':
                batch_t = r3_t
            else:
                batch_t = l_t
            stratified_losses = mu.t_stratified_loss(
                batch_t, loss_dict, loss_name=loss_name)
            for k, v in stratified_losses.items():
                self._log_scalar(
                    f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # Training throughput
        self._log_scalar(
            "train/length", batch['res_mask'].shape[1], prog_bar=False, batch_size=num_batch)
        self._log_scalar(
            "train/batch_size", num_batch, prog_bar=False)
        step_time = time.time() - step_start_time
        self._log_scalar(
            "train/examples_per_second", num_batch / step_time)
        train_loss = total_losses['se3_vf_loss'] + total_losses['cell_loss']
        self._log_scalar(
            "train/loss", train_loss, batch_size=num_batch)
        return train_loss

    def validation_step(self, batch: Any, batch_idx: int):
        assert not torch.is_grad_enabled()

        self.interpolant.set_device(batch['res_mask'].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)
        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            model_sc = self.model(noisy_batch)
            noisy_batch['trans_sc'] = (
                model_sc['pred_trans'] * noisy_batch['diffuse_mask'][..., None]
                + noisy_batch['trans_1'] * (1 - noisy_batch['diffuse_mask'][..., None])
            )
        batch_losses = self.model_step(noisy_batch)
        num_batch = batch_losses['trans_loss'].shape[0]
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
        }
        for k,v in total_losses.items():
            self._log_scalar(
                f"valid/{k}", v, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)

        # Losses to track. Stratified across t.
        so3_t = torch.squeeze(noisy_batch['so3_t'])
        self._log_scalar(
            "valid/so3_t",
            np.mean(du.to_numpy(so3_t)),
            on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)
        r3_t = torch.squeeze(noisy_batch['r3_t'])
        self._log_scalar(
            "valid/r3_t",
            np.mean(du.to_numpy(r3_t)),
            on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)
        l_t = torch.squeeze(noisy_batch['l_t'])
        self._log_scalar(
            "valid/l_t",
            np.mean(du.to_numpy(l_t)),
            on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)
        for loss_name, loss_dict in batch_losses.items():
            if loss_name == 'rots_vf_loss':
                batch_t = so3_t
            elif loss_name == 'trans_loss':
                batch_t = r3_t
            else:
                batch_t = l_t
            stratified_losses = mu.t_stratified_loss(
                batch_t, loss_dict, loss_name=loss_name)
            for k,v in stratified_losses.items():
                self._log_scalar(
                    f"valid/{k}", v, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)

        # Validation throughput
        self._log_scalar(
            "valid/length", batch['res_mask'].shape[1], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=False)
        valid_loss = total_losses['se3_vf_loss'] + total_losses['cell_loss']
        self._log_scalar(
            "valid/loss", valid_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=False)
        return valid_loss
    
    def predict_step(self, batch, batch_idx):
        device = f'cuda:{torch.cuda.current_device()}'    
        interpolant = Interpolant(self._infer_cfg.interpolant)
        interpolant.set_device(device)

        num_batch, num_bb = batch['res_mask'].shape

        now = time.time()
        mof_traj, _ = interpolant.sample(
            num_batch=num_batch,
            num_bb=num_bb,
            model=self.model,
            trans_1=batch['trans_1'],
            rotmats_1=batch['rotmats_1'],
            lattice_1=batch['lattice_1'],
            diffuse_mask=batch['diffuse_mask'],
            atom_types=batch['atom_types'],
            local_coords=batch['local_coords'],
            bb_num_vec=batch['bb_num_vec'],
        )
        elapsed = time.time() - now

        pred_coords, pred_lattice = mof_traj[-1]

        # Get global index
        global_idx = batch_idx * dist.get_world_size() + dist.get_rank()
        
        # Set directory
        sample_dir = os.path.join(
            self.inference_dir,
            f'sample_{global_idx}'
        )
        os.makedirs(sample_dir, exist_ok=True)

        # Create structure files
        atom_types = batch['atom_types'].squeeze().detach().cpu().numpy()
        lattice = batch['lattice_1'].squeeze().detach().cpu().numpy()
        gt_structure = Structure(
            lattice=Lattice.from_parameters(*lattice),
            species=atom_types,
            coords=batch['gt_coords'].squeeze().detach().cpu().numpy(),
            coords_are_cartesian=True
        )
        # Write ground truth structure
        writer = CifWriter(gt_structure)
        writer.write_file(os.path.join(sample_dir, f'gt_{global_idx}.cif'))
        
        try:
            pred_structure = Structure(
                lattice=Lattice.from_parameters(*pred_lattice.squeeze().detach().cpu().numpy()),
                species=atom_types,
                coords=pred_coords.detach().cpu().numpy(),
                coords_are_cartesian=True
            )
            # Write predicted structure
            writer = CifWriter(pred_structure)
            writer.write_file(os.path.join(sample_dir, f'pred_{global_idx}.cif'))
            
            # Compute RMSD with structure matcher
            rms_dist = self.matcher.get_rms_dist(gt_structure, pred_structure)
            rms_dist = None if rms_dist is None else rms_dist[0]
        except Exception as e:
            print(f"ERROR:: {e}")
            rms_dist = None

        # Save results
        self.results.append({
            'sample_idx': global_idx,
            'num_bb': num_bb,
            'num_atoms': len(atom_types),
            'rms_dist': rms_dist,
            'time': elapsed
        })
        
    def on_predict_epoch_end(self):
        # Gather results
        all_results = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(all_results, self.results)

        if dist.get_rank() == 0:
            all_results = [item for sublist in all_results for item in sublist]
            all_df = pd.DataFrame(all_results)
            all_df = all_df.sort_values(by='sample_idx')
        
            # Compute average metrics
            rms_dist = all_df['rms_dist'].dropna()
            match_rate = len(rms_dist) / len(all_df) * 100
            results = {
                'match_rate': match_rate,
                'rms_dist': rms_dist.mean() if len(rms_dist) > 0 else None,
                'avg_time': all_df['time'].dropna().mean()
            }

            # Save average metrics to JSON
            print(f"INFO:: {results}")
            with open(os.path.join(self.inference_dir, 'average.json'), 'w') as f:
                json.dump(results, f)

            # Save results to CSV
            all_df.to_csv(os.path.join(self.inference_dir, 'results.csv'), index=False)
