import os
import time
import random
import numpy as np
import hydra
import torch
import GPUtil
import pytorch_lightning as pl
from tqdm import tqdm
from pathlib import Path
from pytorch_lightning import Trainer
from torch_geometric.data import Batch
from omegaconf import DictConfig, OmegaConf
from experiments import utils as eu
from data.dataset import MOFDataset
from data.dataloader import MOFDatamodule
from data.interpolant import Interpolant
from models.flow_module import FlowModule
from common.utils import PROJECT_ROOT


torch.set_float32_matmul_precision('high')
log = eu.get_pylogger(__name__)


class EvalRunner:

    def __init__(self, cfg: DictConfig):
        """Initialize sampler.

        Args:
            cfg: inference config.
        """
        ckpt_path = cfg.inference.ckpt_path
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, 'config.yaml'))

        # Set-up config
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_struct(ckpt_cfg, False)
        cfg = OmegaConf.merge(cfg, ckpt_cfg)
        cfg.experiment.checkpointer.dirpath = './'
        self._cfg = cfg
        self._infer_cfg = cfg.inference
        self._data_cfg = cfg.data

        # Set seed
        if self._infer_cfg.seed is not None:
            log.info(f'Setting seed to {self._infer_cfg.seed}')
            self._set_seed(self._infer_cfg.seed)
        
        # Set device
        self.device = GPUtil.getAvailable(order='memory')[0]
        log.info(f'Using device {self.device}')

        # Setup output directory
        self.inference_dir = Path(self._infer_cfg.inference_dir)
        self.inference_dir.mkdir(parents=True, exist_ok=True)
        log.info(f'Saving results to {self.inference_dir}')
        
        # Save config to output directory
        config_path = self.inference_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._cfg, f=f)

        # Load model from checkpoint
        self._flow_module = FlowModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            cfg=self._cfg,
        )
        self._flow_module.eval()
        self._flow_module.to(self.device)
        
        # Load interpolant
        self.interpolant = Interpolant(self._infer_cfg.interpolant)
        self.interpolant.set_device(self.device)
        
        self.num_samples = self._infer_cfg.num_samples

    def _set_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Ensuring deterministic behavior in CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def setup_dataloader(self):
        test_dataset = MOFDataset(
            cache_path=os.path.join(self._data_cfg.cache_dir, 'test.pt'),
            dataset_cfg=self._data_cfg,
            is_training=False
        )
        dataloader = MOFDatamodule(
            data_cfg=self._data_cfg,
            train_dataset=None,
            valid_dataset=None,
            test_dataset=test_dataset).test_dataloader()

        return dataloader
    
    def run_sampling(self):
        # Setup dataloader
        dataloader = self.setup_dataloader()
        
        # Predict k samples
        results = {}
        cart_coords = []
        num_atoms = []
        atom_types = []
        lattices = []
        gt_data_list = []
        
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            batch = batch.to(self.device)
            batch_cart_coords, batch_num_atoms, batch_atom_types, batch_lattices = [], [], [], []
            
            for k in range(self.num_samples):
                print(f'batch {batch_idx} / {len(dataloader)}, sample {k} / {self.num_samples}')
                
                # Sample
                mof_traj, _ = self.interpolant.sample(
                    num_batch=batch.num_graphs,
                    num_bbs=batch.num_bbs,
                    model=self._flow_module.model,
                    atom_types=batch.atom_types,
                    local_coords=batch.local_coords,
                    batch_vec=batch.batch,
                    bb_num_vec=batch.bb_num_vec,
                )
                coords, lattice = mof_traj[-1]
                
                # Append to list
                batch_cart_coords.append(coords.detach().cpu())
                batch_lattices.append(lattice.detach().cpu())
                batch_num_atoms.append(batch.num_atoms.detach().cpu())
                batch_atom_types.append(batch.atom_types.detach().cpu())
            
            cart_coords.append(torch.stack(batch_cart_coords, dim=0))
            num_atoms.append(torch.stack(batch_num_atoms, dim=0))
            atom_types.append(torch.stack(batch_atom_types, dim=0))
            lattices.append(torch.stack(batch_lattices, dim=0))

            gt_data_list = gt_data_list + batch.to_data_list()
        
        # Concatenate results
        cart_coords = torch.cat(cart_coords, dim=1)
        num_atoms = torch.cat(num_atoms, dim=1)
        atom_types = torch.cat(atom_types, dim=1)
        lattices = torch.cat(lattices, dim=1)
        gt_data_batch = Batch.from_data_list(gt_data_list)
        
        results = {
            'cart_coords': cart_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lattices': lattices,
            'gt_data_batch': gt_data_batch
        }
        
        return results


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="inference")
def run(cfg: DictConfig) -> None:

    log.info(f'Starting inference with {cfg.inference.num_gpus} GPUs')
    
    # Initialize sampler
    sampler = EvalRunner(cfg)

    # Run sampling
    start_time = time.time()
    results = sampler.run_sampling()
    elapsed_time = time.time() - start_time
    results['elapsed_time'] = elapsed_time

    log.info(f'Finished in {elapsed_time:.2f}s')
    
    # Save results
    torch.save(results, os.path.join(sampler.inference_dir, 'predict.pt'))

if __name__ == '__main__':
    run()