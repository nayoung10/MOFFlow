from collections import defaultdict
import torch
from data import so3_utils
from data import utils as du
from scipy.spatial.transform import Rotation
from torch.distributions import LogNormal, Uniform
import copy


def _centered_gaussian(num_batch, num_res, device):
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)

def _uniform_so3(num_batch, num_res, device):
    return torch.tensor(
        Rotation.random(num_batch*num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)

def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])

def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return (
        rotmats_t * diffuse_mask[..., None, None]
        + rotmats_1 * (1 - diffuse_mask[..., None, None])
    )


class Interpolant:

    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._lattice_cfg = cfg.lattice
        self._sample_cfg = cfg.sampling
        self._igso3 = None
        self._lognormal = LogNormal(
            loc=torch.Tensor(self._lattice_cfg.lognormal.loc), 
            scale=torch.Tensor(self._lattice_cfg.lognormal.scale)
        )
        self._uniform = Uniform(
            low=self._lattice_cfg.uniform.low - self._lattice_cfg.uniform.eps, 
            high=self._lattice_cfg.uniform.high + self._lattice_cfg.uniform.eps
        )

    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(
                1000, sigma_grid, cache_dir='.cache')
        return self._igso3

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2*self._cfg.min_t) + self._cfg.min_t

    def _corrupt_trans(self, trans_1, t, res_mask, diffuse_mask):
        trans_nm_0 = _centered_gaussian(*res_mask.shape, self._device)
        trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        trans_t = _trans_diffuse_mask(trans_t, trans_1, diffuse_mask)
        return trans_t * res_mask[..., None]
    
    def _corrupt_rotmats(self, rotmats_1, t, res_mask, diffuse_mask):
        num_batch, num_res = res_mask.shape
        noisy_rotmats = self.igso3.sample(
            torch.tensor([1.5]),
            num_batch*num_res
        ).to(self._device)
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
        rotmats_0 = torch.einsum(
            "...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = (
            rotmats_t * res_mask[..., None, None]
            + identity[None, None] * (1 - res_mask[..., None, None])
        )
        return _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask)

    def _corrupt_lattice(self, lattice_1, t):
        lengths_0 = self._lognormal.sample((lattice_1.shape[0],)).to(self._device)
        angles_0 = self._uniform.sample(lattice_1[:, 3:].shape).to(self._device)
        lattice_0 = torch.cat([lengths_0, angles_0], dim=-1)
        lattice_t = (1 - t) * lattice_0 + t * lattice_1
        return lattice_t
    
    def corrupt_batch(self, batch):
        noisy_batch = copy.deepcopy(batch)

        # [B, N, 3]
        trans_1 = batch['trans_1']  # Angstrom

        # [B, N, 3, 3]
        rotmats_1 = batch['rotmats_1']

        # [B, 6]
        lattice_1 = batch['lattice_1']

        # [B, N]
        res_mask = batch['res_mask']
        diffuse_mask = batch['diffuse_mask']
        num_batch, _ = diffuse_mask.shape

        # [B, 1]
        t = self.sample_t(num_batch)[:, None]
        so3_t = t
        r3_t = t
        l_t = t
        noisy_batch['so3_t'] = so3_t
        noisy_batch['r3_t'] = r3_t
        noisy_batch['l_t'] = l_t

        # Apply corruptions
        if self._trans_cfg.corrupt:
            trans_t = self._corrupt_trans(
                trans_1, r3_t, res_mask, diffuse_mask)
        else:
            trans_t = trans_1
        if torch.any(torch.isnan(trans_t)):
            raise ValueError('NaN in trans_t during corruption')
        noisy_batch['trans_t'] = trans_t

        if self._rots_cfg.corrupt:
            rotmats_t = self._corrupt_rotmats(
                rotmats_1, so3_t, res_mask, diffuse_mask)
        else:
            rotmats_t = rotmats_1
        if torch.any(torch.isnan(rotmats_t)):
            raise ValueError('NaN in rotmats_t during corruption')
        noisy_batch['rotmats_t'] = rotmats_t

        if self._lattice_cfg.corrupt:
            lattice_t = self._corrupt_lattice(lattice_1, l_t)
        else:
            lattice_t = lattice_1
        if torch.any(torch.isnan(lattice_t)):
            raise ValueError('NaN in lattice_t during corruption')
        noisy_batch['lattice_t'] = lattice_t

        return noisy_batch
    
    def rot_sample_kappa(self, t):
        if self._rots_cfg.sample_schedule == 'exp':
            return 1 - torch.exp(-t*self._rots_cfg.exp_rate)
        elif self._rots_cfg.sample_schedule == 'linear':
            return t
        else:
            raise ValueError(
                f'Invalid schedule: {self._rots_cfg.sample_schedule}')

    def _trans_vector_field(self, t, trans_1, trans_t):
        return (trans_1 - trans_t) / (1 - t)

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        assert d_t > 0
        trans_vf = self._trans_vector_field(t, trans_1, trans_t)
        return trans_t + trans_vf * d_t

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        if self._rots_cfg.sample_schedule == 'linear':
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == 'exp':
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(
                f'Unknown sample schedule {self._rots_cfg.sample_schedule}')
        return so3_utils.geodesic_t(
            scaling * d_t, rotmats_1, rotmats_t)

    def _assemble_coords(self, local_coords, rotmats, trans, bb_num_vec):
        """
        Returns:
            coords: numpy array of shape (n_atoms, 3), where local coordinates 
                have been assembled via X' = X @ rotmats.T + trans
        """

        start_idx = 0 
        final_coords = []
        device = self._device
        for i, num_bb in enumerate(bb_num_vec):
            bb_local_coord = local_coords[start_idx:start_idx+num_bb].to(device)
            bb_rotmats = rotmats[i].to(device)
            bb_trans = trans[i][None].to(device)

            bb_coords = bb_local_coord @ bb_rotmats.t() + bb_trans
            final_coords.append(bb_coords)

            start_idx += num_bb

        final_coords = torch.cat(final_coords, dim=0)

        return final_coords

    def sample(
            self,
            num_batch,
            num_bb,
            model,
            num_timesteps=None,
            trans_potential=None,
            trans_0=None,
            rotmats_0=None,
            lattice_0=None,
            trans_1=None,
            rotmats_1=None,
            lattice_1=None,
            diffuse_mask=None,
            atom_types=None,
            local_coords=None,
            bb_num_vec=None,
            verbose=False,
        ):
        res_mask = torch.ones(num_batch, num_bb, device=self._device)       

        # Set-up initial prior samples
        if trans_0 is None:
            trans_0 = _centered_gaussian(
                num_batch, num_bb, self._device) * du.NM_TO_ANG_SCALE
        if rotmats_0 is None:
            rotmats_0 = _uniform_so3(num_batch, num_bb, self._device)
        if lattice_0 is None:
            lengths_0 = self._lognormal.sample((num_batch,)).to(self._device)
            angles_0 = self._uniform.sample((num_batch, 3)).to(self._device)
            lattice_0 = torch.cat([lengths_0, angles_0], dim=-1)
        batch = {
            'res_mask': res_mask,
            'diffuse_mask': res_mask,
            'atom_types': atom_types,
            'local_coords': local_coords,
            'bb_num_vec': bb_num_vec
        }

        logs_traj = defaultdict(list)

        # Set-up time 
        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]

        mof_traj = [(trans_0, rotmats_0, lattice_0)]
        clean_traj = []
        for i, t_2 in enumerate(ts[1:]):
            if verbose:
                print(f'{i=}, t={t_1.item():.2f}')
                print(torch.cuda.mem_get_info(trans_0.device), torch.cuda.memory_allocated(trans_0.device))
            # Run model
            trans_t_1, rotmats_t_1, lattice_t_1 = mof_traj[-1]
            if self._trans_cfg.corrupt:
                batch['trans_t'] = trans_t_1
            else:
                if trans_1 is None:
                    raise ValueError('Must provide trans_1 if not corrupting.')
                batch['trans_t'] = trans_1
            if self._rots_cfg.corrupt:
                batch['rotmats_t'] = rotmats_t_1
            else:
                if rotmats_1 is None:
                    raise ValueError('Must provide rotmats_1 if not corrupting.')
                batch['rotmats_t'] = rotmats_1
            if self._lattice_cfg.corrupt:
                batch['lattice_t'] = lattice_t_1
            else:
                if lattice_1 is None:
                    raise ValueError('Must provide lattice_1 if not corrupting.')
                batch['lattice_t'] = lattice_1
            batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['so3_t'] = batch['t']
            batch['r3_t'] = batch['t']
            batch['l_t'] = batch['t']
            d_t = t_2 - t_1
        
            with torch.no_grad():
                model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            pred_lattice_1 = model_out['pred_lattice']

            clean_traj.append(
                (pred_trans_1.detach().cpu(), 
                pred_rotmats_1.detach().cpu(),
                pred_lattice_1.detach().cpu())
            )
            if self._cfg.self_condition:
                batch['trans_sc'] = pred_trans_1
            
            # Take reverse step
            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1
            )
            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1
            )
            lattice_t_2 = self._trans_euler_step(
                d_t, t_1, pred_lattice_1, lattice_t_1
            )

            mof_traj.append((trans_t_2, rotmats_t_2, lattice_t_2))
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1, lattice_t_1 = mof_traj[-1]
        if self._trans_cfg.corrupt:
            batch['trans_t'] = trans_t_1
        else:
            if trans_1 is None:
                raise ValueError('Must provide trans_1 if not corrupting.')
            batch['trans_t'] = trans_1
        if self._rots_cfg.corrupt:
            batch['rotmats_t'] = rotmats_t_1
        else:
            if rotmats_1 is None:
                raise ValueError('Must provide rotmats_1 if not corrupting.')
            batch['rotmats_t'] = rotmats_1 
        if self._lattice_cfg.corrupt:
            batch['lattice_t'] = lattice_t_1
        else:
            if lattice_1 is None:
                raise ValueError('Must provide lattice_1 if not corrupting.')
            batch['lattice_t'] = lattice_1
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        pred_lattice_1 = model_out['pred_lattice']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), 
            pred_rotmats_1.detach().cpu(),
            pred_lattice_1.detach().cpu())
        )
        mof_traj.append((pred_trans_1, pred_rotmats_1, pred_lattice_1))

        # Convert trajectories to (coordinates, lattice) format
        mof_final = []
        for trans, rotmats, lattice in mof_traj:
            coords = self._assemble_coords(
                local_coords.squeeze(),
                rotmats.squeeze(),
                trans.squeeze(),
                bb_num_vec.squeeze()
            )
            mof_final.append((coords, lattice))

        clean_mof_final = []
        for trans, rotmats, lattice in clean_traj:
            coords = self._assemble_coords(
                local_coords.squeeze(),
                rotmats.squeeze(),
                trans.squeeze(),
                bb_num_vec.squeeze()
            )
            clean_mof_final.append((coords, lattice))

        return mof_final, clean_mof_final