import logging
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from pytorch_lightning import LightningDataModule


class MOFDatamodule(LightningDataModule):

    def __init__(self, *, data_cfg, train_dataset, valid_dataset, predict_dataset=None):
        super().__init__()
        self.data_cfg = data_cfg
        self.loader_cfg = data_cfg.loader
        self.sampler_cfg = data_cfg.sampler
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        self._predict_dataset = predict_dataset

    def train_dataloader(self, rank=None, num_replicas=None):
        num_workers = self.loader_cfg.num_workers
        return DataLoader(
            self._train_dataset,
            batch_sampler=TimeBatcher(
                sampler_cfg=self.sampler_cfg,
                dataset=self._train_dataset,
                rank=rank,
                num_replicas=num_replicas,
            ),
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self._valid_dataset,
            sampler=DistributedSampler(self._valid_dataset, shuffle=False),
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        num_workers = self.loader_cfg.num_workers
        return DataLoader(
            self._predict_dataset,
            sampler=DistributedSampler(self._predict_dataset, shuffle=False),
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
            persistent_workers=True,
        )


class TimeBatcher:

    def __init__(
            self,
            *,
            sampler_cfg,
            dataset,
            seed=123,
            shuffle=True,
            num_replicas=None,
            rank=None,
        ):
        super().__init__()
        self._log = logging.getLogger(__name__)
        if num_replicas is None:
            self.num_replicas = dist.get_world_size()
        else:
            self.num_replicas = num_replicas
        if rank is None:
            self.rank = dist.get_rank()
        else:
            self.rank = rank
        
        self._sampler_cfg = sampler_cfg
        self._dataset_indices = np.arange(len(dataset))
        self._dataset = dataset

        self._num_batches = math.ceil(len(dataset) / self.num_replicas)
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.max_batch_size = self._sampler_cfg.max_batch_size
        self._log.info(f'Created dataloader rank {self.rank+1} out of {self.num_replicas}')

    def _replica_epoch_batches(self):
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        indices = self._dataset_indices
        if self.shuffle: 
            new_order = torch.randperm(len(indices), generator=rng).numpy().tolist()
            indices = np.array([indices[i] for i in new_order])
        
        if len(indices) > self.num_replicas:
            replica_indices = indices[self.rank::self.num_replicas]
        else:
            replica_indices = indices
        
        # Dynamically determine max batch size
        repeated_indices = []
        for idx in replica_indices:
            num_atoms = self._dataset[idx]['atom_types'].shape[0]
            max_batch_size = min(
                self.max_batch_size,
                self._sampler_cfg.max_num_res_squared // num_atoms**2 + 1,
            )
            repeated_indices.append([idx] * max_batch_size)

        return repeated_indices

    def _create_batches(self):
        all_batches = []
        while len(all_batches) < self._num_batches:
            all_batches.extend(self._replica_epoch_batches())
        if len(all_batches) >= self._num_batches:
            all_batches = all_batches[:self._num_batches]
        self.sample_order = all_batches
    
    def __iter__(self):
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_order)
    
    def __len__(self):
        return len(self.sample_order)