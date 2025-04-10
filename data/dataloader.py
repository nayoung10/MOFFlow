import logging
import math
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule


class MOFDatamodule(LightningDataModule):

    def __init__(self, *, data_cfg, train_dataset, valid_dataset, test_dataset=None):
        super().__init__()
        self.data_cfg = data_cfg
        self.loader_cfg = data_cfg.loader
        self.sampler_cfg = data_cfg.sampler
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        self._test_dataset = test_dataset

    def train_dataloader(self, shuffle=True):
        batch_size = self.loader_cfg.batch_size.train
        num_workers = self.loader_cfg.num_workers
        return DataLoader(
            self._train_dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False,
        )

    def val_dataloader(self):
        batch_size = self.loader_cfg.batch_size.valid
        return DataLoader(
            self._valid_dataset,
            batch_size=batch_size,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
        )

    def test_dataloader(self):
        batch_size=self.loader_cfg.batch_size.test
        num_workers = self.loader_cfg.num_workers
        return DataLoader(
            self._test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
            persistent_workers=True,
        )