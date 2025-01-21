"""Script for running inference and evaluation."""

import os
import time
import random
import numpy as np
import hydra
import torch
import GPUtil
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from omegaconf import DictConfig, OmegaConf
from experiments import utils as eu
from data.dataset import MOFDataset
from data.dataloader import MOFDatamodule
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

        # Set-up config.
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_struct(ckpt_cfg, False)
        cfg = OmegaConf.merge(cfg, ckpt_cfg)
        cfg.experiment.checkpointer.dirpath = './'
        self._cfg = cfg
        self._exp_cfg = cfg.experiment
        self._infer_cfg = cfg.inference
        self._data_cfg = cfg.data
        self._rng = np.random.default_rng(self._infer_cfg.seed)

        if self._infer_cfg.seed is not None:
            log.info(f'Setting seed to {self._infer_cfg.seed}')
            self._set_seed(self._infer_cfg.seed)

        # Set-up output directory only on rank 0
        local_rank = os.environ.get('LOCAL_RANK', 0)
        if local_rank == 0:
            # Create output directory.
            inference_dir = self._infer_cfg.inference_dir
            self._exp_cfg.inference_dir = inference_dir
            os.makedirs(inference_dir, exist_ok=True)
            log.info(f'Saving results to {inference_dir}')

            # Save config.
            config_path = os.path.join(inference_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                OmegaConf.save(config=self._cfg, f=f)
            log.info(f'Saving inference config to {config_path}')

        # Read checkpoint and initialize module.
        self._flow_module = FlowModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            cfg=self._cfg,
        )
        log.info(pl.utilities.model_summary.ModelSummary(self._flow_module))
        self._flow_module.eval()
        self._flow_module._infer_cfg = self._infer_cfg

    @property
    def inference_dir(self):
        return self._flow_module.inference_dir

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
    
    def run_sampling(self):
        devices = GPUtil.getAvailable(
            order='memory', limit = 8)[:self._infer_cfg.num_gpus]
        log.info(f"Using devices: {devices}")
        log.info(f'Evaluating {self._infer_cfg.task}')
        eval_dataset = MOFDataset(
            cache_path=os.path.join(self._data_cfg.cache_dir, 'test.pt'),
            dataset_cfg=self._data_cfg,
            is_training=False
        )
        dataloader = MOFDatamodule(
            data_cfg=self._data_cfg,
            train_dataset=None,
            valid_dataset=None,
            predict_dataset=eval_dataset,
        )

        trainer = Trainer(
            accelerator="gpu",
            strategy="ddp",
            devices=devices,
        )
        trainer.predict(self._flow_module, dataloaders=dataloader)


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="inference")
def run(cfg: DictConfig) -> None:

    # Read model checkpoint.
    log.info(f'Starting inference with {cfg.inference.num_gpus} GPUs')
    start_time = time.time()
    sampler = EvalRunner(cfg)
    sampler.run_sampling()
    elapsed_time = time.time() - start_time
    log.info(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()