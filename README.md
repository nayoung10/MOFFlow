## Environment 

### Create environment 
```bash
mamba env create -f env.yml
pip install git+https://github.com/microsoft/MOFDiff.git
pip install -e .
mamba activate mofflow
```

For **property computation**, we need to install `zeo++` from [here](https://www.zeoplusplus.org/download.html). 

### Create ```.env```

Create ```.env``` file with following entries. 
```bash
export PROJECT_ROOT=<root_dir>
export ZEO_PATH=<path_to_network_binary> # for property computation
```


## To start training

The following code starts training from scratch. All paths/directory information are in ```configs/paths.yaml```; modify as needed. 

```bash
python experiments/train.py experiment.wandb.name=<expname>
```

## To continue training from ckpt 

To continue training from ```<ckpt_path>``` in experiment ```<expname>```

```bash
python experiments/train.py \
    experiment.wandb.name=<expname> \
    experiment.warm_start=<ckpt_path> \
    +experiment.wandb.id=<run_id> \
    +experiment.wandb.resume=must
```

## Inference

```bash
python experiments/inference.py \
    experiment.wandb.name=<expname> \
    inference.ckpt_path=<ckpt_path> \
    inference.num_gpus=<num_gpus> \
    inference.inference_dir=<output_dir>
```

## Evaluation 

### Validity

```bash
mamba activate mofdiff # TODO: install necessary packages
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
python evaluation/valid.py --cif_dir <inference_dir>
```

### RMSD (with varying ```StructureMatcher``` config)
You may need to adjust the code based on how the ground-truth and predicted cif files are structured. 

```bash
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
python evaluation/rmsd.py \
    --cif_dir <inference_dir> \
    --gt_prefix <gt_prefix> \
    --pred_prefix <pred_prefix> \
    --num_cpus <num_cpus>
```

### Property computation (with `zeo++`)

```bash
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
python evaluation/property.py --cif_dir <inference_dir> --ncpu <num_cpus>
```