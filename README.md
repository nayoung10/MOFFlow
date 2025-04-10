## Environment 

### Create environment 
```bash
mamba env create -f env.yml
mamba activate mofflow
pip install git+https://github.com/microsoft/MOFDiff.git
pip install -e .
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

## Predict
To predict the MOF structures for the test set, run the following code. You may specify the number of samples to generate per test sample.

```bash
python experiments/predict.py \
    experiment.wandb.name=<expname> \
    inference.num_samples=<num_samples> \   # default to 1
    inference.interpolant.sampling.num_timesteps=<timesteps> \ # default to 50
    inference.ckpt_path='${paths.ckpt_dir}/last.ckpt' \   # default
    inference.inference_dir='${paths.log_dir}/inference' \  # default 
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
python experiments/evaluate.py
    --save_pt <path/to/predict.pt> \
    --num_samples <k> \ # inferred from data shape if not provided; use if k < k_generated
    --num_cpus <ncpu> \ # default to 48
```

### Property computation (with `zeo++`)

```bash
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
python evaluation/property.py --cif_dir <inference_dir> --ncpu <num_cpus>
```