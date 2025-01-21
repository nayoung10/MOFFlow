import os
import warnings
import argparse
from collections import defaultdict
from pathlib import Path
from functools import partial
from tqdm import tqdm
from p_tqdm import p_map

import numpy as np
import torch

from pymatgen.io.cif import CifParser
from pymatgen.analysis.structure_matcher import StructureMatcher

# Ignore warnings
warnings.filterwarnings("ignore")

def calculate_rms_dist(gt_mof_path, pred_mof_path, matcher):
    try:
        parser = CifParser(gt_mof_path)
        gt_structure = parser.get_structures()[0]

        parser = CifParser(pred_mof_path)
        pred_structure = parser.get_structures()[0]

        rms_dist = matcher.get_rms_dist(gt_structure, pred_structure)
        rms_dist = None if rms_dist is None else rms_dist[0]
    except Exception as e:
        print(f"Error: {e}")
        rms_dist = None
    return rms_dist


def main(cif_dir, gt_prefix="gt", pred_prefix="pred", num_cpus=48):
    
    # Get gt.cif and pred.cif files
    sample_dirs = sorted(Path(cif_dir).glob('sample_*'), key=lambda x: int(x.stem.split('_')[-1]))
    paired_files = [(d / f'{gt_prefix}.cif', d / f'{pred_prefix}.cif' if (d / f'{pred_prefix}.cif').exists() else None) for d in sample_dirs]
        
    # Compute RMSD and match rate
    matcher = StructureMatcher(stol=0.5, ltol=0.3, angle_tol=10) # Default values
    results = defaultdict(list)
    
    # Parallel processing
    p_results = p_map(
        partial(calculate_rms_dist, matcher=matcher),
        [gt for gt, pred in paired_files],
        [pred for gt, pred in paired_files],
        num_cpus=num_cpus
    )

    results["rms_dist"] = p_results
    results["match_rate"] = sum(rms is not None for rms in results["rms_dist"]) / len(results["rms_dist"]) * 100

    # Print results
    print("Average RMSD:", np.mean([rms for rms in results["rms_dist"] if rms is not None]))
    print("Match rate (%):", results["match_rate"])

    # Save results
    print("INFO:: Saving results to", cif_dir)
    torch.save(results, Path(cif_dir) / "eval_metrics.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cif_dir', type=str, default="PATH/inference")
    parser.add_argument('--gt_prefix', type=str, default="gt")
    parser.add_argument('--pred_prefix', type=str, default="pred")
    parser.add_argument('--num_cpus', type=int, default=48)
    args = parser.parse_args()

    main(args.cif_dir, args.gt_prefix, args.pred_prefix, args.num_cpus)
