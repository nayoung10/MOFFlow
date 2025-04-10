import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from p_tqdm import p_map
from pathlib import Path
from functools import partial
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher


def split_data(cart_coords, atom_types, lattices, num_atoms):
    """
    Splits concatenated data into a list of samples.
    Args:
        cart_coords (torch.Tensor): [num_atoms, 3]
        atom_types (torch.Tensor): [num_atoms]
        lattices (torch.Tensor): [num_samples, 3, 3]
        num_atoms (list): [num_samples]
            each element is the number of atoms in a sample
    Returns:
        split_list (list): [num_samples]
            each element is a dict with keys 'cart_coords', 'atom_types', 'lattice'
    """
    split_list = []
    
    cart_coords = cart_coords.split(num_atoms)
    atom_types = atom_types.split(num_atoms)
    
    for coords, atoms, lattice in zip(cart_coords, atom_types, lattices):
        split_list.append({
            'cart_coords': coords,
            'atom_types': atoms,
            'lattice': lattice
        })
    
    return split_list


def process_one(gt, pred, matcher):
    """
    Args:
        gt (dict): keys 'cart_coords', 'atom_types', 'lattice'
        pred (dict): keys 'cart_coords', 'atom_types', 'lattice'
    Returns:
        rmsd (float): RMSD between the two structures
    """
    gt_structure = Structure(
        lattice=Lattice.from_parameters(*gt['lattice']),
        species=gt['atom_types'],
        coords=gt['cart_coords'],
        coords_are_cartesian=True
    )
    pred_structure = Structure(
        lattice=Lattice.from_parameters(*pred['lattice']),
        species=pred['atom_types'],
        coords=pred['cart_coords'],
        coords_are_cartesian=True
    )
    
    rmsd = matcher.get_rms_dist(gt_structure, pred_structure)
    rmsd = rmsd if rmsd is None else rmsd[0]
    
    return rmsd

def main(args):
    # Load the results
    results = torch.load(args.save_pt)
    
    # Prepare ground truth data
    gt_batch = results['gt_data_batch'].to('cpu')
    gt_list = split_data(
        gt_batch['gt_coords'],
        gt_batch['atom_types'],
        gt_batch['lattice_1'],
        gt_batch['num_atoms'].tolist()
    )
    
    # Prepare predicted data
    pred_lists = []
    if args.num_samples is None:
        args.num_samples = results['cart_coords'].shape[0]
    for k in range(args.num_samples):
        pred_list = split_data(
            results['cart_coords'][k],
            results['atom_types'][k],
            results['lattices'][k],
            results['num_atoms'][k].tolist()
        )
        pred_lists.append(pred_list)
    
    # Compute metrics
    matcher = StructureMatcher(stol=args.stol, ltol=args.ltol, angle_tol=args.angle_tol)
    rmsd_df = pd.DataFrame(columns=[k for k in range(args.num_samples)])
    for k in range(args.num_samples):
        rmsd_list = p_map(
            partial(process_one, matcher=matcher), 
            gt_list, 
            pred_lists[k],
            num_cpus=args.num_cpus
        )
        rmsd_df[k] = rmsd_list
    
    rmsd_df['min'] = rmsd_df.min(axis=1)
    
    # Compute summary
    summary = {
        'Match rate (%)': len(rmsd_df['min'].dropna()) / len(rmsd_df['min']) * 100,
        'Mean RMSE': rmsd_df['min'].mean(),
        'Std RMSE': rmsd_df['min'].std()
    }
    print(summary)

    # Save the results
    save_dir = Path(args.save_pt).parent
    rmsd_df.to_csv(save_dir / 'rmsd.csv', index=False)
    with open(save_dir / 'summary.json', 'w') as f:
        json.dump(summary, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_pt', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--num_cpus', type=int, default=48)
    parser.add_argument('--stol', type=float, default=0.5)
    parser.add_argument('--ltol', type=float, default=0.3)
    parser.add_argument('--angle_tol', type=float, default=10)
    args = parser.parse_args()
    main(args)