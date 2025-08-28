"""
Converts tensor data to CIF format. Expects input format to be equal to that of evaluate.py.
"""
import os
import glob
import json
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from torch_geometric.data import Batch
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.cif import CifWriter
from pymatgen.analysis.structure_matcher import StructureMatcher


def merge_prediction_files(pt_files):
    """
    Merges multiple rank-specific .pt prediction files into a single dictionary.
    """
    all_cart_coords = []
    all_num_atoms = []
    all_atom_types = []
    all_lattices = []
    all_gt_data = []

    for f in tqdm(pt_files, desc="Loading prediction files"):
        data = torch.load(f, map_location='cpu')

        all_cart_coords.append(data["cart_coords"])  # [k, num_atoms, 3]
        all_num_atoms.append(data["num_atoms"])      # [k, num_graphs]
        all_atom_types.append(data["atom_types"])    # [k, num_atoms]
        all_lattices.append(data["lattices"])        # [k, num_graphs, 6]

        gt_data = data["gt_data_batch"].to_data_list()
        all_gt_data.extend(gt_data)

    merged = {
        "cart_coords": torch.cat(all_cart_coords, dim=1),   # [k, total_atoms, 3]
        "num_atoms": torch.cat(all_num_atoms, dim=1),       # [k, total_graphs]
        "atom_types": torch.cat(all_atom_types, dim=1),     # [k, total_atoms]
        "lattices": torch.cat(all_lattices, dim=1),         # [k, total_graphs, 6]
        "gt_data_batch": Batch.from_data_list(all_gt_data, exclude_keys=["rotable_atom_mask"])
    }

    return merged

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


def get_structure(cart_coords, atom_types, lattice):
    try:
        structure = Structure(
            lattice=Lattice.from_parameters(*lattice),
            species=atom_types,
            coords=cart_coords,
            coords_are_cartesian=True
        )
        return structure
    except Exception as e:
        print(f"Error processing structure: {e}")
        return None

def process_one(i, pred, save_dir):
    cif_file = save_dir / f"pred_{i}.cif"
    structure = get_structure(pred['cart_coords'], pred['atom_types'], pred['lattice'])
    if structure is not None:
        CifWriter(structure).write_file(cif_file)

def main(args):
    # Find all .pt files
    pt_files = sorted(glob.glob(args.save_pt))
    assert pt_files, f"No files matched pattern {args.save_pt}"

    # Merge
    results = merge_prediction_files(pt_files)

    # Prepare ground truth data
    gt_batch = results['gt_data_batch'].to('cpu')
    gt_list = split_data(
        gt_batch['real_gt_coords'],
        gt_batch['atom_types'],
        gt_batch['lattice_1'],
        gt_batch['num_atoms'].tolist()
    )

    # Save as cif
    save_dir = Path(args.save_pt).parent / "cif" / f"gt"
    save_dir.mkdir(exist_ok=True, parents=True)

    Parallel(n_jobs=args.num_cpus)(
        delayed(process_one)(i, gt, save_dir)
        for i, gt in tqdm(enumerate(gt_list), total=len(gt_list), desc=f"Saving CIF files for ground-truth")
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
    
    # Save as cif
    for k in range(args.num_samples):
        save_dir = Path(args.save_pt).parent / "cif" / f"trial_{k}"
        save_dir.mkdir(exist_ok=True, parents=True)

        Parallel(n_jobs=args.num_cpus)(
            delayed(process_one)(i, pred, save_dir)
            for i, pred in tqdm(enumerate(pred_lists[k]), total=len(pred_lists[k]), desc=f"Saving CIF files for trial {k}")
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_pt', type=str, required=True) # "PATH/TO/*.pt"
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--num_cpus', type=int, default=1)
    args = parser.parse_args()
    main(args)