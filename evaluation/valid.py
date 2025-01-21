import os
import re
from pathlib import Path
from collections import defaultdict
from p_tqdm import p_map
import argparse

import torch

from mofchecker import MOFChecker

EXPECTED_CHECK_VALUES = {
    "has_carbon": True,
    "has_hydrogen": True,
    "has_atomic_overlaps": False,
    "has_overcoordinated_c": False,
    "has_overcoordinated_n": False,
    "has_overcoordinated_h": False,
    "has_undercoordinated_c": False,
    "has_undercoordinated_n": False,
    "has_undercoordinated_rare_earth": False,
    "has_metal": True,
    "has_lone_molecule": False,
    "has_high_charges": False,
    "is_porous": True,
    "has_suspicicious_terminal_oxo": False,
    "has_undercoordinated_alkali_alkaline": False,
    "has_geometrically_exposed_metal": False,
    "has_3d_connected_graph": True
}

def check_criteria(descriptors: dict, expected_values: dict, verbose=False):
    """
    Returns:
    - True if all expected values match the descriptors
    - False if any expected value does not match the descriptors
    """
    for key, expected_value in expected_values.items():
        if descriptors[key] != expected_value:
            if verbose: print(f"Mismatch found for {key}: expected {expected_value}, found {descriptors[key]}")
            return False, key
    return True, None

def process_cif_file(mof_path):
    mofchecker = MOFChecker.from_cif(mof_path)
    descriptors = mofchecker.get_mof_descriptors()
    valid, info = check_criteria(descriptors, EXPECTED_CHECK_VALUES)
    return valid, info

def process_one(mof_path):
    try:
        result = process_cif_file(mof_path)
        return result
    except TimeoutError:
        return (False, "Timeout")
    except Exception as e:
        return (False, str(e))
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cif_dir', type=str, default="PATH/cif")
    parser.add_argument('--prefix', type=str, default="pred")
    parser.add_argument('--save_pt', type=str, default=None)
    
    args = parser.parse_args()
    cif_dir = Path(args.cif_dir)
    
    # Collect all pred.cif files
    pattern = re.compile(r'sample_\d+')
    cif_files = sorted(
        [str(cif_dir / d / "pred.cif") for d in os.listdir(cif_dir) if pattern.match(d)],
        key=lambda x: int(x.split('/')[-2].split('_')[-1])
    )
    
    results = defaultdict(list)
        
    # Process files in parallel
    p_results = p_map(process_one, cif_files, num_cpus=16)
    
    # Separate results into valid and info
    for valid, info in p_results:
        results["valid"].append(valid)
        results["info"].append(info)
    
    # Compute statistics
    num_valid = sum(results["valid"])
    print(f"Percentage of valid MOFs: {num_valid / len(results['valid']) * 100}%")
    
    # Save results
    if args.save_pt is not None:
        print("INFO:: Saving results to", args.save_pt)
        samples = torch.load(args.save_pt)
        samples["valid_check"] = results
        torch.save(samples, args.save_pt)
    else:
        print("INFO:: Saving results to", os.path.join(cif_dir, "valid_check.pt"))
        torch.save(results, os.path.join(cif_dir, "valid_check.pt"))

if __name__ == "__main__":
    main()
