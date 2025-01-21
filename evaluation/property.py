import argparse
import json
import time
from pathlib import Path
from functools import partial
from p_tqdm import p_map
from mofdiff.common.mof_utils import mof_properties


def main(cif_dir, ncpu=64):
    all_files = list((Path(cif_dir)).glob("*.cif"))
    
    # Create save directory
    zeo_dir = Path(cif_dir) / "zeo"
    zeo_dir.mkdir(exist_ok=True)
    print(f"INFO:: Saving ZEO++ properties to {zeo_dir}")

    # Calculate ZEO++ properties
    start_time = time.time()
    zeo_props = p_map(
        partial(mof_properties, zeo_store_path=zeo_dir), all_files, num_cpus=ncpu
    )
    elapsed_time = time.time() - start_time
    print(f"INFO:: ZEO++ properties calculated in {elapsed_time} seconds")
    with open(Path(cif_dir) / "time_zeo.txt", "w") as f:
        f.write(f"Time taken: {elapsed_time} seconds\n")

    # Save ZEO++ properties
    zeo_props = [x for x in zeo_props if x is not None]
    with open(Path(cif_dir) / "zeo_props_relax.json", "w") as f:
        json.dump(zeo_props, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif_dir", type=str)
    parser.add_argument("--ncpu", type=int, default=48)
    args = parser.parse_args()
    main(cif_dir=args.cif_dir, ncpu=args.ncpu)