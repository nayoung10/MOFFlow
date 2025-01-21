from pathlib import Path
from functools import partial
import argparse
import json
import time
from pymatgen.io.cif import CifWriter
from common.relaxation import lammps_relax
from p_tqdm import p_map


def main(cif_dir, ncpu=48):
    all_files = list((Path(cif_dir)).glob("*.cif"))

    # Create save directory
    save_dir = Path(cif_dir) / "relaxed"
    save_dir.mkdir(exist_ok=True, parents=True)

    def relax_mof(ciffile):
        name = ciffile.parts[-1].split(".")[0]
        try:
            struct, relax_info = lammps_relax(str(ciffile), str(save_dir))
        except TimeoutError:
            return None

        # Save relaxed structure
        if struct is not None:
            struct = struct.get_primitive_structure()
            CifWriter(struct).write_file(save_dir / f"{name}.cif")
            relax_info["natoms"] = struct.frac_coords.shape[0]
            relax_info["path"] = str(save_dir / f"{name}.cif")
            return relax_info
        else:
            return None

    # Relax MOFs
    start_time = time.time()
    results = p_map(relax_mof, all_files, num_cpus=ncpu)
    elapsed_time = time.time() - start_time
    print(f"INFO:: MOFs relaxed in {elapsed_time} seconds")
    with open(save_dir / "time_relax.txt", "w") as f:
        f.write(f"Time taken: {elapsed_time} seconds\n")
    
    # Save relax info
    relax_infos = [x for x in results if x is not None]
    with open(save_dir / "relax_info.json", "w") as f:
        json.dump(relax_infos, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif_dir", type=str)
    parser.add_argument("--ncpu", type=int, default=48)
    args = parser.parse_args()
    main(cif_dir=args.cif_dir, ncpu=args.ncpu)