import numpy as np
from pathlib import Path
from tqdm import tqdm
from Bio.PDB import *
from Bio.PDB import PDBExceptions
import warnings
warnings.filterwarnings("ignore")
import os

ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}


def load_structure_np(fname, center):
    """Loads a .ply mesh to return a point cloud and connectivity."""
    # Load the data
    parser = PDBParser()
    structure = parser.get_structure("structure", fname)
    atoms = structure.get_atoms()

    coords = []
    types = []
    for atom in atoms:
        coords.append(atom.get_coord())
        types.append(ele2num[atom.element])

    coords = np.stack(coords)
    types_array = np.zeros((len(types), len(ele2num)))
    for i, t in enumerate(types):
        types_array[i, t] = 1.0

    # Normalize the coordinates, as specified by the user:
    if center:
        coords = coords - np.mean(coords, axis=0, keepdims=True)

    return {"xyz": coords, "types": types_array}

def convert_pdbs(pdb_dir, npy_dir):
    print("Converting PDBs")
    count = 0
    list_not_npy = []
    count_given = 0
    print(len(list(pdb_dir.glob("*.pdb"))))
    for p in tqdm(pdb_dir.glob("*.pdb")):
        if os.path.exists(npy_dir / (p.stem + "_atomxyz.npy")):
            continue
        else:
            try:
                protein = load_structure_np(p, center=False)
                count_given += 1
            except PDBExceptions.PDBConstructionException: # Chain name is a problem, has to be letter at the end not number
                print("PDBConstructionException", p)
                count += 1
                list_not_npy.append(p.stem)
                continue
            except ValueError:
                print("Could not load ValueError", p)
                count += 1
                list_not_npy.append(p.stem)
                continue

            np.save(npy_dir / (p.stem + "_atomxyz.npy"), protein["xyz"])
            np.save(npy_dir / (p.stem + "_atomtypes.npy"), protein["types"])
    #print(count, list_not_npy, count_given)

if __name__ == "__main__":
    #pdb_dir = Path("surface_data/raw/01-benchmark_pdbs/") # Path with PDB files
    pdb_dir = Path("../data_collection/pdb/")
    npy_dir = Path("surface_data/raw/01-benchmark_surfaces_npy/") # Path to save npy files

    convert_pdbs(pdb_dir, npy_dir)