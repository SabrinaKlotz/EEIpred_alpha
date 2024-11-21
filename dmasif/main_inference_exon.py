# Standard imports:
import numpy as np
import torch

from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose
from pathlib import Path
import os

# Custom data loader and model:
from data import ProteinPairsSurfaces, PairData, CenterPairAtoms, load_protein_pair
from data import RandomRotationPairAtoms, NormalizeChemFeatures, iface_valid_filter
from model import dMaSIF
# add an option to save the pre-processed point clouds on hard drive to run ProteinMAE
from data_iteration_protmae import iterate_protmae
from data_iteration_exon import iterate
from helper import *
from Arguments import parser
import shutil


##############################################################################
# mode is test, change to train for background model
##############################################################################
mode = "test"
if mode == "test":
    b_mode = False
else:
    b_mode = True

args = parser.parse_args()
model_path = "models/" + args.experiment_name
save_predictions_path = Path("preds/" + args.experiment_name)

if not os.path.exists(f"surface_data{args.train_no}"):
    os.makedirs(f"surface_data{args.train_no}")

# Ensure reproducability:
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

# Load the train and test datasets:
transformations = (
    Compose([NormalizeChemFeatures(), CenterPairAtoms(), RandomRotationPairAtoms()])
    if args.random_rotation
    else Compose([NormalizeChemFeatures()])
)

if args.single_pdb != "":
    single_data_dir = Path("./data_preprocessing/npys/")
    test_dataset = [load_protein_pair(args.single_pdb, single_data_dir,single_pdb=True)]
    test_pdb_ids = [args.single_pdb]
elif args.pdb_list != "":
    with open(args.pdb_list) as f:
        pdb_list = f.read().splitlines()
    single_data_dir = Path("./data_preprocessing/npys/")
    test_dataset = [load_protein_pair(pdb, single_data_dir,single_pdb=True) for pdb in pdb_list]
    test_pdb_ids = [pdb for pdb in pdb_list]
else:
    ###############################
    #change train to true for background model#
    ###############################
    test_dataset = ProteinPairsSurfaces(
        f"surface_data{args.train_no}", ppi=args.train_no, train=b_mode, transform=None
    )
    # change TRAIN TEST HERE
    test_pdb_ids = np.load(f"surface_data{args.train_no}/processed/{mode}ing_pairs_data_ids_ppi.npy")

# PyTorch geometric expects an explicit list of "batched variables":
batch_vars = ["xyz_p1", "xyz_p2", "atom_coords_p1", "atom_coords_p2"]
test_loader = DataLoader(
    test_dataset, batch_size=args.batch_size, follow_batch=batch_vars
)

net = dMaSIF(args)
# net.load_state_dict(torch.load(model_path, map_location=args.device))
net.load_state_dict(
    torch.load(model_path, map_location=args.device)["model_state_dict"]
)
net = net.to(args.device)

exon_dir = Path("../data_collection/uniprot_EnsemblExonPDB_map/")
lists_dir = Path("../data_collection/cv_splits/")
##############################################################################
# change path to your pdb files including hydrogen atoms
##############################################################################
pdb_dir = Path("surface_data/raw/01-benchmark_pdbs/")
dataset_name = args.experiment_name.split("/")[0]
# load files ones
# change TRAIN TEST HERE
with open(lists_dir / f"{dataset_name}/{mode}_info{args.train_no}.txt") as f_tr:
    testing_protein_names = f_tr.read().splitlines()

if not os.path.exists(f"results/{dataset_name}/fold{args.train_no}/{mode}/"):
    os.makedirs(f"results/{dataset_name}/fold{args.train_no}/{mode}/")

# Perform one pass through the data:
if not args.ppPMAE:
    info = iterate(
        net,
        test_loader,
        None,
        args,
        test=True,
        save_path=save_predictions_path,
        pdb_ids=test_pdb_ids,
        test_info=testing_protein_names,
        exon_dir=exon_dir,
        pdb_dir=pdb_dir,
        dataset_name=args.experiment_name.split("/")[0],
        mode=mode,
    )
else:
    info = iterate_protmae(
        net,
        test_loader,
        None,
        args,
        test=True,
        save_path=save_predictions_path,
        pdb_ids=test_pdb_ids,
        test_info=testing_protein_names,
        exon_dir=exon_dir,
        pdb_dir=pdb_dir,
        dataset_name=args.experiment_name.split("/")[0],
        mode=mode,
    )

# remove the processed files to process different datasets
shutil.rmtree(Path("surface_data" + str(args.train_no) + "/processed"))
