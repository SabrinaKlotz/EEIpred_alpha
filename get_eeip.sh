#!/bin/bash

# Set variables
dataset="EXAMPLE"
###########if dataset is not EXAMPLE, then CHANGE DATASET in dmasif/data.py line 276################
### if an error occurs, please make sure to delete surface_data1/processed folder and run the code again
fold="1"
cuda_no="0" # define your gpu number

# Change directory
cd dmasif

# create directory for the npy files if not exists
mkdir -p surface_data/raw/01-benchmark_surfaces_npy

# Convert PDB to NPY files
python convert_pdb2npy.py

# Run dMaSIF predictions
python -W ignore -u main_inference_exon.py --experiment_name "$dataset/dMaSIF_search_3layer_12_${fold}_$dataset" --batch_size 1 --embedding_layer dMaSIF --search True --device cuda:"$cuda_no" --radius 12.0 --emb_dim 16 --n_layers 3 --train_no "$fold"
# Run preprocessing for ProteinMAE command: 
#python -W ignore -u main_inference_exon.py --ppPMAE True --experiment_name "$dataset/dMaSIF_search_3layer_12_${fold}_$dataset" --batch_size 1 --embedding_layer dMaSIF --search True --device cuda:"$cuda_no" --radius 12.0 --emb_dim 16 --n_layers 3 --train_no "$fold"
# Change directory
cd ..

# create directory for the npy files if not exists
mkdir -p results/dMaSIF_DL

# Run PPDL
python ppDL.py -mth dmasif -d "$dataset" -f "$fold"

# Display the location of the results
path_results="results/dMaSIF_DL/${dataset}_fold${fold}_results.csv"

echo "Please find your EEI prediction in $path_results"