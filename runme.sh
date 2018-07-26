#!/bin/bash

#DATASET_DIR="/vol/vssp/msos/qk/JieJiang/REFIT/edhc/csv"
DATASET_DIR="$(pwd)/ukdale"
WORKSPACE="$(pwd)"

# Pack csv files to hdf5
#python3 pack_csv_to_hdf5.py --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# Train
python3 main_pytorch.py train --workspace=$WORKSPACE --cuda

# Test
#python3 main_pytorch.py test --workspace=$WORKSPACE --iteration=21000 --cuda

# Inference
#python3 main_pytorch.py inference --workspace=$WORKSPACE --iteration=21000 --cuda
