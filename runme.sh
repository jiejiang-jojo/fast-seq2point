DATASET_DIR="/vol/vssp/msos/qk/workspaces/energy_disaggregation/REFIT"
WORKSPACE="/vol/vssp/msos/qk/workspaces/energy_disaggregation"

# Pack csv files to hdf5
python pack_csv_to_hdf5.py --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# Train
CUDA_VISIBLE_DEVICES=1 python main_pytorch.py train --workspace=$WORKSPACE --cuda

# Inference
CUDA_VISIBLE_DEVICES=1 python main_pytorch.py inference --workspace=$WORKSPACE --iteration=1000 --cuda