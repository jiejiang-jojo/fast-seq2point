DATASET_DIR="/vol/vssp/msos/qk/workspaces/energy_disaggregation/REFIT"
WORKSPACE="/vol/vssp/msos/qk/workspaces/energy_disaggregation"

# Pack csv files to hdf5
python pack_csv_to_hdf5.py --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# 
python tmp01.py train --workspace=$WORKSPACE