#!/bin/bash

if [ "$#" -lt 3 ]; then
  echo "USAGE: run_train.sh <config_path> <conda_env_path> <workspace>"
  exit
fi

CONDA_ENV_NAME=py3_pytorch

CONFIG="$1"
shift

# /vol/vssp/msos/qk/anaconda/envs
CONDA_ENV_PATH="$1"
if [ ! -d "$CONDA_ENV_PATH/$CONDA_ENV_NAME" ]; then
  echo "Error: Please make sure the second parameter ($CONDA_ENV_PATH) pointing to a valid conda env folder containing '$CONDA_ENV_NAME'"
  exit
fi
shift

#/vol/vssp/msos/qk/JieJiang/REFIT/edhc
WORKSPACE="$1"
shift

export PATH="$CONDA_ENV_PATH/bin:$PATH"

cd "$WORKSPACE"
source activate $CONDA_ENV_NAME

#$CONDA_ENV_PATH/$CONDA_ENV_NAME/bin/python3.6 main_pytorch.py train --config $CONFIG --workspace=$WORKSPACE --cuda $*
