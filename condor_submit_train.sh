#!/bin/bash

# To use this script a template condor job description is needed.
# It should contain everything (including the executable path) but aruguments
# which will be supplied by this script.

if [ "$#" -lt 3 ]; then
  echo 'condor_submit_train.sh <template> <config file> <conda env path> [--pm- ...]'
  exit
fi

TEMPLATE="$1"
shift

CONFIG="$1"
shift

CONDA_ENV_PATH="$1"
shift

WORKSPACE="$(pwd)"

ARGUMENTS="arguments = $WORKSPACE/$CONFIG $CONDA_ENV_PATH $WORKSPACE $*"

echo "$ARGUMENTS"
condor_submit -a "$ARGUMENTS" "$TEMPLATE"