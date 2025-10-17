#!/usr/bin/env bash

# ***********************************************
#      Filename: run.sh
#        Author: jiff
#         Email: Jiff_Zh@163.com
#   Description: --
#        Create: 2025-03-12 16:58:38
# Last Modified: Year-month-day
# ***********************************************

set -e

if [[ $# -lt 2 ]]; then
    echo "Usage: bash $0 <model> <outdir> *args **kwargs"
    exit
fi

model=$1 && shift
outdir=$1 && shift

# n_proc=1
# n_proc=4
# n_proc=16
# n_proc=128
temperature=0.1
max_new_tokens=128

python pred.py \
    --model $model \
    --save_dir $outdir \
    --temperature ${temperature} \
    --max_new_tokens ${max_new_tokens} \
    $* \

