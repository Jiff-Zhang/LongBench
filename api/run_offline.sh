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

if [[ $# -lt 7 ]]; then
    echo "Usage: bash $0 <model> <model_path> <outdir> <dist_init_addr> <node_rank> <nnodes> <tp> *args **kwargs"
    exit
fi

model=$1 && shift
model_path=$1 && shift
outdir=$1 && shift
dist_init_addr=$1 && shift
node_rank=$1 && shift
nnodes=$1 && shift
tp=$1 && shift
args_kwargs=$*

random_seed=1234

export GLOO_SOCKET_IFNAME=eth0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_LAUNCH_BLOCKING=1 \
    HF_DATASETS_OFFLINE=1 \
    SGL_ENABLE_JIT_DEEPGEMM=false \
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
# dist_init_addr="172.31.0.3:1234"
mkdir -p $outdir

    # --disable-cuda-graph \
echo """
set -e
python -u pred_offline.py \
    --model_name $model \
    --save_dir $outdir \
    --model-path $model_path \
    --random-seed ${random_seed} \
    --dist-init-addr $dist_init_addr \
    --tp $tp \
    --nnodes $nnodes \
    --node-rank ${node_rank} \
    --trust-remote-code \
    --disable-radix-cache \
    --enable-cache-report \
    --log-level debug \
    ${args_kwargs}
python -u result.py -f $outdir/$model.jsonl
""" > $outdir/cmd.sh
    # --attention-backend triton \

# nohup bash $outdir/cmd.sh >> $outdir/log &
bash $outdir/cmd.sh

echo "Pls refer to ${outdir}/log"
