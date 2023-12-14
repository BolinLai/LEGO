#!/bin/bash
# $1 -- path to checkpoint

echo Path:$1

config='configs/generate_ego4d.yaml'

for i in 1 2 3 4 5 6 7 8; do
    sbatch sbatch_inference/ego4d_inference/test_ego4d_sbatch_$i.sh $config $1
done
