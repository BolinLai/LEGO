#!/bin/bash
# $1 -- path to checkpoint

echo Path:$1

config='configs/generate_kitchen.yaml'

for i in 1 2 3 4 5 6 7 8; do
    sbatch sbatch_inference/epickitchen_inference/test_epickitchen_sbatch_$i.sh $config $1
done
