#!/bin/bash

#SBATCH --job-name=epickitchens_2
#SBATCH --partition=learnai4rl
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --chdir=./
#SBATCH --output=./logs/out/epickitchens_%j_infer2.out

### the command to run
srun python metrics/inference.py --config $1 --ckpt $2 --n_chunk 8 --chunk_idx 2
