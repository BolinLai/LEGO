#!/bin/bash

#SBATCH --job-name=ego4d_6
#SBATCH --account genai_interns
#SBATCH --qos genai_interns
#SBATCH --nodes=1
#SBATCH --ntasks=1  # should be equal to --nodes
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --chdir=./
#SBATCH --output=./logs/out/ego4d_%j_infer6.out

### the command to run
srun python metrics/inference.py --config $1 --ckpt $2 --n_chunk 8 --chunk_idx 6