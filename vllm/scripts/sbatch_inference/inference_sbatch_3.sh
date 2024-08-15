#!/bin/bash

#SBATCH --job-name=vllm_3
#SBATCH --account genai_interns
#SBATCH --qos genai_interns
#SBATCH --nodes=1
#SBATCH --ntasks=1  # should be equal to --nodes
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --chdir=./
#SBATCH --output=./vllm/out/logs/job_%j_infer3.out

### the command to run
srun python -m llava.eval.run_llava_in_loop \
    --model-path $1 \
    --image-dir $2 \
    --action-label $3 \
    --query "How does the person properly {} that is displayed in the video frame?" \
    --save-path $4 \
    --save-image-feature-path $5 \
    --save-text-feature-path $6 \
    --seed $7 \
    --num-chunks 5 \
    --chunk-idx 3