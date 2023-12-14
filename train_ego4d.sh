#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --partition=learnai4rl
#SBATCH --time=72:00:00

### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/data/home/bolinlai/Projects/instruct-pix2pix
#SBATCH --error=/fsx/bolinlai/Models/instruct_pix2pix/out/debug.err
#SBATCH --output=/fsx/bolinlai/Models/instruct_pix2pix/out/debug.out

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
# export MASTER_PORT=64720
# export WORLD_SIZE=16

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
# echo "NODELIST="${SLURM_NODELIST}
# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
# echo "MASTER_ADDR="$MASTER_ADDR

# pwd

### the command to run
srun python main.py --name debug --base configs/train_ego4d.yaml --train --gpus 0,1,2,3,4,5,6,7