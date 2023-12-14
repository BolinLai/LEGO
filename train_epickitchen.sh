#!/bin/bash
#SBATCH --job-name=epickitchen
#SBATCH --partition=learnai4rl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --chdir=/data/home/bolinlai/Projects/instruct-pix2pix
## %j is the job id, %u is the user id
#SBATCH --output=/fsx/bolinlai/Models/instruct_pix2pix/out/debug.out
## filename for job standard error output (stderr)
#SBATCH --error=/fsx/bolinlai/Models/instruct_pix2pix/out/debug.err


GPUS_PER_NODE=8

# ******** Number of total processes ********************************************************
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
# ******** Master port, address and world size MUST be passed as variables for DDP to work 
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID 
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
echo "MASTER_PORT"=$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# ********************************************************************************************

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32

export MODULEPATH=/opt/slurm/etc/files/modulesfiles/:$MODULEPATH
module purge
module load cuda/11.6 \
    nccl/2.12.7-cuda.11.6 \
    nccl_efa/1.15.1-nccl.2.12.7-cuda.11.6

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
export NCCL_DEBUG=info

LAUNCHER="accelerate launch \
    --multi_gpu \
    --num_machines $NNODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --role $SLURMD_NODENAME: \
    --rdzv_conf rdzv_backend=c10d \
    --max_restarts 0 "

echo $LAUNCHER

CMD="main.py --name debug --base configs/train_kitchen.yaml --train --gpus 0,1,2,3,4,5,6,7"


clear; srun --nodes=2 --ntasks-per-node=8 --wait=60 --kill-on-bad-exit=1 --jobid $SLURM_JOB_ID bash -c "$LAUNCHER $CMD"

echo "END TIME: $(date)"


