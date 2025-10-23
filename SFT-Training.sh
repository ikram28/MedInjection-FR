#!/bin/bash
#SBATCH --job-name=qwen-gen
#SBATCH -C h100
#SBATCH --nodes=1                  
#SBATCH --ntasks-per-node=2        
#SBATCH --gres=gpu:2               
#SBATCH --cpus-per-task=8          
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=./logs_slurm/qwen-gen%j.out 
#SBATCH --error=./logs_slurm/qwen-gen%j.err  
#SBATCH --account=



module purge


module load arch/h100
module load pytorch-gpu/py3/2.8.0



set -x -e

export OMP_NUM_THREADS=8

export CUDA_LAUNCH_BLOCKING=1


export NCCL_ASYNC_ERROR_HANDLING=1

srun -l python -u SFT-Training-Generated.py \
   --model_name="./models/Qwen3-4B-Instruct-2507/" \
   --path_train_dataset="./SFT-DATA/train_gen_balanced_chat.json" \
   --path_eval_dataset="./SFT-DATA/val_gen_balanced_chat.json" \
   --output_dir="./SFT-qwen-gen/" \
   --logging_dir="./SFT-qwen-gen-logs/" \
   --epochs=10 \
   --batch_size=12 \
   --save_steps=50 \
   --logging_steps=10 \
   --seed=42 \
   --learning_rate=1e-04
