#!/bin/bash
#SBATCH --job-name=gpu_example
#SBATCH --partition=debug       # Specify the GPU partition
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=1              # Number of tasks (total)
#SBATCH --cpus-per-task=16       # Number of CPU cores (threads) per task
#SBATCH --mem-per-cpu=8G        # Memory limit per CPU core (there is no --mem-per-task)
#SBATCH --time=24:00:00         # Job timeout
#SBATCH --output=sbatch_log/debug.log      # Redirect stdout to a log file
#SBATCH --error=sbatch_log/myjob.error     # Redirect stderr to a separate error log file
#SBATCH --mail-type=ALL         # Send updates via email



JOB_START_TIME=$(date)
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}" 
echo "Running on node: $(hostname)"
echo "Starting on:     ${JOB_START_TIME}" 

python -m experiments.generate_cifar_dataset_siren output_dir=./outputs/cifar_inrs_data/train/ data.train=true

python -m experiments.generate_cifar_dataset_siren output_dir=./outputs/cifar_inrs_data/val/ data.train=false
