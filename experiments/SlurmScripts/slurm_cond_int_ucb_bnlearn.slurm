#!/bin/bash

# SLURM Directives
#SBATCH --job-name=alarm_cond_bandits         # Job name
#SBATCH --output=output_%j.log           # Output file (%j = Job ID)
#SBATCH --error=error_%j.log             # Error file (%j = Job ID)
#SBATCH --ntasks=1                       # Number of tasks (1 task for a single Python script)
#SBATCH --cpus-per-task=30                # Number of CPU cores per task
#SBATCH --mem=500GB                        # Memory allocation
#SBATCH --time=6-10:00:00                  # Time limit (D-HH:MM:SS)
#SBATCH --partition=gpua100             # Partition name (gpua100 -> use cs02)
#SBATCH --mail-type=ALL                  # Email notifications (BEGIN, END, FAIL, or ALL)
#SBATCH --mail-user=f.simoes@uu.nl       # Your email address

# Load Conda (if not already in your PATH)
# module load anaconda   # Use the appropriate module for Conda on your system

conda activate cond-bandits
python3 -u run_cond_int_ucb_on_bnlearn_datasets.py # -u for outputting to file constantly, for real-time monitoring

