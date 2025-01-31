#!/bin/bash
#
#SBATCH --mail-user=robinett@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/robinett/GitHub/quantum/notebooks/phase2-fs-eval/outputs/%j.stdout
#SBATCH --error=/home/robinett/GitHub/quantum/notebooks/phase2-fs-eval/outputs/%j.stderr
#SBATCH --chdir=/home/robinett/GitHub/quantum/notebooks/phase2-fs-eval
#SBATCH --account=pi-sriesenfeld
#SBATCH --partition=sriesenfeld
#SBATCH --job-name=check_hostname_of_node
#SBATCH --time=0:30:00
#SBATCH --array 0-79
#SBATCH --mem=8G
python3 compute_han_statistics.py $SLURM_ARRAY_TASK_ID
