#!/bin/bash
#SBATCH --mail-user=lanceknight26@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --mem=500M
#SBATCH --output=test_slurm.out

python tox21_col_self.py
