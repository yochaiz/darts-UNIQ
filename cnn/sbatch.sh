#!/bin/bash
#SBATCH -c 3 # number of cores
#SBATCH --gres=gpu:2 # number of gpu requested
#SBATCH -J "F-BANNAS"
#SBATCH -t 03-00:00:00
#SBATCH -p gip,all
#SBATCH -w gaon6
#SBATCH --mail-user=yochaiz@cs.technion.ac.il
#SBATCH --mail-type=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

# source /etc/profile.d/modules.sh
# module load cuda
source ~/venv/bin/activate # activate python3 virtual environment
cd ~/F-BANNAS/cnn
PYTHONPATH=../ python3 sbatch.py

