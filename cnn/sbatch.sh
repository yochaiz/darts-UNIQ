#!/bin/bash
#SBATCH -c 3 # number of cores
#SBATCH --gres=gpu:1 # number of gpu requested
#-J "20181114-104311-32-1"
#SBATCH -t 03-00:00:00
# -p gip,all
# -w gaon2
#SBATCH --mail-user=yochaiz@cs.technion.ac.il
#SBATCH --mail-type=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

source /etc/profile.d/modules.sh
module load cuda
source ~/venv/bin/activate # activate python3 virtual environment
cd ~/F-BANNAS/cnn
PYTHONPATH=../ python3 sbatch.py

