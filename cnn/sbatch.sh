#!/bin/bash
#SBATCH -c 3 # number of cores
#SBATCH --gres=gpu:1 # number of gpu requested
#SBATCH -J "F-BANNAS"
#SBATCH -t 02-00:00:00
#SBATCH -p gip,all
#SBATCH -w gaon4
#SBATCH --mail-user=yochaiz@cs.technion.ac.il
#SBATCH --mail-type=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

# source /etc/profile.d/modules.sh
# module load cuda
# cd ~/DropDarts/cnn
# source ~/tf3/bin/activate # activate python3 virtual environment
export HOME=/tmp/yochaiz
cd /tmp/yochaiz/F-BANNAS/cnn
PYTHONPATH=../ python3 sbatch.py

