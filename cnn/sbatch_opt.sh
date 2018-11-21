#!/bin/bash
#SBATCH -c 3 # number of cores
#SBATCH -t 03-00:00:00
#SBATCH --gres=gpu:1 # number of gpu requested
# -J "[0,0,0,16]"
#SBATCH -p all
# -w gaon2
#SBATCH --mail-user=yochaiz@cs.technion.ac.il
#SBATCH --mail-type=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

source /etc/profile.d/modules.sh
module load cuda
source ~/venv/bin/activate # activate python3 virtual environment
cd ~/F-BANNAS/cnn
chmod a+x sbatch_opt.py
echo $1
echo $2
PYTHONPATH=../ python3 sbatch_opt.py $1 "$2"