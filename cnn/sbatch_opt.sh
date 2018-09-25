#!/bin/bash
#SBATCH -c 3 # number of cores
#SBATCH --gres=gpu:1 # number of gpu requested
#SBATCH -J "OPT"
#SBATCH -t 01-00:00:00
#SBATCH -p all
#SBATCH -w gaon6
#SBATCH --mail-user=yochaiz@cs.technion.ac.il
#SBATCH --mail-type=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

source /etc/profile.d/modules.sh
module load cuda
cd /home/yochaiz/DropDarts/cnn
source ~/tf3/bin/activate # activate python3 virtual environment
echo $1
echo $2
PYTHONPATH=/home/yochaiz/DropDarts python3 sbatch_opt.py $1 $2

