from os import getpid
from sys import executable
from time import sleep
from subprocess import Popen
from datetime import datetime
from signal import signal, SIGTERM
from shutil import copy2


def handleSIGTERM(procs):
    def sigHandler(signal, frame):
        for p in procs:
            p.send_signal(SIGTERM)

    signal(SIGTERM, sigHandler)


now = datetime.now()
outputFile = '{}.out'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

commands = [
    [executable, './train_search.py', '--gpu', '0',
     '--epochs', '5', '--batch_size', '100',
     '--bitwidth', '1,2,3,4,5', '--MaxBopsBits', '3',
     '--bopsCounter', 'continuous',
     '--pre_trained', '/home/yochaiz/UNIQ/results/resnet_cifar10_trained_32_bit_deeper/model_best.pth.tar',
     '--data', '/home/yochaiz/UNIQ/results/'
     ]
]

# # resume training
# folders = ['2018-07-16_16-31-25']
# commands = []
# for f in folders:
#     commands.append([executable,
#                      './results/{}/code/ddpg/main-ddpg.py'.format(f),
#                      '--DDPG-checkpoint', './results/{}'.format(f),
#                      '--gpus', '0'])

# ===========================================================================

procs = []

with open(outputFile, mode='w') as out:
    # log pid
    out.write('PID:[{}]\n'.format(getpid()))
    # run processes
    for cmd in commands:
        # copy2(file, dstFile)
        # out.write('copied [{}] to [{}]'.format(file, dstFile))
        p = Popen(cmd, stdout=out, stderr=out)
        procs.append(p)
        # print command
        out.write('***{}***\n'.format(cmd))
        # pause until next command
        sleep(30)

# handle sbatch scancel
handleSIGTERM(procs)

for p in procs:
    p.wait()

# switch files
# dstFile = 'Policy.py'
# commands = [
#     ([sys.executable, './train.py', '0', '--random', '--k', '32', '--desc', '"step 13 with critic main model"'],
#      'ReplayBuffer-1.py'),
#     ([sys.executable, './train.py', '0', '--random', '--k', '32', '--desc', '"step 13 with critic & actor main model"'],
#      'ReplayBuffer-2.py')
# ]
