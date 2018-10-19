from os import getpid
from sys import executable
from time import sleep
from subprocess import Popen
from datetime import datetime
from signal import signal, SIGTERM


def handleSIGTERM(procs):
    def sigHandler(signal, frame):
        for p in procs:
            p.send_signal(SIGTERM)

    signal(SIGTERM, sigHandler)


now = datetime.now()
outputFile = '{}.out'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

commands = [
    [executable, './train_search.py',
     '--data', '../data/', '--batch_size', '250', '--arch_learning_rate', '1.0', '--lmbda', '10.0', '--dataset', 'cifar10',
     '--bitwidth', '3#8', '--baselineBits', '3', '--epochs', '1', '--bopsCounter', 'discrete', '--alphas_data_parts', '2',
     '--model', 'resnet', '--learning_rate', '0.01', '--nCopies', '1', '--train_portion', '0.5', '--gpu', '0,1',
     '--grad_estimator', 'layer_same_path', '--alphas_regime', 'alphas_weights_loop', '--workers', '2', '--nSamples', '200',
     '--pre_trained', '../pre_trained/cifar10/train_portion_0.5/resnet_w:[8]_a:[8]/train/model_checkpoint.updated_stats.pth.tar'
     ]  #
]

# '../pre_trained/cifar10/train_portion_0.5/resnet_w:[8]_a:[8]/train/model_checkpoint.updated_stats.pth.tar'
# '../pre_trained/cifar100/train_portion_0.5/resnet_w:[8]_a:[8]/train/model_opt.updated_stats.pth.tar'

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
