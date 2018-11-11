from os import getpid
from sys import executable
from time import sleep
from subprocess import Popen
from datetime import datetime
from signal import signal, SIGTERM
from argparse import ArgumentParser


def handleSIGTERM(procs):
    def sigHandler(signal, frame):
        for p in procs:
            p.send_signal(SIGTERM)

    signal(SIGTERM, sigHandler)


parser = ArgumentParser()
parser.add_argument('--json', type=str, required=True, help='JSON file path')

args = parser.parse_args()

now = datetime.now()
outputFile = '{}.out'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

# parse JSONs to list
jsonFilesList = args.json.split('?')
print(jsonFilesList)

commandsList = []
for gpuID, jsonFileName in enumerate(jsonFilesList):
    command = [executable, './train_opt2.py', '--json', jsonFileName, '--gpu', '{}'.format(gpuID), '--data', '../data/']
    commandsList.append(command)

# commands = [
#     [executable, './train_opt2.py',
#      '--data', args.data,
#      '--gpu', '0'
#      ]
# ]

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
    for cmd in commandsList:
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
