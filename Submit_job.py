import time
from subprocess import Popen
import shlex
import os
list = ['Deepwalk','n2v','Walklet','mnmf','boost','diff','LEM','netmf','GAE','randne']

def File(algo,i):
    return f'''#!/bin/bash

## specify the job and project name
#SBATCH --job-name={algo}
#SBATCH --account=jgb21

## specify the required resources
#SBATCH --partition=arza
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000
#SBATCH --time=1-00:00:00

module purge
module load python/3.10
python Main.py --algo={algo} --Thread={i}
'''
for algo in list:
    for i in range(30):
        if os.path.exists(f'Label_mu/0.7/{algo}/{i}.npy'):
            continue
        with open('temp.sh','w')as F:
            F.write(File(algo,i))
        cmd = shlex.split('sbatch temp.sh')
        process = Popen(cmd)
        process.wait()
        time.sleep(1)