import argparse
import itertools
import os
from os.path import join
import socket
import time
import sys

def get_expdir(fname):
    hostname = socket.gethostname()
    if hostname == "Roberts-MacBook-Pro.local":
        expdir = "test_expdir"
    elif hostname == "vremote":
        expdir = join(join("/scratch/hdd001/home/adragnar/experiments/civil_comments", fname), str(time.time()))
    elif hostname ==  "q.vector.local":
        expdir = join(join("/scratch/gobi1/adragnar/experiments/civil_comments", fname), str(time.time()))
    else:
        raise Exception('Unrecognized Computer')

    os.makedirs(expdir)
    return expdir

def get_datapath():
    hostname = socket.gethostname()
    if hostname == "Roberts-MacBook-Pro.local":
        return "data/all_data.csv"
    else:
        return "~/civil_comments/data/all_data.csv"

def get_wordvecspath():
    hostname = socket.gethostname()
    if hostname == "Roberts-MacBook-Pro.local":
        return "data/crawl-300d-2M.vec"
    else:
        return "~/civil_comments/data/crawl-300d-2M.vec"


def setup():
    expdir = get_expdir('cmnist')
    data_fname = get_datapath()
    seeds = [10000, 8079, 501]
    splits = [0, 1, 2]
    label_noise = [0, 0.25]

    cmdfile = join(expdir, 'cmdfile.sh')
    with open(cmdfile, 'w') as cmdf:
        for id, combo in enumerate(itertools.product(seeds, splits, label_noise)):
            command_str = \
            '''python main.py {id} {expdir} {data_fname} {seed} {env_split} {label_noise}\n'''
            command_str = command_str.format(
                id=id,
                expdir=expdir,
                data_fname=data_fname,
                seed=combo[0],
                env_split=combo[1],
                label_noise=combo[2]
            )
            cmdf.write(command_str)

    #Return cmdfile name
    sys.stdout.write(cmdfile)
    sys.stdout.flush()

if __name__ == '__main__':
    setup()
