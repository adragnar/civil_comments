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

    try:
        os.makedirs(expdir)
    except FileExistsError:
        pass

    return expdir

def get_datapath():
    hostname = socket.gethostname()
    if hostname == "Roberts-MacBook-Pro.local":
        return "data/all_data.csv"
    else:
        return "~/civil_comments/data/all_data.csv"

def get_wordfreqpath():
    hostname = socket.gethostname()
    if hostname == "Roberts-MacBook-Pro.local":
        return "data/wordfreq.pkl"
    else:
        return "~/civil_comments/data/wordfreq.pkl"

def get_wordvecspath():
    hostname = socket.gethostname()
    if hostname == "Roberts-MacBook-Pro.local":
        return "data/crawl-300d-2M.vec"
    else:
        return "~/civil_comments/data/crawl-300d-2M.vec"

def hyp_setup():
    expdir = get_expdir('cmnist')
    data_fname = get_datapath()
    seeds = [10000, 8079]
    splits = [1]
    label_noise = [0.05]
    sens_att = ['LGBTQ']
    w_enc = ['embed']  #embed
    model = ['mlp']
    shift_type = ['ltype']

    lr = [0.0001, 0.01]
    niter = [10000, 50000]
    l2 = [0.0, 1.0, 5.0]
    penalty_weight = [1000, 5000, 10000]
    penalty_anneal = [1]
    hid_layers = [50, 100, 200]


    cmdfile = join(expdir, 'cmdfile.sh')
    with open(cmdfile, 'w') as cmdf:
        for id, combo in enumerate(itertools.product(seeds, splits, label_noise, \
                          sens_att, w_enc, model, shift_type, lr, niter, l2, penalty_weight, \
                          penalty_anneal, hid_layers)):
            command_str = \
            '''python main.py {id} {expdir} {data_fname} {seed} {env_split} {label_noise} {sens_att} {w_enc} {model} \
               {stype} -inc_hyperparams 1 -lr {lr} -niter {niter} -l2 {l2} -penalty_weight {penwgt} -penalty_anneal {penann} \
               -hid_layers {hid}\n'''
            command_str = command_str.format(
                id=id,
                expdir=expdir,
                data_fname=data_fname,
                seed=combo[0],
                env_split=combo[1],
                label_noise=combo[2],
                sens_att=combo[3],
                w_enc=combo[4],
                model=combo[5],
                stype=combo[6],
                lr=combo[7],
                niter=combo[8],
                l2=combo[9],
                penwgt=combo[10],
                penann=combo[11],
                hid=combo[12]
            )
            cmdf.write(command_str)

    #Return cmdfile name
    sys.stdout.write(cmdfile)
    sys.stdout.flush()


def setup():
    expdir = get_expdir('cmnist')
    data_fname = get_datapath()
    seeds = [10000, 8079, 501]
    splits = [0, 1, 2]
    label_noise = [0, 0.10, 0.25]
    sens_att = ['LGBTQ', 'muslim']
    w_enc = ['embed']  #embed
    model = ['logreg']
    shift_type = ['lshift']  #lshift, zshift 

    cmdfile = join(expdir, 'cmdfile.sh')
    with open(cmdfile, 'w') as cmdf:
        for id, combo in enumerate(itertools.product(seeds, splits, label_noise, sens_att, w_enc, model, shift_type)):
            command_str = \
            '''python main.py {id} {expdir} {data_fname} {seed} {env_split} {label_noise} {sens_att} {w_enc} {model} {stype}\n'''
            command_str = command_str.format(
                id=id,
                expdir=expdir,
                data_fname=data_fname,
                seed=combo[0],
                env_split=combo[1],
                label_noise=combo[2],
                sens_att=combo[3],
                w_enc=combo[4],
                model=combo[5],
                stype=combo[6]
            )
            cmdf.write(command_str)

    #Return cmdfile name
    sys.stdout.write(cmdfile)
    sys.stdout.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument("hparams", type=int, default=None)
    args = parser.parse_args()
    if args.hparams == 0:
        setup()
    else:
        hyp_setup()
