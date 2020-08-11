import argparse
import itertools
import os
from os.path import join
import socket
import time
import sys
sys.path.append(join(os.getcwd(), 'launchfiles'))

import setup_params as setup

def get_reddit_datapath(v):
    if v == 'gendered':
        return "reddit/data/labeled/2014_gendered_labeled.csv"
    elif v == 'baseline':
        return "reddit/data/labeled/2014b_labeled.csv"
    else:
        assert False


def hyp_subreddit_oodgen_setup():
    expdir = setup.get_expdir('subreddit_oodgen')
    data_fname = join(os.getcwd(), get_reddit_datapath('gendered'))
    #Params
    seeds = [10000, 8079, 501]
    epochs = [50]
    n_batches = [2000]
    hid_layers = [1]
    lr = [0.001, 0.0001]
    l2 = [0.1, 1.0]
    pen_wgt = [1000, 5000, 10000]
    pen_ann = [1]

    cmdfile = join(expdir, 'cmdfile.sh')
    with open(cmdfile, 'w') as cmdf:
        for id, combo in enumerate(itertools.product(seeds, epochs, n_batches, \
                                       hid_layers, lr, l2, pen_wgt, pen_ann)):
            command_str = \
            '''python reddit/r_main.py {id} {expdir} {data_fname} {l_noise} -seed {seed} -inc_hyperparams 1 -epochs {epoch} -n_batches {batch} -hid_layers {hid} -lr {lr} -l2 {l2} -pen_wgt {pen_wgt} -pen_ann {pen_ann}\n'''
            command_str = command_str.format(
                id=id,
                expdir=expdir,
                data_fname=data_fname,
                l_noise=0,
                seed=combo[0],
                epoch=combo[1],
                batch=combo[2],
                hid=combo[3],
                lr=combo[4],
                l2=combo[5],
                pen_wgt=combo[6],
                pen_ann=combo[7]
            )
            cmdf.write(command_str)

    #Return cmdfile name
    sys.stdout.write(expdir)
    sys.stdout.flush()


#LABELGEN
def hyp_labelgen_setup():
    expdir = setup.get_expdir('reddit_label_gen')
    data_fname = setup.get_datapath()
    #Params
    seeds = [10000, 8079]
    epochs = [150, 300]
    batch_size = [5000, 500]
    hid_layers = [200, 500]
    lr = [0.01, 0.001, 0.0001]
    l2 = [0.1, 1.0, 5.0]

    cmdfile = join(expdir, 'cmdfile.sh')
    with open(cmdfile, 'w') as cmdf:
        for id, combo in enumerate(itertools.product(seeds, epochs, batch_size, hid_layers, lr, l2)):
            command_str = \
            '''python reddit/r_labelgen.py {id} {expdir} {data_fname} -inc_hyperparams 1 -seed {seed} -epochs {epoch} -batch_size {batch} -hid_layers {hid} -lr {lr} -l2 {l2}\n'''
            command_str = command_str.format(
                id=id,
                expdir=expdir,
                data_fname=data_fname,
                seed=combo[0],
                epoch=combo[1],
                batch=combo[2],
                hid=combo[3],
                lr=combo[4],
                l2=combo[5]
            )
            cmdf.write(command_str)

    #Return cmdfile name
    sys.stdout.write(expdir)
    sys.stdout.flush()

# def labelgen_setup():
#     expdir = setup.get_expdir('reddit_label_gen')
#     data_fname = setup.get_datapath()
#     #Params
#     seed = [10000, 8079, 501]
#     epochs = [5000, 10000, 50000]
#     batch_size = [10000, 40000]
#     hid_layers = [200, 500, 800]
#     lr = [0.01, 0.001, 0.0001]
#     l2 = [1.0, 10.0, 20.0]
#
#     cmdfile = join(expdir, 'cmdfile.sh')
#     with open(cmdfile, 'w') as cmdf:
#         for id, combo in enumerate(itertools.product(seeds, label_noise, sens_cat, w_enc)):
#             command_str = \
#             '''python intra_main.py {id} {expdir} {data_fname} {seed} {label_noise} {sens_cat} {w_enc}\n'''
#             command_str = command_str.format(
#                 id=id,
#                 expdir=expdir,
#                 data_fname=data_fname,
#                 seed=combo[0],
#                 label_noise=combo[1],
#                 sens_cat=combo[2],
#                 w_enc=combo[3]
#             )
#             cmdf.write(command_str)
#
#     #Return cmdfile name
#     sys.stdout.write(expdir)
#     sys.stdout.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument("label_gen", type=int, default=None)
    parser.add_argument("hparams", type=int, default=None)
    args = parser.parse_args()

    if args.label_gen == 0:
        if args.hparams == 0:
            subreddit_oodgen_setup()
        else:
            hyp_subreddit_oodgen_setup()
    elif args.label_gen == 1:
        if args.hparams == 0:
            labelgen_setup()
        else:
            hyp_labelgen_setup()
