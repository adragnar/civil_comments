import argparse
import itertools
import os
from os.path import join
import socket
import time
import sys
sys.path.append(join(os.getcwd(), 'launchfiles'))

import setup_params as setup

def hyp_labelgen_setup():
    expdir = setup.get_expdir('reddit_label_gen')
    data_fname = setup.get_datapath()
    #Params
    seeds = [10000, 8079]
    epochs = [450, 900]
    batch_size = [60000, 10000]
    hid_layers = [200, 500, 800]
    lr = [0.01, 0.001, 0.0001]
    l2 = [1.0, 10.0, 20.0]

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
    parser.add_argument("hparams", type=int, default=None)
    parser.add_argument("label_gen", type=int, default=None)
    args = parser.parse_args()

    if args.label_gen == 1:
        if args.hparams == 0:
            labelgen_setup()
        else:
            hyp_labelgen_setup()
