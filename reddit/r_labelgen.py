import argparse
import os
from os.path import join, dirname

import sys

sys.path.append(join(os.getcwd(), 'launchfiles'))
sys.path.append(os.getcwd())

import copy
import pickle
import socket
import logging

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from tqdm import tqdm

import algo_hyperparams
import data_proc
import models
import setup_params as setup
import ref
from ref import make_tensor

import r_preprocessing

def generate_data(t, seed, homedir=''):
    full_data = pd.read_csv(setup.get_datapath(homedir))

    #Dataset Level Preprocessing
    full_data = r_preprocessing.preprocess_data(full_data, \
                                                 {'data':'comment_text', \
                                                 'labels':'toxicity'}, \
                                                 tox_thresh=0.4, c_len=15, \
                                                 social_media=True)
                                                 
    #Remove Out-ofDomain test set
    full_data['gender'] = full_data[['male', 'female', 'transgender', \
                                     'other_gender', 'heterosexual', \
                                     'homosexual_gay_or_lesbian', 'bisexual', \
                                     'other_sexual_orientation', ]].max(axis=1)
    test_index = (full_data['gender'] > 0)
    test_data = full_data[test_index]
    test_data = data_proc.ToxicityDataset(test_data, transform=t)
    full_data = full_data[np.logical_not(test_index)]

    #Constuct Balanced training/validation dataset
    toxic_data = full_data[full_data['toxicity'] == 0]
    nontoxic_data = full_data[full_data['toxicity'] == 1]
    nsamples = min(len(toxic_data), len(nontoxic_data))
    toxic_data = toxic_data.sample(n=nsamples, random_state=seed)
    nontoxic_data = nontoxic_data.sample(n=nsamples, random_state=seed)
    full_data = pd.concat([toxic_data, nontoxic_data], ignore_index=True).sample(frac=1)


    #Process training and validation data
    train_data, val_data = train_test_split(full_data, test_size=0.2, random_state=seed)
    train_data.reset_index(drop=True, inplace=True)
    val_data.reset_index(drop=True, inplace=True)
    train_data = data_proc.ToxicityDataset(train_data, transform=t)  #NOTE FOR AMPLER TO WORK, INDEX MUST BE SEQUENTIAL 0-LEN(DSET)
    val_data = data_proc.ToxicityDataset(val_data, transform=t)
    return train_data, val_data, test_data

def reddit_labelgen(id, expdir, data_fname, args):
    #Logging Setup
    logger_fname = os.path.join(expdir, 'log_{}.txt'.format(str(id)))
    logging.basicConfig(filename=logger_fname, level=logging.DEBUG)

    #Generate word embedding transform
    t = data_proc.get_word_transform('embed', setup.get_wordvecspath(), proc=False)
    logging.info('WEs loaded')
    train_data, val_data, __ = generate_data(t, args['seed'])
    dataloader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True)
    logging.info('data loaded')

    #Train model to distinguish.
    base = models.MLP()
    losses = base.run(dataloader, args, batching=True)
    to_save_model = {'base_model':models.MLP(), 'model_arch':base.model_arch, \
                      'model_params':base.model.state_dict()}
    ##Now validate
    final_results = {}
    final_results['model'] = to_save_model
    final_data = {'train':train_data[:], 'val':val_data[:]}
    for p, d in final_data.items():
        preds = base.predict(d['x'])
        labels = d['y']
        final_results[p+'_loss'] = ref.compute_loss(preds, labels, ltype='BCE')
        final_results[p+'_acc'] = ref.compute_loss(preds, labels, ltype='ACC')
    final_results['losses'] = losses
    final_results['id'] = id
    final_results['params'] = args

    pickle.dump(final_results, open(join(expdir, '{}_redlgen.pkl'.format(id)), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('id', type=str, \
                        help='unique id of exp')
    parser.add_argument("expdir", type=str, default=None,
                        help="path to location to save files")
    parser.add_argument("data_fname", type=str,
                        help="filename adult.csv")

    #Hyperparams
    parser.add_argument('-inc_hyperparams', type=int, default=0)
    parser.add_argument("-seed", type=int, default=None)
    parser.add_argument("-epochs", type=int, default=None)
    parser.add_argument("-batch_size", type=int, default=None)
    parser.add_argument("-hid_layers", type=int, default=None)
    parser.add_argument("-lr", type=float, default=None)
    parser.add_argument("-l2", type=float, default=None)
    args = parser.parse_args()

    if args.inc_hyperparams == 0:
        assert False
    else:
        algo_params = {'seed':args.seed, \
                      'n_iterations':args.epochs, \
                      'batch_size':args.batch_size, \
                      'hid_layers':args.hid_layers, \
                      'lr': args.lr, \
                      'l2_reg':args.l2
                      }

    reddit_labelgen(args.id, args.expdir, args.data_fname, algo_params)
