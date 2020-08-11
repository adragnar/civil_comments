#Modify existing reddit datasets with a toxicity column
import argparse
import os
from os.path import join, dirname

import sys

sys.path.append(join(os.getcwd(), 'launchfiles'))
sys.path.append(os.getcwd())

import copy
import pickle
import socket

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from tqdm import tqdm

import data_proc
import models
import setup_params as setup
import r_setup_params as reddit_setup

def preprocess_data(data):
    '''Clean up just-downloaded reddit datasets for later in pipeline
    :param data: THe raw datset (pd df)
    :return: pd Df
    '''

    #remove Nans in comment text column
    data['test_nan']= data['body'].apply(lambda x: 1 if type(x) == str else 0)

    #Drop comments that are too short
    data['test_len'] = data['body'].apply(lambda x: 1 if (len(str(x)) > 15) else 0)


    data = data[(data['test_nan'] == 1) & (data['test_len'] == 1)]
    data.drop(['test_nan', 'test_len'], axis=1, inplace=True)
    return data

def add_toxlabel(old_fpaths, new_fpaths, mpath, e_path):
    ''':param fname: path to reddit dataset
       :param new_fname: path to labelled reddit dataset
       :param mpath: path to model file
       :param epath: path to embeddings file'''

    t = data_proc.get_word_transform('embed', e_path)
    print('WE loaded')
    for fname, new_fname in zip(old_fpaths, new_fpaths):
        data = pd.read_csv(fname)
        import pdb; pdb.set_trace()
        data = preprocess_data(data)

        data_embed = t(data['body'])
        print('data')
        model = pickle.load(open(mpath, 'rb'))
        base = models.MLP()
        print('model')
        data['toxicity'] = base.predict(data_embed, model['model'], \
                                   args={'hid_layers':model['params']['hid_layers']}).values
        data.to_csv(new_fname)


if __name__ == '__main__':
    assert (os.getcwd().split('/')[-1] == 'civil_comments') or \
                    (os.getcwd().split('/')[-1] == 'civil_liberties')

    old_fpaths = ['reddit/data/orig/2014b.csv']
    new_fpaths = ['reddit/data/labeled/2014b_labeled.csv']
    # old_fpaths = ['reddit/data/orig/test.csv']
    # new_fpaths = ['reddit/data/labeled/test_labeled.csv']
    m_path = 'reddit/labelgen_models/0810_labelgen.pkl'
    add_toxlabel(old_fpaths, \
                 new_fpaths, \
                 m_path, \
                 setup.get_wordvecspath())
