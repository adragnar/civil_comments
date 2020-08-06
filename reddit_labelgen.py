import argparse
import os
from os.path import join

import sys
sys.path.insert(1, join(os.getcwd(), 'launchfiles'))

import copy
import pickle
import socket

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
import main
import models
import setup_params as setup
from ref import make_tensor

args = {'seed':1000, \
        'epochs':5000, \
        'batch_size':20, \
        'hid_layers':100, \
        'lr':0.001, \
        'l2_reg':1.0}

nsamples = 2300

#Generate word embedding transform
t = data_proc.get_word_transform('embed', setup.get_wordvecspath())

#Generate full_data
g_data = pd.read_csv(setup.get_reddit_datapath('gendered'))
toxic_data = g_data[g_data['subreddit'] == 'TheRedPill'][['body']].sample(n=nsamples, random_state=args['seed'])
toxic_data['toxicity'] = np.ones(len(toxic_data))

baseline_data = pd.read_csv(setup.get_reddit_datapath('baseline'))[['body']].sample(n=nsamples, random_state=args['seed'])
baseline_data['toxicity'] = np.zeros(len(baseline_data))

full_data = pd.concat([baseline_data, toxic_data], ignore_index=True).sample(frac=1, random_state=args['seed'])
full_data['comment_text'] = full_data['body']
full_data.drop('body', axis=1, inplace=True)

train_data, val_data = train_test_split(full_data, test_size=0.2, random_state=args['seed'])
train_data.reset_index(drop=True, inplace=True)
val_data.reset_index(drop=True, inplace=True)
train_data = data_proc.ToxicityDataset(train_data, transform=t)  #NOTE FOR AMPLER TO WORK, INDEX MUST BE SEQUENTIAL 0-LEN(DSET)
dataloader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True)

#Train model to distinguish.
losses = []
model = models.BaseMLP(300, args['hid_layers'])
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])

for epoch in tqdm(range(args['epochs'])):
    for i_batch, sample_batch in enumerate(dataloader):
        logits = model(sample_batch['x'].float()).squeeze()
        labels = sample_batch['y'].double()
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)

        weight_norm = model.weight_norm()
        loss += args['l2_reg'] * weight_norm

        #Do the backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #Accounting
        losses.append(loss.detach().numpy())
