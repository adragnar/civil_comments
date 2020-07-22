import argparse
import os
from os.path import join

import sys
sys.path.insert(1, join(os.getcwd(), 'launchfiles'))

import itertools
import json
import logging
import pickle
import warnings

from abc import ABC, abstractmethod
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import torch.autograd as autograd
from torch import nn
from scipy.stats import f as fdist
from scipy.stats import ttest_ind
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression

import torch
import torch.nn.functional as F

import data_proc
import ref
import setup_params as setup


import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
STOPWORDS = set(stopwords.words('english'))
thresh = 0.2

esplit_from_id = {0:np.array([[0.1, 0.9], [0.2, 0.8], [0.9, 0.1]]), \
                  1:np.array([[0.1, 0.9], [0.3, 0.7], [0.9, 0.1]]), \
                  2:np.array([[0.05, 0.95], [0.35, 0.65], [0.9, 0.1]])}



def main(id, expdir, data_fname, seed, env_id, label_noise):
    ''':param env_splits: the envrionments, infinite possible, each binary of
                          form np.array([[.1, 0.9], [0.2, 0.8], [0.9, 0.1]])'''
    word2vec, _, _ = data_proc.load_word_vectors(setup.get_wordvecspath())
    full_data = pd.read_csv(data_fname)
    full_data['LGTBQ'] = full_data[['homosexual_gay_or_lesbian', 'bisexual', 'other_sexual_orientation']].max(axis=1)

    #Data Processing
    full_partition = full_data[(full_data['LGTBQ'] > 0)]
    toxic, non_toxic = full_partition[full_data['toxicity'] >= thresh].sample(frac=1).reset_index(drop=True), \
                            full_partition[full_data['toxicity'] < thresh].sample(frac=1).reset_index(drop=True)
    toxic['toxicity'], non_toxic['toxicity'] = toxic['toxicity'].apply((lambda x: 1 if x > thresh else 0)), \
                        non_toxic['toxicity'].apply((lambda x: 1 if x > thresh else 0))

    totals = {'nt':len(non_toxic), 't':len(toxic)}
    env_splits = esplit_from_id[env_id]
    weights = {'nt':env_splits.mean(axis=0)[0], 't':env_splits.mean(axis=0)[1]}

    #Adjust so that desired env splits possible
    if float(totals['t']/(totals['t'] + totals['nt'])) >= weights['t']:  #see who has the bigger proportion
        ns = int(totals['nt']/weights['nt'] - totals['nt'])   #     int((len(full_partition) - weights['nt']*totals['nt'])/weights['t'])
        toxic = toxic.sample(n=ns, random_state=seed)
    else:
        ns = int(totals['t']/weights['t'] - totals['t'])
        non_toxic = non_toxic.sample(n=ns, random_state=seed)

    #partition env splits
    nenvs = env_splits.shape[0]
    e_props = env_splits/env_splits.sum(axis=0) #proprotion of vector in each env

    env_partitions = []  #Note - last env is the test env
    for i in range(nenvs):  #Note - tehre might be an error here that excludes  single sample from diff envs
        #Get both componenets of envs
        past_ind = int(np.array(e_props[:i, 0]).sum() * len(non_toxic))
        pres_ind = int(np.array(e_props[:(i+1), 0]).sum() * len(non_toxic))
        nt = non_toxic.iloc[past_ind:pres_ind]

        past_ind = int(np.array(e_props[:i, 1]).sum() * len(toxic))
        pres_ind = int(np.array(e_props[:(i+1), 1]).sum() * len(toxic))
        t = toxic.iloc[past_ind:pres_ind]

        #Make full env
        env = pd.concat([nt, t], ignore_index=True).sample(frac=1)
        if label_noise > 0:
            lnoise_fnc = lambda x: np.random.binomial(1, 1-label_noise) if x > thresh else np.random.binomial(1, label_noise)
            env['toxicity'] = env['toxicity'].apply(lnoise_fnc)
        env_partitions.append(env)


    t = data_proc.GetEmbedding(word2vec, stopwords=STOPWORDS)

    #Baseline Logistic Regression
    train_partition = data_proc.ToxicityDataset(pd.concat([e for e in env_partitions[:-1]], \
                                                ignore_index=True)[['id', 'toxicity', 'comment_text']], transform=t)[:]
    test_partition = data_proc.ToxicityDataset(env_partitions[-1][['id', 'toxicity', 'comment_text']], transform=t)[:]

    print(train_partition['x'].shape, test_partition['x'].shape)
    baseline_model = LogisticRegression(fit_intercept = True, penalty = 'l2').fit(train_partition['x'], train_partition['y'])
    baseline_train_score = baseline_model.score(train_partition['x'], train_partition['y'])
    baseline_test_score = baseline_model.score(test_partition['x'], test_partition['y'])

    baseline_res = {'id':{'seed':seed, 'env_splits':env_id, 'label_noise':label_noise}, \
                    'results':{'train':baseline_train_score, 'test':baseline_test_score}, \
                    'model':baseline_model}
    pickle.dump(baseline_res, open(join(expdir, '{}_baseline.pkl'.format(id)), 'wb'))

    #IRM Logistic Regression
    train_envs = [data_proc.ToxicityDataset(e[['id', 'toxicity', 'comment_text']], transform=t)[:] for e in env_partitions[:-1]]
    test_partition = data_proc.ToxicityDataset(env_partitions[-1][['id', 'toxicity', 'comment_text']], transform=t)[:]

    print(train_envs[0]['x'].shape, test_partition['x'].shape)

    args = {'lr': 0.001, \
             'n_iterations':70000, \
             'penalty_anneal_iters':1, \
             'l2_reg':1.0, \
             'pen_wgt':10, \
             'hid_layers':1, \
             'verbose':False}
    base = ref.LinearInvariantRiskMinimization('cls')
    irm_model, errors, penalties, losses = base.train(train_envs, seed, args)

    train_logits = base.predict(np.concatenate([train_envs[i]['x'] for i in range(len(train_envs))]), irm_model)
    train_labels = np.concatenate([train_envs[i]['y'] for i in range(len(train_envs))])
    test_logits = base.predict(test_partition['x'], irm_model)
    test_labels = test_partition['y']

    irm_train_acc = ref.compute_loss(np.expand_dims(train_logits, axis=1), train_labels, ltype='ACC')
    irm_test_acc = ref.compute_loss(np.expand_dims(test_logits, axis=1), test_labels, ltype='ACC')
    irm_res = {'id':{'seed':seed, 'env_splits':env_id, 'label_noise':label_noise}, \
                    'results':{'train':irm_train_acc, 'test':irm_test_acc}, \
                    'model':irm_model}

    pickle.dump(irm_res, open(join(expdir, '{}_irm.pkl'.format(id)), 'wb'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('id', type=str, \
                        help='unique id of exp')
    parser.add_argument("expdir", type=str, default=None,
                        help="path to location to save files")
    parser.add_argument("data_fname", type=str,
                        help="filename adult.csv")
    parser.add_argument("seed", type=str, default=None)
    parser.add_argument("env_split", type=str, default=None)
    parser.add_argument("label_noise", type=str, default=None)
    args = parser.parse_args()

    main(args.id, args.expdir, args.data_fname, int(args.seed), int(args.env_split), \
           float(args.label_noise))
