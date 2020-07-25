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

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
STOPWORDS = set(stopwords.words('english'))

import algo_hyperparams
import data_proc
import models
import ref
import setup_params as setup
import partition_environments

def evaluate_model(train, test, base, model, hid_layers=None,  ltype='ACC'):
    '''Given a model and its data, evaluate
    :param trainenvs: list of dictionaries of np arrays, each which is
    the dataset for a train env {'x':arr, 'y':arr}
    :param testenv: dict of np arrays dataset for a test env  {'x':arr, 'y':arr}
    :param model: trained mdoel used for prediction
    :param base: the model needed for prediction method'''
    if hid_layers is not None:
        train_logits = base.predict(train['x'], model, hid_layers=hid_layers)
        test_logits = base.predict(test['x'], model, hid_layers=hid_layers)
    else:
        train_logits = base.predict(train['x'], model)
        test_logits = base.predict(test['x'], model)
    train_labels = train['y']
    test_labels = test['y']

    train_acc = ref.compute_loss(np.expand_dims(train_logits, axis=1), train_labels, ltype=ltype)
    test_acc = ref.compute_loss(np.expand_dims(test_logits, axis=1), test_labels, ltype=ltype)
    return train_acc, test_acc

def main(id, expdir, data_fname, args, algo_args):
    ''':param env_splits: the envrionments, infinite possible, each binary of
                          form np.array([[.1, 0.9], [0.2, 0.8], [0.9, 0.1]])'''



    #Set up dataset with proper embeddings
    if args['word_encoding'] == 'embed':
        word2vec, _, _ = data_proc.load_word_vectors(setup.get_wordvecspath())
        t = data_proc.GetEmbedding(word2vec, stopwords=STOPWORDS)
    elif args['word_encoding'] == 'BOW':
        word_freq = pickle.load(open(setup.get_wordfreqpath(), 'rb'))
        vocabulary = sorted([x for x in word_freq if ((word_freq[x] > 20) and (word_freq[x] < 1e3))])
        vocabulary = {vocabulary[i]:i for i in range(len(vocabulary))}
        t = data_proc.GetBOW(vocabulary, lem=WordNetLemmatizer(), stopwords=STOPWORDS)

    #Get Environment Data
    env_partitions = partition_environments.partition_envs(data_fname, args)

    #Baseline Logistic Regression
    train_partition = data_proc.ToxicityDataset(pd.concat([e for e in env_partitions[:-1]], \
                                                ignore_index=True)[['id', 'toxicity', 'comment_text']], transform=t)[:]
    test_partition = data_proc.ToxicityDataset(env_partitions[-1][['id', 'toxicity', 'comment_text']], transform=t)[:]

    print(train_partition['x'].shape, test_partition['x'].shape)

    if args['base_model'] == 'logreg':
        baseline_model = LogisticRegression(fit_intercept = True, penalty = 'l2').fit(train_partition['x'], train_partition['y'])
        baseline_train_score = baseline_model.score(train_partition['x'], train_partition['y'])
        baseline_test_score = baseline_model.score(test_partition['x'], test_partition['y'])
    elif args['base_model'] == 'mlp':
        base = models.MLP()
        baseline_model, _ = base.run(train_partition['x'], train_partition['y'], algo_args)
        baseline_train_score, baseline_test_score = \
                  evaluate_model(train_partition, test_partition, base, baseline_model, \
                                   hid_layers=algo_args['hid_layers'], ltype='ACC')

    baseline_res = {'id':{'seed':args['seed'], 'env_splits':args['env_id'], 'label_noise':args['label_noise'], 'algo_params':algo_args}, \
                    'results':{'train':baseline_train_score, 'test':baseline_test_score}, \
                    'model':baseline_model}
    pickle.dump(baseline_res, open(join(expdir, '{}_baseline.pkl'.format(id)), 'wb'))

    #IRM Logistic Regression
    train_envs = [data_proc.ToxicityDataset(e[['id', 'toxicity', 'comment_text']], transform=t)[:] for e in env_partitions[:-1]]
    test_partition = data_proc.ToxicityDataset(env_partitions[-1][['id', 'toxicity', 'comment_text']], transform=t)[:]
    print(train_envs[0]['x'].shape, test_partition['x'].shape)

    if args['base_model'] == 'logreg':
        base = models.LinearInvariantRiskMinimization('cls')
        irm_model, errors, penalties, losses = base.train(train_envs, args['seed'], algo_args)
        irm_train_acc, irm_test_acc = \
             evaluate_model({'x': np.concatenate([train_envs[i]['x'] for i in range(len(train_envs))]), \
                             'y':np.concatenate([train_envs[i]['y'] for i in range(len(train_envs))])}, \
                             test_partition, base, irm_model, ltype='ACC')
    elif args['base_model'] == 'mlp':
        base = models.InvariantRiskMinimization('cls')
        irm_model, errors, penalties, losses = base.train(train_envs, args['seed'], algo_args)
        irm_train_acc, irm_test_acc = \
             evaluate_model({'x': np.concatenate([train_envs[i]['x'] for i in range(len(train_envs))]), \
                             'y':np.concatenate([train_envs[i]['y'] for i in range(len(train_envs))])}, \
                             test_partition, base, irm_model, hid_layers=algo_args['hid_layers'], ltype='ACC')

    irm_res = {'id':{'seed':args['seed'], 'env_splits':args['env_id'], 'label_noise':args['label_noise'], 'algo_params':algo_args}, \
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
    parser.add_argument("env_id", type=str, default=None)
    parser.add_argument("label_noise", type=str, default=None)
    parser.add_argument("sens_att", type=str, default=None)
    parser.add_argument("word_encoding", type=str, default=None)
    parser.add_argument("model", type=str, default=None)

    #Hyperparams
    parser.add_argument('-inc_hyperparams', type=int, default=0)
    parser.add_argument('-lr', type=float, default=None)
    parser.add_argument('-niter', type=int, default=None)
    parser.add_argument('-l2', type=float, default=None)
    parser.add_argument('-penalty_weight', type=float, default=None)
    parser.add_argument('-penalty_anneal', type=float, default=None)
    parser.add_argument('-hid_layers', type=int, default=None)
    args = parser.parse_args()

    params = {'seed':int(args.seed), \
              'env_id':int(args.env_id), \
              'label_noise':float(args.label_noise), \
              'sens_att':args.sens_att, \
              'word_encoding':args.word_encoding, \
              'base_model':args.model \
              }

    if args.inc_hyperparams == 0:
        algo_params = algo_hyperparams.get_hyperparams(args.model)
    else:
        algo_params =  {'lr': args.lr, \
                         'n_iterations':args.niter, \
                         'penalty_anneal_iters':args.penalty_anneal, \
                         'l2_reg':args.l2, \
                         'pen_wgt':args.penalty_weight, \
                         'hid_layers':args.hid_layers, \
                         'verbose':False}

    main(args.id, args.expdir, args.data_fname, params, algo_params)
