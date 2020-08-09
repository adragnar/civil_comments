import argparse
import logging
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


import algo_hyperparams
import data_proc
import models
import ref
import setup_params as setup
import partition_environments

def evaluate_ensemble(data, base, model=None, hid_layers=None, ltype='ACC'):
    '''Given an ensemble of models and its data, evaluate
    :param trainenvs: list of dictionaries of np arrays, each which is
    the dataset for a train env {'x':arr, 'y':arr}
    :param testenv: dict of np arrays dataset for a test env  {'x':arr, 'y':arr}
    :param base: list of trained models used for prediction
    :param model: the model_params needed for .predict method (neeed for all but sklearn)'''

    preds = []
    if type(base) == list:  ##In case we're using sklearn bb classifiers
        assert 'sklearn' in str(type(base[0]))
        for b in base:
            preds.append(b.predict(data['x']))

    else:  #Assuming these will be my standard interface fucntions
        for m in models:
            preds.append(base.predict(data['x'], m, args={'hid_layers':hid_layers}))

    avg_pred = np.mean(preds, axis=0)
    labels = data['y']

    acc = ref.compute_loss(avg_pred, labels, ltype=ltype)
    return acc


def evaluate_model(data, base, model=None, hid_layers=None, ltype='ACC'):
    '''Given a model and its data, evaluate
    :param data: dict of np arrays dataset for a test env  {'x':arr, 'y':arr}
    :param model: trained mdoel used for prediction
    :param base: the model needed for prediction method'''
    if 'sklearn' in str(type(base)):
        preds = base.predict(data['x'])
    else:
        preds = base.predict(data['x'], model, args={'hid_layers':hid_layers})
    labels = data['y']
    acc = ref.compute_loss(preds, labels, ltype=ltype)
    return acc

def main(id, expdir, data_fname, args, algo_args):
    ''':param env_splits: the envrionments, infinite possible, each binary of
                          form np.array([[.1, 0.9], [0.2, 0.8], [0.9, 0.1]])'''

    logger_fname = os.path.join(expdir, 'log_{}.txt'.format(id))
    logging.basicConfig(filename=logger_fname, level=logging.DEBUG)

    #Get Data Embedding Function

    if args['word_encoding'] == 'embed':
        t = data_proc.get_word_transform(args['word_encoding'], setup.get_wordvecspath())
    elif args['word_encoding'] == 'BOW':
        t = data_proc.get_word_transform(args['word_encoding'], setup.get_wordfreqpath())

    #Get Environment Data
    if args['exptype'] == 'cmnist':
        env_partitions = partition_environments.partition_envs_cmnist(data_fname, args)
    elif 'lshift' in args['exptype']:
        env_partitions = partition_environments.partition_envs_labelshift(data_fname, args)
    else:
        raise Exception('Data Partition Unimplemented')

    train_envs = [data_proc.ToxicityDataset(e[['id', 'toxicity', 'comment_text']], transform=t)[:] for e in env_partitions[:-1]]
    train_partition = data_proc.ToxicityDataset(pd.concat([e for e in env_partitions[:-1]], \
                                                ignore_index=True)[['id', 'toxicity', 'comment_text']], transform=t)[:]
    test_partition = data_proc.ToxicityDataset(env_partitions[-1][['id', 'toxicity', 'comment_text']], transform=t)[:]


    print(len(train_envs), train_partition['x'].shape, test_partition['x'].shape)

    #Baseline Logistic Regression
    if args['base_model'] == 'logreg':
        baseline_model = LogisticRegression(fit_intercept=True, penalty='l2', C=float(1/algo_args['l2_reg'])).fit(train_partition['x'], train_partition['y'])
        baseline_train_score = baseline_model.score(train_partition['x'], train_partition['y'])
        baseline_test_score = baseline_model.score(test_partition['x'], test_partition['y'])
    elif args['base_model'] == 'mlp':
        base = models.MLP()
        baseline_model, _ = base.run(train_partition, algo_args)
        baseline_train_score = evaluate_model(train_partition, base, model=baseline_model, \
                                   hid_layers=algo_args['hid_layers'], ltype='ACC')
        baseline_test_score = evaluate_model(test_partition, base, model=baseline_model, \
                                   hid_layers=algo_args['hid_layers'], ltype='ACC')
        print(baseline_train_score, baseline_test_score)

    baseline_res = {'id':{'params':args, 'algo_params':algo_args}, \
                    'results':{'train':baseline_train_score, 'test':baseline_test_score}, \
                    'model':baseline_model}
    pickle.dump(baseline_res, open(join(expdir, '{}_baseline.pkl'.format(id)), 'wb'))

    #Ensemble Logistic Regression
    ensemble_models = []
    if args['base_model'] == 'logreg':
        for e in train_envs:
            ensemble_models.append(LogisticRegression(fit_intercept=True, penalty='l2', C=float(1/algo_args['l2_reg'])).fit(e['x'], e['y']))
        ensemble_train_score = evaluate_ensemble(train_partition, ensemble_models, \
                                   hid_layers=algo_args['hid_layers'], ltype='ACC')
        ensemble_test_score = evaluate_ensemble(test_partition, ensemble_models, \
                                   hid_layers=algo_args['hid_layers'], ltype='ACC')

    elif args['base_model'] == 'mlp':
        for e in train_envs:
            base = models.MLP()
            bm = base.run(e, algo_args)
            ensemble_models.append(bm)
        ensemble_train_score = evaluate_ensemble(train_partition, base, model=ensemble_models, \
                               hid_layers=algo_args['hid_layers'], ltype='ACC')
        ensemble_test_score = evaluate_ensemble(test_partition, base, model=ensemble_models, \
                         hid_layers=algo_args['hid_layers'], ltype='ACC')

    ensemble_res = {'id':{'params':args, 'algo_params':algo_args}, \
                    'results':{'train':ensemble_train_score, 'test':ensemble_test_score}, \
                    'model':ensemble_models}
    pickle.dump(ensemble_res, open(join(expdir, '{}_ensemble.pkl'.format(id)), 'wb'))


    #IRM Logistic Regression
    if args['base_model'] == 'logreg':
        base = models.LinearInvariantRiskMinimization('cls')
        irm_model, errors, penalties, losses = base.train(train_envs, args['seed'], algo_args)
        irm_train_acc =  evaluate_model({'x': np.concatenate([train_envs[i]['x'] for i in range(len(train_envs))]), \
                             'y':np.concatenate([train_envs[i]['y'] for i in range(len(train_envs))])}, \
                             base, model=irm_model, ltype='ACC')
        irm_test_acc = evaluate_model(test_partition, base, model=irm_model, ltype='ACC')

    elif args['base_model'] == 'mlp':
        base = models.InvariantRiskMinimization('cls')
        irm_model, errors, penalties, losses = base.train(train_envs, args['seed'], algo_args)
        irm_train_acc =  evaluate_model({'x': np.concatenate([train_envs[i]['x'] for i in range(len(train_envs))]), \
                             'y':np.concatenate([train_envs[i]['y'] for i in range(len(train_envs))])}, \
                             base, model=irm_model, hid_layers=args['hid_layers'], ltype='ACC')
        irm_test_acc = evaluate_model(test_partition, base, model=irm_model, hid_layers=args['hid_layers'], ltype='ACC')

    irm_res = {'id':{'params':args, 'algo_params':algo_args}, \
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
    parser.add_argument("exptype", type=str, default=None)

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
              'base_model':args.model, \
              'exptype':args.exptype
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
