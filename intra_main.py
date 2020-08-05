import argparse
import os
from os.path import join

import sys
sys.path.insert(1, join(os.getcwd(), 'launchfiles'))

import copy
import pickle

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression

import algo_hyperparams
import data_proc
import main
import models
import setup_params as setup


#INTRA EXPERIMENTS
intra_cats_all = {'race':['black', 'white', 'asian', \
                            'latino', 'other_race_or_ethnicity'], \
              'gender':['male', 'female', 'transgender', 'other_gender', \
                        'heterosexual', 'homosexual_gay_or_lesbian', 'bisexual',
                        'other_sexual_orientation'], \
              'religion':['christian', 'jewish', 'muslim', 'hindu', \
                          'buddhist', 'atheist', 'other_religion'], \
              'disability':['physical_disability', 'intellectual_or_learning_disability', \
                            'psychiatric_or_mental_illness', 'other_disability'], \
                            }

intra_cats_test = {'race':'asian', \
              'gender':'transgender', \
              'religion':'atheist', \
              'disability':'intellectual_or_learning_disability', \
            }

def load_data_intra(fname):
    thresh = 0.2
    full_data = pd.read_csv(fname)
    full_data['toxicity'] = full_data['toxicity'].apply((lambda x: 1 if x > thresh else 0))
    return full_data

def intra_add_label_noise(data, lcol, lnoise):
    '''Given a pandas dataframe with binary labels, flip with probability lnoise'''
    data[lcol] = data[lcol].apply(lambda x: np.random.binomial(1, lnoise) if (x == 0) else np.random.binomial(1, (1 - lnoise)))
    return data

def intra_main(id, expdir, fname, args, algo_args):

    #Load the data
    full_data = load_data_intra(fname)
    if args['word_encoding'] == 'embed':
        t = data_proc.get_word_transform(args['word_encoding'], setup.get_wordvecspath())
    elif args['word_encoding'] == 'BOW':
        t = data_proc.get_word_transform(args['word_encoding'], setup.get_wordfreqpath())

    #Get the names of training and test environments
    train_names = []
    test_names = {}
    for cat, dems in intra_cats_all.items():
        if cat == args['sens_cat']:
            train_names = copy.deepcopy(dems); train_names.remove(intra_cats_test[cat])
            test_names[cat] = [intra_cats_test[cat]]
        else:
            test_names[cat] = dems


    #Get training environment information
    train_partitions = {args['sens_cat']:[]}
    train_indicies = set()
    for dem in train_names:
        new_p = full_data[(full_data[dem] > 0)]
        if args['label_noise'] > 0:
            new_p = intra_add_label_noise(new_p, 'toxicity', args['label_noise'])
        train_indicies.union(set(new_p.index))
        train_partitions[args['sens_cat']].append(new_p)
    agg_train_partition = data_proc.ToxicityDataset(pd.concat([e for e in train_partitions[args['sens_cat']]], \
                                                ignore_index=True)[['id', 'toxicity', 'comment_text']], transform=t)[:]
    train_partitions = [data_proc.ToxicityDataset(e[['id', 'toxicity', 'comment_text']], transform=t)[:] for e in train_partitions[args['sens_cat']]]


    #Get Test Partitions
    test_partitions = {}
    for cat, dems in test_names.items():
        data = {}
        for d in dems:
            new_p = full_data[(full_data[d] > 0) & (np.logical_not(full_data.index.isin(train_indicies)))]
            if args['label_noise'] > 0:
                new_p = intra_add_label_noise(new_p, 'toxicity', args['label_noise'])
            data[d] = data_proc.ToxicityDataset(new_p[['id', 'toxicity', 'comment_text']], transform=t)[:]
        test_partitions[cat] = data

    #Add

    #Train Models
    #Baseline Logistic Regression
    baseline_model = LogisticRegression(fit_intercept=True, penalty='l2', C=float(1/algo_args['l2_reg'])).fit(agg_train_partition['x'], agg_train_partition['y'])
    baseline_train_score = baseline_model.score(agg_train_partition['x'], agg_train_partition['y'])
    baseline_res = {'id':{'params':args, 'algo_params':algo_args}, \
                    'results':{'train':baseline_train_score}, \
                    'model':baseline_model}
    pickle.dump(baseline_res, open(join(expdir, '{}_baseline.pkl'.format(id)), 'wb'))

    #IRM Logistic Regression
    base = models.LinearInvariantRiskMinimization('cls')
    irm_model, errors, penalties, losses = base.train(train_partitions, args['seed'], algo_args)
    irm_train_acc =  main.evaluate_model({'x': np.concatenate([train_partitions[i]['x'] for i in range(len(train_partitions))]), \
                                     'y':np.concatenate([train_partitions[i]['y'] for i in range(len(train_partitions))])}, \
                                     base, model=irm_model, ltype='ACC')
    irm_res = {'id':{'params':args, 'algo_params':algo_args}, \
                    'results':{'train':irm_train_acc}, \
                    'model':irm_model}
    pickle.dump(irm_res, open(join(expdir, '{}_irm.pkl'.format(id)), 'wb'))

    #Test Models
    results = []
    for cat, dems in test_partitions.items():
        for d, vals in dems.items():
            #Run inference on test partitions
            baseline_test_score = baseline_model.score(vals['x'], vals['y'])
            irm_test_acc = main.evaluate_model(vals, \
                                             base, model=irm_model, ltype='ACC')
            results.append([cat, d, baseline_test_score, irm_test_acc])

    results = pd.DataFrame(results)
    results.columns = ['cat', 'demo', 'base', 'irm']
    intra_res = {'id':{'params':args, 'algo_params':algo_args}, \
                    'results':results}
    pickle.dump(intra_res, open(join(expdir, '{}_intra.pkl'.format(id)), 'wb'))

    # import psutil
    # process = psutil.Process(os.getpid())
    # print(float(process.memory_info().rss/(1e9)))  # in bytes
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('id', type=str, \
                        help='unique id of exp')
    parser.add_argument("expdir", type=str, default=None,
                        help="path to location to save files")
    parser.add_argument("data_fname", type=str,
                        help="filename adult.csv")
    parser.add_argument("seed", type=str, default=None)
    parser.add_argument("label_noise", type=str, default=None)
    parser.add_argument("sens_cat", type=str, default=None)
    parser.add_argument("word_encoding", type=str, default=None)
    # parser.add_argument("model", type=str, default=None)

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
              'label_noise':float(args.label_noise), \
              'sens_cat':args.sens_cat, \
              'word_encoding':args.word_encoding, \
              }

    if args.inc_hyperparams == 0:
        algo_params = algo_hyperparams.get_hyperparams('logreg')
    else:
        algo_params =  {'lr': args.lr, \
                         'n_iterations':args.niter, \
                         'penalty_anneal_iters':args.penalty_anneal, \
                         'l2_reg':args.l2, \
                         'pen_wgt':args.penalty_weight, \
                         'hid_layers':args.hid_layers, \
                         'verbose':False}

    intra_main(args.id, args.expdir, args.data_fname, params, algo_params)
