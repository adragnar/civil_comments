import argparse
import os
from os.path import join
import logging

import sys
sys.path.insert(1, join(os.getcwd(), 'launchfiles'))
sys.path.append(os.getcwd())

import copy
import math
import pickle
import socket

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from torch.utils.data import DataLoader

import algo_hyperparams
import data_proc
import main
import models
import setup_params as setup
import ref

def add_label_noise(data, lnoise):
    '''Given a pandas series with binary labels, flip with probability lnoise
    :return pd.Series'''
    return data.apply(lambda x: np.random.binomial(1, lnoise) if (x == 0)\
                                    else np.random.binomial(1, (1 - lnoise)))

def evaluate(envs, model, ltype='ACC'):
    '''Envs - list of dataloaders, each with data from diff env'''
    num_corr = 0
    for dl in envs:
        for batch_idx, sample_batch in enumerate(dl):
            probs = model.predict(sample_batch['x'].detach().numpy())
            preds = ref.pred_binarize(probs)
            labels = sample_batch['y'].detach().numpy().squeeze()
            if ltype == 'ACC':
                num_corr += np.abs(preds - labels).sum()
            elif ltype == 'BCE':
                ce = np.sum((labels * np.log(preds)) + ((1 - labels)*np.log(1-preds)))
                num_corr += ce
            else:
                raise Exception('Unimplemented')

    return float(num_corr/(sum([len(dl) for dl in envs])))

def generate_data(t, data_fname, label_noise, nbatches):
    thresh = 0.4
    full_data = pd.read_csv(data_fname); full_Data = full_data.sample(n=1000)
    full_data['comment_len'] = full_data['body'].apply(lambda x: 1 if (len(str(x)) > 15) else 0)
    full_data = full_data[full_data['comment_len'] == 1]; full_data.reset_index(inplace=True)
    full_data['toxicity'] = full_data['toxicity'].apply((lambda x: 1 if x > thresh else 0))


    #Get partition data into environments
    train_subs = ['TheRedPill', 'MensRights']
    test_subs = ['TwoXChromosomes']
    train_envs = []  #ASsume that this data comes with a 'toxicity' label associated
    test_envs = []  #ASsume that this data comes with a 'toxicity' label associated
    for sub in (train_subs + test_subs):
        #Generate raw dataframe
        df = full_data[full_data['subreddit'] == sub][['body', 'toxicity']]; df.reset_index(inplace=True)
        df = df.reset_index()
        if label_noise > 0:  #Add label noise if needed
            df['toxicity'] = add_label_noise(df['toxicity'], label_noise)

        #Make formal data sturctures
        df = data_proc.ToxicityDataset(df, \
                     rel_cols={'data':'body', 'labels':'toxicity'}, transform=t)
        df = DataLoader(df, batch_size=math.ceil(float(len(df)/nbatches)), shuffle=True)

        #Place in train or test
        if sub in train_subs:
            train_envs.append(df)
        elif sub in test_subs:
            test_envs.append(df)
    logging.info('Data Loaded')

    return train_envs, test_envs

def subreddit_oodgen(id, expdir, data_fname, args, algo_args, load_model=False):
    logger_fname = os.path.join(expdir, 'log_{}.txt'.format(str(id)))
    logging.basicConfig(filename=logger_fname, level=logging.DEBUG)

    #Load Data
    t = data_proc.get_word_transform('embed', setup.get_wordvecspath())
    logging.info('WEs Loaded')

    train_envs, test_envs = generate_data(t, data_fname, args['label_noise'], \
                                           algo_args['n_batches'])

    #Train Models
    if not load_model:
        base = models.LinearInvariantRiskMinimization('cls')
        errors, penalties, losses = base.train(train_envs, \
                                            args['seed'], algo_args, batching=True)
        to_save_model = {'base_model':models.LinearInvariantRiskMinimization('cls'), \
                         'model':base.model}

        #Evaluate
        train_loss = evaluate(train_envs, base, ltype='BCE')
        train_acc = evaluate(train_envs, base, ltype='ACC')
        test_loss = evaluate(test_envs, base, ltype='BCE')
        test_acc = evaluate(test_envs, base, ltype='ACC')

        irm_res = {'id':{'params':args, 'algo_params':algo_args}, \
                        'results':{'train_acc':train_acc, 'test_acc':test_acc, \
                                  'train_loss':train_loss, 'test_loss':test_loss}, \
                        'losses':losses, \
                        'model':to_save_model}
        pickle.dump(irm_res, open(join(expdir, '{}_irm.pkl'.format(id)), 'wb'))

    else:
        for f in os.listdir(load_model):
            if 'irm' not in f:
                continue
            fpath = join(load_model, f)
            import pdb; pdb.set_trace()
            base_data = data_proc.load_saved_model(fpath)

            #Evaluate
            base = base_data['model']
            train_loss = evaluate(train_envs, base, ltype='BCE')
            train_acc = evaluate(train_envs, base, ltype='ACC')
            test_loss = evaluate(test_envs, base, ltype='BCE')
            test_acc = evaluate(test_envs, base, ltype='ACC')

            #Save Data
            base_data['results'] = {'train_acc':train_acc, 'test_acc':test_acc, \
                      'train_loss':train_loss, 'test_loss':test_loss}
            pickle.dump(base_data, open(fpath, 'wb'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('id', type=str, \
                        help='unique id of exp')
    parser.add_argument("expdir", type=str, default=None,
                        help="path to location to save files")
    parser.add_argument("data_fname", type=str,
                        help="filename adult.csv")
    parser.add_argument("label_noise", type=float, default=None)
    parser.add_argument("-seed", type=int, default=1000)


    #Hyperparams
    parser.add_argument('-inc_hyperparams', type=int, default=0)
    parser.add_argument("-epochs", type=int, default=None)
    parser.add_argument("-n_batches", type=int, default=None)
    parser.add_argument("-hid_layers", type=int, default=None)
    parser.add_argument("-lr", type=float, default=None)
    parser.add_argument("-l2", type=float, default=None)
    parser.add_argument('-pen_wgt', type=float, default=None)
    parser.add_argument('-pen_ann', type=float, default=None)
    args = parser.parse_args()

    params = {'seed':args.seed, \
              'label_noise':args.label_noise}

    if args.inc_hyperparams == 0:
        assert False
    else:
        algo_params = { \
                      'n_iterations':args.epochs, \
                      'n_batches':args.n_batches, \
                      'hid_layers':args.hid_layers, \
                      'lr': args.lr, \
                      'l2_reg':args.l2,
                      'pen_wgt':args.pen_wgt,
                      'penalty_anneal_iters':args.pen_ann
                      }
    subreddit_oodgen(args.id, args.expdir, args.data_fname, params, algo_params, \
    load_model='/scratch/hdd001/home/adragnar/experiments/civil_comments/subreddit_oodgen/1597183186.5170383')
