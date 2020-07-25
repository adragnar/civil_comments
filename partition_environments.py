import argparse
import os
from os.path import join

import pandas as pd
import numpy as np


thresh = 0.2

esplit_from_id = {0:np.array([[0.1, 0.9], [0.2, 0.8], [0.9, 0.1]]), \
                  1:np.array([[0.1, 0.9], [0.3, 0.7], [0.9, 0.1]]), \
                  2:np.array([[0.05, 0.95], [0.35, 0.65], [0.9, 0.1]])}

def get_sensatt_column(data, satt):
    '''From the sensitive attribute described, return a pd series with that
       indicator'''

    if satt == 'LGBTQ':
        return data[['homosexual_gay_or_lesbian', 'bisexual', 'other_sexual_orientation']].max(axis=1)
    elif satt == 'muslim':
        return data['muslim']
    else:
        raise Exception('satt not implemented')

def partition_envs(fname, args):
    '''Take a pd Dataframe and split it into list of environments
    :param fname: path to dataset
    :param args['seed']: random seed (int)
    :param args['env_id']: id mapping to environment partition (int)
    :param args['label_noise']: chance that label is flipped (float)
    :params args['sens_att']: col in dataset whose data to include
    :return list of numpy arrays, last is testenv'''

    full_data = pd.read_csv(fname)
    full_data[args['sens_att']] = get_sensatt_column(full_data, args['sens_att'])

    #Data Processing
    full_partition = full_data[(full_data[args['sens_att']] > 0)]
    toxic, non_toxic = full_partition[full_data['toxicity'] >= thresh].sample(frac=1).reset_index(drop=True), \
                            full_partition[full_data['toxicity'] < thresh].sample(frac=1).reset_index(drop=True)
    toxic['toxicity'], non_toxic['toxicity'] = toxic['toxicity'].apply((lambda x: 1 if x > thresh else 0)), \
                        non_toxic['toxicity'].apply((lambda x: 1 if x > thresh else 0))

    totals = {'nt':len(non_toxic), 't':len(toxic)}
    env_splits = esplit_from_id[args['env_id']]
    weights = {'nt':env_splits.mean(axis=0)[0], 't':env_splits.mean(axis=0)[1]}

    #Adjust so that desired env splits possible
    if float(totals['t']/(totals['t'] + totals['nt'])) >= weights['t']:  #see who has the bigger proportion
        ns = int(totals['nt']/weights['nt'] - totals['nt'])   #     int((len(full_partition) - weights['nt']*totals['nt'])/weights['t'])
        toxic = toxic.sample(n=ns, random_state=args['seed'])
    else:
        ns = int(totals['t']/weights['t'] - totals['t'])
        non_toxic = non_toxic.sample(n=ns, random_state=args['seed'])

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
        if args['label_noise'] > 0:
            lnoise_fnc = lambda x: np.random.binomial(1, 1-args['label_noise']) if x > thresh else np.random.binomial(1, args['label_noise'])
            env['toxicity'] = env['toxicity'].apply(lnoise_fnc)
        env_partitions.append(env)
    return env_partitions

if __name__ == '__main__':
    #Generate data for hyperparam tuning
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument("data_fname", type=str,
                        help="filename adult.csv")
    parser.add_argument("env_id", type=str, default=None)
    args = parser.parse_args()

    params = {'seed':1000, \
              'env_id':int(args.env_id), \
              'label_noise':0.0, \
              'sens_att':'LGBTQ'}

    envs = partition_envs(args.data_fname, params)
    import pickle
    pickle.dump(envs, open('data/irmval_envs.pkl', 'wb'))
