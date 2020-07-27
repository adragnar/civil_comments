import argparse
import os
from os.path import join

import pandas as pd
import numpy as np


thresh = 0.2

# esplit_from_id = {0:np.array([[0.1, 0.9], [0.2, 0.8], [0.9, 0.1]]), \
#                   1:np.array([[0.1, 0.9], [0.3, 0.7], [0.9, 0.1]]), \
#                   2:np.array([[0.05, 0.95], [0.35, 0.65], [0.9, 0.1]])}
esplit_from_id = {0:[0.1, 0.2, 0.9], \
                  1:[0.1, 0.3, 0.9], \
                  2:[0.05, 0.35, 0.9]
                  }

def get_sensatt_column(data, satt):
    '''From the sensitive attribute described, return a pd series with that
       indicator'''

    if satt == 'LGBTQ':
        return data[['homosexual_gay_or_lesbian', 'bisexual', 'other_sexual_orientation']].max(axis=1)
    elif satt == 'muslim':
        return data['muslim']
    else:
        raise Exception('satt not implemented')

def balance_sa_partitions(data, env_splits, label, seed):
    '''Given DataFrame of data tagged with SA (z=1), split into y=1 and y=0
    in proportions given by esplit
    :param data: dataframe. Labels binarized (pd DataFrame)
    :param esplit: list of p_e for each env, last is test
    :return data where (y_Bar=0, z=1), (y_Bar=1, z=1)'''
    toxic, non_toxic = data[data[label] == 1].sample(frac=1).reset_index(drop=True), \
                            data[data[label] == 0].sample(frac=1).reset_index(drop=True)

    totals = {'nt':len(non_toxic), 't':len(toxic)}

    weights = {'nt':sum(env_splits)/len(env_splits), 't':sum((1-e) for e in env_splits)/len(env_splits)}

    #Adjust so that desired env splits possible
    req_t = int(float(weights['t']/weights['nt']) * totals['nt']) #req if other max used
    req_nt = int(float(weights['nt']/weights['t']) * totals['t'])

    if req_nt < totals['nt']:  #figure out limiting data source
        non_toxic = non_toxic.sample(n=req_nt, random_state=seed)
    elif req_nt > totals['nt']:
        toxic = toxic.sample(n=req_t, random_state=seed)

    return non_toxic, toxic

def compute_proportions_of_env(pe, diff_z, l_noise):
    '''Given a probability of spurious label flip, return relative probabilites
    of different classes. Includes option where different SA labels
    :param diff_z bool - whetehr or not there is spurious corr with some class
    :param l_noise: amount of label noise present (float)'''
    if (not diff_z) and (l_noise == 0):
        ret =  {'y0_sa1':pe, 'y1_sa1':(1 - pe)}
    elif diff_z and (l_noise == 0):
        ret = {'y0_sa1':pe/2, 'y1_sa1':(1 - pe)/2, 'y0_sa0':(1 - pe)/2, 'y1_sa0':pe/2}
    elif (not diff_z) and (l_noise > 0):
        ret = {'y0_sa1':pe * (1 - l_noise), 'y1_sa1':(1 - pe) * (1 - l_noise), \
                'y0_sa1_l':pe * l_noise, 'y1_sa1_l':(1 - pe) * l_noise}
    elif diff_z and (l_noise > 0):
        ret = {'y0_sa1':pe/2 * (1 - l_noise), 'y1_sa1':(1 - pe)/2 * (1 - l_noise), 'y0_sa0':(1 - pe)/2 * (1 - l_noise), 'y1_sa0':pe/2 * (1 - l_noise), \
                'y0_sa1_l':pe/2 * l_noise, 'y1_sa1_l':(1 - pe)/2 * l_noise, 'y0_sa0_l':(1 - pe)/2 * l_noise, 'y1_sa0_l':pe/2* l_noise}
    else:
        raise Exception('Not implemented combination')

    assert sum(ret.values()) == 1.0
    return ret

def split_envs(src_dict, prop_dict, env_size, seed):
    '''Given two repositoreis of data that are in the desired proportions to
    eachother overall, split into specified number of environments (equal size)
    with proprotions for each env given in p_list

    :param src_dict: dict of the data partitions to sample from
    :param prop_dict: dict of relative proporitons from each ith sample
    :param env_size: Ultimate size of the final envirornment
    :return a version of src list with the sampled data removed'''

    assert len(src_dict) == len(prop_dict)

    sampled_src = {}
    env_partitions = []
    for p, d_source in src_dict.items():
        nsamples = int(prop_dict[p] * env_size)
        e = d_source.sample(n=nsamples, random_state=seed)
        env_partitions.append(e)
        sampled_src[p] = d_source.drop(e.index, axis=0)

    return pd.concat(env_partitions, ignore_index=True).sample(frac=1), \
              sampled_src

def add_label_noise(ds, lnoise, seed):
    '''Given a dict of data sources, return a new dict with double the sources,
    where each original is spilt into 2 - where the 2nd has labels flipped.
    Note - only to be applied when all other partitions have been defined
    :param ds: dicitonary of data sources {name:DataFrame}
    :param lnoise: probability with which to flip label
    :return a dicitonary of all paritions, new ones denoted with _l'''

    tmp_datal = {}
    for k, v in ds.items():
        tmp_datal[(k+'_l')] = ds[k].sample(frac=lnoise, random_state=seed)
        tmp_datal[(k+'_l')]['toxicity'] = tmp_datal[(k+'_l')]['toxicity'].apply(lambda x: 1 if x == 0 else 0)
        ds[k] = ds[k].drop(tmp_datal[(k+'_l')].index, axis=0)
    #Merge label flipped and regular data
    return {**ds, **tmp_datal}


def load_data(fname, args):
    full_data = pd.read_csv(fname)
    full_data[args['sens_att']] = get_sensatt_column(full_data, args['sens_att'])
    full_data['toxicity'] = full_data['toxicity'].apply((lambda x: 1 if x > thresh else 0))
    return full_data

def partition_envs_labelshift(fname, args):
    '''Partition environments such that occurence of SA (or its lack) does not
    vary across envs, but the label associaion does.'''
    def compute_proportions_of_env(pe, l_noise):
        '''Given a probability of spurious label flip, return relative probabilites
        of different classes. Includes option where different SA labels
        :param diff_z bool - whetehr or not there is spurious corr with some class
        :param l_noise: amount of label noise present (float)'''
        if (l_noise == 0):
            ret =  {'y0':pe, 'y1':(1 - pe)}

        else:
            ret = {'y0':pe * (1 - l_noise), 'y1':(1 - pe) * (1 - l_noise), \
                    'y0_l':pe * l_noise, 'y1_l':(1 - pe) * l_noise}

        assert sum(ret.values()) == 1.0
        return ret

    full_data = load_data(fname, args)

    #Data Processing
    if args['exptype'] == 'lshift_sa':
        full_partition = full_data[(full_data[args['sens_att']] > 0)]
    elif args['exptype'] == 'lshift_reg':
        full_partition = full_data[(full_data[args['sens_att']] == 0)]
    else:
        assert False

    data_sources = {}
    data_sources['y0'], data_sources['y1'] = balance_sa_partitions(full_partition, \
                      esplit_from_id[args['env_id']], 'toxicity', args['seed'])

    if args['label_noise'] > 0:
        data_sources = add_label_noise(data_sources, args['label_noise'], args['seed'])

    nenvs = len(esplit_from_id[args['env_id']])
    total_nsamples = sum([d.shape[0] for d in list(data_sources.values())])

    #Make each environment
    env_partitions = []
    for pe in esplit_from_id[args['env_id']]:
        src_props = compute_proportions_of_env(pe, args['label_noise'])
        env, data_sources = split_envs({k:data_sources[k] for k in src_props}, src_props, int(total_nsamples/nenvs), args['seed'])
        env_partitions.append(env)

    return env_partitions


def partition_envs_cmnist(fname, args):
    '''Take a pd Dataframe and split it into list of environments
    :param fname: path to dataset
    :param args['seed']: random seed (int)
    :param args['env_id']: id mapping to environment partition (int)
    :param args['label_noise']: chance that label is flipped (float)
    :params args['sens_att']: col in dataset whose data to include
    :return list of numpy arrays, last is testenv'''

    def compute_proportions_of_env(pe, l_noise):
        '''Given a probability of spurious label flip, return relative probabilites
        of different classes. Includes option where different SA labels
        :param diff_z bool - whetehr or not there is spurious corr with some class
        :param l_noise: amount of label noise present (float)'''

        if l_noise == 0:
            ret = {'y0_sa1':pe/2, 'y1_sa1':(1 - pe)/2, 'y0_sa0':(1 - pe)/2, 'y1_sa0':pe/2}
        else:
            ret = {'y0_sa1':pe/2 * (1 - l_noise), 'y1_sa1':(1 - pe)/2 * (1 - l_noise), 'y0_sa0':(1 - pe)/2 * (1 - l_noise), 'y1_sa0':pe/2 * (1 - l_noise), \
                    'y0_sa1_l':pe/2 * l_noise, 'y1_sa1_l':(1 - pe)/2 * l_noise, 'y0_sa0_l':(1 - pe)/2 * l_noise, 'y1_sa0_l':pe/2* l_noise}

        assert sum(ret.values()) == 1.0
        return ret

    #Start Function
    full_data = load_data(fname, args)

    #Data Processing
    sa_partition = full_data[(full_data[args['sens_att']] > 0)]
    non_sa_partition = full_data[(full_data[args['sens_att']] == 0)]

    data_sources = {}
    data_sources['y0_sa1'], data_sources['y1_sa1'] = balance_sa_partitions(sa_partition, \
                      esplit_from_id[args['env_id']], 'toxicity', args['seed'])

    # #Get Non SA Attributes if needed
    # if args['shift_type'] == 'zshift':
    yi_size = len(data_sources['y0_sa1']) + len(data_sources['y1_sa1'])  #Assuming labels balanced in all envs so |y0_sa1| = |y1_sa0|
    data_sources['y0_sa0'] = non_sa_partition[(non_sa_partition['toxicity'] == 0)].sample( \
                          n=len(data_sources['y1_sa1']), random_state=args['seed'])
    data_sources['y1_sa0'] = non_sa_partition[(non_sa_partition['toxicity'] == 1)].sample( \
                          n=len(data_sources['y0_sa1']), random_state=args['seed'])

    #Apply Label Noise if needed
    if args['label_noise'] > 0:
        data_sources = add_label_noise(data_sources, args['label_noise'], args['seed'])

    nenvs = len(esplit_from_id[args['env_id']])
    total_nsamples = sum([d.shape[0] for d in list(data_sources.values())])

    #Make each environment
    env_partitions = []
    for pe in esplit_from_id[args['env_id']]:
        src_props = compute_proportions_of_env(pe, args['label_noise'])
        env, data_sources = split_envs({k:data_sources[k] for k in src_props}, src_props, int(total_nsamples/nenvs), args['seed'])
        env_partitions.append(env)
    import pdb; pdb.set_trace()
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
              'label_noise':0.25, \
              'shift_type':'zshift', \
              'sens_att':'LGBTQ'}

    envs = partition_envs(args.data_fname, params)
    import pickle
    pickle.dump(envs, open('data/irmval_envs.pkl', 'wb'))
