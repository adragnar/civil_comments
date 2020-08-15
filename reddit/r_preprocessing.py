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

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
STOPWORDS = set(stopwords.words('english'))

from tqdm import tqdm

import data_proc
import models
import setup_params as setup
import r_setup_params as reddit_setup



def preprocess_data(data, rel_cols, tox_thresh=None, c_len=15, \
                                 social_media=False, stopwords=[]):
    '''Raw preprocessing to be done before data is used on both data All non nan
       values stuff controlled by kwargs
    :param data: pandas dataframe.
    :param rel_cols dictionary mapping keys (data, labels) to names in given dset
    :param tox_thresh: threshold for binarizing tox label (if dataset has labels)
    :param c_len: min length of comment
    :param social_media: Whether to do a variety of social-media specific preprocessing
    :param stopwords: words to take out if social_media=True'''

    #Clean Target
    #remove Nans in comment text column
    data['test_nan']= data[rel_cols['data']].apply(lambda x: 1 if type(x) == str else 0)
    data = data[data['test_nan'] == 1]; data.reset_index(inplace=True)
    data.drop(['test_nan'], axis=1, inplace=True)

    #Do social media preprocess
    if social_media:
        def proc_social(words, stopwords=[]):
            '''Remove html tags and non-words from text preproc
               words: list of tokens'''
            #html
            words = [re.sub(r'<.*?>', '', w).lower() for w in words]

            #stopwords and alphanumeric
            words = [re.sub(r'\W+', '', w).lower() for w in words \
                     if re.sub(r'\W+', '', w).lower() not in stopwords]

            #Remove letter artefacts
            words = [w for w in words if ((len(w) > 1) or (w in ['a', 'i', 'u']))]
            return words

        text_processor = TextPreProcessor(
            # terms that will be normalized
            normalize=['url', 'email', 'percent', 'money', 'phone', 'user', \
                       'time', 'date', 'number'],
            fix_html=True,  # fix HTML tokens
            segmenter="english",
            corrector="english",
            unpack_hashtags=True,  # perform word segmentation on hashtags
            unpack_contractions=True,  # Unpack contractions (can't -> can not)
            spell_correct_elong=False,  # spell correction for elongated words
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
        )

        data[rel_cols['data']] = data[rel_cols['data']].apply( \
            lambda s: " ".join(proc_social(text_processor.pre_process_doc(s), \
                                            stopwords=STOPWORDS)))
    else:  #If no social media
        def proc(txt, stopwords=[]):
            words = txt.split(" ")
            words = [re.sub(r'\W+', '', w).lower() for w in words \
                     if re.sub(r'\W+', '', w).lower() not in stopwords]
            return words

        data[rel_cols['data']] = data[rel_cols['data']].apply(\
                        lambda s: " ".join(proc(s, stopwords=STOPWORDS)))

    #Remove too small comments
    if c_len is not None:
        data['test_len'] = data[rel_cols['data']].apply(lambda x: 1 if (len(str(x)) > c_len) else 0)
        data = data[data['test_len'] == 1]; data.reset_index(inplace=True)
        data.drop(['test_len'], axis=1, inplace=True)

    #Binarize labels if labelled
    if tox_thresh is not None:
        data[rel_cols['labels']] = data[rel_cols['labels']].apply((lambda x: 1 if x > tox_thresh else 0))

    return data

# def preprocess_data(data):
#     '''Clean up just-downloaded reddit datasets for later in pipeline
#     :param data: THe raw datset (pd df)
#     :return: pd Df
#     '''
#
#     #remove Nans in comment text column
#     data['test_nan']= data['body'].apply(lambda x: 1 if type(x) == str else 0)
#
#     #Drop comments that are too short
#     data['test_len'] = data['body'].apply(lambda x: 1 if (len(str(x)) > 15) else 0)
#
#
#     data = data[(data['test_nan'] == 1) & (data['test_len'] == 1)]
#     data.drop(['test_nan', 'test_len'], axis=1, inplace=True)
#     return data

def add_toxlabel(old_fpaths, new_fpaths, rel_cols, mpath, e_path):
    ''':param fname: path to reddit dataset
       :param new_fname: path to labelled reddit dataset
       :param mpath: path to model file
       :param epath: path to embeddings file'''

    t = data_proc.get_word_transform('embed', e_path, proc=False)
    print('WE loaded')
    for fname, new_fname in zip(old_fpaths, new_fpaths):
        data = pd.read_csv(fname)
        data = preprocess_data(data, rel_cols, c_len=0, social_media=False)
        data_embed = t(data['body'])
        print('data')
        model = data_proc.load_saved_model(mpath)
        print('model')
        data['toxicity'] = model.predict(data_embed)
        data.to_csv(new_fname)


if __name__ == '__main__':
    assert (os.getcwd().split('/')[-1] == 'civil_comments') or \
                    (os.getcwd().split('/')[-1] == 'civil_liberties')

    # old_fpaths = ['reddit/data/orig/2014b.csv']
    # new_fpaths = ['reddit/data/labeled/2014b_labeled.csv']
    old_fpaths = ['reddit/data/orig/2014_gendered.csv']
    new_fpaths = ['reddit/data/labeled/2014_gendered_labeled.csv']
    m_path = 'reddit/labelgen_models/0810_labelgen.pkl'
    rel_cols = {'data':'body'}
    add_toxlabel(old_fpaths, \
                 new_fpaths, \
                 rel_cols, \
                 m_path, \
                 setup.get_wordvecspath())
