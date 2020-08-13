import gensim
import codecs
import json

from math import ceil

from gensim.models.keyedvectors import Word2VecKeyedVectors
from gensim.models import KeyedVectors
from gensim.models import FastText
from gensim.scripts.glove2word2vec import glove2word2vec

import collections
import copy
import itertools
import random
import sklearn

import re
import tqdm
import socket

import warnings
warnings.filterwarnings("ignore")

import importlib
import pickle
from collections import defaultdict, Counter
from typing import List, Dict

import torch
from torch import utils
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import ref

STOPWORDS = set(stopwords.words('english'))




class ToxicityDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, rel_cols={'data':'comment_text', 'labels':'toxicity'}, transform=None):
        """
        Args:
            data (pd dataframe): The pd dataframe with (uid, tox_label, text)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset = data[list(rel_cols.values())]
        self.cols = rel_cols
        self.transform = transform
        self.dim = 300

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset.loc[idx, self.cols['data']]
        if self.transform:
            sample = self.transform(sample).astype(np.float64)
        target = self.dataset.loc[idx, self.cols['labels']].astype(np.float64)
        if 'numpy' not in str(type(target)):
            target = target.values
        #
        return {'x':sample, 'y':target} #Fix this without hack

class GetEmbedding(object):
    """Given a sentence of text, generate sentence embedding

    Args:
        model: word embedding model, dictionary of wrods -> embeds
    """

    def __init__(self, model, stopwords=[]):
        self.model = model
        self.stopwords = stopwords
        self.unknown_embed = np.zeros(300)  #NOTE - this may not be same for all WEs

    def __call__(self, sample):
        ''':param sample: pd.Series'''
        if type(sample) == str:
            words = sent_proc(sample, stopwords=self.stopwords)
            if len(words) == 0:
                words = set(np.zeros(300))
            sent_embedding = np.sum([self.model[w] if w in self.model else self.unknown_embed for w in words], axis = 0)
        elif type(sample) == pd.Series:
            sent_embedding = np.zeros((len(sample), 300))
            for i, txt in enumerate(sample):
                words = sent_proc(txt, stopwords=self.stopwords)
                if len(words) == 0:
                    words = set(np.zeros(300))
                sent_embedding[i, :] = np.sum([self.model[w] if w in self.model else self.unknown_embed for w in words], axis = 0)

        return sent_embedding

class GetBOW(object):
    """Given a sentence of text, generate BOW rep

    Args:
        vocab: dictionary, word-->index in array (assume contigious) """

    def __init__(self, vocab, lem=None, stopwords=[]):
        self.vocab = vocab
        self.stopwords = stopwords
        self.lem = lem

    def __call__(self, sample):
        ''':param sample: str or pd.Series'''
        def get_rep(txt):
            rep = np.zeros(len(self.vocab))
            words = sent_proc(txt, stopwords=self.stopwords, lem=self.lem)
            for w in words:
                try:
                    rep[self.vocab[w]] = 1
                except KeyError:
                    continue
            return rep

        if type(sample) == str:
            bow_embed = get_rep(sample)
        elif type(sample) == pd.Series:
            bow_embed = np.zeros((len(sample), len(self.vocab)))
            for i, txt in enumerate(sample):
                bow_embed[i, :] = get_rep(txt)

        return bow_embed

def generate_dataset(d, elg, t=None):
    full = d[['id', 'toxicity', 'comment_text']]
    full = full[elg]

    #convert to pytorch formatting
    full = ToxicityDataset(full, transform=t)
    return full

def generate_dataloader(d, elg, nbatch=1, t=None):
    full = generate_dataset(d, elg, t=t)
    full = DataLoader(full, ceil(len(full)/nbatch), shuffle=False)
    return full


#Word Embeddings/Processing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
STOPWORDS = set(stopwords.words('english'))

def get_word_transform(e_type, fpath):
    if e_type == 'embed':
        word2vec, _, _ = load_word_vectors(fpath)
        t = GetEmbedding(word2vec, stopwords=STOPWORDS)
    elif e_type == 'BOW':
        word_freq = pickle.load(open(fpath, 'rb'))
        vocabulary = sorted([x for x in word_freq if ((word_freq[x] > 20) and (word_freq[x] < 1e3))])
        vocabulary = {vocabulary[i]:i for i in range(len(vocabulary))}
        t = GetBOW(vocabulary, lem=WordNetLemmatizer(), stopwords=STOPWORDS)
    return t

def load_word_vectors(fname):
    def get_nvecs():
        hostname = socket.gethostname()
        if hostname == "Roberts-MacBook-Pro.local":
            return 1000
        else:
            return None
    model = KeyedVectors.load_word2vec_format(fname, limit=get_nvecs(), binary=False)
    vecs = model.vectors
    words = list(model.vocab.keys())
    return model, vecs, words

def sent_proc(txt, stopwords=[], lem=None):
    words = txt.split(" ")
    words = {re.sub(r'\W+', '', w).lower() for w in words \
             if re.sub(r'\W+', '', w).lower() not in stopwords}
    if lem is not None:
        words = {lem.lemmatize(w) for w in words}
    return words

def load_saved_model(mpath):
    '''Given path to a model data structure manually saved, reconstruct the
    appropiate model objext (ie - such that .predict() can be called)'''
    model_info = pickle.load(open(mpath, 'rb'))

    if type(model_info['model']) != dict:  #If model can be stored straight
        return model_info['model']

    #reconsturcted MLP
    elif set(['base_model', 'model_arch', 'model_params']) ==  set(model_info['model'].keys()):
        base = model_info['model']['base_model']
        base.model = model_info['model']['model_arch']
        base.model.load_state_dict(model_info['model']['model_params'])

    #Reconsturcted Linear IRM
    elif set(['base_model', 'model']) ==  set(model_info['model'].keys()): 
        base = model_info['model']['base_model']
        base.model = model_info['model']['model']
    else:
        raise Exception ('Bad format')

    return base
