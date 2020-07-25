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

    def __init__(self, data, transform=None):
        """
        Args:
            data (pd dataframe): The pd dataframe with (uid, tox_label, text)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset = data
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset.loc[idx, 'comment_text']
        if self.transform:
            sample = self.transform(sample)

        target = self.dataset.loc[idx, 'toxicity']

        return {'x':sample, 'y':np.expand_dims(target.values, axis=1)} #Fix this without hack

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
            words = sample.split(" ")
            words = [w for w in words if w.lower() not in self.stopwords]
            sent_embedding = np.sum([self.model[w] if w in self.model else self.unknown_embed for w in words], axis = 0)
        elif type(sample) == pd.Series:
            sent_embedding = np.zeros((len(sample), 300))
            for i, txt in enumerate(sample):
                words = txt.split(" ")
                words = [w for w in words if w.lower() not in self.stopwords]
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
def load_word_vectors(fname):
    model = KeyedVectors.load_word2vec_format(fname, limit=int(1e6), binary=False)
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
