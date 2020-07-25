import itertools
import json
import logging
import os
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

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def pred_binarize(v):
    '''Convert all values to 0 if <0.5, 1 otherwise'''
    def thresh(x):
        if (x >= 0.5): return 1
        else: return 0
    print(v.shape)
    return np.apply_along_axis(thresh, 1, v)

def compute_loss(pred, ground, ltype='MSE'):
    '''Compute loss between two prediction vectors'''
    if ltype == 'MSE':
        return F.mse_loss(torch.tensor(pred).float(), torch.tensor(ground).float()).numpy()
    elif ltype == 'BCE':
        return F.binary_cross_entropy_with_logits(torch.tensor(pred).float(), torch.tensor(ground).float()).numpy()
    if ltype == 'ACC':
        pred = pred_binarize(pred)
        return 1 - F.mse_loss(torch.tensor(pred).float(), torch.tensor(ground).float()).numpy()

def make_tensor(arr):
    '''Convert np array into a float tensor'''
    return torch.from_numpy(arr).float()
