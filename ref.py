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

def pred_binarize(v):
    '''Convert all values to 0 if <0.5, 1 otherwise
    :param v: npArray with dim=1'''
    def thresh(x):
        if (x >= 0.5): return 1
        else: return 0
    return np.array([thresh(e) for e in v])

def compute_loss(pred, ground, ltype='MSE'):
    '''Compute loss between two prediction vectors
    :param pred: The final predictions (not logits) of classifier - {0-1} npArr
    :param ground: The ground truth labels - {0-1} npArr'''
    #Inputs can be any sort of vector - normalize dims
    pred, ground = pred.squeeze(), ground.squeeze()
    assert (pred.shape == ground.shape) and (len(pred.shape) == 1)

    if ltype == 'MSE':
        return F.mse_loss(torch.tensor(pred).float(), torch.tensor(ground).float()).numpy()
    elif ltype == 'BCE':
        return F.binary_cross_entropy(torch.tensor(pred).float(), torch.tensor(ground).float()).numpy()
    if ltype == 'ACC':
        pred = pred_binarize(pred)
        return 1 - F.mse_loss(torch.tensor(pred.squeeze()).float(), torch.tensor(ground.squeeze()).float()).numpy()

def make_tensor(arr):
    '''Convert np array into a float tensor, or pass throiugh a regular tensor'''
    if type(arr) == torch.Tensor:
        return arr.float()
    elif type(arr) == np.ndarray:
        return torch.from_numpy(arr).float()
    else:
        raise Exception('Unimplemented')
