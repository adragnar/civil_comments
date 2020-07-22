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

class IRMBase(ABC):
    '''Base class for all IRM implementations'''
    def __init__(self, ptype):
        '''Ptype: p for regression problem, cls for binary classification'''
        self.ptype = ptype

    @abstractmethod
    def train(self, data, y_all, environments, seed, args):
        pass

    @abstractmethod
    def predict(self, data, phi_params, hid_layers=100):
        pass

    def mean_nll(self, logits, y):
        if self.ptype == 'cls':
            return nn.functional.binary_cross_entropy_with_logits(logits, y)
        elif self.ptype == 'reg':
            return nn.functional.mse_loss(logits, y)
        else:
            raise Exception('Unimplemented Problem')

    def mean_accuracy(self, logits, y):
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-2).float().mean()

    def penalty(self, logits, y):
        scale = torch.tensor(1.).requires_grad_()
        loss = self.mean_nll(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    def run(self, data, y_all, d_atts, unid, expdir, seed, \
                env_atts_types, eq_estrat, args):
        assert False #Not implemented for this case yet
        phi_fname = os.path.join(expdir, 'phi_{}.pt'.format(unid))
        errors_fname = os.path.join(expdir, 'errors_{}.npy'.format(unid))
        penalties_fname = os.path.join(expdir, 'penalties_{}.npy'.format(unid))
        losses_fname = os.path.join(expdir, 'losses_{}.npy'.format(unid))


        #Now start with IRM itself
        phi, errors, penalties, losses = self.train(envs, seed, args)

        #Save Results
        torch.save(phi, phi_fname)
        np.save(errors_fname, errors)
        np.save(penalties_fname, penalties)
        np.save(losses_fname, losses)


class LinearInvariantRiskMinimization(IRMBase):
    """Object Wrapper around IRM"""

    def __init__(self, ptype):
        super().__init__(ptype)

    def train(self, envs, seed, args):
        ''':param envs: list of training env data structures, of form
                         {'x':data (npArray), 'y':labels (npArray)}'''
        dim_x = envs[0]['x'].shape[1]

        errors = []
        penalties = []
        losses = []

        phi = torch.nn.Parameter(torch.empty(dim_x, \
                                            args['hid_layers']).normal_(\
                                            generator=torch.manual_seed(seed)))
        w = torch.ones(args['hid_layers'], 1)
        w.requires_grad = True
        optimizer = torch.optim.Adam([phi], lr=args['lr'])

        logging.info('[step, train nll, train acc, train penalty, test acc]')

        #Start the training loop
        for step in tqdm(range(args['n_iterations'])):
            e_comp = {}
            for i, e in enumerate(envs):
                e_comp[i] = {}
                data, y_all = e['x'], e['y']
                logits = make_tensor(data) @ phi @ w #Note - for given loss this is raw output
                labels = make_tensor(y_all)
                e_comp[i]['nll'] = self.mean_nll(logits, labels)
                e_comp[i]['acc'] = self.mean_accuracy(logits, labels)
                e_comp[i]['penalty'] = self.penalty(logits, labels)

            train_nll = torch.stack([e_comp[e]['nll'] \
                                     for e in e_comp]).mean()
            train_acc = torch.stack([e_comp[e]['acc']
                                     for e in e_comp]).mean()
            train_penalty = torch.stack([e_comp[e]['penalty']
                                         for e in e_comp]).mean()
            loss = train_nll.clone()

            #Regularize the weights
            weight_norm = phi.norm().pow(2)
            loss += args['l2_reg'] * weight_norm

            #Add the invariance penalty
            penalty_weight = (args['pen_wgt']
                              if step >= args['penalty_anneal_iters'] else 1.0)
            loss += penalty_weight * train_penalty
            if penalty_weight > 1.0: # Rescale big loss
                loss /= penalty_weight

            #Do the backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #Printing and Logging
            if step % 1000 == 0:
                logging.info([np.int32(step),
                              train_nll.detach().cpu().numpy(),
                              train_acc.detach().cpu().numpy(),
                              train_penalty.detach().cpu().numpy()]
                             )


            errors.append(train_nll.detach().numpy())
            penalties.append(train_penalty.detach().numpy())
            losses.append(loss.detach().numpy())

        return phi, errors, penalties, losses

    def predict(self, data, phi_params):
        '''
        :param data: the dataset (nparray)
        :param phi_params: The state dict of the MLP'''
        def sigmoid(x):
            return 1/(1 + np.exp(-x))

        #Handle case of no data
        if data.shape[0] == 0:
            assert False

        phi = phi_params.detach().numpy()
        w = np.ones([phi.shape[1], 1])
        return sigmoid(data @ (phi @ w).ravel())
