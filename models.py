import itertools
import json
import logging
import os
import warnings

from abc import ABC, abstractmethod
import math
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import torch.autograd as autograd
from torch import nn
from scipy.stats import f as fdist
from scipy.stats import ttest_ind
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression

from ref import make_tensor


class BaseMLP(nn.Module):
    def __init__(self, d, hid_dim):
        super(BaseMLP, self).__init__()
        lin1 = nn.Linear(d, hid_dim)
        lin2 = nn.Linear(hid_dim, hid_dim)
        lin3 = nn.Linear(hid_dim, 1)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), \
                                   lin3)

    def weight_norm(self):
        '''Returns the l1 norm of all weights in the model'''
        # import pdb; pdb.set_trace()
        weight_norm = torch.tensor(0.)
        for w in self.parameters():
            weight_norm += w.norm().pow(2)
        return weight_norm

    def forward(self, input_data):
        out = self._main(input_data)
        return out


class MLP():
    '''Wrapper around BaseMLP class to use as standalone prediction model
       :param data: if batching=False, dstructs {'x':data (npArray), 'y':labels (npArray)}.
                    if true pytorch dataloader with {'x':np, 'y':np} as batches'''
    def __init__(self):
        self.model_arch = None  #Included for purposes of saving archetecture
        self.model = None


    def run(self, data, args, batching=False):
        ''':param data: if batching=False, dstructs {'x':data (npArray), 'y':labels (npArray)}.
                        if true pytorch dataloader with {'x':np, 'y':np} as batches'''

        def update_params(data, y_all, model, optim, l2_reg=None):
            logits = model(make_tensor(data)).squeeze()
            labels = make_tensor(y_all).squeeze()
            loss = nn.functional.binary_cross_entropy_with_logits(logits, \
                                                                  labels)
            weight_norm = model.weight_norm()
            loss += l2_reg * weight_norm

            #Do the backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # def sigmoid(x):
            #     return 1/(1 + np.exp(-x))
            # def thresh(x):
            #     if (x >= 0.5): return 1
            #     else: return 0
            # acc = np.abs(np.array([thresh(e) for e in sigmoid(logits.detach().numpy())]) - labels.detach().numpy()).mean()
            # print(acc)

            return loss

        if batching:
            dim_x = data.dataset.dim
        else:
            dim_x = data['x'].shape[1]

        losses = []
        self.model_arch = BaseMLP(dim_x, args['hid_layers'])
        self.model = BaseMLP(dim_x, args['hid_layers'])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args['lr'])

        for step in tqdm(range(args['n_iterations'])):
            if batching:
                for idx, mbatch in enumerate(data):
                    loss = update_params(mbatch['x'].detach().numpy(), \
                                         mbatch['y'].detach().numpy(), self.model, \
                                         optimizer, l2_reg=args['l2_reg'])
            else:
                loss = update_params(data['x'], data['y'], self.model, \
                                     optimizer, l2_reg=args['l2_reg'])

            #Printing and Logging
            if step % 1000 == 0:
                logging.info([np.int32(step), loss.detach().cpu().numpy()])
            losses.append(loss.detach().numpy())
        return losses

    def predict(self, data):
        '''
        :param data: the dataset (nparray)
        :param self.model - full model object (ie- callable for inference)
        '''
        assert self.model is not None

        #Handle case of no data
        if data.shape[0] == 0:
            return np.array()

        logits = self.model(make_tensor(data))
        preds = nn.functional.sigmoid(logits).detach().numpy()
        return preds

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

class LinearInvariantRiskMinimization(IRMBase):
    """Object Wrapper around IRM"""

    def __init__(self, ptype):
        super().__init__(ptype)
        self.model = None

    def train(self, envs, seed, args, batching=False):
        ''':param envs: if batching=False two possibilities list of training env
                        dstructs {'x':data (npArray), 'y':labels (npArray)}.
                        If true list of dataloaders of each envs'''

        def update_params(envs, model, optimizer, args):
            ''':param envs: list of training env data structures, of form
                             {'x':data (npArray) or (torch.Tensor), 'y':labels (npArray) or torch.Tensor}'''
            e_comp = {}
            for i, e in enumerate(envs):
                e_comp[i] = {}
                data, y_all = e['x'], e['y']
                logits = (make_tensor(data) @ phi @ w).squeeze() #Note - for given loss this is raw output
                labels = make_tensor(y_all).squeeze()
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

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss, train_nll, train_acc, train_penalty

        #######
        if batching:
            dim_x = envs[0].dataset.dim
        else:
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
            if batching:
                nbatch = math.ceil((len(envs[0])/envs[0].batch_size))  #Assume all envs have the same #batches/iter
                for input_envs in zip(*envs):
                    loss, train_nll, train_acc, train_penalty = \
                        update_params(input_envs, phi, optimizer, {'l2_reg':args['l2_reg'], \
                                        'pen_wgt':args['pen_wgt'], \
                                        'penalty_anneal_iters':args['penalty_anneal_iters']})

                    print(loss)

                # for i in range(nbatch):
                #     import pdb; pdb.set_trace()
                #     input_envs = [next(dl) for dl in envs_iters]
                #     loss, train_nll, train_acc, train_penalty = \
                #               update_params(input_envs, phi, optimizer, {'l2_reg':args['l2_reg'], \
                #                                 'pen_wgt':args['pen_wgt'], \
                #                                 'penalty_anneal_iters':args['penalty_anneal_iters']})
            else:
                loss, train_nll, train_acc, train_penalty = \
                              update_params(envs, phi, optimizer, {'l2_reg':args['l2_reg'], \
                                            'pen_wgt':args['pen_wgt'], \
                                            'penalty_anneal_iters':args['penalty_anneal_iters']})

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

        self.model = phi
        return errors, penalties, losses

    def predict(self, data):
        '''
        :param data: the dataset (nparray)
        :param phi_params: The state dict of the MLP'''
        def sigmoid(x):
            return 1/(1 + np.exp(-x))

        assert self.model is not None
        #Handle case of no data
        if data.shape[0] == 0:
            assert False

        phi = self.model.detach().numpy()
        w = np.ones([phi.shape[1], 1])
        return sigmoid(data @ (phi @ w).ravel()).squeeze()


class InvariantRiskMinimization(IRMBase):
    """Object Wrapper around IRM"""

    def __init__(self, ptype):
        super().__init__(ptype)


    def train(self, envs, seed, args, batching=False):
        ''':param envs: if batching=False two possibilities list of training env
                        dstructs {'x':data (npArray), 'y':labels (npArray)}.
                        If true list of dataloaders of each envs'''
        def update_params(envs, model, optimizer, args):
            ''':param envs: list of training env data structures, of form
                             {'x':data (npArray), 'y':labels (npArray)}'''
            e_comp = {}
            for i, e in enumerate(envs):
                e_comp[i] = {}
                data, y_all = e['x'], e['y']
                logits = (make_tensor(data) @ phi @ w).squeeze() #Note - for given loss this is raw output
                labels = make_tensor(y_all).squeeze()
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

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss, train_nll, train_acc, train_penalty

        ###############
        if batching:
            dim_x = dim(envs[0])
        else:
            dim_x = envs[0]['x'].shape[1]

        errors = []
        penalties = []
        losses = []

        phi = BaseMLP(dim_x, args['hid_layers'])
        optimizer = torch.optim.Adam(phi.parameters(), lr=args['lr'])

        logging.info('[step, train nll, train acc, train penalty, test acc]')

        #Start the training loop
        for step in tqdm(range(args['n_iterations'])):
            e_comp = {}
            for i, e in enumerate(envs):
                e_comp[i] = {}
                data, y_all = e['x'], e['y']
                # import pdb; pdb.set_trace()
                # d = make_tensor(data.loc[e_in].values)
                # import pdb; pdb.set_trace()
                logits = phi(make_tensor(data)).squeeze()
                labels = make_tensor(y_all).squeeze()
                e_comp[i]['nll'] = self.mean_nll(logits, labels)
                e_comp[i]['acc'] = self.mean_accuracy(logits, labels)
                e_comp[i]['penalty'] = self.penalty(logits, labels)

            train_nll = torch.stack([e_comp[e]['nll'] \
                                    for e in e_comp]).mean()
            train_acc = torch.stack([e_comp[e]['acc'] \
                                    for e in e_comp]).mean()
            train_penalty = torch.stack([e_comp[e]['penalty'] \
                                        for e in e_comp]).mean()
            loss = train_nll.clone()

            #Regularize the weights
            weight_norm = torch.tensor(0.)
            for w in phi.parameters():
                weight_norm += w.norm().pow(2)
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

        return phi.state_dict(), errors, penalties, losses

    def predict(self, data, phi_params, args={}):
        '''
        :param data: the dataset (nparray)
        :param phi_params: The state dict of the MLP'''
        #Handle case of no data
        if data.shape[0] == 0:
            return pd.DataFrame()

        phi = BaseMLP(data.shape[1], args['hid_layers'])
        phi.load_state_dict(phi_params)
        logits = phi(make_tensor(data))
        preds = nn.functional.sigmoid(logits).detach().numpy()
        return pd.DataFrame(preds)
