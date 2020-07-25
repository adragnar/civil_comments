def get_hyperparams(algo):
    if algo == 'logreg':
        return  {'lr': 0.001, \
                 'n_iterations':70000, \
                 'penalty_anneal_iters':1, \
                 'l2_reg':1.0, \
                 'pen_wgt':5000, \
                 'hid_layers':1, \
                 'verbose':False}

    elif algo == 'mlp':
        assert False 
