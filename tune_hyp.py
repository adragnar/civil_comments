import ref
import matplotlib.pyplot as plt
import itertools
import pickle
from sklearn.model_selection import train_test_split

import models

def tune_hyp(id, expdir, data_fname, args):
    base_envs = pickle.load(open(data_fname, 'rb'))
    train_envs = base_envs[:-1]
    val_env = base_envs[:-1]


# lr = [0.001]
# n_iterations = [50000]
# pen_wgt = [10, 50, 1]

    if args['model'] == 'logreg'
        base = models.LinearInvariantRiskMinimization('cls')
    elif args['model'] == 'mlp'
        base = models.InvariantRiskMinimization('cls')
    model, errors, penalties, losses = base.train(train_envs, 1000, args)

    train_logits = base.predict(np.concatenate([train_envs[i]['x'] for i in range(len(train_envs))]), irm_model)
    train_labels = np.concatenate([train_envs[i]['y'] for i in range(len(train_envs))])
    test_logits = base.predict(test_partition['x'], irm_model)
    test_labels = test_partition['y']
    train_acc = ref.compute_loss(np.expand_dims(train_logits, axis=1), train_labels, ltype='ACC')
    test_acc = ref.compute_loss(np.expand_dims(test_logits, axis=1), test_labels, ltype='ACC')

    res[(a,b,c)] = {'model':model, 'errors':errors, 'penalties':penalties, 'losses':losses}
`

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('id', type=str, \
                        help='unique id of exp')
    parser.add_argument("expdir", type=str, default=None,
                        help="path to location to save files")
    parser.add_argument("data_fname", type=str,
                        help="filename adult.csv")
    parser.add_argument("model", type=str, default=None)
    args = parser.parse_args()

    params = {'seed':args.seed, \
              'model':args.model, \
              'lr': args.lr, \
              'n_iterations': args.n_iters, \
              'penalty_anneal_iters':1, \
              'l2_reg':args.l2_reg, \
              'pen_wgt': args.pen_wgt, \
              'hid_layers':args.hid_layers, \
              'verbose':False}
              }


    tune_hyp(args.id, args.expdir, args.data_fname, params)
