import os
import numpy as np
import cPickle as pickle
import tan.experiments.runner as runner
import matplotlib; matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt  # noqa
from ..data import hepmass as hepmass
from ..model import transforms as trans


def main(download=True, run_org=True, run_new=True, ntrls=1):
    # Download and setup data.
    home = os.path.expanduser('~')
    datadir = os.path.join(home, 'data/tan')
    data_path = os.path.join(datadir, 'hepmass/hepmass.p')
    if download:
        hepmass.download_and_make_data(datadir)
    data = pickle.load(open(data_path, 'rb'))
    # Train as in paper.
    if run_org:
        ac = {
            'lr_decay': (0.5, ),
            'init_lr': (0.005, ),
            'first_trainable_A': (True, ),
            'trans_funcs': ([trans.simple_rnn_transform, ], ),
            'cond_func': (runner.conds.rnn_model, ),
            'param_nlayers': (2, ),
            'trial': range(ntrls),
        }
        ret = runner.run_experiment(
            data_path, arg_list=runner.misc.make_arguments(ac))
        test_llks = ret[0]['results']['test_llks']
        mean_test_llks = np.mean(test_llks)
        stderr_test_llks = np.std(test_llks)
        print('{}\nAverage Test Log-Likelihood: {} +/- {}'.format(
            data_path, mean_test_llks,
            2*stderr_test_llks/np.sqrt(len(test_llks))))
        # plot samples
        samples = ret[0]['results']['samples']
        perm = np.random.permutation(data['test'].shape[0])[:1024]
        fig = plt.figure()
        testax = plt.scatter(
            data['test'][perm, 0], data['test'][perm, -1], c='blue')
        sampax = plt.scatter(samples[:, 0], samples[:, -1], c='red')
        plt.xlabel('first dim')
        plt.ylabel('last dim')
        fig.legend((testax, sampax), ('Test', 'Sampled'))
        fig_path = os.path.join(datadir, 'hepmass/firstlast.png')
        fig.savefig(fig_path)
    else:
        ret = None
    # Train new config using intermediate Linear, leaky relu, and RNN shifts
    if run_new:
        ac = {
            'lr_decay': (0.5, ),
            'init_lr': (0.005, ),
            'first_trainable_A': (True, ),
            'trans_funcs': (
                [trans.leaky_transformation,
                 trans.log_rescale, trans.rnn_coupling, trans.reverse,
                 trans.linear_map, trans.leaky_transformation,
                 trans.log_rescale, trans.rnn_coupling, trans.reverse,
                 trans.linear_map, trans.leaky_transformation,
                 trans.log_rescale, trans.rnn_coupling, trans.reverse,
                 trans.linear_map, trans.leaky_transformation,
                 trans.log_rescale, trans.rnn_coupling, trans.reverse,
                 trans.linear_map, trans.leaky_transformation,
                 trans.log_rescale, ], ),
            'cond_func': (runner.conds.rnn_model, ),
            'param_nlayers': (2, ),
            'train_iters': (60000, ),
            'batch_size': (1024, ),
            'relu_alpha': (None, ),
            'trial': range(ntrls),
        }
        ret_new = runner.run_experiment(
            data_path, arg_list=runner.misc.make_arguments(ac))
        test_llks = ret_new[0]['results']['test_llks']
        mean_test_llks = np.mean(test_llks)
        stderr_test_llks = np.std(test_llks)
        print('{}\nAverage Test Log-Likelihood: {} +/- {}'.format(
            data_path, mean_test_llks,
            2*stderr_test_llks/np.sqrt(len(test_llks))))
        # plot samples
        samples = ret_new[0]['results']['samples']
        perm = np.random.permutation(data['test'].shape[0])[:1024]
        fig = plt.figure()
        testax = plt.scatter(
            data['test'][perm, 0], data['test'][perm, -1], c='blue')
        sampax = plt.scatter(samples[:, 0], samples[:, -1], c='red')
        plt.xlabel('first dim')
        plt.ylabel('last dim')
        fig.legend((testax, sampax), ('Test', 'Sampled'))
        fig_path = os.path.join(datadir, 'hepmass/new_firstlast.png')
        fig.savefig(fig_path)
    else:
        ret_new = None
    return ret, ret_new
