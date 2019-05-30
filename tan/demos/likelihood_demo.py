import os
import numpy as np
import cPickle as pickle
import matplotlib; matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt  # noqa
from datetime import datetime
from ..experiments import runner
from ..data import bsds, hepmass, miniboone, power, gas, miniboone_noprune
from ..model import transforms as trans


def download_and_make_data(dataset, datadir):
    if dataset == 'bsds':
        # NOTE: makes rather big (~7.5G) pickle file, which takes time to
        # load/save.
        bsds.download_and_make_data(datadir)
    elif dataset == 'hepmass':
        hepmass.download_and_make_data(datadir)
    elif dataset == 'miniboone':
        miniboone.download_and_make_data(datadir)
    elif dataset == 'miniboone_noprune':
        miniboone_noprune.download_and_make_data(datadir)
    elif dataset == 'power':
        power.download_and_make_data(datadir)
    elif dataset == 'gas':
        gas.download_and_make_data(datadir)
    else:
        raise NotImplementedError
    return None


def main(dataset='hepmass', download=True, ntrls=10, home=None, init_lr=None):
    # Download and setup data.
    if home is None:
        home = os.path.expanduser('~')
    datadir = os.path.join(home, 'data/tan')
    datapath = os.path.join(datadir, '{name}/{name}.p'.format(name=dataset))
    if download:
        download_and_make_data(dataset, datadir)
    print('Loading test data for plotting.')
    data = pickle.load(open(datapath, 'rb'))['test']
    # Train using intermediate Linear, leaky relu, and RNN shifts
    if init_lr is None:
        init_lr = 0.0015 if dataset == 'miniboone' else 0.005
    ac = {
        'init_lr': (init_lr, ),
        'lr_decay': (0.5, ),
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
        datapath, arg_list=runner.misc.make_arguments(ac))
    results = ret_new[0]['results']
    # Get test likelihoods
    test_llks = results['test_llks']
    results['mean_test_llks'] = np.mean(test_llks)
    results['std_test_llks'] = np.std(test_llks)
    results['stderr_test_llks'] = \
        2*results['std_test_llks']/np.sqrt(len(test_llks))
    print('{}\nAverage Test Log-Likelihood: {} +/- {}'.format(
        datapath, results['mean_test_llks'], results['stderr_test_llks']))
    # plot samples
    samples = results['samples']
    perm = np.random.permutation(data.shape[0])[:1024]
    fig = plt.figure()
    testax = plt.scatter(
        data[perm, 0], data[perm, -1], c='blue')
    sampax = plt.scatter(samples[:, 0], samples[:, -1], c='red')
    plt.xlabel('first dim')
    plt.ylabel('last dim')
    fig.legend((testax, sampax), ('Test', 'Sampled'))
    fig_path = os.path.join(datadir, '{}/{}_firstlast.png'.format(
        dataset, datetime.now().strftime('%m_%d_%Y_%H_%M_%S')))
    fig.savefig(fig_path)
    return results
