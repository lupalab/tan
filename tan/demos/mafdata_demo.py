import os
import numpy as np
import cPickle as pickle
import urllib
import tarfile
import matplotlib; matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt  # noqa
from datetime import datetime
from ..experiments import runner
from ..external_maf import bsds300, hepmass, miniboone, power, gas
from ..model import transforms as trans
from ..utils import misc


def save_data(trn, val, tst, dataset, datadir):
    path = os.path.join(datadir, dataset+'/')
    misc.make_path(path)
    outfile = os.path.join(path, dataset+'.p')
    pickle.dump(
        {'train': trn.x.astype(np.float32),
         'valid': val.x.astype(np.float32),
         'test': tst.x.astype(np.float32)},
        open(outfile, 'wb')
    )
    return None


def download_and_make_data(home=None, download=True):
    """ Download MAF preprocessed datasets and convert to pickle files to use
    for training. Assumes that this is being called from outer tan directory.
    Saves pickle files in specifed home directory.
    """
    # Download data from MAF repository
    if home is None:
        home = os.path.expanduser('~')
    datadir = os.path.join(home, 'data/tan')
    if download:
        print('Downloading MAF data...')
        savedir = 'tan/external_maf/datasets'
        url = 'https://zenodo.org/record/1161203/files/data.tar.gz'
        local_filename = os.path.join(savedir, 'data.tar.gz')
        urllib.urlretrieve(url, local_filename)
        tar = tarfile.open(local_filename, "r:gz")
        tar.extractall(savedir)
        tar.close()
        os.remove(local_filename)
    # bsds
    print('\nMake BSDS...')
    dataset = 'maf_bsds'
    data = bsds300.BSDS300()
    save_data(data.trn, data.val, data.tst, dataset, datadir)
    # hepmass
    print('Making HEPMASS...')
    dataset = 'maf_hepmass'
    data = hepmass.HEPMASS()
    save_data(data.trn, data.val, data.tst, dataset, datadir)
    # miniboone
    print('Making MINIBOONE...')
    dataset = 'maf_miniboone'
    data = miniboone.MINIBOONE()
    save_data(data.trn, data.val, data.tst, dataset, datadir)
    # power
    print('Making POWER...')
    dataset = 'maf_power'
    data = power.POWER()
    save_data(data.trn, data.val, data.tst, dataset, datadir)
    # gas
    print('Making GAS...')
    dataset = 'maf_gas'
    data = gas.GAS()
    save_data(data.trn, data.val, data.tst, dataset, datadir)
    return None


def main(dataset='maf_hepmass', ntrls=10, home=None, init_lr=None):
    # Download and setup data.
    if home is None:
        home = os.path.expanduser('~')
    datadir = os.path.join(home, 'data/tan')
    datapath = os.path.join(datadir, '{name}/{name}.p'.format(name=dataset))
    print('Loading test data for plotting.')
    data = pickle.load(open(datapath, 'rb'))['test']
    # Train using intermediate Linear, leaky relu, and RNN shifts
    if init_lr is None:
        init_lr = 0.0015 if dataset == 'maf_miniboone' else 0.005
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
        datapath, arg_list=runner.misc.make_arguments(ac), home=home)
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
