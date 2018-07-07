""" Based on https://github.com/gpapamak/maf/blob/master/datasets/hepmass.py
Code to recreate dataset used in MAF paper.
"""
import pandas as pd
import numpy as np
import gzip
import wget
import shutil
import cPickle as pickle
from collections import Counter
from os.path import join
from ..utils import misc


def load_data(path):
    data_train = pd.read_csv(filepath_or_buffer=join(path, "1000_train.csv"),
                             index_col=False)
    data_test = pd.read_csv(filepath_or_buffer=join(path, "1000_test.csv"),
                            index_col=False)
    return data_train, data_test


def load_data_no_discrete(path):
    """
    Loads the positive class examples from the first 10 percent of the dataset.
    """
    data_train, data_test = load_data(path)

    # Gets rid of any background noise examples i.e. class label 0.
    data_train = data_train[data_train[data_train.columns[0]] == 1]
    data_train = data_train.drop(data_train.columns[0], axis=1)
    data_test = data_test[data_test[data_test.columns[0]] == 1]
    data_test = data_test.drop(data_test.columns[0], axis=1)
    # Because the data set is messed up!
    data_test = data_test.drop(data_test.columns[-1], axis=1)
    ntrain = int(0.1*len(data_train))
    ntest = int(0.1*len(data_test))
    data_train = data_train[:ntrain]
    data_test = data_test[:ntest]

    return data_train, data_test


def load_data_no_discrete_normalised(path):

    data_train, data_test = load_data_no_discrete(path)
    mu = data_train.mean()
    s = data_train.std()
    data_train = (data_train - mu)/s
    data_test = (data_test - mu)/s

    return data_train, data_test


def load_data_no_discrete_normalised_as_array(path):

    data_train, data_test = load_data_no_discrete_normalised(path)
    data_train, data_test = data_train.as_matrix(), data_test.as_matrix()

    # Remove any features that have too many re-occurring real values.
    i = 0
    features_to_remove = []
    for feature in data_train.T:
        c = Counter(feature)
        max_count = np.array([v for k, v in sorted(c.iteritems())])[0]
        if max_count > 5:
            features_to_remove.append(i)
        i += 1
    data_train = data_train[:, np.array(
        [j for j in range(data_train.shape[1]) if j not in features_to_remove])]
    data_test = data_test[:, np.array(
        [j for j in range(data_test.shape[1]) if j not in features_to_remove])]

    N = data_train.shape[0]
    N_validate = int(N*0.1)
    data_validate = data_train[-N_validate:]
    data_train = data_train[0:-N_validate]

    return data_train, data_validate, data_test


def download_and_make_data(datapath):
    url_train = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                 '00347/1000_train.csv.gz')
    url_test = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                '00347/1000_test.csv.gz')
    path = join(datapath, 'hepmass/')
    misc.make_path(path)
    print('Downloading...')
    filename_train = wget.download(url_train, path)
    filename_test = wget.download(url_test, path)
    print('\nExtracting...')
    with gzip.open(filename_train, 'rb') as f_in:
        with open(join(path, '1000_train.csv'), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    with gzip.open(filename_test, 'rb') as f_in:
        with open(join(path, '1000_test.csv'), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print('Processing...')
    trn, val, tst = load_data_no_discrete_normalised_as_array(path)
    print('Saving...')
    outfile = join(path, 'hepmass.p')
    pickle.dump(
        {'train': trn.astype(np.float32),
         'valid': val.astype(np.float32),
         'test': tst.astype(np.float32)},
        open(outfile, 'wb')
    )
