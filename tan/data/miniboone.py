""" Based on https://github.com/gpapamak/maf/blob/master/datasets/miniboone.py
Code to recreate dataset used in MAF paper.
"""
import numpy as np
import pandas as pd
import os.path
import wget
import cPickle as pickle
from collections import Counter
from ..utils import misc


def get_top_lines(datafile, N, data_save_file):
    with open(datafile) as myfile:
        head = [next(myfile) for x in xrange(N)]
    fl = open(data_save_file, 'wb')
    for i in range(1,N):
        fl.write(head[i])
    fl.close()


def load_data(root_path):
    # NOTE: To remember how the pre-processing was done.
    data = pd.read_csv(
        root_path, names=[str(x) for x in range(50)], delim_whitespace=True)
    data = data.as_matrix()
    # Remove some random outliers
    indices = (data[:, 0] < -100)
    data = data[~indices]

    i = 0
    # Remove any features that have too many re-occuring real values.
    features_to_remove = []
    for feature in data.T:
        c = Counter(feature)
        max_count = np.array([v for k, v in sorted(c.iteritems())])[0]
        if max_count > 5:
            features_to_remove.append(i)
        i += 1
    data = data[:, np.array(
        [i for i in range(data.shape[1]) if i not in features_to_remove])]
    svpath = root_path + 'data_created'
    np.save(svpath, data)

    #data = np.load(root_path)
    N_test = int(0.1*data.shape[0])
    data_test = data[-N_test:]
    data = data[0:-N_test]
    N_validate = int(0.1*data.shape[0])
    data_validate = data[-N_validate:]
    data_train = data[0:-N_validate]

    return data_train, data_validate, data_test


def load_data_normalised(root_path):
    root_path1 = root_path + '_only_positive.txt'
    get_top_lines(root_path, 36500, root_path1)
    data_train, data_validate, data_test = load_data(root_path1)
    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu)/s
    data_validate = (data_validate - mu)/s
    data_test = (data_test - mu)/s

    return data_train, data_validate, data_test


def download_and_make_data(datapath):
    url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
           '00199/MiniBooNE_PID.txt')
    path = os.path.join(datapath, 'miniboone/')
    misc.make_path(path)
    print('Downloading...')
    filename = wget.download(url, path)
    print('\nProcessing...')
    trn, val, tst = load_data_normalised(filename)
    print('Saving...')
    outfile = os.path.join(path, 'miniboone.p')
    pickle.dump(
        {'train': trn.astype(np.float32),
         'valid': val.astype(np.float32),
         'test': tst.astype(np.float32)},
        open(outfile, 'wb')
    )
