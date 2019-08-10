import pandas as pd
import numpy as np
import wget
import zipfile
import re
import os.path
import cPickle as pickle
from ..utils import misc


def load_data(filename):
    df = pd.read_csv(filename, sep=';',
                     parse_dates=['Unnamed: 0'],
                     decimal=",", dtype=np.float32)
    df.set_index('Unnamed: 0', inplace=True)
    df.fillna(0, inplace=True)

    df.drop(columns=[
        'MT_133', 'MT_223', 'MT_178', 'MT_066', 'MT_181', 'MT_012',
        'MT_119', 'MT_003', 'MT_359',
        'MT_015', 'MT_030', 'MT_039', 'MT_041', 'MT_092', 'MT_106', 'MT_107',
        'MT_108', 'MT_109', 'MT_110', 'MT_111', 'MT_112', 'MT_113', 'MT_115',
        'MT_116', 'MT_117', 'MT_120', 'MT_121', 'MT_122', 'MT_134', 'MT_152',
        'MT_160', 'MT_165', 'MT_170', 'MT_179', 'MT_185', 'MT_186', 'MT_224',
        'MT_289', 'MT_305', 'MT_322', 'MT_337', 'MT_370', 'MT_332'], inplace=True)

    return df.values


def load_data_split(filename):
    data = load_data(filename)

    N_test = int(0.2*data.shape[0])
    data_test = data[-N_test:]
    data = data[0:-N_test]
    N_validate = int(0.1*data.shape[0])
    data_validate = data[-N_validate:]
    data_train = data[0:-N_validate]

    return data_train, data_validate, data_test


def load_data_normalised(filename):
    data_train, data_validate, data_test = load_data_split(filename)
    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu)/s
    data_validate = (data_validate - mu)/s
    data_test = (data_test - mu)/s

    return data_train, data_validate, data_test


def download_and_make_data(datapath):
    url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
           '00321/LD2011_2014.txt.zip')
    txtname = 'LD2011_2014.txt'
    path = os.path.join(datapath, 'electricity/')
    misc.make_path(path)
    print('Downloading...')
    filename = wget.download(url, path)
    print('\nExtracting...')
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(path)
    zip_ref.close()
    print('Processing...')
    trn, val, tst = load_data_normalised(os.path.join(path, txtname))
    print('Saving...')
    outfile = os.path.join(path, 'electricity.p')
    pickle.dump(
        {'train': trn.astype(np.float32),
         'valid': val.astype(np.float32),
         'test': tst.astype(np.float32)},
        open(outfile, 'wb')
    )
