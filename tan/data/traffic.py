import numpy as np
import wget
import zipfile
import re
import os.path
import cPickle as pickle
from ..utils import misc


def load_data(filename):
    data_file = open(filename)
    data = []
    for line in data_file:
        regex = re.findall(r'[^\[\;\]]+', line)
        for i in range(963):
            sp = regex[i].split()
            data.append([float(i) for i in sp])

    data = np.asarray(data)
    data = data.reshape(-1, 963, 144)
    data = data.transpose((0, 2, 1))
    return data.reshape(-1, 963)


def load_data_split(path):
    data_train = load_data(os.path.join(path, 'PEMS_train'))
    data_test = load_data(os.path.join(path, 'PEMS_test'))

    rng = np.random.RandomState(42)
    rng.shuffle(data_train)

    N = data_train.shape[0]
    N_validate = int(0.05*N)
    data_validate = data_train[-N_validate:]
    data_train = data_train[0:-N_validate]

    return data_train, data_validate, data_test


def download_and_make_data(datapath):
    url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
           '00204/PEMS-SF.zip')
    path = os.path.join(datapath, 'traffic/')
    misc.make_path(path)
    print('Downloading...')
    filename = wget.download(url, path)
    print('\nExtracting...')
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(path)
    zip_ref.close()
    print('Processing...')
    trn, val, tst = load_data_split(path)
    print('Saving...')
    outfile = os.path.join(path, 'traffic.p')
    pickle.dump(
        {'train': trn.astype(np.float32),
         'valid': val.astype(np.float32),
         'test': tst.astype(np.float32)},
        open(outfile, 'wb')
    )
