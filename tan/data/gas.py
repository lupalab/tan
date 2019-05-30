import pandas as pd
import numpy as np
import re
import zipfile
import wget
import cPickle as pickle
import os.path
from ..utils import misc


class GAS:
    # http://archive.ics.uci.edu/ml/datasets/gas+sensor+array+under+dynamic+gas+mixtures
    # http://archive.ics.uci.edu/ml/machine-learning-databases/00322/data.zip
    class Data:

        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, file):

        # file = file + 'gas/ethylene_CO.pickle'
        trn, val, tst = load_data_and_clean_and_split(file)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]

    def show_histograms(self, split):

        data_split = getattr(self, split, None)
        if data_split is None:
            raise ValueError('Invalid data split')

        # util.plot_hist_marginals(data_split.x)
        # plt.show()


def load_data(fl):

    # data = pd.read_pickle(file)
    data = pd.read_csv(fl, sep='\t').sample(frac=0.25)
    # data = pd.read_pickle(file).sample(frac=0.25)
    # data.to_pickle(file)
    data.drop("Meth", axis=1, inplace=True)
    data.drop("Eth", axis=1, inplace=True)
    data.drop("Time", axis=1, inplace=True)
    return data


def get_correlation_numbers(data):
    C = data.corr()
    A = C > 0.98
    B = A.as_matrix().sum(axis=1)
    return B


def write_csv_file(original_file, converted_file):
    with open(original_file) as f:
        content = f.readlines()
    cv_fl = open(converted_file, 'wb')
    cv_fl.write("Time\tMeth\tEth")
    cv_fl.write("\tA\tB\tC\tD\tE\tF\tG\tH\tI\tJ\tK\tL\tM\tN\tO\tP\n")
    for i in range(1, len(content)):
        # content[i] = re.sub(' +', ' ', content[i])
        content[i] = " ".join(content[i].split())
        content[i] = re.sub(' ', "\t", content[i])
        cv_fl.write(content[i])
        cv_fl.write("\n")
    cv_fl.close()


def load_data_and_clean(file):

    data = load_data(file)
    B = get_correlation_numbers(data)

    while np.any(B > 1):
        col_to_remove = np.where(B > 1)[0][0]
        col_name = data.columns[col_to_remove]
        data.drop(col_name, axis=1, inplace=True)
        B = get_correlation_numbers(data)
    # print(data.corr())
    data = (data-data.mean())/data.std()

    return data


def load_data_and_clean_and_split(file):

    new_fl = file + "changed"
    write_csv_file(file, new_fl)
    data = load_data_and_clean(new_fl).as_matrix()
    N_test = int(0.1*data.shape[0])
    data_test = data[-N_test:]
    data_train = data[0:-N_test]
    N_validate = int(0.1*data_train.shape[0])
    data_validate = data_train[-N_validate:]
    data_train = data_train[0:-N_validate]

    return data_train, data_validate, data_test


def download_and_make_data(datapath):
    url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
           '00322/data.zip')
    txtname = 'ethylene_CO.txt'
    path = os.path.join(datapath, 'gas/')
    misc.make_path(path)
    print('Downloading...')
    filename = wget.download(url, path)
    print('\nExtracting...')
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(path)
    zip_ref.close()
    print('Processing...')
    trn, val, tst = load_data_and_clean_and_split(os.path.join(path, txtname))
    print('Saving...')
    outfile = os.path.join(path, 'gas.p')
    pickle.dump(
        {'train': trn.astype(np.float32),
         'valid': val.astype(np.float32),
         'test': tst.astype(np.float32)},
        open(outfile, 'wb')
    )
