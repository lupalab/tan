import os
import runner
import copy
import matplotlib; matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt  # noqa
from mpl_toolkits.mplot3d import Axes3D  # noqa
import numpy as np
import requests
from datetime import datetime
from ..data import pointcloud_fetcher as pfetcher
from ..model import transforms as trans  # noqa
import embed_experiment as emb_exp
from ..utils import misc as umisc


# https://stackoverflow.com/a/39225039
def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def main(download=True):
    # Download and setup data.
    home = os.path.expanduser('~')
    datadir = os.path.join(home, 'data/tan/pntcloud')
    data_path = os.path.join(datadir, 'plane.p')
    umisc.make_path(datadir)
    FILEID = "16ho7MYBc0mTFjfcPaaH6Tzzkar9gv-fm"
    if download:
        print('Downloading data.')
        download_file_from_google_drive(FILEID, data_path)
    print('Loading test data for plotting.')
    # Options
    ac = copy.copy(runner.ARG_CHOICES_UNTIED)
    ac['trans_funcs'] = (ac['trans_funcs'][1], )
    ac['first_trainable_A'] = (True, )
    ac['first_do_linear_map'] = (True, )
    ac['lr_decay'] = (0.5, )
    ac['train_iters'] = (30000, )
    ac['hold_iters'] = (64, )
    ac['batch_size'] = (99, )
    ac['do_init_cond_trans'] = (True, )
    ac['do_final_cond_trans'] = (True, )
    ac['conditional_conditioning'] = (True, )
    ac['samp_per_cond'] = (2000, )
    ac['embed_size'] = (256, )
    ac['embed_layers'] = (128, )
    noisestd = 0.0
    # Train
    ret = runner.run_experiment(
        data_path,
        arg_list=runner.misc.make_arguments(ac),
        exp_class=emb_exp.EmbedExperiment,
        fetcher_class=pfetcher.generate_fetchers(
            subsamp_test=1000, subsamp_valid=1000, noisestd=noisestd),
        no_log=True,  # TODO: logging hangs?
    )
    # Plot
    print('Plotting samples.')
    samps = ret[0]['results']['samples']
    samps_cond = ret[0]['results']['samples_cond']
    ename = 'noise_{}-embedlayers_{}-embedsize_{}-time_{}'.format(
        noisestd, ac['embed_layers'], ac['embed_size'], datetime.now())
    print(ename)
    for ii in range(3):
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        i = np.random.randint(128)
        # plot original
        x_in = samps_cond[0]['inp_val'][i]
        ax.scatter(x_in[:, 0], x_in[:, 1], x_in[:, 2])
        ax.set_title('Original')
        # plot sample
        x = samps[i]
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], c='r')
        ax.set_title('Samples')
        fig_path = os.path.join(
            datadir, 'airplane_pnts{}_{}.png'.format(ii, ename))
        fig.savefig(fig_path)
