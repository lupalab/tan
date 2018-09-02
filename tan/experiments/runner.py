import tensorflow as tf  # noqa
import copy
import os
import cPickle as pickle
import numpy as np
import hashlib
from ..data import helpers as helpers
from ..utils import misc as misc
from ..data import batch_fetcher as bfetchers
from ..experiments import experiment
from ..experiments import config as econfig
from ..model import conditionals as conds
from ..model import transforms as trans # noqa
from ..model import likelihoods as likes # noqa
from datetime import datetime

# Hyperparameters.
DEF_ARGS = {
    'train_iters': 30000,
    'hold_iters': 100,
    'hold_interval': 2500,
    'ncomps': 40,
    'decay_interval': 5000,
    'dropout_keeprate_val': None,
    'optimizer_class': tf.train.AdamOptimizer,
    'momentum': None,
    'momentum_iter': 5000,
    'max_grad_norm': 1.0,
    'trans_alpha': None,
    'rescale_init_constant': 1.0,
    'trans_state_activation': tf.nn.tanh,
    'cond_param_irange': 1e-6,
    'first_do_linear_map': True,
    'standardize': True,
    'base_distribution': 'gaussian',
}
# Base configs for different transformations.
BASE_ARG_CHOICES = {
    'lr_decay': (0.5, 0.1),
    'init_lr': (0.005, ),
    'first_trainable_A': (True, False),
    'trans_funcs': [
        None,
        [trans.additive_coupling, trans.reverse, trans.additive_coupling,
         trans.reverse, trans.additive_coupling, trans.reverse,
         trans.additive_coupling, trans.log_rescale],  # NICE Type
        [trans.simple_rnn_transform, ],  # 1xRNN
        [trans.simple_rnn_transform, trans.reverse,
         trans.simple_rnn_transform],  # 2xRNN
        [trans.rnn_coupling, trans.reverse, trans.rnn_coupling, trans.reverse,
         trans.rnn_coupling, trans.reverse, trans.rnn_coupling,
         trans.log_rescale],  # 4xRNN Coup
        [trans.simple_rnn_transform, trans.reverse,
         trans.rnn_coupling, trans.reverse, trans.rnn_coupling, trans.reverse,
         trans.rnn_coupling, trans.reverse, trans.rnn_coupling,
         trans.log_rescale],  # 1xRNN + RNN Coupling
        [trans.simple_rnn_transform, trans.reverse, trans.additive_coupling,
         trans.reverse, trans.additive_coupling, trans.reverse,
         trans.additive_coupling, trans.reverse, trans.additive_coupling,
         trans.log_rescale],  # 1xRNN + NICE
    ],
}
# Get configs for standard Gaussian conditional model.
ARG_CHOICES_STDGAU = copy.copy(BASE_ARG_CHOICES)
ARG_CHOICES_STDGAU['single_marginal'] = (True,)
ARG_CHOICES_STDGAU['standard'] = (True,)
ARG_CHOICES_STDGAU['ncomps'] = (1, )
ARG_CHOICES_STDGAU['cond_func'] = (conds.independent_model,)
ARG_LIST_STDGAU = misc.make_arguments(ARG_CHOICES_STDGAU)
ARG_LIST_STDGAU = filter(
    lambda conf: conf['first_trainable_A'] or conf['trans_funcs'] is not None,
    ARG_LIST_STDGAU)  # Avoid models that have no variables to optimize.
# Get configs for independent GMMs
ARG_CHOICES_IND = copy.copy(BASE_ARG_CHOICES)
ARG_CHOICES_IND['single_marginal'] = (False,)
ARG_CHOICES_IND['standard'] = (False,)
ARG_CHOICES_IND['cond_func'] = (conds.independent_model,)
ARG_LIST_IND = misc.make_arguments(ARG_CHOICES_IND)
# Get config for Tied conditional model.
ARG_CHOICES_TIED = copy.copy(BASE_ARG_CHOICES)
ARG_CHOICES_TIED['cond_tied_model'] = (True,)
ARG_CHOICES_TIED['param_nlayers'] = (2,)
ARG_CHOICES_TIED['cond_func'] = (conds.cond_model,)
ARG_LIST_TIED = misc.make_arguments(ARG_CHOICES_TIED)
# Get config for Untied conditional model.
ARG_CHOICES_UNTIED = copy.copy(BASE_ARG_CHOICES)
ARG_CHOICES_UNTIED['cond_tied_model'] = (False,)
ARG_CHOICES_UNTIED['param_nlayers'] = (2,)
ARG_CHOICES_UNTIED['cond_func'] = (conds.cond_model,)
ARG_LIST_UNTIED = misc.make_arguments(ARG_CHOICES_UNTIED)
# Get config for RNN conditional model.
ARG_CHOICES_RNN = copy.copy(BASE_ARG_CHOICES)
ARG_CHOICES_RNN['param_nlayers'] = (None, 2)
ARG_CHOICES_RNN['cond_func'] = (conds.rnn_model,)
ARG_LIST_RNN = misc.make_arguments(ARG_CHOICES_RNN)
# Get config for RNN conditional model.
ARG_CHOICES_RNN_FC = copy.copy(BASE_ARG_CHOICES)
ARG_CHOICES_RNN_FC['param_nlayers'] = (2, )
ARG_CHOICES_RNN_FC['cond_func'] = (conds.rnn_model,)
ARG_LIST_RNN_FC = misc.make_arguments(ARG_CHOICES_RNN_FC)
# Make the default be RNN conditional models.
ARG_LIST = misc.make_arguments(ARG_CHOICES_RNN)


def shorten(obj):
    """ Helper function to shorten stringeds from long options, uses hash to
    ensure shortening without collision """
    string = str(obj)
    if len(string) >= 255:
        hash_object = hashlib.md5(string)
        string_hash = str(hash_object.hexdigest())
        return string[:50] + '...' + string[-50:] + '_' + string_hash
    return string


def print_value(value):
    """ Helper function to print functions, lists, and dictionaries for
        filenames and printouts. """
    if isinstance(value, str):
        return value
    try:
        try:
            string = reduce(lambda x, y: x+'-'+y,
                            [print_value(v) for v in value.items()])
        except AttributeError:  # Not dictionary
            string = reduce(
                lambda x, y: x+','+y, [print_value(v) for v in value])
    except TypeError:  # Not iterable
        try:
            string = value.func_name
        except AttributeError:  # Not function
            string = str(value)
    return string


def get_exp_name(args):
    sorted_keys = np.sort(args.keys())
    exp_name = reduce(lambda x, y: x+y,
                      ['{}--{}/'.format(k, shorten(print_value(args[k])))
                       for k in sorted_keys], '')
    return exp_name


def make_trainer(dataset, base_save_path, base_log_path,
                 nepochs=None, exp_class=experiment.Experiment,
                 fetcher_class=bfetchers.DatasetFetchers, **kwargs):
    # Options.
    # Load data.
    # TODO: general data load
    if isinstance(dataset, str):
        print('Loading {}...'.format(dataset))
        dataset = pickle.load(open(dataset, 'rb'))
        print('Loaded.')
    # Make the data fetchers.
    if 'train_labels' in dataset and 'valid_labels' in dataset and \
            'test_labels' in dataset:
        # Labeled data.
        fetchers = fetcher_class(
            (dataset['train'], dataset['train_labels']),
            (dataset['valid'], dataset['valid_labels']),
            (dataset['test'], dataset['test_labels']))
    else:
        fetchers = fetcher_class(
            (dataset['train'],), (dataset['valid'],), (dataset['test'],))

    def main(args):
        # Make config for trial with defualt and given arguments.
        trial_args = copy.copy(kwargs)
        for ind in args:
            trial_args[ind] = args[ind]
        # Data preprocessing
        standardize = misc.get_default(trial_args, 'standardize', False)
        cov_func = misc.get_default(trial_args, 'cov_func', None)
        trial_args['first_do_linear_map'] = misc.get_default(
            trial_args, 'first_do_linear_map', False)
        # Get initial linear map parameters.
        if trial_args['first_do_linear_map']:
            try:
                (imp, ib, ip) = helpers.get_initmap(
                    dataset['train'], standardize=standardize,
                    cov_func=cov_func)
                trial_args['first_init_mat_params'] = imp
                trial_args['first_init_b'] = ib
                trial_args['first_perm'] = ip
            except (TypeError, ValueError) as error:
                print('No initial linear parameters due to error:\n{}'.format(
                    error))
        # Determine the number of iterations to run nepochs
        trial_args['batch_size'] = misc.get_default(
            trial_args, 'batch_size', 256)
        if nepochs is not None:
            N, d = dataset['train'].shape
            iters_per_epoch = N/float(trial_args['batch_size'])
            trial_args['train_iters'] = int(nepochs*iters_per_epoch)

        config = econfig.RedConfig(**trial_args)
        # Make directories specific to experiment trial.
        if base_save_path is not None:
            save_path = os.path.join(base_save_path, get_exp_name(args))
            misc.make_path(save_path)
        else:
            AttributeError('Must provide save path for validating model')
        if base_log_path is not None:
            log_path = os.path.join(base_log_path, get_exp_name(args))
            misc.make_path(log_path)
        else:
            log_path = None
        # Save config for easy model loading.
        try:
            pickle.dump(
                trial_args, open(os.path.join(save_path, 'trial_args.p'), 'wb'))
        except TypeError:
            print('Could not save trial arg pickle file.')
        # Set up trial and train.
        exp = exp_class(
            config, log_path, save_path, fetchers)
        with exp.graph.as_default():
            res_dicts = exp.main()
        # Save results.
        if log_path is not None:
            pickle.dump(
                res_dicts, open(os.path.join(log_path, 'result.p'), 'wb'))
        else:
            pickle.dump(
                res_dicts, open(os.path.join(save_path, 'result.p'), 'wb'))
        return res_dicts

    return main


def invalid_result(result):
    return result is None or np.isnan(result['loss'])


def run_experiment(data, arg_list=ARG_LIST, def_args=DEF_ARGS,
                   exp_class=experiment.Experiment,
                   fetcher_class=bfetchers.DatasetFetchers,
                   estimator='TAN', retries=1,
                   log_path=None, save_path=None, experiments_name=None,
                   no_log=False):
    # Set up paths.
    if log_path is None or save_path is None:
        home = os.path.expanduser('~')
        data_name = os.path.basename(data)
        experiments_name = \
            experiments_name if experiments_name is not None else \
            datetime.now().strftime('%Y_%m_%d_%H:%M:%S.%f')
        log_path = log_path if log_path is not None else \
            os.path.join(
                home, 'de_logs', estimator, data_name, experiments_name)
        save_path = save_path if save_path is not None else \
            os.path.join(
                home, 'de_models', estimator, data_name, experiments_name)
    if no_log:
        log_path = None
    else:
        misc.make_path(log_path)
    misc.make_path(save_path)
    print('log path: {}\nsave path: {}'.format(log_path, save_path))
    # Get results for all hyperparameter choices
    main = make_trainer(data, save_path, log_path, exp_class=exp_class,
                        fetcher_class=fetcher_class, **def_args)
    if no_log:
        log_path = save_path
    results = []
    best = None
    for ai in range(len(arg_list)):
        args = arg_list[ai]
        retries_left = retries
        print('RUNNING {}'.format(experiments_name))
        print('[{}/{}] {}'.format(ai+1, len(arg_list), args))
        results.append(main(args))
        while invalid_result(results[-1]) and retries_left > 0:
            print('[{}/{}] Retrying {}'.format(ai+1, len(arg_list), args))
            retries_left -= 1
            results[-1] = main(args)

        better_result = not invalid_result(results[-1]) and (
            invalid_result(best) or best['loss'] > results[-1]['loss']
        )
        if better_result:
            best = {}
            best['loss'] = results[-1]['loss']
            best['results'] = results[-1]
            best['args'] = args
        pickle.dump(
            {'best': best, 'trial_results': results,
             'trial_args': arg_list[:ai+1]},
            open(os.path.join(log_path, experiments_name+'_all_trials.p'),
                 'wb'))
    if best is not None:
        best['save_path'] = save_path
        best['log_path'] = log_path
        best['def_args'] = def_args
    pickle.dump(
        best,
        open(os.path.join(save_path, experiments_name+'_best_trial.p'), 'wb'))
    return best, results
