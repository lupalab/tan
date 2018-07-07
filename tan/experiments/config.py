import tensorflow as tf
from ..utils import misc
from ..model import transforms as trans
from ..model import likelihoods as likes
from ..model import conditionals as conds
from ..rnn import cells


# TODO: rename.
class RedConfig:
    def __init__(self, **kwargs):
        # Transformation configurations.
        #
        # Initial permutation.
        self.first_perm = misc.get_default(kwargs, 'first_perm')
        # Initial linear mapping.
        self.first_do_linear_map = misc.get_default(
            kwargs, 'first_do_linear_map', False)
        self.first_init_mat_params = misc.get_default(
            kwargs, 'first_init_mat_params')
        self.first_init_b = misc.get_default(kwargs, 'first_init_b')
        self.first_trainable_A = misc.get_default(
            kwargs, 'first_trainable_A', False)
        self.first_trainable_b = misc.get_default(
            kwargs, 'first_trainable_b', False)
        # Subsequent linear mappings.
        self.init_mat_params = misc.get_default(kwargs, 'init_mat_params')
        self.init_b = misc.get_default(kwargs, 'init_b')
        self.trainable_A = misc.get_default(kwargs, 'trainable_A', True)
        self.trainable_b = misc.get_default(kwargs, 'trainable_b', True)
        self.mat_func = misc.get_default(kwargs, 'mat_func', trans.get_LU_map)
        # RNN transformation parameters.
        self.trans_state_size = misc.get_default(
            kwargs, 'trans_state_size', 16)
        self.trans_alpha = misc.get_default(kwargs, 'trans_alpha', None)
        self.trans_state_activation = misc.get_default(
            kwargs, 'trans_state_activation', tf.nn.relu
        )
        # RNN coupling parameters.
        self.rnn_coupling_params = misc.get_default(
            kwargs, 'rnn_coupling_params', {'nunits': 256, 'num_layers': 1})
        self.rnn_coupling_type = misc.get_default(kwargs, 'rnn_coupling_type',
                                                  cells.GRUCell)
        self.rnn_coupling_class = self.rnn_coupling_type(
            **self.rnn_coupling_params)
        # Image tiling arguments.
        self.tile_hw = misc.get_default(kwargs, 'tile_hw', 28)
        self.tile_chans = misc.get_default(kwargs, 'tile_chans', 1)
        # Additive transformation parameters.
        self.add_hidden_sizes = misc.get_default(
            kwargs, 'add_hidden_sizes', [256]*2)
        self.add_irange = misc.get_default(
            kwargs, 'add_irange', None)
        self.add_output_irange = misc.get_default(
            kwargs, 'add_output_irange', None)
        # Rescaling transformation parameters.
        self.rescale_init_constant = misc.get_default(
            kwargs, 'rescale_init_constant', 1.0)
        self.trans_funcs = misc.get_default(kwargs, 'trans_funcs')
        # Conditional transformation parameters.
        self.do_init_cond_trans = misc.get_default(
            kwargs, 'do_init_cond_trans', False)
        self.do_final_cond_trans = misc.get_default(
            kwargs, 'do_final_cond_trans', False)
        self.cond_hidden_sizes = misc.get_default(
            kwargs, 'cond_hidden_sizes', [256]*2)
        self.cond_irange = misc.get_default(
            kwargs, 'cond_irange', None)
        self.cond_output_irange = misc.get_default(
            kwargs, 'cond_output_irange', 1e-6)
        # Leaky ReLU parameters
        self.relu_alpha = misc.get_default(
            kwargs, 'relu_alpha', None)

        # Make list of transformation functions.
        #
        # Do an initial permuation and linear transformation.
        self.first_transformations = []
        if self.first_perm is not None:
            self.first_transformations += [
                lambda x: trans.permute(x, self.first_perm)
            ]
        if self.first_do_linear_map:
            self.first_transformations += [
                lambda x: trans.linear_map(
                    x,
                    init_mat_params=self.first_init_mat_params,
                    init_b=self.first_init_b,
                    mat_func=self.mat_func,
                    trainable_A=self.first_trainable_A,
                    trainable_b=self.first_trainable_b
                )
            ]

        # TODO: untracked transformations should still work.
        # Make the default arguments for the various transformations.
        self.transform_arguments = {
            trans.linear_map: {
                'init_mat_params': self.init_mat_params,
                'init_b': self.init_b,
                'mat_func': self.mat_func,
                'trainable_A': self.trainable_A,
                'trainable_b': self.trainable_b},
            trans.additive_coupling: {
                'hidden_sizes': self.add_hidden_sizes,
                'irange': self.add_irange,
                'output_irange': self.add_output_irange},
            trans.rescale: {
                'init_constant': self.rescale_init_constant},
            trans.simple_rnn_transform: {
                'state_size': self.trans_state_size,
                'alpha': self.trans_alpha,
                'state_activation': self.trans_state_activation},
            trans.rnn_coupling: {
                'rnn_class': self.rnn_coupling_class},
            trans.negate: {},
            trans.reverse: {},
            trans.log_rescale: {},
            trans.shift: {},
            trans.leaky_transformation: {
                'alpha': self.relu_alpha},
        }

        # Likelihood parameters.
        #
        self.ncomps = misc.get_default(kwargs, 'ncomps', 40)
        self.nparams = 3*self.ncomps
        self.base_distribution = misc.get_default(
            kwargs, 'base_distribution', 'gaussian')
        self.likefunc = misc.get_default(
            kwargs, 'likefunc', likes.mixture_likelihoods)
        self.param_nlayers = misc.get_default(kwargs, 'param_nlayers')

        # Conditional density parameters.
        # TODO: Add options to use conditioning information at fc and/or
        #       autoregressive levels.
        #
        self.nhidparams = misc.get_default(kwargs, 'nhidparams', self.nparams)
        # Independent model.
        self.single_marginal = misc.get_default(kwargs, 'single_marginal',
                                                False)
        self.standard = misc.get_default(kwargs, 'standard', False)
        # Conditional model.
        self.cond_tied_model = misc.get_default(kwargs, 'cond_tied_model', True)
        # RNN Model.
        self.rnn_params = misc.get_default(
            kwargs, 'rnn_params', {'nunits': 256, 'num_layers': 1})
        self.rnn_type = misc.get_default(kwargs, 'rnn_type', cells.GRUCell)
        self.rnn_class = self.rnn_type(**self.rnn_params)
        # Conditional model arguments.
        self.conditional_argument = {
            conds.independent_model: {'single_marginal': self.single_marginal,
                                      'standard': self.standard},
            conds.cond_model: {'tied_model': self.cond_tied_model},
            conds.rnn_model: {'rnn_class': self.rnn_class}
        }
        # Orderless methods that get all covariates as inputs.
        self.orderless_models = set([])  # TODO: remove
        # Make conditional function.
        self.cond_func = misc.get_default(kwargs, 'cond_func', conds.rnn_model)
        self.orderless = self.cond_func in self.orderless_models  # TODO: remove
        self.conditional_model = lambda x, f, c: self.cond_func(
            x, self.nhidparams, param_func=f, conditioning=c,
            **self.conditional_argument[self.cond_func]
        )
        self.hidden_activation = misc.get_default(
            kwargs, 'hidden_activation', tf.nn.relu)
        self.cond_param_irange = misc.get_default(
            kwargs, 'cond_param_irange', None)

        # Conditioning value usage.
        #
        self.trans_conditioning = misc.get_default(
            kwargs, 'trans_conditioning', True)
        self.conditional_conditioning = misc.get_default(
            kwargs, 'conditional_conditioning', True)
        self.fc_conditioning = misc.get_default(kwargs, 'fc_conditioning', True)

        # Sampling Options.
        #
        self.sample_batch_size = misc.get_default(
            kwargs, 'sample_batch_size', 128)
        self.nsample_batches = misc.get_default(kwargs, 'nsample_batches', 10)
        self.samp_per_cond = misc.get_default(kwargs, 'samp_per_cond', 500)

        # Training configurations.
        #
        self.initializer_class = misc.get_default(kwargs, 'initializer_class',
                                                  None)
        self.initializer_args = misc.get_default(
            kwargs, 'initializer_args', {'minval': -0.05, 'maxval': 0.05})
        self.initializer = None if self.initializer_class is None else \
            self.initializer_class(**self.initializer_args)
        self.batch_size = misc.get_default(kwargs, 'batch_size', 128)
        self.init_lr = misc.get_default(kwargs, 'init_lr', 0.1)
        self.min_lr = misc.get_default(kwargs, 'min_lr', None)
        self.lr_decay = misc.get_default(kwargs, 'lr_decay', 0.9)
        self.decay_interval = misc.get_default(kwargs, 'decay_interval', 10000)
        self.penalty = misc.get_default(kwargs, 'penalty', 0.0)
        self.dropout_keeprate_val = misc.get_default(
            kwargs, 'dropout_keeprate_val', 1.0)
        self.train_iters = misc.get_default(kwargs, 'train_iters', 30000)
        self.hold_iters = misc.get_default(kwargs, 'hold_iters', 100)
        self.print_iters = misc.get_default(kwargs, 'print_iters', 250)
        self.hold_interval = misc.get_default(kwargs, 'hold_interval', 2500)
        self.optimizer_class = misc.get_default(
            kwargs, 'optimizer_class', tf.train.AdamOptimizer)
        self.max_grad_norm = misc.get_default(kwargs, 'max_grad_norm', None)
        self.do_check = misc.get_default(kwargs, 'do_check', False)
        self.momentum = misc.get_default(kwargs, 'momentum', None)
        self.momentum_iter = misc.get_default(
            kwargs, 'momentum_iter', 10000000000)
        self.pretrain_scope = misc.get_default(kwargs, 'pretrain_scope', None)
        self.pretrain_iters = misc.get_default(kwargs, 'pretrain_iters', 5000)
        self.noise_scale = misc.get_default(kwargs, 'noise_scale', None)

        # Image configurations.
        #
        self.img_size = misc.get_default(kwargs, 'img_size', (64, 64, 3))
        self.downsample = misc.get_default(kwargs, 'downsample', 1)
        self.do_resize = misc.get_default(kwargs, 'do_resize', True)
        self.center_crop = misc.get_default(kwargs, 'center_crop', None)
        self.do_bw = misc.get_default(kwargs,  'do_bw', False)
        self.do_read_logit = misc.get_default(kwargs, 'do_read_logit', False)
        self.do_init_logit = misc.get_default(kwargs, 'do_init_logit', False)
        self.seq_feats = misc.get_default(kwargs, 'seq_feats', 256)
        self.mean_layers = misc.get_default(kwargs, 'mean_layers', None)
        self.conv_feats = misc.get_default(kwargs, 'conv_feats', 64)
        self.conv_hid = misc.get_default(kwargs, 'conv_hid', [64, 64, 64])
        self.use_markov_feats = misc.get_default(
            kwargs, 'use_markov_feats', False)
        # self.use_discrete_probs = misc.get_default(
        #     kwargs, 'use_discrete_probs', False)
        # flat, column, double_seq, sub_column
        self.image_model = misc.get_default(kwargs, 'image_model', 'column')
        self.use_2nd_markov = misc.get_default(kwargs, 'use_2nd_markov', False)
        self.image_sublevels = misc.get_default(kwargs, 'image_sublevels', 2)
        self.image_cell_class = misc.get_default(kwargs, 'image_cell_class',
                                                 cells.GRUCell)
        self.logit_iters = misc.get_default(kwargs, 'logit_iters', None)
        self.seq_logit_trans = misc.get_default(
            kwargs, 'seq_logit_trans', False)
        self.seq_cell_layers = misc.get_default(kwargs, 'seq_cell_layers', 1)
        self.seq_fc_layers = misc.get_default(kwargs, 'seq_fc_layers', None)
        self.predict_layers = misc.get_default(kwargs, 'predict_layers', None)
        self.predict_alpha = misc.get_default(kwargs, 'predict_alpha', 0.5)

        # Embeddings.
        self.embed_size = misc.get_default(kwargs, 'embed_size', 256)
        self.embed_layers = misc.get_default(kwargs, 'embed_layers', [256, 256])
        self.embed_irange = misc.get_default(kwargs, 'embed_irange', 1e-6)
        self.embed_activation = misc.get_default(
            kwargs, 'embed_activation', tf.nn.relu)
        self.set_regression = misc.get_default(kwargs, 'set_regression', False)
        self.set_classification = misc.get_default(
            kwargs, 'set_classification', False)
        self.set_loss_alpha = misc.get_default(kwargs, 'set_loss_alpha', 0.5)
        self.set_layers = misc.get_default(kwargs, 'set_layers', [256, 256])

    def transformations_generator(self):
        """ Helper function to dynamically build transformation functions based
        on configuration parameters. """
        for func in self.first_transformations:
            yield func
        # TODO: Move to front?
        if self.do_init_cond_trans:
            yield lambda x, c: trans.conditioning_transformation(
                x, c, hidden_sizes=self.cond_hidden_sizes,
                irange=self.cond_irange, output_irange=self.cond_output_irange)
        if self.trans_funcs is not None:
            for func in self.trans_funcs:
                yield lambda x: func(
                    x, **self.transform_arguments[func])
        if self.do_final_cond_trans:
            yield lambda x, c: trans.conditioning_transformation(
                x, c, hidden_sizes=self.cond_hidden_sizes,
                irange=self.cond_irange, output_irange=self.cond_output_irange)

    @property
    def transformations(self):
        return [t for t in self.transformations_generator()]
