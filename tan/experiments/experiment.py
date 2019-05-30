import tensorflow as tf
from . import trainer
from ..model import model as mod


class Experiment:

    # TODO: make all arguements optional, load config from
    # save_location+'config.p' when not given, resolve dimension, and add
    # option to restore
    def __init__(self, config, summary_location, save_location, fetchers=None,
                 graph=None, inputs_pl=None, conditioning_pl=None,
                 get_trainer=True):
        self.config = config
        self.summary_location = summary_location
        self.save_location = save_location
        self.fetchers = fetchers
        # Set up model/trainer in graph
        if graph is None:
            tf.reset_default_graph()
            self.graph = tf.Graph()
        else:
            self.graph = graph
        with self.graph.as_default():
            if fetchers is not None:  # Use given inputs_pl and conditioning_pl
                if fetchers.train.ndatasets > 1:
                    # Labeled data.
                    inputs_pl = tf.placeholder(
                        tf.float32, (None, fetchers.dim[0]), 'inputs')
                    conditioning_pl = tf.placeholder(
                        tf.float32, (None, fetchers.dim[1]), 'conditioning')
                else:
                    # Unlabeled data.
                    inputs_pl = tf.placeholder(
                        tf.float32, (None, fetchers.dim), 'inputs')
                    conditioning_pl = None
            if config.dropout_keeprate_val is not None and \
                    config.dropout_keeprate_val < 1.0:
                dropout_keeprate = tf.placeholder(tf.float32, [],
                                                  'dropout_keeprate')
            else:
                dropout_keeprate = None
            with tf.variable_scope('TAN', initializer=config.initializer):
                with tf.variable_scope('model'):
                    self.model = mod.TANModel(
                        config.transformations,
                        conditional_model=config.conditional_model,
                        likefunc=config.likefunc,
                        param_nlayers=config.param_nlayers,
                        hidden_activation=config.hidden_activation,
                        cond_param_irange=config.cond_param_irange,
                        dropout_keep_prob=dropout_keeprate,
                        nparams=config.nparams,
                        base_distribution=config.base_distribution,
                        sample_size=config.sample_batch_size)
                    self.nll, self.llikes, self.sampler = \
                        self.model.build_graph(inputs_pl, conditioning_pl)
                if get_trainer:
                    tf_config = tf.ConfigProto()
                    tf_config.gpu_options.allow_growth = True
                    self.sess = tf.Session(config=tf_config)
                    with tf.variable_scope('train'):
                        self.trn = trainer.RedTrainer(
                            fetchers, self.nll, inputs_pl,
                            self.llikes,
                            conditioning_data=conditioning_pl,
                            batch_size=config.batch_size,
                            sess=self.sess,
                            init_lr=config.init_lr,
                            min_lr=config.min_lr,
                            lr_decay=config.lr_decay,
                            decay_interval=config.decay_interval,
                            penalty=config.penalty,
                            dropout_keeprate=dropout_keeprate,
                            dropout_keeprate_val=config.dropout_keeprate_val,
                            train_iters=config.train_iters,
                            hold_iters=config.hold_iters,
                            print_iters=config.print_iters,
                            hold_interval=config.hold_interval,
                            optimizer_class=config.optimizer_class,
                            max_grad_norm=config.max_grad_norm,
                            do_check=config.do_check,
                            momentum=config.momentum,
                            momentum_iter=config.momentum_iter,
                            pretrain_scope=config.pretrain_scope,
                            pretrain_iters=config.pretrain_iters,
                            summary_log_path=summary_location,
                            save_path=save_location,
                            sampler=self.sampler,
                            nsamp=config.nsample_batches)

    @property
    def main(self):
        return self.trn.main
