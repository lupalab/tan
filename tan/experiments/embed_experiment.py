import tensorflow as tf
import numpy as np
import trainer
from ..model import model as mod
from ..utils import nn


def DeepSetNetwork(inputs, config):
    with tf.variable_scope('embed'):
        N = tf.shape(inputs)[0]
        n = tf.shape(inputs)[1]
        d = int(inputs.get_shape()[-1])
        points = tf.reshape(inputs, [-1, d])
        point_feats = nn.fc_network(
             points, config.embed_size, config.embed_layers,
             output_init_range=config.embed_irange,
             activation=config.embed_activation,
             name='embed_network')
        point_feats_tens = tf.reshape(point_feats, [N, n, config.embed_size])
        samp_embed_feats = tf.concat(
             (tf.reduce_mean(point_feats_tens, 1, False),
              tf.reduce_max(point_feats_tens, 1, False)), -1)
        embed_feats = tf.reshape(
             tf.tile(tf.expand_dims(samp_embed_feats, 1), [1, n, 1]),
             [N*n, 2*config.embed_size])
        return points, embed_feats, samp_embed_feats


def DeepSetNetwork2(inputs, config):
    with tf.variable_scope('embed'):
        N = tf.shape(inputs)[0]
        n = tf.shape(inputs)[1]
        d = int(inputs.get_shape()[-1])
        points = tf.reshape(inputs, [N*n, d])

        # weights
        W0 = tf.get_variable('W0', shape=(d, 256), dtype=tf.float32)
        b0 = tf.get_variable('b0', shape=(256, ), dtype=tf.float32)
        U1 = tf.get_variable('U1', shape=(256, 256), dtype=tf.float32)
        W1 = tf.get_variable('W1', shape=(256, 256), dtype=tf.float32)
        b1 = tf.get_variable('b1', shape=(256, ), dtype=tf.float32)
        U2 = tf.get_variable('U2', shape=(256, 256), dtype=tf.float32)
        W2 = tf.get_variable('W2', shape=(256, 256), dtype=tf.float32)
        b2 = tf.get_variable('b2', shape=(256, ), dtype=tf.float32)
        W3 = tf.get_variable('W3', shape=(256, config.embed_size), dtype=tf.float32)
        b3 = tf.get_variable('b3', shape=(config.embed_size, ), dtype=tf.float32)

        # forward
        # Standard Layer0
        y = tf.nn.elu(tf.nn.xw_plus_b(points, W0, b0, 'linear0'))  # [N*n, 256]
        # PermEqui Layer1
        ym = tf.reshape(y, [N, n, 256])                            # [N, n, 256]
        ym = tf.reduce_mean(ym, 1, False)                          # [N, 256]
        ym = tf.matmul(ym, U1, name='linear12')                    # [N, 256]
        ym = tf.expand_dims(ym, 1)                                 # [N, 1, 256]
        y = tf.nn.xw_plus_b(y, W1, b1, name='linear11')            # [N*n, 256]
        y = tf.reshape(y, [N, n, 256])                             # [N, n, 256]
        y = tf.nn.elu(y + ym)                                      # [N, n, 256]
        # PermEqui Layer2
        ym = tf.reduce_mean(y, 1, False)                           # [N, 256]
        ym = tf.matmul(ym, U1, name='linear22')                    # [N, 256]
        ym = tf.expand_dims(ym, 1)                                 # [N, 1, 256]
        y = tf.reshape(y, [N*n, 256])                              # [N*n, 256]
        y = tf.nn.xw_plus_b(y, W2, b2, name='linear21')            # [N*n, 256]
        y = tf.reshape(y, [N, n, 256])                             # [N, n, 256]
        y = tf.nn.elu(y + ym)                                      # [N, n, 256]
        y = tf.reshape(y, [N*n, 256])                              # [N*n, 256]
        # Standard Layer3
        point_feats = tf.nn.xw_plus_b(y, W3, b3, 'linear3')        # [N*n, config.embed_size]

        point_feats_tens = tf.reshape(point_feats, [N, n, config.embed_size])
        samp_embed_feats = tf.concat(
             (tf.reduce_mean(point_feats_tens, 1, False),
              tf.reduce_max(point_feats_tens, 1, False)), -1)
        embed_feats = tf.reshape(
             tf.tile(tf.expand_dims(samp_embed_feats, 1), [1, n, 1]),
             [N*n, 2*config.embed_size])
        return points, embed_feats, samp_embed_feats


def DeepSetNetwork3(inputs, config):
    with tf.variable_scope('embed'):
        N = tf.shape(inputs)[0]
        n = tf.shape(inputs)[1]
        d = int(inputs.get_shape()[-1])
        points = tf.reshape(inputs, [N*n, d])

        # weights
        embed_layers = config.embed_layers
        W0 = tf.get_variable('W0', shape=(d, embed_layers), dtype=tf.float32)
        b0 = tf.get_variable('b0', shape=(embed_layers, ), dtype=tf.float32)
        W1 = tf.get_variable('W1', shape=(embed_layers, embed_layers), dtype=tf.float32)
        b1 = tf.get_variable('b1', shape=(embed_layers, ), dtype=tf.float32)
        W2 = tf.get_variable('W2', shape=(embed_layers, embed_layers), dtype=tf.float32)
        b2 = tf.get_variable('b2', shape=(embed_layers, ), dtype=tf.float32)
        W3 = tf.get_variable('W3', shape=(embed_layers, config.embed_size), dtype=tf.float32)
        b3 = tf.get_variable('b3', shape=(config.embed_size, ), dtype=tf.float32)

        # forward
        # Standard Layer0
        y = tf.nn.elu(tf.nn.xw_plus_b(points, W0, b0, 'linear0'))  # [N*n, embed_layers]
        # PermEqui Layer1
        y = tf.reshape(y, [N, n, embed_layers])                             # [N, n, embed_layers]
        ym = tf.reduce_max(y, 1, True)                             # [N, 1, embed_layers]
        y = y - ym                                                 # [N, n. embed_layers]
        y = tf.reshape(y, [N*n, embed_layers])                              # [N*n, embed_layers]
        y = tf.nn.elu(tf.nn.xw_plus_b(y, W1, b1, name='linear1'))  # [N*n, embed_layers]
        # PermEqui Layer2
        y = tf.reshape(y, [N, n, embed_layers])                             # [N, n, embed_layers]
        ym = tf.reduce_max(y, 1, True)                             # [N, 1, embed_layers]
        y = y - ym                                                 # [N, n. embed_layers]
        y = tf.reshape(y, [N*n, embed_layers])                              # [N*n, embed_layers]
        y = tf.nn.elu(tf.nn.xw_plus_b(y, W2, b2, name='linear2'))  # [N*n, embed_layers]
        # Standard Layer3
        point_feats = tf.nn.xw_plus_b(y, W3, b3, 'linear3')        # [N*n, config.embed_size]

        point_feats_tens = tf.reshape(point_feats, [N, n, config.embed_size])
        samp_embed_feats = tf.concat(
             (tf.reduce_mean(point_feats_tens, 1, False),
              tf.reduce_max(point_feats_tens, 1, False)), -1)
        embed_feats = tf.reshape(
             tf.tile(tf.expand_dims(samp_embed_feats, 1), [1, n, 1]),
             [N*n, 2*config.embed_size])
        return points, embed_feats, samp_embed_feats


class EmbedExperiment:

    # TODO: make all arguements optional, load config from
    # save_location+'config.p' when not given, resolve dimension, and add
    # option to restore
    def __init__(self, config, summary_location, save_location, fetchers):
        self.config = config
        self.summary_location = summary_location
        self.save_location = save_location
        self.fetchers = fetchers
        # Set up model/trainer in graph
        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            # TODO: inputs should be determined by fetchers.
            # TODO: placeholder for set lengths
            if fetchers.train.ndatasets > 1:
                print('Labeled Images')
                inputs_pl = tf.placeholder(
                    tf.float32, (None, None, fetchers.dim[0]), 'inputs')
                # Labeled data. Assumes one-hot
                # TODO: allow for indexed classes.
                conditioning_pl = tf.placeholder(
                    tf.float32, (None, fetchers.dim[1]), 'conditioning')
                log_prior = np.log(np.mean(fetchers.train._datasets[1], 0))
            else:
                # Unlabeled data.
                inputs_pl = tf.placeholder(
                    tf.float32, (None, None, fetchers.dim), 'inputs')
                conditioning_pl = None
            if config.dropout_keeprate_val is not None and \
                    config.dropout_keeprate_val < 1.0:
                self.config.dropout_keeprate = tf.placeholder(
                    tf.float32, [], 'dropout_keeprate')
            else:
                self.config.dropout_keeprate = None
            self.sess = tf.Session()

            with tf.variable_scope('TAN', initializer=config.initializer):
                points, embed_feats, samp_embed_feats = DeepSetNetwork3(inputs_pl, config)
                if conditioning_pl is not None \
                        and config.set_classification or config.set_regression:
                    with tf.variable_scope('set_learning'):
                        set_feats = nn.fc_network(
                            samp_embed_feats,
                            int(conditioning_pl.get_shape()[-1]),
                            config.set_layers,
                            output_init_range=config.embed_irange,
                            activation=config.embed_activation,
                            name='setfeat_network')
                        if config.set_classification:
                            class_llikes = set_feats \
                                + tf.expand_dims(log_prior, 0)
                            self.lbl_loss = tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits(
                                    labels=conditioning_pl, logits=class_llikes)
                            )
                            prediction = tf.argmax(class_llikes, axis=1)
                            true_labels = tf.argmax(conditioning_pl, axis=1)
                            valid_score = tf.cast(
                                tf.equal(prediction, true_labels), tf.float32)
                else:
                    # TODO: append the conditioning_pl?
                    valid_score = None
                with tf.variable_scope('model'):
                    self.model = mod.TANModel(
                        config.transformations,
                        conditional_model=config.conditional_model,
                        likefunc=config.likefunc,
                        param_nlayers=config.param_nlayers,
                        hidden_activation=config.hidden_activation,
                        cond_param_irange=config.cond_param_irange,
                        dropout_keep_prob=config.dropout_keeprate,
                        nparams=config.nparams,
                        base_distribution=config.base_distribution,
                        sample_size=config.sample_batch_size)
                    # TODO: append the conditioning_pl
                    self.nll, self.llikes, self.sampler = \
                        self.model.build_graph(
                            points, embed_feats, samp_embed_feats)
                    if conditioning_pl is not None \
                            and config.set_classification or\
                            config.set_regression:
                        self.loss = (1.0-config.set_loss_alpha)*self.nll + \
                            config.set_loss_alpha*self.lbl_loss
                        self.loss = tf.Print(
                            self.loss, [self.nll, self.lbl_loss], 'nll/lbl')
                    else:
                        self.loss = self.nll
                with tf.variable_scope('train'):
                    if valid_score is None:
                        valid_score = self.llikes
                    self.trn = trainer.RedTrainer(
                        fetchers, self.loss, inputs_pl,
                        valid_score,
                        conditioning_data=conditioning_pl,
                        batch_size=config.batch_size,
                        sess=self.sess,
                        init_lr=config.init_lr,
                        min_lr=config.min_lr,
                        lr_decay=config.lr_decay,
                        decay_interval=config.decay_interval,
                        penalty=config.penalty,
                        dropout_keeprate=self.config.dropout_keeprate,
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
                        input_sample=True,
                        samp_per_cond=config.samp_per_cond,
                        nsamp=config.nsample_batches)

    @property
    def main(self):
        return self.trn.main

    @property
    def sample(self):
	return self.trn.sample
