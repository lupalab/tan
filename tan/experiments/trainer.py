import tensorflow as tf
import numpy as np
import os


TRAIN = 'train'
VALID = 'valid'
TEST = 'test'


# TODO: chance name.
class RedTrainer:
    """Encapsulates the logic for training a sequence model.
    Args:
        fetchers: A dictionary of fetchers for training, validation, and testing
            datasets-
            {TRAIN: train_fetcher, VALID: valid_fetcher, TEST: test_fetcher}.
            Each fetcher implements functions
                ...
        loss: scalar, loss to optimize.
        input_data: placeholder for N x d covariates
        batch_size: int, size of batches for training.
        sess: tf session to train in.
        init_lr: scalar, initial learning rate.
        lr_decay: scalar, multiplicative decay factor of learning rate.
        decay_interval: int, number of batches between decay of learning rate.
        min_lr: scalar, minimum value of learning rate to use.
        penalty: scalar, multiplier to ridge penalty.
        dropout_keeprate: scalar placeholder for dropout value
        dropout_keeprate_val: real 0< <=1 of kept dropout rate for training
        train_iters: int, number of batches to train model for.
        hold_iters: int, number validation batches to use.
        print_iters: int, print training stats (like loss) every print_iters
            batches.
        hold_interval: int, print validation stats every hold_intervals.
        iters_pl: optional placeholder/tensor for iterations.
        # TODO: what is this for?
        iters_func:
        optimizer_class: class of tf optimizer to use.
        max_grad_norm: scalar, norm to clip gradients to.
        do_check: boolean indicating whether to use check_ops for debugging.
        momentum: Deprecated.
        momentum_iter: Deprecated.
        rms_decay: Deprecated.
        rms_eps: Deprecated.
        pretrain_scope: variable scope to match variables with re.match to
            pretrain.
        pretrain_iters: int, number of batches to pretrain for.
        conditioning_data: placeholder of N x p extraneous covariates.
        summary_log_path: string, path to save log files to.
        save_path: string, path to save the graph to.
        sampler:
        input_sample:
        nsamp:
        samp_per_cond:
    """
    def __init__(self, fetchers, loss, input_data, llikes,
                 batch_size=128, sess=None,
                 # Learning rate.
                 init_lr=0.1, lr_decay=0.9, decay_interval=10000, min_lr=None,
                 # Regularization.
                 penalty=0.0, dropout_keeprate=None, dropout_keeprate_val=1.0,
                 # Iteration configs.
                 # TODO: change hold_iter to valid_interval
                 # TODO: change iters to intervals
                 train_iters=100000, hold_iters=1000, print_iters=100,
                 hold_interval=1000,
                 iters_pl=None, iters_func=lambda x: x,
                 # Optimizer configs.
                 optimizer_class=tf.train.GradientDescentOptimizer,
                 max_grad_norm=None, do_check=False,
                 # Momentum.
                 # TODO: remove.
                 momentum=None, momentum_iter=1500, rms_decay=0.9,
                 rms_eps=1e-10,
                 # Pretraining.
                 pretrain_scope=None, pretrain_iters=5000,
                 # Conditioning.
                 conditioning_data=None,
                 # Saving.
                 summary_log_path=None, save_path=None,
                 # Sampling.
                 sampler=None, input_sample=False, nsamp=10, samp_per_cond=1):
        self._input_data = input_data
        self._conditioning_data = conditioning_data

        # Training parameters.
        self._train_iters = train_iters
        self._valid_iters = hold_iters
        self._print_iters = print_iters
        self._hold_interval = hold_interval
        self._lr_decay = lr_decay
        self._decay_interval = decay_interval
        self._batch_size = batch_size
        self._iters_pl = iters_pl
        self._iters_func = iters_func

        # Make losses
        self._llikes = llikes
        self._loss_op = loss
        if penalty > 0.0:
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self._loss_op += penalty*sum(reg_losses)

        # Training operations.
        self._lr = tf.Variable(init_lr, trainable=False)
        self._optimizer = optimizer = optimizer_class(self._lr)
        self.tvars = tvars = tf.trainable_variables()
        grads_tvars = optimizer.compute_gradients(self._loss_op, tvars)
        if max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(
                [gt[0] for gt in grads_tvars], max_grad_norm)
            grads_tvars = zip(grads, tvars)
        self._train_op = optimizer.apply_gradients(grads_tvars)
        if do_check:
            check_op = tf.add_check_numerics_ops()
            self._train_op = tf.group(self._train_op, check_op,
                                      tf.check_numerics(self._loss_op, 'check'))
        if pretrain_scope is not None and pretrain_iters is not None:
            self._do_pretrain = True
            self._pretrain_iters = pretrain_iters
            self.ptvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            pretrain_scope)
            ptgrads = tf.gradients(self._loss_op, self.ptvars)
            if max_grad_norm is not None:
                ptgrads, _ = tf.clip_by_global_norm(ptgrads, max_grad_norm)
            self._pretrain_op = optimizer.apply_gradients(
                zip(ptgrads, self.ptvars)
            )
        else:
            self._do_pretrain = False
        if momentum is not None:
            # mom_optimizer = tf.train.MomentumOptimizer(self._lr, momentum)
            mom_optimizer = tf.train.RMSPropOptimizer(
                self._lr, momentum=momentum, decay=rms_decay, epsilon=rms_eps)
            self._momentum_iter = momentum_iter
            self._momentum_op = mom_optimizer.apply_gradients(grads_tvars)
            # self._momentum_op = tf.group(
            #     mom_optimizer.apply_gradients(grads_tvars), self._train_op)
        else:
            self._momentum_op = None
            self._momentum_iter = None
        if min_lr is None:
            self._lr_update = tf.assign(self._lr, self._lr * self._lr_decay)
        else:
            self._lr_update = tf.assign(
                self._lr, tf.maximum(min_lr, self._lr * self._lr_decay))

        # Make session if needed.
        if sess is None:
            sess = tf.Session()
        self._sess = sess
        # Set up fetchers.
        self._dropout_keeprate = dropout_keeprate
        self._dropout_keeprate_val = dropout_keeprate_val
        self._fetchers = fetchers

        # Sampling.
        self._sampler = sampler
        self._input_sample = input_sample
        self._nsamps = nsamp
        self._samp_per_cond = samp_per_cond

        # Summarization variables.
        self._summary_log_path = summary_log_path
        self._average_pl = tf.placeholder(tf.float32, name='average_pl')
        self._average_summary = tf.summary.scalar('average_loss',
                                                  self._average_pl)
        if self._summary_log_path is not None:
            self._train_writer, self._val_writer, self._test_writer = \
                make_writers(self._summary_log_path, self._sess)
        else:
            self._train_writer, self._val_writer, self._test_writer = \
                (None, None, None)
        if save_path is not None:
            self._saver = tf.train.Saver()
            self._save_path = os.path.join(save_path, 'model.ckpt')
        else:
            self._saver = None
            self._save_path = None

    def update_lr(self):
        self._sess.run(self._lr_update)

    def _setup_feed_dict(self, batch, testing=False, iters=None):
        if self._conditioning_data is None:
            feed_dict = {self._input_data: batch}
        else:
            feed_dict = {self._input_data: batch[0],
                         self._conditioning_data: batch[1]}
        if self._dropout_keeprate is not None:
            if not testing:
                feed_dict[self._dropout_keeprate] = self._dropout_keeprate_val
            else:
                feed_dict[self._dropout_keeprate] = 0.0
        if self._iters_pl is not None:
            if not testing and iters is not None:
                feed_dict[self._iters_pl] = self._iters_func(iters)
            else:
                feed_dict[self._iters_pl] = self._iters_func(self._train_iters)
        return feed_dict

    def _pretrain(self):
        if not self._do_pretrain:
            return True

        for i in xrange(self._pretrain_iters):
            # Decay the learning rate.
            if i % self._decay_interval == 0:
                if i > 0:
                    self.update_lr()
                print('Iter: {} lrate: {}'.format(i, self._sess.run(self._lr)))
            # Setup feed_dict.
            batch = self._fetchers.train.next_batch(self._batch_size)
            feed_dict = self._setup_feed_dict(batch, testing=False, iters=0)
            #  Print to screen and save summary.
            if i % self._print_iters == 0:
                train_loss, _ = self._sess.run(
                    (self._loss_op, self._pretrain_op), feed_dict=feed_dict)
                print('Pretrain Iter: {} Train Loss: {}'.format(i, train_loss))
                # Abort training if we have NaN loss
                if np.isnan(train_loss):
                    return False
            else:
                self._sess.run(self._pretrain_op, feed_dict=feed_dict)
        return True

    def _save(self):
        if self._saver is not None:
            print('Saving {}...'.format(self._save_path))
            self._saver.save(self._sess, self._save_path)

    def _print_loss(self, i, loss, msg='Train Loss', writer=None):
        print('Iter: {} {}: {}'.format(i, msg, loss))
        if writer is not None:
            writer.add_summary(
                self._sess.run(self._average_summary,
                               feed_dict={self._average_pl: loss}), i
            )

    def main(self):
        """Runs the model on the given data.
        Args:
            summary_log_path: path to save tensorboard summaries.
            save_path: path to save best validation set model.
            print_iters: number of iterations to print to screen at.
        Returns:
            tuple of (best_validation_value, test_validation_value)
        """
        self._sess.run(tf.global_variables_initializer())
        # Pretrain if needed.
        if not self._pretrain():
            return {'loss': np.NaN, 'test_llks': None}

        # Main train loop.
        best_loss = None
        train_operation = self._train_op
        for i in xrange(self._train_iters):
            # if i >= 2250:
            #     import pdb; pdb.set_trace()  # XXX BREAKPOINT
            # Decay the learning rate.
            if i % self._decay_interval == 0:
                if i > 0:
                    self.update_lr()
                print('Iter: {} lrate: {}'.format(i, self._sess.run(self._lr)))
            # Use a momentum operator if it is over the momentum iterations.
            if self._momentum_op is not None and i == self._momentum_iter:
                print('Using RMSProp with Momentum.')
                train_operation = self._momentum_op
            # Training.
            batch = self._fetchers.train.next_batch(self._batch_size)
            feed_dict = self._setup_feed_dict(batch, testing=False, iters=i)
            # Print to screen and save summary.
            if i % self._print_iters == 0:
                train_loss, _ = self._sess.run(
                    (self._loss_op, train_operation), feed_dict=feed_dict
                )
                self._print_loss(i, train_loss, writer=self._train_writer)
                # self._print_loss(i, train_loss, writer=None)
                # Abort training if we have NaN loss
                # TODO: use the last saved model with a lower learning rate?
                if np.isnan(train_loss):
                    return {'loss': np.NaN, 'test_llks': None}
            else:
                self._sess.run(train_operation, feed_dict=feed_dict)

            # Validation.
            if i == 0 or i % self._hold_interval == 0 \
                    or i+1 == self._train_iters:
                # Get validation validation value on validation set.
                valid_loss = self.validation_loss(i)
                # If this is the best validation value, record and save model.
                if best_loss is None or best_loss > valid_loss:
                    best_loss = valid_loss
                    self._save()

        # Testing.
        # Get validation value on test set.
        test_llks = self.test_llikelihoods(load_saved_model=True)
        print('Mean test nll {}'.format(-np.mean(test_llks)))

        # Sample using best model.
        if self._sampler is not None:
            samples, samples_cond = self.sample(load_saved_model=True)
            return {'loss': best_loss, 'test_llks': test_llks,
                    'samples': samples, 'samples_cond': samples_cond}
        return {'loss': best_loss, 'test_llks': test_llks}

    def validation_loss(self, i):
        loss = 0.0
        for j in xrange(self._valid_iters):
            batch = self._fetchers.validation.next_batch(self._batch_size)
            feed_dict = self._setup_feed_dict(batch, testing=True)
            loss_batch = -np.mean(
                self._sess.run(self._llikes, feed_dict=feed_dict))
            loss += loss_batch
        loss = loss/self._valid_iters
        if self._val_writer is not None:
            self._val_writer.add_summary(
                self._sess.run(self._average_summary,
                               feed_dict={self._average_pl: loss}), i
            )
        print('Validation nll: {}'.format(loss))
        return loss

    def restore_model(self):
        if self._saver is not None and self._save_path is not None:
            self._saver.restore(self._sess, self._save_path)

    def test_llikelihoods(self, load_saved_model=False):
        if load_saved_model:
            self.restore_model()
        test_list = []
        try:
            while True:
                batch = self._fetchers.test.next_batch(self._batch_size)
                feed_dict = self._setup_feed_dict(batch, testing=True)
                llikes = self._sess.run(self._llikes, feed_dict=feed_dict)
                test_list += [llikes]
        except IndexError:
            self._fetchers.test.reset_index()
            print('REACHED END')
        test_list = np.concatenate(test_list, 0)
        return test_list

    def sample(self, load_saved_model=False):
        if load_saved_model:
            self.restore_model()
        samples = []
        samples_cond = []
        nsamp = int(self._sampler.get_shape()[0])
        for si in range(self._nsamps):
            cond_dict = {}
            if self._dropout_keeprate is not None:
                feed_dict = {self._dropout_keeprate: 0.0}
            else:
                feed_dict = None
            batch = self._fetchers.validation.next_batch(nsamp)
            if self._conditioning_data is not None:
                # Get validation labels to condition on.
                samp_cond = batch[1]
                feed_dict = {} if feed_dict is None else feed_dict
                feed_dict[self._conditioning_data] = samp_cond
                cond_dict['cond_val'] = samp_cond
                if self._input_sample:
                    feed_dict[self._input_data] = batch[0]
                    cond_dict['inp_val'] = batch[0]
            elif self._input_sample:
                feed_dict = {} if feed_dict is None else feed_dict
                feed_dict[self._input_data] = batch
                cond_dict['inp_val'] = batch
            samples_cond.append(cond_dict)
            if self._samp_per_cond == 1:
                samp = self._sess.run(self._sampler, feed_dict=feed_dict)
            else:
                samp = []
                for ci in range(self._samp_per_cond):
                    samp.append(
                        self._sess.run(self._sampler, feed_dict=feed_dict))
                samp = np.stack(samp, 1)
            samples.append(samp)
        samples = np.concatenate(samples, 0)
        # if len(samples_cond) > 0:
        #     return samples, np.concatenate(samples_cond, 0)
        # return samples
        return samples, samples_cond


def make_writers(summary_log_path, sess):
    train_writer = tf.summary.FileWriter(
        os.path.join(summary_log_path, TRAIN), sess.graph
    )
    val_writer = tf.summary.FileWriter(
        os.path.join(summary_log_path, VALID), sess.graph
    )
    test_writer = tf.summary.FileWriter(
        os.path.join(summary_log_path, TEST), sess.graph
    )
    return train_writer, val_writer, test_writer
