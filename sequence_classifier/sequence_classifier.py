import copy
import os
import random
import tempfile
import time
from typing import List, Tuple, Union
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, label_ranking_average_precision_score
from sklearn.utils.validation import check_is_fitted
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def calculate_overall_lwlrap_sklearn(truth: np.ndarray, scores: np.ndarray) -> float:
    sample_weight = np.sum(truth > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    overall_lwlrap = label_ranking_average_precision_score(
        truth[nonzero_weight_sample_indices, :] > 0,
        scores[nonzero_weight_sample_indices, :],
        sample_weight=sample_weight[nonzero_weight_sample_indices])
    return overall_lwlrap


class SequenceClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, num_recurrent_units: Union[List[int], tuple], max_seq_length: int, batch_size: int=32,
                 learning_rate: float=1e-3, l2_reg: float=1e-5, clipnorm: float=1.0, max_epochs: int=1000,
                 patience: int=5, gpu_memory_frac: float=0.9, multioutput: bool=False, warm_start: bool=False,
                 verbose: bool=False, random_seed: Union[int, None]=None):
        self.num_recurrent_units = num_recurrent_units
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.multioutput = multioutput
        self.warm_start = warm_start
        self.max_seq_length = max_seq_length
        self.gpu_memory_frac = gpu_memory_frac
        self.l2_reg = l2_reg
        self.clipnorm = clipnorm
        self.random_seed = random_seed
        self.verbose = verbose

    def fit(self, X: Union[list, tuple, np.ndarray], y: Union[list, tuple, np.ndarray],
            validation_data: Union[Tuple[Union[list, tuple, np.ndarray], Union[list, tuple, np.ndarray]], None]=None):
        self.check_params(
            warm_start=self.warm_start, multioutput=self.multioutput, num_recurrent_units=self.num_recurrent_units,
            batch_size=self.batch_size, max_seq_length=self.max_seq_length, learning_rate=self.learning_rate,
            l2_reg=self.l2_reg, max_epochs=self.max_epochs, patience=self.patience, verbose=self.verbose,
            gpu_memory_frac=self.gpu_memory_frac, clipnorm=self.clipnorm, random_seed=self.random_seed
        )
        n_classes = self.check_Xy(X, 'X', y, 'y')
        self.prepare_to_train()
        if validation_data is not None:
            if (not isinstance(validation_data, list)) and (not isinstance(validation_data, tuple)) and \
                    (not isinstance(validation_data, np.ndarray)):
                raise ValueError('`validation_data` is wrong argument! Expected `{0}` or `{1}`, but got `{2}`.'.format(
                    type([1, 2]), type((1, 2)), type(validation_data)
                ))
            if isinstance(validation_data, np.ndarray):
                if len(validation_data.shape) != 1:
                    raise ValueError('`validation_data` is wrong argument! Expected a 1-D array, but got a '
                                     '{0}-D one.'.format(len(validation_data.shape)))
            if len(validation_data) != 2:
                raise ValueError('`validation_data` is wrong argument! Expected a two-element sequence (inputs and '
                                 'targets), but got a {0}-element one.'.format(len(validation_data)))
            n_val_classes = self.check_Xy(validation_data[0], 'X_val', validation_data[1], 'y_val')
            if n_val_classes != n_classes:
                raise ValueError('`validation_data` is wrong argument! Number of classes in validation set is not '
                                 'correspond to number of classes in training set! '
                                 '{0} != {1}.'.format(n_val_classes, n_classes))
        if self.warm_start:
            self.is_fitted()
            new_layers = (self.num_recurrent_units if isinstance(self.num_recurrent_units, tuple) else
                          (tuple(self.num_recurrent_units.tolist()) if isinstance(self.num_recurrent_units, np.ndarray)
                           else tuple(self.num_recurrent_units)))
            new_input_shape = (self.max_seq_length, self.get_feature_vector_size(X))
            if new_input_shape[1] != self.input_shape_[1]:
                raise ValueError('Structure of the training data does not correspond to the neural network structure! '
                                 'Feature vector size must be equal to {0}, but it is equal to {1}.'.format(
                    self.input_shape_[1], new_input_shape[1]))
            if new_layers != self.layers_:
                raise ValueError('Old structure of recurrent layers does not correspond to new structure! '
                                 '{0} != {1}'.format(self.layers_, new_layers))
            if self.n_classes_ != n_classes:
                raise ValueError('Old number of classes does not correspond to new number! {0} != {1}'.format(
                    self.n_classes_, n_classes))
        if self.random_seed is None:
            self.random_seed = int(round(time.time()))
        tf.set_random_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        self.n_classes_ = n_classes
        if not self.warm_start:
            self.finalize_model()
            self.input_shape_ = (self.max_seq_length, self.get_feature_vector_size(X))
        self.layers_ = (self.num_recurrent_units if isinstance(self.num_recurrent_units, tuple) else
                        (tuple(self.num_recurrent_units.tolist()) if isinstance(self.num_recurrent_units, np.ndarray)
                         else tuple(self.num_recurrent_units)))
        train_op, eval_loss = self.build_model(self.input_shape_, self.n_classes_)
        n_train_batches = int(np.ceil(len(y) / float(self.batch_size)))
        bounds_of_batches_for_training = []
        for iteration in range(n_train_batches):
            batch_start = iteration * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(y))
            bounds_of_batches_for_training.append((batch_start, batch_end))
        bounds_of_batches_for_validation = []
        if validation_data is None:
            n_val_batches = 0
        else:
            n_val_batches = int(np.ceil(len(validation_data[1]) / float(self.batch_size)))
            for iteration in range(n_val_batches):
                batch_start = iteration * self.batch_size
                batch_end = min(batch_start + self.batch_size, len(validation_data[1]))
                bounds_of_batches_for_validation.append((batch_start, batch_end))
        init = tf.global_variables_initializer()
        init.run(session=self.sess_)
        tmp_model_name = self.get_temp_model_name()
        if self.verbose:
            if n_val_batches > 0:
                if self.multioutput:
                    print('Epoch   Train loss   Val. loss   Val. LwLRAP   Duration (secs)')
                else:
                    print('Epoch   Train loss   Val. loss    Val. F1   Duration (secs)')
            else:
                print('Epoch   Train loss   Duration (secs)')
        n_epochs_without_improving = 0
        try:
            best_quality = None
            for epoch in range(self.max_epochs):
                start_time = time.time()
                train_loss_value = self.do_training_epoch(X, y, bounds_of_batches_for_training, train_op, eval_loss)
                if n_val_batches > 0:
                    val_loss_value, cur_quality = self.do_validation(validation_data[0], validation_data[1],
                                                                     bounds_of_batches_for_validation, eval_loss)
                    epoch_duration = time.time() - start_time
                    if self.verbose:
                        if self.multioutput:
                            print('{0:>5}   {1:>10.6f}   {2:>9.6f}   {3:>11.6f}   {4:>15.3f}'.format(
                                epoch + 1, train_loss_value, val_loss_value, cur_quality, epoch_duration))
                        else:
                            print('{0:>5}   {1:>10.6f}   {2:>9.6f}   {3:>8.6f}   {4:>15.3f}'.format(
                                epoch + 1, train_loss_value, val_loss_value, cur_quality, epoch_duration))
                else:
                    epoch_duration = time.time() - start_time
                    if self.verbose:
                        print('{0:>5}   {1:>10.6f}   {2:>15.3f}'.format(epoch + 1, train_loss_value, epoch_duration))
                    cur_quality = -train_loss_value
                if best_quality is None:
                    best_quality = cur_quality
                    self.save_model(tmp_model_name)
                    n_epochs_without_improving = 0
                else:
                    if cur_quality > best_quality:
                        best_quality = cur_quality
                        self.save_model(tmp_model_name)
                        n_epochs_without_improving = 0
                    else:
                        n_epochs_without_improving += 1
                if n_epochs_without_improving >= self.patience:
                    if self.verbose:
                        print('Epoch {0}: early stopping...'.format(epoch + 1))
                    break
            if best_quality is not None:
                self.finalize_model()
                if self.warm_start:
                    self.warm_start = False
                    _, log_likelihood = self.build_model(self.input_shape_, self.n_classes_)
                    self.warm_start = True
                else:
                    _, log_likelihood = self.build_model(self.input_shape_, self.n_classes_)
                self.load_model(tmp_model_name)
        finally:
            for cur_name in self.find_all_model_files(tmp_model_name):
                os.remove(cur_name)
        return self

    def do_training_epoch(self, X: Union[list, tuple, np.ndarray], y: Union[list, tuple, np.ndarray],
                          batches: List[Tuple[int, int]], train_op, eval_loss) -> float:
        indices_in_batch = np.arange(0, self.batch_size, 1, dtype=np.int32)
        random.shuffle(batches)
        feed_dict_for_batch = None
        for batch_start, batch_end in batches:
            X_batch = []
            y_batch = []
            for idx in range(batch_start, batch_end):
                X_inst = self.get_data_input(X, idx, True)
                y_inst = self.get_data_target(y, idx)
                if not isinstance(X_inst, np.ndarray):
                    raise ValueError('Training sample {0} is wrong! Expected `{1}`, got `{2}`.'.format(
                        batch_start + idx, type(np.array([1, 2])), type(X_inst)))
                if X_inst.shape[1] != self.input_shape_[1]:
                    raise ValueError('Feature size of training sample {0} is wrong! Expected {1}, got '
                                     '{2}.'.format(batch_start + idx, self.input_shape_[1], X_inst.shape[1]))
                if X_inst.shape[0] == self.input_shape_[0]:
                    X_batch.append(X_inst)
                elif X_inst.shape[0] > self.input_shape_[0]:
                    X_batch.append((X_inst[0:self.input_shape_[0]]))
                else:
                    X_batch.append(
                        np.vstack(
                            (
                                X_inst,
                                np.zeros((self.input_shape_[0] - X_inst.shape[0], X_inst.shape[1]),
                                         dtype=X_inst.dtype)
                            )
                        )
                    )
                y_batch.append(y_inst)
                del X_inst, y_inst
            X_batch = np.concatenate([np.reshape(cur, (1, cur.shape[0], cur.shape[1])) for cur in X_batch], axis=0)
            y_batch = np.concatenate([np.reshape(cur, (1, cur.shape[0])) for cur in y_batch], axis=0)
            if feed_dict_for_batch is not None:
                del feed_dict_for_batch
            if X_batch.shape[0] < self.batch_size:
                if X_batch.shape[0] < (self.batch_size - X_batch.shape[0]):
                    idx = list(range(X_batch.shape[0]))
                    n = X_batch.shape[0] * 2
                    while n <= (self.batch_size - X_batch.shape[0]):
                        idx += list(range(X_batch.shape[0]))
                        n += X_batch.shape[0]
                    idx += random.sample(list(range(X_batch.shape[0])), self.batch_size - n)
                else:
                    idx = random.sample(list(range(X_batch.shape[0])), self.batch_size - X_batch.shape[0])
                X_batch = np.concatenate((X_batch, X_batch[idx]), axis=0)
                y_batch = np.concatenate((y_batch, y_batch[idx]), axis=0)
                del idx
            np.random.shuffle(indices_in_batch)
            feed_dict_for_batch = self.fill_feed_dict(X_batch[indices_in_batch], y_batch[indices_in_batch])
            self.sess_.run(train_op, feed_dict=feed_dict_for_batch)
            del X_batch, y_batch
        del indices_in_batch
        return eval_loss.eval(feed_dict=feed_dict_for_batch, session=self.sess_)

    def do_validation(self, X: Union[list, tuple, np.ndarray], y: Union[list, tuple, np.ndarray],
                      batches: List[Tuple[int, int]], eval_loss) -> Tuple[float, float]:
        y_true = []
        y_pred = []
        feed_dict_for_batch = None
        total_loss = 0.0
        for batch_start, batch_end in batches:
            X_batch = []
            y_batch = []
            for idx in range(batch_start, batch_end):
                X_inst = self.get_data_input(X, idx, False)
                y_inst = self.get_data_target(y, idx)
                if not isinstance(X_inst, np.ndarray):
                    raise ValueError('Validation sample {0} is wrong! Expected `{1}`, got `{2}`.'.format(
                        batch_start + idx, type(np.array([1, 2])), type(X_inst)))
                if X_inst.shape[1] != self.input_shape_[1]:
                    raise ValueError('Feature size of validation sample {0} is wrong! Expected {1}, got '
                                     '{2}.'.format(batch_start + idx, self.input_shape_[1],
                                                   X_inst.shape[1]))
                if X_inst.shape[0] == self.input_shape_[0]:
                    X_batch.append(X_inst)
                elif X_inst.shape[0] > self.input_shape_[0]:
                    X_batch.append((X_inst[0:self.input_shape_[0]]))
                else:
                    X_batch.append(
                        np.vstack(
                            (
                                X_inst,
                                np.zeros((self.input_shape_[0] - X_inst.shape[0], X_inst.shape[1]),
                                         dtype=X_inst.dtype)
                            )
                        )
                    )
                y_batch.append(y_inst)
                del X_inst, y_inst
            X_batch = np.concatenate([np.reshape(cur, (1, cur.shape[0], cur.shape[1])) for cur in X_batch], axis=0)
            y_batch = np.concatenate([np.reshape(cur, (1, cur.shape[0])) for cur in y_batch], axis=0)
            if feed_dict_for_batch is not None:
                del feed_dict_for_batch
            if X_batch.shape[0] < self.batch_size:
                idx = [(X_batch.shape[0] - 1) for _ in range(self.batch_size - X_batch.shape[0])]
                X_batch = np.concatenate((X_batch, X_batch[idx]), axis=0)
                y_batch = np.concatenate((y_batch, y_batch[idx]), axis=0)
                feed_dict_for_batch = self.fill_feed_dict(X_batch, y_batch)
                val_loss_batch, logits = self.sess_.run(
                    [eval_loss, self.logits_], feed_dict=feed_dict_for_batch
                )
                y_true.append(y_batch[:(batch_end - batch_start)])
                del idx
            else:
                feed_dict_for_batch = self.fill_feed_dict(X_batch, y_batch)
                val_loss_batch, logits = self.sess_.run([eval_loss, self.logits_], feed_dict=feed_dict_for_batch)
                y_true.append(y_batch)
            y_pred.append(logits[:(batch_end - batch_start)])
            total_loss += val_loss_batch
            del X_batch, y_batch, logits
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        total_loss /= float(len(batches))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.multioutput:
                res = calculate_overall_lwlrap_sklearn(y_true, y_pred)
            else:
                res = f1_score(y_true.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
        return total_loss, res

    def predict(self, X: Union[list, tuple, np.ndarray]) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X: Union[list, tuple, np.ndarray]) -> np.ndarray:
        self.is_fitted()
        n_samples = self.check_X(X, 'X')
        res = np.zeros((n_samples, self.n_classes_), dtype=np.float32)
        n_batches = int(np.ceil(n_samples / float(self.batch_size)))
        feed_dict_for_batch = None
        for iteration in range(n_batches):
            batch_start = iteration * self.batch_size
            batch_end = min(batch_start + self.batch_size, n_samples)
            X_batch = []
            for idx in range(batch_start, batch_end):
                X_inst = self.get_data_input(X, idx, False)
                if not isinstance(X_inst, np.ndarray):
                    raise ValueError('Validation sample {0} is wrong! Expected `{1}`, got `{2}`.'.format(
                        batch_start + idx, type(np.array([1, 2])), type(X_inst)))
                if X_inst.shape[1] != self.input_shape_[1]:
                    raise ValueError('Feature size of validation sample {0} is wrong! Expected {1}, got '
                                     '{2}.'.format(batch_start + idx, self.input_shape_[1],
                                                   X_inst.shape[1]))
                if X_inst.shape[0] == self.input_shape_[0]:
                    X_batch.append(X_inst)
                elif X_inst.shape[0] > self.input_shape_[0]:
                    X_batch.append((X_inst[0:self.input_shape_[0]]))
                else:
                    X_batch.append(
                        np.vstack(
                            (
                                X_inst,
                                np.zeros((self.input_shape_[0] - X_inst.shape[0], X_inst.shape[1]),
                                         dtype=X_inst.dtype)
                            )
                        )
                    )
                del X_inst
            X_batch = np.concatenate([np.reshape(cur, (1, cur.shape[0], cur.shape[1])) for cur in X_batch], axis=0)
            if feed_dict_for_batch is not None:
                del feed_dict_for_batch
            if X_batch.shape[0] < self.batch_size:
                idx = [(X_batch.shape[0] - 1) for _ in range(self.batch_size - X_batch.shape[0])]
                X_batch = np.concatenate((X_batch, X_batch[idx]), axis=0)
                feed_dict_for_batch = self.fill_feed_dict(X_batch)
                logits = self.logits_.eval(feed_dict=feed_dict_for_batch,
                                           session=self.sess_)[:(batch_end - batch_start)]
                del idx
            else:
                feed_dict_for_batch = self.fill_feed_dict(X_batch)
                logits = self.logits_.eval(feed_dict=feed_dict_for_batch, session=self.sess_)
            for idx in range(batch_start, batch_end):
                res[idx] = logits[idx - batch_start]
            del X_batch, logits
        return res

    def predict_log_proba(self, X: Union[list, tuple, np.ndarray]) -> np.ndarray:
        return np.log(self.predict_proba(X) + 1e-5)

    def fit_predict(self, X: Union[list, tuple, np.array], y: Union[list, tuple, np.array], **kwargs):
        return self.fit(X, y).predict(X)

    def score(self, X: Union[list, tuple, np.ndarray], y: Union[list, tuple, np.ndarray],
              sample_weight: Union[list, tuple, np.ndarray, None]=None):
        self.is_fitted()
        n_samples = self.check_X(X, 'X')
        y_true = np.zeros((n_samples, self.n_classes_), dtype=np.float32)
        y_pred = np.zeros((n_samples, self.n_classes_), dtype=np.float32)
        n_batches = int(np.ceil(n_samples / float(self.batch_size)))
        feed_dict_for_batch = None
        for iteration in range(n_batches):
            batch_start = iteration * self.batch_size
            batch_end = min(batch_start + self.batch_size, n_samples)
            X_batch = []
            for idx in range(batch_start, batch_end):
                X_inst = self.get_data_input(X, idx, False)
                y_inst = self.get_data_target(y, idx)
                if not isinstance(X_inst, np.ndarray):
                    raise ValueError('Validation sample {0} is wrong! Expected `{1}`, got `{2}`.'.format(
                        batch_start + idx, type(np.array([1, 2])), type(X_inst)))
                if X_inst.shape[1] != self.input_shape_[1]:
                    raise ValueError('Feature size of validation sample {0} is wrong! Expected {1}, got '
                                     '{2}.'.format(batch_start + idx, self.input_shape_[1],
                                                   X_inst.shape[1]))
                if X_inst.shape[0] == self.input_shape_[0]:
                    X_batch.append(X_inst)
                elif X_inst.shape[0] > self.input_shape_[0]:
                    X_batch.append((X_inst[0:self.input_shape_[0]]))
                else:
                    X_batch.append(
                        np.vstack(
                            (
                                X_inst,
                                np.zeros((self.input_shape_[0] - X_inst.shape[0], X_inst.shape[1]),
                                         dtype=X_inst.dtype)
                            )
                        )
                    )
                y_true[idx] = y_inst
                del X_inst, y_inst
            X_batch = np.concatenate([np.reshape(cur, (1, cur.shape[0], cur.shape[1])) for cur in X_batch], axis=0)
            if feed_dict_for_batch is not None:
                del feed_dict_for_batch
            if X_batch.shape[0] < self.batch_size:
                idx = [(X_batch.shape[0] - 1) for _ in range(self.batch_size - X_batch.shape[0])]
                X_batch = np.concatenate((X_batch, X_batch[idx]), axis=0)
                feed_dict_for_batch = self.fill_feed_dict(X_batch)
                logits = self.logits_.eval(feed_dict=feed_dict_for_batch,
                                           session=self.sess_)[:(batch_end - batch_start)]
                del idx
            else:
                feed_dict_for_batch = self.fill_feed_dict(X_batch)
                logits = self.logits_.eval(feed_dict=feed_dict_for_batch, session=self.sess_)
            for idx in range(batch_start, batch_end):
                y_pred[idx] = logits[idx - batch_start]
            del X_batch, logits
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.multioutput:
                quality = calculate_overall_lwlrap_sklearn(y_true, y_pred)
            else:
                quality = f1_score(y_true.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
        return quality

    def is_fitted(self):
        check_is_fitted(self, ['n_classes_', 'rnn_output_', 'logits_', 'input_data_', 'input_shape_', 'layers_',
                               'y_ph_', 'sess_'])

    def fill_feed_dict(self, X: np.ndarray, y: np.ndarray = None) -> dict:
        assert len(X.shape) == 3
        assert X.shape[0] == self.batch_size
        feed_dict = {self.input_data_: X}
        if y is not None:
            feed_dict[self.y_ph_] = y
        return feed_dict

    def prepare_to_train(self):
        pass

    def get_data_input(self, X: Union[list, tuple, np.ndarray], idx: int, training_phase: bool) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            raise ValueError('`X` is wrong! Expected `{0}`, got `{1}`.'.format(type(np.array([1, 2])), type(X)))
        if len(X.shape) != 3:
            raise ValueError('`X` is wrong! Expected a 3-D array, got a {0}-D one.'.format(len(X.shape)))
        return X[idx]

    def get_data_target(self, y: Union[list, tuple, np.ndarray], idx: int) -> np.ndarray:
        if isinstance(y, np.ndarray):
            if len(y.shape) == 1:
                classes_distr = np.zeros((self.n_classes_,), dtype=np.float32)
                if isinstance(y[idx], set):
                    set_of_classes = y[idx]
                else:
                    if isinstance(y[idx], int):
                        set_of_classes = {y[idx]}
                    else:
                        set_of_classes = {int(y[idx])}
                for class_idx in set_of_classes:
                    classes_distr[class_idx] = 1.0
            else:
                classes_distr = y[idx]
        else:
            classes_distr = np.zeros((self.n_classes_,), dtype=np.float32)
            if isinstance(y[idx], set):
                set_of_classes = y[idx]
            else:
                if isinstance(y[idx], int):
                    set_of_classes = {y[idx]}
                else:
                    set_of_classes = {int(y[idx])}
            for class_idx in set_of_classes:
                classes_distr[class_idx] = 1.0
        return classes_distr

    def prepare_to_predict(self):
        pass

    def get_feature_vector_size(self, X: Union[list, tuple, np.ndarray]) -> int:
        if not isinstance(X, np.ndarray):
            raise ValueError('`X` is wrong! Expected `{0}`, got `{1}`.'.format(type(np.array([1, 2])), type(X)))
        if len(X.shape) != 3:
            raise ValueError('`X` is wrong! Expected a 3-D array, got a {0}-D one.'.format(len(X.shape)))
        return X.shape[2]

    def build_model(self, input_size: Tuple[int, int], n_classes: int):
        if not self.warm_start:
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_frac
            self.sess_ = tf.Session(config=config)
            self.input_data_ = tf.placeholder(
                shape=(self.batch_size, input_size[0], input_size[1]), dtype=tf.float32,
                name='InputSequences'
            )
            with tf.name_scope('Masking'):
                sequence_lengths = tf.reduce_sum(tf.sign(tf.reduce_sum(tf.abs(self.input_data_), axis=-1)), axis=-1)
                sequence_lengths = tf.cast(x=sequence_lengths, dtype=tf.int32)
            self.y_ph_ = tf.placeholder(shape=(self.batch_size, n_classes), dtype=tf.float32, name='OutputClasses')
            with tf.name_scope('BiRNN'):
                forward_cells = [
                    tf.contrib.rnn.SRUCell(num_units=self.num_recurrent_units[layer_idx], activation=tf.nn.tanh,
                                           reuse=False, name='ForwardSRUCell{0}'.format(layer_idx + 1))
                    for layer_idx in range(len(self.num_recurrent_units))
                ]
                backward_cells = [
                    tf.contrib.rnn.SRUCell(num_units=self.num_recurrent_units[layer_idx], activation=tf.nn.tanh,
                                           reuse=False, name='BackwardSRUCell{0}'.format(layer_idx + 1))
                    for layer_idx in range(len(self.num_recurrent_units))
                ]
                _, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    cells_fw=forward_cells, cells_bw=backward_cells, inputs=self.input_data_,
                    sequence_length=sequence_lengths, dtype=tf.float32, time_major=False
                )
                self.rnn_output_ = tf.concat([output_state_bw[-1], output_state_bw[-1]], axis=-1)
        glorot_init = tf.keras.initializers.glorot_uniform(seed=self.random_seed)
        self.logits_ = tf.keras.layers.Dense(
            units=n_classes, activation=('sigmoid' if self.multioutput else 'softmax'),
            kernel_regularizer=tf.nn.l2_loss, kernel_initializer=glorot_init, name='FinalOutputs'
        )(self.rnn_output_)
        if self.multioutput:
            loss_tensor = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_ph_, logits=self.logits_,
                                                                  name='SigmoidXEntropyLoss')
        else:
            loss_tensor = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_ph_, logits=self.logits_,
                                                                     name='SoftmaxXEntropyLoss')
        if self.l2_reg > 0.0:
            base_loss = tf.reduce_mean(loss_tensor)
            regularization_loss = self.l2_reg * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            final_loss = base_loss + regularization_loss
        else:
            final_loss = tf.reduce_mean(loss_tensor)
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
            grads_and_vars = optimizer.compute_gradients(final_loss)
            capped_gvs = [
                (grad, var) if grad is None else (
                    tf.clip_by_norm(grad, self.clipnorm, name='grad_clipping_{0}'.format(idx + 1)),
                    var
                )
                for idx, (grad, var) in enumerate(grads_and_vars)
            ]
            train_op = optimizer.apply_gradients(capped_gvs)
        with tf.name_scope('eval'):
            if self.multioutput:
                loss_tensor_eval = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_ph_, logits=self.logits_,
                                                                           name='SigmoidXEntropyEvalLoss')
            else:
                loss_tensor_eval = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_ph_, logits=self.logits_,
                                                                              name='SoftmaxXEntropyEvalLoss')
            eval_loss = tf.reduce_mean(loss_tensor_eval)
        return train_op, eval_loss

    def finalize_model(self):
        if hasattr(self, 'input_data_'):
            del self.input_data_
        if hasattr(self, 'y_ph_'):
            del self.y_ph_
        if hasattr(self, 'logits_'):
            del self.logits_
        if hasattr(self, 'rnn_output_'):
            del self.rnn_output_
        if hasattr(self, 'sess_'):
            for k in list(self.sess_.graph.get_all_collection_keys()):
                self.sess_.graph.clear_collection(k)
            self.sess_.close()
            del self.sess_
        tf.reset_default_graph()

    def save_model(self, file_name: str):
        saver = tf.train.Saver()
        saver.save(self.sess_, file_name)

    def load_model(self, file_name: str):
        saver = tf.train.Saver()
        saver.restore(self.sess_, file_name)

    def get_params(self, deep=True) -> dict:
        return {'multioutput': self.multioutput, 'warm_start': self.warm_start,
                'num_recurrent_units': self.num_recurrent_units, 'batch_size': self.batch_size,
                'max_seq_length': self.max_seq_length, 'learning_rate': self.learning_rate, 'l2_reg': self.l2_reg,
                'clipnorm': self.clipnorm, 'max_epochs': self.max_epochs, 'patience': self.patience,
                'gpu_memory_frac': self.gpu_memory_frac, 'verbose': self.verbose, 'random_seed': self.random_seed}

    def set_params(self, **params):
        for parameter, value in params.items():
            self.__setattr__(parameter, value)
        return self

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.set_params(
            warm_start=self.warm_start, multioutput=self.multioutput, num_recurrent_units=self.num_recurrent_units,
            batch_size=self.batch_size, max_seq_length=self.max_seq_length, learning_rate=self.learning_rate,
            l2_reg=self.l2_reg, max_epochs=self.max_epochs, patience=self.patience, verbose=self.verbose,
            gpu_memory_frac=self.gpu_memory_frac, clipnorm=self.clipnorm, random_seed=self.random_seed
        )
        try:
            self.is_fitted()
            is_fitted = True
        except:
            is_fitted = False
        if is_fitted:
            result.n_classes_ = self.n_classes_
            result.rnn_output_ = self.rnn_output_
            result.logits_ = self.logits_
            result.input_data_ = self.input_data_
            result.input_shape_ = self.input_shape_
            result.layers_ = self.layers_
            result.y_ph_ = self.y_ph_
            result.sess_ = self.sess_
        return result

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result.set_params(
            warm_start=self.warm_start, multioutput=self.multioutput, num_recurrent_units=self.num_recurrent_units,
            batch_size=self.batch_size, max_seq_length=self.max_seq_length, learning_rate=self.learning_rate,
            l2_reg=self.l2_reg, max_epochs=self.max_epochs, patience=self.patience, verbose=self.verbose,
            gpu_memory_frac=self.gpu_memory_frac, clipnorm=self.clipnorm, random_seed=self.random_seed
        )
        try:
            self.is_fitted()
            is_fitted = True
        except:
            is_fitted = False
        if is_fitted:
            result.n_classes_ = self.n_classes_
            result.rnn_output_ = self.rnn_output_
            result.logits_ = self.logits_
            result.input_data_ = self.input_data_
            result.input_shape_ = self.input_shape_
            result.layers_ = self.layers_
            result.y_ph_ = self.y_ph_
            result.sess_ = self.sess_
        return result

    def __getstate__(self):
        return self.dump_all()

    def __setstate__(self, state: dict):
        self.load_all(state)

    def dump_all(self):
        try:
            self.is_fitted()
            is_fitted = True
        except:
            is_fitted = False
        params = self.get_params(True)
        if is_fitted:
            params['layers_'] = copy.copy(self.layers_)
            params['input_shape_'] = copy.copy(self.input_shape_)
            params['n_classes_'] = self.n_classes_
            model_file_name = self.get_temp_model_name()
            try:
                params['model_name_'] = os.path.basename(model_file_name)
                self.save_model(model_file_name)
                for cur_name in self.find_all_model_files(model_file_name):
                    with open(cur_name, 'rb') as fp:
                        model_data = fp.read()
                    params['model.' + os.path.basename(cur_name)] = model_data
                    del model_data
            finally:
                for cur_name in self.find_all_model_files(model_file_name):
                    os.remove(cur_name)
        return params

    def load_all(self, new_params: dict):
        if not isinstance(new_params, dict):
            raise ValueError('`new_params` is wrong! Expected `{0}`, got `{1}`.'.format(type({0: 1}), type(new_params)))
        self.check_params(**new_params)
        self.finalize_model()
        is_fitted = ('layers_' in new_params) and ('input_shape_' in new_params) and \
                    ('n_classes_' in new_params)
        model_files = list(
            filter(
                lambda it3: len(it3) > 0,
                map(
                    lambda it2: it2[len('model.'):].strip(),
                    filter(
                        lambda it1: it1.startswith('model.') and (len(it1) > len('model.')),
                        new_params.keys()
                    )
                )
            )
        )
        if is_fitted and (len(model_files) == 0):
            is_fitted = False
        if is_fitted:
            tmp_dir_name = tempfile.gettempdir()
            tmp_file_names = [os.path.join(tmp_dir_name, cur) for cur in model_files]
            for cur in tmp_file_names:
                if os.path.isfile(cur):
                    raise ValueError('File `{0}` exists, and so it cannot be used for data transmission!'.format(cur))
            self.set_params(**new_params)
            self.layers_ = copy.copy(new_params['layers_'])
            self.input_shape_ = copy.copy(new_params['input_shape_'])
            self.n_classes_ = new_params['n_classes_']
            if self.random_seed is None:
                self.random_seed = int(round(time.time()))
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            try:
                for idx in range(len(model_files)):
                    with open(tmp_file_names[idx], 'wb') as fp:
                        fp.write(new_params['model.' + model_files[idx]])
                if self.warm_start:
                    self.warm_start = False
                    self.build_model(self.input_shape_, self.n_classes_)
                    self.warm_start = True
                else:
                    self.build_model(self.input_shape_, self.n_classes_)
                self.load_model(os.path.join(tmp_dir_name, new_params['model_name_']))
            finally:
                for cur in tmp_file_names:
                    if os.path.isfile(cur):
                        os.remove(cur)
        else:
            self.set_params(**new_params)
        return self

    @staticmethod
    def get_temp_model_name() -> str:
        return tempfile.NamedTemporaryFile(mode='w', suffix='bert_crf.ckpt').name

    @staticmethod
    def find_all_model_files(model_name: str) -> List[str]:
        model_files = []
        if os.path.isfile(model_name):
            model_files.append(model_name)
        dir_name = os.path.dirname(model_name)
        base_name = os.path.basename(model_name)
        for cur in filter(lambda it: it.lower().find(base_name.lower()) >= 0, os.listdir(dir_name)):
            model_files.append(os.path.join(dir_name, cur))
        return sorted(model_files)

    @staticmethod
    def check_params(**kwargs):
        if 'batch_size' not in kwargs:
            raise ValueError('`batch_size` is not specified!')
        if (not isinstance(kwargs['batch_size'], int)) and (not isinstance(kwargs['batch_size'], np.int32)) and \
                (not isinstance(kwargs['batch_size'], np.uint32)):
            raise ValueError('`batch_size` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3), type(kwargs['batch_size'])))
        if kwargs['batch_size'] < 1:
            raise ValueError('`batch_size` is wrong! Expected a positive integer value, '
                             'but {0} is not positive.'.format(kwargs['batch_size']))
        if 'num_recurrent_units' not in kwargs:
            raise ValueError('`num_recurrent_units` is not specified!')
        if (not isinstance(kwargs['num_recurrent_units'], list)) and (not isinstance(kwargs['num_recurrent_units'], tuple)) and \
                (not isinstance(kwargs['num_recurrent_units'], np.ndarray)):
            raise ValueError('`num_recurrent_units` is wrong! Expected `{0}`, got `{1}`.'.format(
                type([3, 4]), type(kwargs['num_recurrent_units'])))
        if isinstance(kwargs['num_recurrent_units'], np.ndarray):
            if len(kwargs['num_recurrent_units'].shape) != 1:
                raise ValueError('`num_recurrent_units` is wrong! Expected a 1-D array, got a {0}-D one.'.format(
                    len(kwargs['num_recurrent_units'].shape)))
        if len(kwargs['num_recurrent_units']) < 1:
            raise ValueError('`num_recurrent_units` is wrong! Expected a nonempty list.')
        for layer_idx in range(len(kwargs['num_recurrent_units'])):
            if kwargs['num_recurrent_units'][layer_idx] < 1:
                raise ValueError('Item {0} of `num_recurrent_units` is wrong! Expected a positive integer value, but '
                                 '{1} is not positive.'.format(layer_idx, kwargs['num_recurrent_units'][layer_idx]))
        if 'learning_rate' not in kwargs:
            raise ValueError('`learning_rate` is not specified!')
        if (not isinstance(kwargs['learning_rate'], float)) and (not isinstance(kwargs['learning_rate'], np.float32)) \
                and (not isinstance(kwargs['learning_rate'], np.float64)):
            raise ValueError('`learning_rate` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3.5), type(kwargs['learning_rate'])))
        if kwargs['learning_rate'] <= 0.0:
            raise ValueError('`learning_rate` is wrong! Expected a positive floating-point value, '
                             'but {0} is not positive.'.format(kwargs['learning_rate']))
        if 'l2_reg' not in kwargs:
            raise ValueError('`l2_reg` is not specified!')
        if (not isinstance(kwargs['l2_reg'], float)) and (not isinstance(kwargs['l2_reg'], np.float32)) and \
                (not isinstance(kwargs['l2_reg'], np.float64)):
            raise ValueError('`l2_reg` is wrong! Expected `{0}`, got `{1}`.'.format(type(3.5), type(kwargs['l2_reg'])))
        if kwargs['l2_reg'] < 0.0:
            raise ValueError('`l2_reg` is wrong! Expected a non-negative floating-point value, '
                             'but {0} is negative.'.format(kwargs['l2_reg']))
        if 'clipnorm' not in kwargs:
            raise ValueError('`clipnorm` is not specified!')
        if kwargs['clipnorm'] is not None:
            if (not isinstance(kwargs['clipnorm'], float)) and (not isinstance(kwargs['clipnorm'], np.float32)) and \
                    (not isinstance(kwargs['clipnorm'], np.float64)):
                raise ValueError('`clipnorm` is wrong! Expected `{0}`, got `{1}`.'.format(
                    type(3.5), type(kwargs['clipnorm'])))
            if kwargs['clipnorm'] <= 0.0:
                raise ValueError('`clipnorm` is wrong! Expected a positive floating-point value, '
                                 'but {0} is not positive.'.format(kwargs['clipnorm']))
        if 'warm_start' not in kwargs:
            raise ValueError('`warm_start` is not specified!')
        if (not isinstance(kwargs['warm_start'], int)) and (not isinstance(kwargs['warm_start'], np.int32)) and \
                (not isinstance(kwargs['warm_start'], np.uint32)) and \
                (not isinstance(kwargs['warm_start'], bool)) and (not isinstance(kwargs['warm_start'], np.bool)):
            raise ValueError('`warm_start` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(True), type(kwargs['warm_start'])))
        if 'multioutput' not in kwargs:
            raise ValueError('`multioutput` is not specified!')
        if (not isinstance(kwargs['multioutput'], int)) and (not isinstance(kwargs['multioutput'], np.int32)) and \
                (not isinstance(kwargs['multioutput'], np.uint32)) and \
                (not isinstance(kwargs['multioutput'], bool)) and (not isinstance(kwargs['multioutput'], np.bool)):
            raise ValueError('`multioutput` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(True), type(kwargs['multioutput'])))
        if 'max_epochs' not in kwargs:
            raise ValueError('`max_epochs` is not specified!')
        if (not isinstance(kwargs['max_epochs'], int)) and (not isinstance(kwargs['max_epochs'], np.int32)) and \
                (not isinstance(kwargs['max_epochs'], np.uint32)):
            raise ValueError('`max_epochs` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3), type(kwargs['max_epochs'])))
        if kwargs['max_epochs'] < 1:
            raise ValueError('`max_epochs` is wrong! Expected a positive integer value, '
                             'but {0} is not positive.'.format(kwargs['max_epochs']))
        if 'patience' not in kwargs:
            raise ValueError('`patience` is not specified!')
        if (not isinstance(kwargs['patience'], int)) and (not isinstance(kwargs['patience'], np.int32)) and \
                (not isinstance(kwargs['patience'], np.uint32)):
            raise ValueError('`patience` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3), type(kwargs['patience'])))
        if kwargs['patience'] < 1:
            raise ValueError('`patience` is wrong! Expected a positive integer value, '
                             'but {0} is not positive.'.format(kwargs['patience']))
        if 'random_seed' not in kwargs:
            raise ValueError('`random_seed` is not specified!')
        if kwargs['random_seed'] is not None:
            if (not isinstance(kwargs['random_seed'], int)) and (not isinstance(kwargs['random_seed'], np.int32)) and \
                    (not isinstance(kwargs['random_seed'], np.uint32)):
                raise ValueError('`random_seed` is wrong! Expected `{0}`, got `{1}`.'.format(
                    type(3), type(kwargs['random_seed'])))
        if 'gpu_memory_frac' not in kwargs:
            raise ValueError('`gpu_memory_frac` is not specified!')
        if (not isinstance(kwargs['gpu_memory_frac'], float)) and \
                (not isinstance(kwargs['gpu_memory_frac'], np.float32)) and \
                (not isinstance(kwargs['gpu_memory_frac'], np.float64)):
            raise ValueError('`gpu_memory_frac` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3.5), type(kwargs['gpu_memory_frac'])))
        if (kwargs['gpu_memory_frac'] <= 0.0) or (kwargs['gpu_memory_frac'] > 1.0):
            raise ValueError('`gpu_memory_frac` is wrong! Expected a floating-point value in the (0.0, 1.0], '
                             'but {0} is not proper.'.format(kwargs['gpu_memory_frac']))
        if 'max_seq_length' not in kwargs:
            raise ValueError('`max_seq_length` is not specified!')
        if (not isinstance(kwargs['max_seq_length'], int)) and \
                (not isinstance(kwargs['max_seq_length'], np.int32)) and \
                (not isinstance(kwargs['max_seq_length'], np.uint32)):
            raise ValueError('`max_seq_length` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3), type(kwargs['max_seq_length'])))
        if kwargs['max_seq_length'] < 1:
            raise ValueError('`max_seq_length` is wrong! Expected a positive integer value, '
                             'but {0} is not positive.'.format(kwargs['max_seq_length']))
        if 'verbose' not in kwargs:
            raise ValueError('`verbose` is not specified!')
        if (not isinstance(kwargs['verbose'], int)) and (not isinstance(kwargs['verbose'], np.int32)) and \
                (not isinstance(kwargs['verbose'], np.uint32)) and \
                (not isinstance(kwargs['verbose'], bool)) and (not isinstance(kwargs['verbose'], np.bool)):
            raise ValueError('`verbose` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(True), type(kwargs['verbose'])))

    def check_X(self, X: Union[list, tuple, np.array], X_name: str) -> int:
        if (not hasattr(X, '__len__')) or (not hasattr(X, '__getitem__')):
            raise ValueError('`{0}` is wrong, because it is not list-like object!'.format(X_name))
        n = X.shape[0] if isinstance(X, np.ndarray) else len(X)
        for idx in range(n):
            if not self.check_X_item(X[idx]):
                raise ValueError('Item {0} of `{1}` is wrong!'.format(idx, X_name))
        return n

    def check_X_item(self, item) -> bool:
        if not isinstance(item, np.ndarray):
            return False
        if len(item.shape) != 2:
            return False
        return (item.shape[0] > 0) and (item.shape[1] > 0)

    def check_Xy(self, X: Union[list, tuple, np.array], X_name: str,
                 y: Union[list, tuple, np.array], y_name: str) -> int:
        n = self.check_X(X, X_name)
        if (not hasattr(y, '__len__')) or (not hasattr(y, '__getitem__')):
            raise ValueError('`{0}` is wrong, because it is not a list-like object!'.format(y_name))
        if isinstance(y, np.ndarray):
            if (len(y.shape) != 1) and (len(y.shape) != 2):
                raise ValueError('`{0}` is wrong, because it is neither 1-D list nor 2-D one!'.format(y_name))
            n_y = y.shape[0]
        else:
            n_y = len(y)
        if n != n_y:
            raise ValueError('Length of `{0}` does not correspond to length of `{1}`! {2} != {3}'.format(
                X_name, y_name, n, n_y))
        classes_list = set()
        if isinstance(y, np.ndarray) and (len(y.shape) == 2):
            for idx in range(n):
                min_value = y[idx].min()
                max_value = y[idx].max()
                if min_value < 0.0:
                    raise ValueError('Item {0} of `{1}` is wrong, because some class probability is less '
                                     'than 0.0!'.format(idx, y_name))
                if max_value > 1.0:
                    raise ValueError('Item {0} of `{1}` is wrong, because some class probability is greater '
                                     'than 1.0!'.format(idx, y_name))
                if (max_value - min_value) <= 1e-5:
                    raise ValueError('Item {0} of `{1}` is wrong, because all class probabilities are '
                                     'same!'.format(idx, y_name))
                classes_list.add(int(y[idx].argmax()))
            if len(classes_list) != y.shape[1]:
                raise ValueError('All class labels must be in the interval from 0 to number of classes '
                                 '(not include this value)!')
        else:
            for idx in range(n):
                if (not isinstance(y[idx], int)) and (not isinstance(y[idx], np.int32)) and \
                        (not isinstance(y[idx], np.uint32)) and (not isinstance(y[idx], np.int64)) and \
                        (not isinstance(y[idx], np.uint64)) and (not isinstance(y[idx], np.int8)) and \
                        (not isinstance(y[idx], np.uint8)) and (not isinstance(y[idx], np.int16)) and \
                        (not isinstance(y[idx], np.uint16)) and (not isinstance(y[idx], set)):
                    raise ValueError('Item {0} of `{1}` is wrong, because it is neither integer object nor set!'.format(
                        idx, y_name
                    ))
                if isinstance(y[idx], set):
                    for class_idx in y[idx]:
                        if (not isinstance(class_idx, int)) and (not isinstance(class_idx, np.int32)) and \
                                (not isinstance(class_idx, np.uint32)) and (not isinstance(class_idx, np.int64)) and \
                                (not isinstance(class_idx, np.uint64)) and (not isinstance(class_idx, np.int8)) and \
                                (not isinstance(class_idx, np.uint8)) and (not isinstance(class_idx, np.int16)) and \
                                (not isinstance(class_idx, np.uint16)):
                            raise ValueError('Item {0} of `{1}` is wrong, because it is not set of integers!'.format(
                                idx, y_name
                            ))
                        if class_idx < 0:
                            raise ValueError('Item {0} of `{1}` is wrong, because it is not set of non-negative '
                                             'integers!'.format(idx, y_name))
                    classes_list |= set(map(
                        lambda class_idx: class_idx if isinstance(class_idx, int) else int(class_idx), y[idx]
                    ))
                else:
                    if y[idx] < 0:
                        raise ValueError('Item {0} of `{1}` is wrong, because it\'s value is negative!'.format(
                            idx, y_name
                        ))
                    classes_list.add(y[idx] if isinstance(y[idx], int) else int(y[idx]))
        classes_list = sorted(list(classes_list))
        if (classes_list[0] != 0) or (classes_list[-1] != (len(classes_list) - 1)):
            raise ValueError('All class labels must be in the interval from 0 to number of classes '
                             '(not include this value)!')
        return len(classes_list)
