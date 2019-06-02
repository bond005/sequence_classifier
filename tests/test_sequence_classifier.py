import codecs
import gc
import os
import pickle
import random
import re
import requests
import sys
import tempfile
from typing import Dict, Tuple
import unittest
import zipfile

from nltk.tokenize.nist import NISTTokenizer
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import StratifiedShuffleSplit

try:
    from sequence_classifier.sequence_classifier import SequenceClassifier
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from sequence_classifier.sequence_classifier import SequenceClassifier


class TestSequenceClassifier(unittest.TestCase):
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    @classmethod
    def setUpClass(cls):
        cls.load_data()

    @classmethod
    def tearDownClass(cls):
        del cls.X_train
        del cls.y_train
        del cls.X_test
        del cls.y_test

    def tearDown(self):
        if hasattr(self, 'cls'):
            del self.cls
        if hasattr(self, 'another_cls'):
            del self.another_cls
        if hasattr(self, 'temp_file_name'):
            if os.path.isfile(self.temp_file_name):
                os.remove(self.temp_file_name)

    def test_creation(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50)
        self.assertIsInstance(self.cls, SequenceClassifier)
        self.assertTrue(hasattr(self.cls, 'batch_size'))
        self.assertTrue(hasattr(self.cls, 'num_recurrent_units'))
        self.assertTrue(hasattr(self.cls, 'learning_rate'))
        self.assertTrue(hasattr(self.cls, 'l2_reg'))
        self.assertTrue(hasattr(self.cls, 'clipnorm'))
        self.assertTrue(hasattr(self.cls, 'multioutput'))
        self.assertTrue(hasattr(self.cls, 'warm_start'))
        self.assertTrue(hasattr(self.cls, 'max_epochs'))
        self.assertTrue(hasattr(self.cls, 'patience'))
        self.assertTrue(hasattr(self.cls, 'random_seed'))
        self.assertTrue(hasattr(self.cls, 'gpu_memory_frac'))
        self.assertTrue(hasattr(self.cls, 'max_seq_length'))
        self.assertTrue(hasattr(self.cls, 'verbose'))
        self.assertIsInstance(self.cls.batch_size, int)
        self.assertIsInstance(self.cls.num_recurrent_units, tuple)
        self.assertIsInstance(self.cls.learning_rate, float)
        self.assertIsInstance(self.cls.l2_reg, float)
        self.assertIsInstance(self.cls.clipnorm, float)
        self.assertIsInstance(self.cls.multioutput, bool)
        self.assertIsInstance(self.cls.warm_start, bool)
        self.assertIsInstance(self.cls.max_epochs, int)
        self.assertIsInstance(self.cls.patience, int)
        self.assertIsNone(self.cls.random_seed)
        self.assertIsInstance(self.cls.gpu_memory_frac, float)
        self.assertIsInstance(self.cls.max_seq_length, int)
        self.assertIsInstance(self.cls.verbose, bool)
        self.assertEqual(self.cls.num_recurrent_units, (10, 3))
        self.assertEqual(self.cls.max_seq_length, 50)

    def test_check_params_positive(self):
        SequenceClassifier.check_params(
            warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length=100,
            learning_rate=1e-3, l2_reg=1e-5, max_epochs=1000, patience=5, verbose=True, gpu_memory_frac=0.9,
            clipnorm=5.0, random_seed=3
        )
        self.assertTrue(True)

    def test_check_params_negative001(self):
        true_err_msg = re.escape('`batch_size` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], max_seq_length=100,
                learning_rate=1e-3, l2_reg=1e-5, max_epochs=1000, patience=5, verbose=True, gpu_memory_frac=0.9,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative002(self):
        true_err_msg = re.escape('`batch_size` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size='3', max_seq_length=100,
                learning_rate=1e-3, l2_reg=1e-5, max_epochs=1000, patience=5, verbose=True, gpu_memory_frac=0.9,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative003(self):
        true_err_msg = re.escape('`batch_size` is wrong! Expected a positive integer value, but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size=-3, max_seq_length=100,
                learning_rate=1e-3, l2_reg=1e-5, max_epochs=1000, patience=5, verbose=True, gpu_memory_frac=0.9,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative004(self):
        true_err_msg = re.escape('`max_epochs` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length=100,
                learning_rate=1e-3, l2_reg=1e-5, patience=5, verbose=True, gpu_memory_frac=0.9,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative005(self):
        true_err_msg = re.escape('`max_epochs` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length=100,
                learning_rate=1e-3, l2_reg=1e-5, max_epochs='3', patience=5, verbose=True, gpu_memory_frac=0.9,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative006(self):
        true_err_msg = re.escape('`max_epochs` is wrong! Expected a positive integer value, but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length=100,
                learning_rate=1e-3, l2_reg=1e-5, max_epochs=-3, patience=5, verbose=True, gpu_memory_frac=0.9,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative007(self):
        true_err_msg = re.escape('`patience` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length=100,
                learning_rate=1e-3, l2_reg=1e-5, max_epochs=1000, verbose=True, gpu_memory_frac=0.9,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative008(self):
        true_err_msg = re.escape('`patience` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length=100,
                learning_rate=1e-3, l2_reg=1e-5, max_epochs=1000, patience='3', verbose=True, gpu_memory_frac=0.9,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative009(self):
        true_err_msg = re.escape('`patience` is wrong! Expected a positive integer value, but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length=100,
                learning_rate=1e-3, l2_reg=1e-5, max_epochs=1000, patience=-3, verbose=True, gpu_memory_frac=0.9,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative010(self):
        true_err_msg = re.escape('`max_seq_length` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size=32,
                learning_rate=1e-3, l2_reg=1e-5, max_epochs=1000, patience=5, verbose=True, gpu_memory_frac=0.9,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative011(self):
        true_err_msg = re.escape('`max_seq_length` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length='3',
                learning_rate=1e-3, l2_reg=1e-5, max_epochs=1000, patience=5, verbose=True, gpu_memory_frac=0.9,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative012(self):
        true_err_msg = re.escape('`max_seq_length` is wrong! Expected a positive integer value, but -3 is not '
                                 'positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length=-3,
                learning_rate=1e-3, l2_reg=1e-5, max_epochs=1000, patience=5, verbose=True, gpu_memory_frac=0.9,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative013(self):
        true_err_msg = re.escape('`gpu_memory_frac` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length=100,
                learning_rate=1e-3, l2_reg=1e-5, max_epochs=1000, patience=5, verbose=True,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative014(self):
        true_err_msg = re.escape('`gpu_memory_frac` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3.5), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length=100,
                learning_rate=1e-3, l2_reg=1e-5, max_epochs=1000, patience=5, verbose=True, gpu_memory_frac='0.9',
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative015(self):
        true_err_msg = re.escape('`gpu_memory_frac` is wrong! Expected a floating-point value in the (0.0, 1.0], '
                                 'but {0} is not proper.'.format(-1.0))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length=100,
                learning_rate=1e-3, l2_reg=1e-5, max_epochs=1000, patience=5, verbose=True, gpu_memory_frac=-1.0,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative016(self):
        true_err_msg = re.escape('`gpu_memory_frac` is wrong! Expected a floating-point value in the (0.0, 1.0], '
                                 'but {0} is not proper.'.format(1.3))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length=100,
                learning_rate=1e-3, l2_reg=1e-5, max_epochs=1000, patience=5, verbose=True, gpu_memory_frac=1.3,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative017(self):
        true_err_msg = re.escape('`learning_rate` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length=100,
                l2_reg=1e-5, max_epochs=1000, patience=5, verbose=True, gpu_memory_frac=0.9,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative018(self):
        true_err_msg = re.escape('`learning_rate` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3.5), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length=100,
                learning_rate='1e-3', l2_reg=1e-5, max_epochs=1000, patience=5, verbose=True, gpu_memory_frac=0.9,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative019(self):
        true_err_msg = re.escape('`learning_rate` is wrong! Expected a positive floating-point value, but {0} is not '
                                 'positive.'.format(0.0))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length=100,
                learning_rate=0.0, l2_reg=1e-5, max_epochs=1000, patience=5, verbose=True, gpu_memory_frac=0.9,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative020(self):
        true_err_msg = re.escape('`l2_reg` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length=100,
                learning_rate=1e-3, max_epochs=1000, patience=5, verbose=True, gpu_memory_frac=0.9,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative021(self):
        true_err_msg = re.escape('`l2_reg` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3.5), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length=100,
                learning_rate=1e-3, l2_reg='1e-5', max_epochs=1000, patience=5, verbose=True, gpu_memory_frac=0.9,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative022(self):
        true_err_msg = re.escape('`l2_reg` is wrong! Expected a non-negative floating-point value, but {0} is '
                                 'negative.'.format(-2.0))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length=100,
                learning_rate=1e-3, l2_reg=-2.0, max_epochs=1000, patience=5, verbose=True, gpu_memory_frac=0.9,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative023(self):
        true_err_msg = re.escape('`warm_start` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                multioutput=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length=100,
                learning_rate=1e-3, l2_reg=1e-5, max_epochs=1000, patience=5, verbose=True, gpu_memory_frac=0.9,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative024(self):
        true_err_msg = re.escape('`warm_start` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(True), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start='True', multioutput=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length=100,
                learning_rate=1e-3, l2_reg=1e-5, max_epochs=1000, patience=5, verbose=True, gpu_memory_frac=0.9,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative025(self):
        true_err_msg = re.escape('`multioutput` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length=100,
                learning_rate=1e-3, l2_reg=1e-5, max_epochs=1000, patience=5, verbose=True, gpu_memory_frac=0.9,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative026(self):
        true_err_msg = re.escape('`multioutput` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(True), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput='False', num_recurrent_units=[10, 3], batch_size=32, max_seq_length=100,
                learning_rate=1e-3, l2_reg=1e-5, max_epochs=1000, patience=5, verbose=True, gpu_memory_frac=0.9,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative027(self):
        true_err_msg = re.escape('`verbose` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length=100,
                learning_rate=1e-3, l2_reg=1e-5, max_epochs=1000, patience=5, gpu_memory_frac=0.9,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative028(self):
        true_err_msg = re.escape('`verbose` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(True), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length=100,
                learning_rate=1e-3, l2_reg=1e-5, max_epochs=1000, patience=5, verbose='True', gpu_memory_frac=0.9,
                clipnorm=5.0, random_seed=3
            )

    def test_check_params_negative029(self):
        true_err_msg = re.escape('`clipnorm` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length=100,
                l2_reg=1e-5, max_epochs=1000, patience=5, verbose=True, gpu_memory_frac=0.9,
                learning_rate=1e-3, random_seed=3
            )

    def test_check_params_negative030(self):
        true_err_msg = re.escape('`clipnorm` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3.5), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length=100,
                learning_rate=1e-3, l2_reg=1e-5, max_epochs=1000, patience=5, verbose=True, gpu_memory_frac=0.9,
                clipnorm='5.0', random_seed=3
            )

    def test_check_params_negative031(self):
        true_err_msg = re.escape('`clipnorm` is wrong! Expected a positive floating-point value, but {0} is not '
                                 'positive.'.format(0.0))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, num_recurrent_units=[10, 3], batch_size=32, max_seq_length=100,
                learning_rate=1e-3, l2_reg=1e-5, max_epochs=1000, patience=5, verbose=True, gpu_memory_frac=0.9,
                clipnorm=0.0, random_seed=3
            )

    def test_check_params_negative032(self):
        true_err_msg = re.escape('`num_recurrent_units` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                warm_start=True, multioutput=False, batch_size=32, max_seq_length=100, l2_reg=1e-5, max_epochs=1000,
                patience=5, verbose=True, gpu_memory_frac=0.9, clipnorm=5.0, learning_rate=1e-3, random_seed=3
            )

    def test_check_params_negative033(self):
        true_err_msg = re.escape('`num_recurrent_units` is wrong! Expected `{0}`, got `{1}`.'.format(
            type([3, 4]), type({3, 4})))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                num_recurrent_units={10, 3}, warm_start=True, multioutput=False, batch_size=32, max_seq_length=100,
                l2_reg=1e-5, max_epochs=1000, patience=5, verbose=True, gpu_memory_frac=0.9, clipnorm=5.0,
                learning_rate=1e-3, random_seed=3
            )

    def test_check_params_negative034(self):
        true_err_msg = re.escape('`num_recurrent_units` is wrong! Expected a 1-D array, got a 2-D one.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                num_recurrent_units=np.array([[10, 3], [20, 6]], dtype=np.int32), warm_start=True, multioutput=False,
                batch_size=32, max_seq_length=100, l2_reg=1e-5, max_epochs=1000, patience=5, verbose=True,
                gpu_memory_frac=0.9, clipnorm=5.0, learning_rate=1e-3, random_seed=3
            )

    def test_check_params_negative035(self):
        true_err_msg = re.escape('`num_recurrent_units` is wrong! Expected a nonempty list.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                num_recurrent_units=[], warm_start=True, multioutput=False,
                batch_size=32, max_seq_length=100, l2_reg=1e-5, max_epochs=1000, patience=5, verbose=True,
                gpu_memory_frac=0.9, clipnorm=5.0, learning_rate=1e-3, random_seed=3
            )

    def test_check_params_negative036(self):
        true_err_msg = re.escape('Item 1 of `num_recurrent_units` is wrong! Expected a positive integer value, but -3 '
                                 'is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            SequenceClassifier.check_params(
                num_recurrent_units=(10, -3), warm_start=True, multioutput=False,
                batch_size=32, max_seq_length=100, l2_reg=1e-5, max_epochs=1000, patience=5, verbose=True,
                gpu_memory_frac=0.9, clipnorm=5.0, learning_rate=1e-3, random_seed=3
            )

    def test_check_X_positive01(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50)
        X = np.random.uniform(-1.0, 1.0, (10, 50, 7))
        true_size = 10
        predicted_size = self.cls.check_X(X, 'X')
        self.assertEqual(true_size, predicted_size)

    def test_check_X_positive02(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50)
        X = np.array([np.random.uniform(-1.0, 1.0, (random.randint(30, 60), 7)) for _ in range(15)], dtype=object)
        true_size = 15
        predicted_size = self.cls.check_X(X, 'X')
        self.assertEqual(true_size, predicted_size)

    def test_check_X_positive03(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50)
        X = [np.random.uniform(-1.0, 1.0, (random.randint(30, 60), 7)) for _ in range(17)]
        true_size = 17
        predicted_size = self.cls.check_X(X, 'X')
        self.assertEqual(true_size, predicted_size)

    def test_check_X_negative01(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50)
        true_err_msg = re.escape('`X_train` is wrong, because it is not list-like object!')
        X = set([
            tuple(map(
                lambda it: tuple(it), np.random.uniform(-1.0, 1.0, (random.randint(30, 60), 7)).tolist()
            )) for _ in range(17)
        ])
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.cls.check_X(X, 'X_train')

    def test_check_X_negative02(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50)
        X = [np.random.uniform(-1.0, 1.0, (random.randint(30, 60), 7)) for _ in range(3)]
        X.append(np.random.uniform(-1.0, 1.0, (random.randint(30, 60), 7)).tolist())
        X += [np.random.uniform(-1.0, 1.0, (random.randint(30, 60), 7)) for _ in range(10)]
        true_err_msg = re.escape('Item 3 of `X_val` is wrong!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.cls.check_X(X, 'X_val')

    def test_check_Xy_positive01(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50)
        X = np.random.uniform(-1.0, 1.0, (100, 50, 7))
        y = np.concatenate(
            (np.zeros((40,), dtype=np.int32), np.full((35,), 1, dtype=np.int32), np.full((25,), 2, dtype=np.int32))
        )
        true_number_of_classes = 3
        number_of_classes = self.cls.check_Xy(X, 'X_train', y, 'y_train')
        self.assertEqual(true_number_of_classes, number_of_classes)

    def test_check_Xy_positive02(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50)
        X = np.random.uniform(-1.0, 1.0, (100, 50, 7))
        y = np.concatenate(
            (np.zeros((40,), dtype=np.int32), np.full((35,), 1, dtype=np.int32), np.full((25,), 2, dtype=np.int32))
        ).tolist()
        true_number_of_classes = 3
        number_of_classes = self.cls.check_Xy(X, 'X_train', y, 'y_train')
        self.assertEqual(true_number_of_classes, number_of_classes)

    def test_check_Xy_positive03(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50)
        X = np.random.uniform(-1.0, 1.0, (100, 50, 7))
        y = np.concatenate(
            (np.zeros((40,), dtype=np.int32), np.full((25,), 1, dtype=np.int32), np.full((15,), 2, dtype=np.int32))
        ).tolist()
        y += ([{0, 2} for _ in range(7)] + [{1, 2} for _ in range(13)])
        true_number_of_classes = 3
        number_of_classes = self.cls.check_Xy(X, 'X_train', y, 'y_train')
        self.assertEqual(true_number_of_classes, number_of_classes)

    def test_check_Xy_positive04(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50)
        X = np.random.uniform(-1.0, 1.0, (100, 50, 7))
        indices = [0 for _ in range(40)] + [1 for _ in range(25)] + [2 for _ in range(15)]
        indices += [{0, 2} for _ in range(7)]
        indices += [{1, 2} for _ in range(13)]
        true_number_of_classes = 3
        y = np.zeros((len(indices), true_number_of_classes), dtype=np.float32)
        for sample_idx in range(len(indices)):
            if isinstance(indices[sample_idx], set):
                set_of_classes = indices[sample_idx]
            else:
                set_of_classes = {indices[sample_idx]}
            for class_idx in set_of_classes:
                y[sample_idx][class_idx] = 1.0
        number_of_classes = self.cls.check_Xy(X, 'X_train', y, 'y_train')
        self.assertEqual(true_number_of_classes, number_of_classes)

    def test_check_Xy_negative01(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50)
        X = np.random.uniform(-1.0, 1.0, (100, 50, 7))
        y = 3.4
        true_err_msg = re.escape('`y_train` is wrong, because it is not a list-like object!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.cls.check_Xy(X, 'X_train', y, 'y_train')

    def test_check_Xy_negative02(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50)
        X = np.random.uniform(-1.0, 1.0, (100, 50, 7))
        y = np.random.uniform(0.0, 1.0, (100, 3, 2))
        true_err_msg = re.escape('`y_val` is wrong, because it is neither 1-D list nor 2-D one!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.cls.check_Xy(X, 'X_val', y, 'y_val')

    def test_check_Xy_negative03(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50)
        X = np.random.uniform(-1.0, 1.0, (100, 50, 7))
        y = np.concatenate(
            (np.zeros((40,), dtype=np.int32), np.full((35,), 1, dtype=np.int32), np.full((24,), 2, dtype=np.int32))
        )
        true_err_msg = re.escape('Length of `X_val` does not correspond to length of `y_val`! 100 != 99')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.cls.check_Xy(X, 'X_val', y, 'y_val')

    def test_check_Xy_negative04(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50)
        X = np.random.uniform(-1.0, 1.0, (100, 50, 7))
        indices = [0 for _ in range(40)] + [1 for _ in range(25)] + [2 for _ in range(15)]
        indices += [{0, 2} for _ in range(7)]
        indices += [{1, 2} for _ in range(13)]
        true_number_of_classes = 3
        y = np.zeros((len(indices), true_number_of_classes), dtype=np.float32)
        for sample_idx in range(len(indices)):
            if isinstance(indices[sample_idx], set):
                set_of_classes = indices[sample_idx]
            else:
                set_of_classes = {indices[sample_idx]}
            for class_idx in set_of_classes:
                y[sample_idx][class_idx] = 1.0
        y[2][1] = -0.5
        true_err_msg = re.escape('Item 2 of `y` is wrong, because some class probability is less than 0.0!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.cls.check_Xy(X, 'X', y, 'y')

    def test_check_Xy_negative05(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50)
        X = np.random.uniform(-1.0, 1.0, (100, 50, 7))
        indices = [0 for _ in range(40)] + [1 for _ in range(25)] + [2 for _ in range(15)]
        indices += [{0, 2} for _ in range(7)]
        indices += [{1, 2} for _ in range(13)]
        true_number_of_classes = 3
        y = np.zeros((len(indices), true_number_of_classes), dtype=np.float32)
        for sample_idx in range(len(indices)):
            if isinstance(indices[sample_idx], set):
                set_of_classes = indices[sample_idx]
            else:
                set_of_classes = {indices[sample_idx]}
            for class_idx in set_of_classes:
                y[sample_idx][class_idx] = 1.0
        y[3][2] = 1.1
        true_err_msg = re.escape('Item 3 of `y` is wrong, because some class probability is greater than 1.0!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.cls.check_Xy(X, 'X', y, 'y')

    def test_check_Xy_negative06(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50)
        X = np.random.uniform(-1.0, 1.0, (100, 50, 7))
        indices = [0 for _ in range(40)] + [1 for _ in range(25)] + [2 for _ in range(15)]
        indices += [{0, 2} for _ in range(7)]
        indices += [{1, 2} for _ in range(13)]
        true_number_of_classes = 3
        y = np.zeros((len(indices), true_number_of_classes), dtype=np.float32)
        for sample_idx in range(len(indices)):
            if isinstance(indices[sample_idx], set):
                set_of_classes = indices[sample_idx]
            else:
                set_of_classes = {indices[sample_idx]}
            for class_idx in set_of_classes:
                y[sample_idx][class_idx] = 1.0
        y[4][0] = 0.5
        y[4][1] = 0.5
        y[4][2] = 0.5
        true_err_msg = re.escape('Item 4 of `y` is wrong, because all class probabilities are same!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.cls.check_Xy(X, 'X', y, 'y')

    def test_check_Xy_negative07(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50)
        X = np.random.uniform(-1.0, 1.0, (100, 50, 7))
        indices = [0 for _ in range(40)] + [1 for _ in range(40)]
        indices += [{0, 1} for _ in range(20)]
        y = np.zeros((len(indices), 3), dtype=np.float32)
        for sample_idx in range(len(indices)):
            if isinstance(indices[sample_idx], set):
                set_of_classes = indices[sample_idx]
            else:
                set_of_classes = {indices[sample_idx]}
            for class_idx in set_of_classes:
                y[sample_idx][class_idx] = 1.0
        true_err_msg = re.escape('All class labels must be in the interval from 0 to number of classes '
                                 '(not include this value)!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.cls.check_Xy(X, 'X', y, 'y')

    def test_check_Xy_negative08(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50)
        X = np.random.uniform(-1.0, 1.0, (100, 50, 7))
        y = [0 for _ in range(40)] + [1 for _ in range(35)] + [2 for _ in range(25)]
        y[7] = 0.3
        true_err_msg = re.escape('Item 7 of `y_train` is wrong, because it is neither integer object nor set!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.cls.check_Xy(X, 'X_train', y, 'y_train')

    def test_check_Xy_negative09(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50)
        X = np.random.uniform(-1.0, 1.0, (100, 50, 7))
        y = [0 for _ in range(40)] + [1 for _ in range(35)] + [2 for _ in range(25)]
        y[19] = {'0', 1}
        true_err_msg = re.escape('Item 19 of `y_train` is wrong, because it is not set of integers!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.cls.check_Xy(X, 'X_train', y, 'y_train')

    def test_check_Xy_negative10(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50)
        X = np.random.uniform(-1.0, 1.0, (100, 50, 7))
        y = [0 for _ in range(40)] + [1 for _ in range(35)] + [2 for _ in range(25)]
        y[7] = -2
        true_err_msg = re.escape('Item 7 of `y_train` is wrong, because it\'s value is negative!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.cls.check_Xy(X, 'X_train', y, 'y_train')

    def test_check_Xy_negative11(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50)
        X = np.random.uniform(-1.0, 1.0, (100, 50, 7))
        y = [0 for _ in range(40)] + [1 for _ in range(35)] + [2 for _ in range(25)]
        y[19] = {-1, 1}
        true_err_msg = re.escape('Item 19 of `y_val` is wrong, because it is not set of non-negative integers!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.cls.check_Xy(X, 'X_val', y, 'y_val')

    def test_check_Xy_negative12(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50)
        X = np.random.uniform(-1.0, 1.0, (100, 50, 7))
        y = [0 for _ in range(40)] + [1 for _ in range(35)] + [2 for _ in range(25)]
        y[7] = 5
        true_err_msg = re.escape('All class labels must be in the interval from 0 to number of classes '
                                 '(not include this value)!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.cls.check_Xy(X, 'X_train', y, 'y_train')

    def test_fit_predict01(self):
        n_classes = len(set(self.y_train.tolist()))
        structure_of_recurrent_layers = (2 * n_classes, (3 * n_classes) // 2)
        self.cls = SequenceClassifier(num_recurrent_units=structure_of_recurrent_layers, gpu_memory_frac=0.8,
                                      max_seq_length=self.X_train.shape[1], multioutput=False, batch_size=16,
                                      verbose=True)
        res = self.cls.fit(X=self.X_train, y=self.y_train, validation_data=(self.X_test, self.y_test))
        self.assertIsInstance(res, SequenceClassifier)
        self.assertTrue(hasattr(res, 'n_classes_'))
        self.assertTrue(hasattr(res, 'rnn_output_'))
        self.assertTrue(hasattr(res, 'logits_'))
        self.assertTrue(hasattr(res, 'input_data_'))
        self.assertTrue(hasattr(res, 'input_shape_'))
        self.assertTrue(hasattr(res, 'layers_'))
        self.assertTrue(hasattr(res, 'y_ph_'))
        self.assertTrue(hasattr(res, 'sess_'))
        self.assertIsInstance(res.n_classes_, int)
        self.assertIsInstance(res.input_shape_, tuple)
        self.assertIsInstance(res.layers_, tuple)
        self.assertEqual(res.n_classes_, n_classes)
        self.assertEqual(res.input_shape_, self.X_train.shape[1:])
        self.assertEqual(res.layers_, structure_of_recurrent_layers)
        score = res.score(self.X_test, self.y_test)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
        probabilities = res.predict_proba(self.X_test)
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(probabilities.shape, (self.X_test.shape[0], n_classes))
        for sample_idx in range(probabilities.shape[0]):
            for class_idx in range(probabilities.shape[1]):
                self.assertGreaterEqual(probabilities[sample_idx][class_idx], 0.0)
                self.assertLessEqual(probabilities[sample_idx][class_idx], 1.0)
        y_pred = res.predict(self.X_test)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(y_pred.shape, (self.X_test.shape[0],))
        self.assertEqual(y_pred.tolist(), probabilities.argmax(axis=1).tolist())
        log_probabilities = res.predict_log_proba(self.X_test)
        self.assertIsInstance(log_probabilities, np.ndarray)
        self.assertEqual(log_probabilities.shape, (self.X_test.shape[0], n_classes))
        for sample_idx in range(probabilities.shape[0]):
            for class_idx in range(probabilities.shape[1]):
                self.assertAlmostEqual(np.log(probabilities[sample_idx][class_idx] + 1e-5),
                                       log_probabilities[sample_idx][class_idx], places=4)

    def test_fit_predict02(self):
        n_classes = len(set(self.y_train.tolist()))
        structure_of_recurrent_layers = (2 * n_classes, (3 * n_classes) // 2)
        self.cls = SequenceClassifier(num_recurrent_units=structure_of_recurrent_layers, gpu_memory_frac=0.8,
                                      max_seq_length=self.X_train.shape[1], multioutput=True, batch_size=16,
                                      verbose=True)
        y_train = [{class_idx} for class_idx in self.y_train]
        for sample_idx in range(len(y_train)):
            if random.random() > 0.8:
                y_train[sample_idx].add(random.randint(0, n_classes - 1))
        y_test = [{class_idx} for class_idx in self.y_test]
        for sample_idx in range(len(y_test)):
            if random.random() > 0.8:
                y_test[sample_idx].add(random.randint(0, n_classes - 1))
        res = self.cls.fit(X=self.X_train, y=y_train, validation_data=(self.X_test, y_test))
        self.assertIsInstance(res, SequenceClassifier)
        self.assertTrue(hasattr(res, 'n_classes_'))
        self.assertTrue(hasattr(res, 'rnn_output_'))
        self.assertTrue(hasattr(res, 'logits_'))
        self.assertTrue(hasattr(res, 'input_data_'))
        self.assertTrue(hasattr(res, 'input_shape_'))
        self.assertTrue(hasattr(res, 'layers_'))
        self.assertTrue(hasattr(res, 'y_ph_'))
        self.assertTrue(hasattr(res, 'sess_'))
        self.assertIsInstance(res.n_classes_, int)
        self.assertIsInstance(res.input_shape_, tuple)
        self.assertIsInstance(res.layers_, tuple)
        self.assertEqual(res.n_classes_, n_classes)
        self.assertEqual(res.input_shape_, self.X_train.shape[1:])
        self.assertEqual(res.layers_, structure_of_recurrent_layers)
        score = res.score(self.X_test, y_test)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
        probabilities = res.predict_proba(self.X_test)
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(probabilities.shape, (self.X_test.shape[0], n_classes))
        for sample_idx in range(probabilities.shape[0]):
            for class_idx in range(probabilities.shape[1]):
                self.assertGreaterEqual(probabilities[sample_idx][class_idx], 0.0)
                self.assertLessEqual(probabilities[sample_idx][class_idx], 1.0)
        y_pred = res.predict(self.X_test)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(y_pred.shape, (self.X_test.shape[0],))
        self.assertEqual(y_pred.tolist(), probabilities.argmax(axis=1).tolist())
        log_probabilities = res.predict_log_proba(self.X_test)
        self.assertIsInstance(log_probabilities, np.ndarray)
        self.assertEqual(log_probabilities.shape, (self.X_test.shape[0], n_classes))
        for sample_idx in range(probabilities.shape[0]):
            for class_idx in range(probabilities.shape[1]):
                self.assertAlmostEqual(np.log(probabilities[sample_idx][class_idx] + 1e-5),
                                       log_probabilities[sample_idx][class_idx], places=4)

    def test_fit_predict03(self):
        n_classes = len(set(self.y_train.tolist()))
        structure_of_recurrent_layers = (2 * n_classes, (3 * n_classes) // 2)
        self.cls = SequenceClassifier(num_recurrent_units=structure_of_recurrent_layers, batch_size=16,
                                      max_seq_length=self.X_train.shape[1], multioutput=False, max_epochs=5,
                                      gpu_memory_frac=0.8, verbose=True)
        res = self.cls.fit(X=self.X_train, y=self.y_train)
        self.assertIsInstance(res, SequenceClassifier)
        self.assertTrue(hasattr(res, 'n_classes_'))
        self.assertTrue(hasattr(res, 'rnn_output_'))
        self.assertTrue(hasattr(res, 'logits_'))
        self.assertTrue(hasattr(res, 'input_data_'))
        self.assertTrue(hasattr(res, 'input_shape_'))
        self.assertTrue(hasattr(res, 'layers_'))
        self.assertTrue(hasattr(res, 'y_ph_'))
        self.assertTrue(hasattr(res, 'sess_'))
        self.assertIsInstance(res.n_classes_, int)
        self.assertIsInstance(res.input_shape_, tuple)
        self.assertIsInstance(res.layers_, tuple)
        self.assertEqual(res.n_classes_, n_classes)
        self.assertEqual(res.input_shape_, self.X_train.shape[1:])
        self.assertEqual(res.layers_, structure_of_recurrent_layers)
        score = res.score(self.X_test, self.y_test)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
        probabilities = res.predict_proba(self.X_test)
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(probabilities.shape, (self.X_test.shape[0], n_classes))
        for sample_idx in range(probabilities.shape[0]):
            for class_idx in range(probabilities.shape[1]):
                self.assertGreaterEqual(probabilities[sample_idx][class_idx], 0.0)
                self.assertLessEqual(probabilities[sample_idx][class_idx], 1.0)
        y_pred = res.predict(self.X_test)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(y_pred.shape, (self.X_test.shape[0],))
        self.assertEqual(y_pred.tolist(), probabilities.argmax(axis=1).tolist())
        log_probabilities = res.predict_log_proba(self.X_test)
        self.assertIsInstance(log_probabilities, np.ndarray)
        self.assertEqual(log_probabilities.shape, (self.X_test.shape[0], n_classes))
        for sample_idx in range(probabilities.shape[0]):
            for class_idx in range(probabilities.shape[1]):
                self.assertAlmostEqual(np.log(probabilities[sample_idx][class_idx] + 1e-5),
                                       log_probabilities[sample_idx][class_idx], places=4)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
        train_indices, test_indices = next(sss.split(self.X_test, self.y_test))
        res.warm_start = True
        res.fit(X=self.X_test[train_indices], y=self.y_test[train_indices],
                validation_data=(self.X_test[test_indices], self.y_test[test_indices]))
        score = res.score(self.X_test[test_indices], self.y_test[test_indices])
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_fit_negative01(self):
        n_classes = len(set(self.y_train.tolist()))
        structure_of_recurrent_layers = (2 * n_classes, (3 * n_classes) // 2)
        self.cls = SequenceClassifier(num_recurrent_units=structure_of_recurrent_layers,
                                      max_seq_length=self.X_train.shape[1], multioutput=False)
        true_err_msg = re.escape('`validation_data` is wrong argument! Expected `{0}` or `{1}`, but got `{2}`.'.format(
            type([1, 2]), type((1, 2)), type({1, 2})))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.cls.fit(self.X_train, self.y_train, validation_data={4, 'b'})

    def test_fit_negative02(self):
        n_classes = len(set(self.y_train.tolist()))
        structure_of_recurrent_layers = (2 * n_classes, (3 * n_classes) // 2)
        self.cls = SequenceClassifier(num_recurrent_units=structure_of_recurrent_layers,
                                      max_seq_length=self.X_train.shape[1], multioutput=False)
        true_err_msg = re.escape('`validation_data` is wrong argument! Expected a two-element sequence (inputs and '
                                 'targets), but got a 1-element one.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.cls.fit(self.X_train, self.y_train, validation_data=(self.X_test,))

    def test_fit_negative03(self):
        n_classes = len(set(self.y_train.tolist()))
        structure_of_recurrent_layers = (2 * n_classes, (3 * n_classes) // 2)
        n_classes = len(set(self.y_test.tolist()))
        X_test = np.concatenate((self.X_test, self.X_test[-3:]), axis=0)
        y_test = np.concatenate((self.y_test, np.full(shape=(3,), fill_value=n_classes, dtype=self.y_test.dtype)))
        self.cls = SequenceClassifier(num_recurrent_units=structure_of_recurrent_layers,
                                      max_seq_length=self.X_train.shape[1], multioutput=False)
        true_err_msg = re.escape('`validation_data` is wrong argument! Number of classes in validation set is not '
                                 'correspond to number of classes in training set! '
                                 '{0} != {1}.'.format(n_classes + 1, n_classes))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.cls.fit(self.X_train, self.y_train, validation_data=(X_test, y_test))

    def test_fit_negative04(self):
        n_classes = len(set(self.y_train.tolist()))
        structure_of_recurrent_layers = (2 * n_classes, (3 * n_classes) // 2)
        self.cls = SequenceClassifier(num_recurrent_units=structure_of_recurrent_layers,
                                      max_seq_length=self.X_train.shape[1], multioutput=False, warm_start=True)
        with self.assertRaises(NotFittedError):
            self.cls.fit(X=self.X_train, y=self.y_train, validation_data=(self.X_test, self.y_test))

    def test_fit_negative05(self):
        n_classes = len(set(self.y_train.tolist()))
        structure_of_recurrent_layers = (2 * n_classes, (3 * n_classes) // 2)
        self.cls = SequenceClassifier(num_recurrent_units=structure_of_recurrent_layers, gpu_memory_frac=0.8,
                                      max_seq_length=self.X_train.shape[1], multioutput=False, max_epochs=5,
                                      batch_size=16)
        res = self.cls.fit(X=self.X_train, y=self.y_train)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
        train_indices, test_indices = next(sss.split(self.X_test, self.y_test))
        res.warm_start = True
        true_err_msg = re.escape('Old structure of recurrent layers does not correspond to new structure! {0} != '
                                 '{1}'.format(structure_of_recurrent_layers, (structure_of_recurrent_layers[0],)))
        res.num_recurrent_units = (structure_of_recurrent_layers[0],)
        with self.assertRaisesRegex(ValueError, true_err_msg):
            res.fit(X=self.X_test[train_indices], y=self.y_test[train_indices],
                    validation_data=(self.X_test[test_indices], self.y_test[test_indices]))

    def test_fit_negative06(self):
        n_classes = len(set(self.y_train.tolist()))
        structure_of_recurrent_layers = (2 * n_classes, (3 * n_classes) // 2)
        self.cls = SequenceClassifier(num_recurrent_units=structure_of_recurrent_layers, batch_size=16,  max_epochs=5,
                                      gpu_memory_frac=0.8, max_seq_length=self.X_train.shape[1], multioutput=False)
        res = self.cls.fit(X=self.X_train, y=self.y_train)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
        y_test = np.copy(self.y_test)
        for sample_idx in range(y_test.shape[0]):
            if y_test[sample_idx] == (n_classes - 1):
                y_test[sample_idx] = n_classes - 2
        train_indices, test_indices = next(sss.split(self.X_test, y_test))
        X_train = self.X_test[train_indices]
        y_train = y_test[train_indices]
        X_test = self.X_test[test_indices]
        y_test = y_test[test_indices]
        res.warm_start = True
        true_err_msg = re.escape('Old number of classes does not correspond to new number! {0} != {1}'.format(
            n_classes, n_classes - 1))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            res.fit(X=X_train, y=y_train, validation_data=(X_test, y_test))

    def test_fit_negative07(self):
        n_classes = len(set(self.y_train.tolist()))
        structure_of_recurrent_layers = (2 * n_classes, (3 * n_classes) // 2)
        self.cls = SequenceClassifier(num_recurrent_units=structure_of_recurrent_layers, batch_size=16,
                                      max_seq_length=self.X_train.shape[1], multioutput=False, max_epochs=5,
                                      gpu_memory_frac=0.8)
        res = self.cls.fit(X=self.X_train, y=self.y_train)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
        train_indices, test_indices = next(sss.split(self.X_test, self.y_test))
        X_train = np.zeros((len(train_indices), self.X_test.shape[1], self.X_test.shape[2] + 1),
                           dtype=self.X_test.dtype)
        X_test = np.zeros((len(test_indices), X_train.shape[1], X_train.shape[2]), dtype=X_train.dtype)
        sample_idx_ = 0
        for sample_idx in train_indices:
            X_train[sample_idx_, 0:self.X_test.shape[1], 0:self.X_test.shape[2]] = self.X_test[sample_idx]
            sample_idx_ += 1
        sample_idx_ = 0
        for sample_idx in test_indices:
            X_test[sample_idx_, 0:self.X_test.shape[1], 0:self.X_test.shape[2]] = self.X_test[sample_idx]
            sample_idx_ += 1
        y_train = self.y_test[train_indices]
        y_test = self.y_test[test_indices]
        res.warm_start = True
        true_err_msg = re.escape('Structure of the training data does not correspond to the neural network structure! '
                                 'Feature vector size must be equal to {0}, but it is equal to {1}.'.format(
            self.X_test.shape[2], self.X_test.shape[2] + 1))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            res.fit(X=X_train, y=y_train, validation_data=(X_test, y_test))

    def test_serialize_positive01(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50, batch_size=16, learning_rate=1e-2,
                                      l2_reg=1e-6, clipnorm=5.0, max_epochs=100, patience=3, gpu_memory_frac=0.8,
                                      multioutput=True, warm_start=True, verbose=True, random_seed=42)
        self.temp_file_name = tempfile.NamedTemporaryFile(mode='w', dir=base_dir).name
        with open(self.temp_file_name, mode='wb') as fp:
            pickle.dump(self.cls, fp)
        del self.cls
        with open(self.temp_file_name, mode='rb') as fp:
            self.another_cls = pickle.load(fp)
        self.assertIsInstance(self.another_cls, SequenceClassifier)
        self.assertTrue(hasattr(self.another_cls, 'batch_size'))
        self.assertTrue(hasattr(self.another_cls, 'num_recurrent_units'))
        self.assertTrue(hasattr(self.another_cls, 'learning_rate'))
        self.assertTrue(hasattr(self.another_cls, 'l2_reg'))
        self.assertTrue(hasattr(self.another_cls, 'clipnorm'))
        self.assertTrue(hasattr(self.another_cls, 'multioutput'))
        self.assertTrue(hasattr(self.another_cls, 'warm_start'))
        self.assertTrue(hasattr(self.another_cls, 'max_epochs'))
        self.assertTrue(hasattr(self.another_cls, 'patience'))
        self.assertTrue(hasattr(self.another_cls, 'random_seed'))
        self.assertTrue(hasattr(self.another_cls, 'gpu_memory_frac'))
        self.assertTrue(hasattr(self.another_cls, 'max_seq_length'))
        self.assertTrue(hasattr(self.another_cls, 'verbose'))
        self.assertIsInstance(self.another_cls.batch_size, int)
        self.assertIsInstance(self.another_cls.num_recurrent_units, tuple)
        self.assertIsInstance(self.another_cls.learning_rate, float)
        self.assertIsInstance(self.another_cls.l2_reg, float)
        self.assertIsInstance(self.another_cls.clipnorm, float)
        self.assertIsInstance(self.another_cls.multioutput, bool)
        self.assertIsInstance(self.another_cls.warm_start, bool)
        self.assertIsInstance(self.another_cls.max_epochs, int)
        self.assertIsInstance(self.another_cls.patience, int)
        self.assertIsInstance(self.another_cls.random_seed, int)
        self.assertIsInstance(self.another_cls.gpu_memory_frac, float)
        self.assertIsInstance(self.another_cls.max_seq_length, int)
        self.assertIsInstance(self.another_cls.verbose, bool)
        self.assertEqual(self.another_cls.num_recurrent_units, (10, 3))
        self.assertEqual(self.another_cls.max_seq_length, 50)
        self.assertEqual(self.another_cls.batch_size, 16)
        self.assertAlmostEqual(self.another_cls.learning_rate, 1e-2)
        self.assertAlmostEqual(self.another_cls.l2_reg, 1e-6)
        self.assertAlmostEqual(self.another_cls.clipnorm, 5.0)
        self.assertEqual(self.another_cls.max_epochs, 100)
        self.assertEqual(self.another_cls.patience, 3)
        self.assertAlmostEqual(self.another_cls.gpu_memory_frac, 0.8)
        self.assertTrue(self.another_cls.multioutput)
        self.assertTrue(self.another_cls.warm_start)
        self.assertTrue(self.another_cls.verbose)
        self.assertEqual(self.another_cls.random_seed, 42)

    def test_serialize_positive02(self):
        n_classes = len(set(self.y_train.tolist()))
        structure_of_recurrent_layers = [2 * n_classes, (3 * n_classes) // 2]
        base_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        self.cls = SequenceClassifier(num_recurrent_units=structure_of_recurrent_layers,
                                      max_seq_length=self.X_train.shape[1], batch_size=16, learning_rate=1e-2,
                                      l2_reg=1e-6, clipnorm=5.0, max_epochs=3, patience=3, gpu_memory_frac=0.8,
                                      multioutput=False, warm_start=False, verbose=True, random_seed=42)
        self.cls.fit(self.X_train, self.y_train)
        old_score = self.cls.score(self.X_test, self.y_test)
        self.temp_file_name = tempfile.NamedTemporaryFile(mode='w', dir=base_dir).name
        with open(self.temp_file_name, mode='wb') as fp:
            pickle.dump(self.cls, fp)
        del self.cls
        with open(self.temp_file_name, mode='rb') as fp:
            self.another_cls = pickle.load(fp)
        self.assertIsInstance(self.another_cls, SequenceClassifier)
        self.assertTrue(hasattr(self.another_cls, 'batch_size'))
        self.assertTrue(hasattr(self.another_cls, 'num_recurrent_units'))
        self.assertTrue(hasattr(self.another_cls, 'learning_rate'))
        self.assertTrue(hasattr(self.another_cls, 'l2_reg'))
        self.assertTrue(hasattr(self.another_cls, 'clipnorm'))
        self.assertTrue(hasattr(self.another_cls, 'multioutput'))
        self.assertTrue(hasattr(self.another_cls, 'warm_start'))
        self.assertTrue(hasattr(self.another_cls, 'max_epochs'))
        self.assertTrue(hasattr(self.another_cls, 'patience'))
        self.assertTrue(hasattr(self.another_cls, 'random_seed'))
        self.assertTrue(hasattr(self.another_cls, 'gpu_memory_frac'))
        self.assertTrue(hasattr(self.another_cls, 'max_seq_length'))
        self.assertTrue(hasattr(self.another_cls, 'verbose'))
        self.assertIsInstance(self.another_cls.batch_size, int)
        self.assertIsInstance(self.another_cls.num_recurrent_units, list)
        self.assertIsInstance(self.another_cls.learning_rate, float)
        self.assertIsInstance(self.another_cls.l2_reg, float)
        self.assertIsInstance(self.another_cls.clipnorm, float)
        self.assertIsInstance(self.another_cls.multioutput, bool)
        self.assertIsInstance(self.another_cls.warm_start, bool)
        self.assertIsInstance(self.another_cls.max_epochs, int)
        self.assertIsInstance(self.another_cls.patience, int)
        self.assertIsInstance(self.another_cls.random_seed, int)
        self.assertIsInstance(self.another_cls.gpu_memory_frac, float)
        self.assertIsInstance(self.another_cls.max_seq_length, int)
        self.assertIsInstance(self.another_cls.verbose, bool)
        self.assertEqual(self.another_cls.num_recurrent_units, structure_of_recurrent_layers)
        self.assertEqual(self.another_cls.max_seq_length, self.X_train.shape[1])
        self.assertEqual(self.another_cls.batch_size, 16)
        self.assertAlmostEqual(self.another_cls.learning_rate, 1e-2)
        self.assertAlmostEqual(self.another_cls.l2_reg, 1e-6)
        self.assertAlmostEqual(self.another_cls.clipnorm, 5.0)
        self.assertEqual(self.another_cls.max_epochs, 3)
        self.assertEqual(self.another_cls.patience, 3)
        self.assertAlmostEqual(self.another_cls.gpu_memory_frac, 0.8)
        self.assertFalse(self.another_cls.multioutput)
        self.assertFalse(self.another_cls.warm_start)
        self.assertTrue(self.another_cls.verbose)
        self.assertEqual(self.another_cls.random_seed, 42)
        self.assertTrue(hasattr(self.another_cls, 'n_classes_'))
        self.assertTrue(hasattr(self.another_cls, 'rnn_output_'))
        self.assertTrue(hasattr(self.another_cls, 'logits_'))
        self.assertTrue(hasattr(self.another_cls, 'input_data_'))
        self.assertTrue(hasattr(self.another_cls, 'input_shape_'))
        self.assertTrue(hasattr(self.another_cls, 'layers_'))
        self.assertTrue(hasattr(self.another_cls, 'y_ph_'))
        self.assertTrue(hasattr(self.another_cls, 'sess_'))
        self.assertIsInstance(self.another_cls.n_classes_, int)
        self.assertIsInstance(self.another_cls.input_shape_, tuple)
        self.assertIsInstance(self.another_cls.layers_, tuple)
        self.assertEqual(self.another_cls.n_classes_, n_classes)
        self.assertEqual(self.another_cls.input_shape_, self.X_train.shape[1:])
        self.assertEqual(self.another_cls.layers_, tuple(structure_of_recurrent_layers))
        score = self.another_cls.score(self.X_test, self.y_test)
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, old_score, places=3)

    def test_get_data_target_positive01(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50)
        y = np.array([0, 1, 2, 0, 0, 2, 1, 1, 0], dtype=np.int32)
        self.cls.n_classes_ = 3
        true_target = np.array([0, 0, 1], dtype=np.float32)
        real_target = self.cls.get_data_target(y, 2)
        self.assertIsInstance(real_target, np.ndarray)
        self.assertEqual(true_target.shape, real_target.shape)
        self.assertEqual(true_target.tolist(), real_target.tolist())

    def test_get_data_target_positive02(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50)
        y = [{0}, {1}, {2}, {0}, {0}, {0, 2}, {1}, {1}, {0}]
        self.cls.n_classes_ = 3
        true_target = np.array([1, 0, 1], dtype=np.float32)
        real_target = self.cls.get_data_target(y, 5)
        self.assertIsInstance(real_target, np.ndarray)
        self.assertEqual(true_target.shape, real_target.shape)
        self.assertEqual(true_target.tolist(), real_target.tolist())

    def test_get_data_target_positive03(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50)
        y = [{0}, {1}, {2}, {0}, {0}, {0, 2}, {1}, {1}, set()]
        self.cls.n_classes_ = 3
        true_target = np.array([0, 0, 0], dtype=np.float32)
        real_target = self.cls.get_data_target(y, 8)
        self.assertIsInstance(real_target, np.ndarray)
        self.assertEqual(true_target.shape, real_target.shape)
        self.assertEqual(true_target.tolist(), real_target.tolist())

    def test_get_data_target_positive04(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50)
        y = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 1],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 0]
            ],
            dtype=np.float32
        )
        self.cls.n_classes_ = 3
        true_target = np.array([1, 0, 0], dtype=np.float32)
        real_target = self.cls.get_data_target(y, 3)
        self.assertIsInstance(real_target, np.ndarray)
        self.assertEqual(true_target.shape, real_target.shape)
        self.assertEqual(true_target.tolist(), real_target.tolist())

    def test_generate_bounds_of_batches_positive01(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50, batch_size=4)
        data_size = 16
        true_bounds_of_batches = [(0, 4), (4, 8), (8, 12), (12, 16)]
        calc_bounds_of_batches = self.cls.generate_bounds_of_batches(data_size)
        self.assertEqual(true_bounds_of_batches, calc_bounds_of_batches)

    def test_generate_bounds_of_batches_positive02(self):
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50, batch_size=4)
        data_size = 13
        true_bounds_of_batches = [(0, 4), (4, 8), (8, 12), (12, 13)]
        calc_bounds_of_batches = self.cls.generate_bounds_of_batches(data_size)
        self.assertEqual(true_bounds_of_batches, calc_bounds_of_batches)

    def test_generate_new_batch_positive01(self):
        X = np.vstack(
            (
                np.random.uniform(0.0, 1.0, (3, 7)),  # 0 3
                np.random.uniform(-1.5, 0.5, (4, 7)),  # 3 7
                np.random.uniform(1.0, 2.3, (5, 7)),  # 7 12
                np.random.uniform(2.5, 3.7, (1, 7))  # 12 13
            )
        )
        y = np.concatenate(
            (
                np.full((3,), 1, dtype=np.int32),  # 0 3
                np.full((4,), 0, dtype=np.int32),  # 3 7
                np.full((5,), 2, dtype=np.int32),  # 7 12
                np.full((1,), 3, dtype=np.int32),  # 12 13
            )
        )
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50, batch_size=4)
        true_batch = (X[4:8], y[4:8])
        calculated_batch = self.cls.generate_new_batch(4, 8, X, y, False)
        self.assertIsInstance(calculated_batch, tuple)
        self.assertEqual(len(calculated_batch), 2)
        self.assertIsInstance(calculated_batch[0], np.ndarray)
        self.assertIsInstance(calculated_batch[1], np.ndarray)
        self.assertEqual(calculated_batch[0].shape, true_batch[0].shape)
        self.assertEqual(calculated_batch[1].shape, true_batch[1].shape)
        self.assertEqual(calculated_batch[1].tolist(), true_batch[1].tolist())
        for row_idx in range(true_batch[0].shape[0]):
            for col_idx in range(true_batch[0].shape[1]):
                self.assertAlmostEqual(calculated_batch[0][row_idx][col_idx], true_batch[0][row_idx][col_idx], places=5,
                                       msg='Row {0}, column {1}.'.format(row_idx, col_idx))

    def test_generate_new_batch_positive02(self):
        X = np.vstack(
            (
                np.random.uniform(0.0, 1.0, (3, 7)),  # 0 3
                np.random.uniform(-1.5, 0.5, (4, 7)),  # 3 7
                np.random.uniform(1.0, 2.3, (5, 7)),  # 7 12
                np.random.uniform(2.5, 3.7, (1, 7))  # 12 13
            )
        )
        y = np.concatenate(
            (
                np.full((3,), 1, dtype=np.int32),  # 0 3
                np.full((4,), 0, dtype=np.int32),  # 3 7
                np.full((5,), 2, dtype=np.int32),  # 7 12
                np.full((1,), 3, dtype=np.int32),  # 12 13
            )
        )
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50, batch_size=4)
        true_batch = (
            np.vstack((X[12:13], X[12:13], X[12:13], X[12:13])),
            np.concatenate((y[12:13], y[12:13], y[12:13], y[12:13]))
        )
        calculated_batch = self.cls.generate_new_batch(12, 13, X, y, False)
        self.assertIsInstance(calculated_batch, tuple)
        self.assertEqual(len(calculated_batch), 2)
        self.assertIsInstance(calculated_batch[0], np.ndarray)
        self.assertIsInstance(calculated_batch[1], np.ndarray)
        self.assertEqual(calculated_batch[0].shape, true_batch[0].shape)
        self.assertEqual(calculated_batch[1].shape, true_batch[1].shape)
        self.assertEqual(calculated_batch[1].tolist(), true_batch[1].tolist())
        for row_idx in range(true_batch[0].shape[0]):
            for col_idx in range(true_batch[0].shape[1]):
                self.assertAlmostEqual(calculated_batch[0][row_idx][col_idx], true_batch[0][row_idx][col_idx], places=5,
                                       msg='Row {0}, column {1}.'.format(row_idx, col_idx))

    def test_generate_new_batch_positive03(self):
        X = np.vstack(
            (
                np.random.uniform(0.0, 1.0, (3, 7)),  # 0 3
                np.random.uniform(-1.5, 0.5, (4, 7)),  # 3 7
                np.random.uniform(1.0, 2.3, (5, 7)),  # 7 12
                np.random.uniform(2.5, 3.7, (1, 7))  # 12 13
            )
        )
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50, batch_size=4)
        true_batch = X[4:8]
        calculated_batch = self.cls.generate_new_batch(4, 8, X, shuffle=False)
        self.assertIsInstance(calculated_batch, np.ndarray)
        self.assertEqual(calculated_batch.shape, true_batch.shape)
        for row_idx in range(true_batch.shape[0]):
            for col_idx in range(true_batch.shape[1]):
                self.assertAlmostEqual(calculated_batch[row_idx][col_idx], true_batch[row_idx][col_idx], places=5,
                                       msg='Row {0}, column {1}.'.format(row_idx, col_idx))

    def test_generate_new_batch_positive04(self):
        X = np.vstack(
            (
                np.random.uniform(0.0, 1.0, (3, 7)),  # 0 3
                np.random.uniform(-1.5, 0.5, (4, 7)),  # 3 7
                np.random.uniform(1.0, 2.3, (5, 7)),  # 7 12
                np.random.uniform(2.5, 3.7, (1, 7))  # 12 13
            )
        )
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50, batch_size=4)
        true_batch = np.vstack((X[12:13], X[12:13], X[12:13], X[12:13]))
        calculated_batch = self.cls.generate_new_batch(12, 13, X, shuffle=False)
        self.assertIsInstance(calculated_batch, np.ndarray)
        self.assertEqual(calculated_batch.shape, true_batch.shape)
        for row_idx in range(true_batch.shape[0]):
            for col_idx in range(true_batch.shape[1]):
                self.assertAlmostEqual(calculated_batch[row_idx][col_idx], true_batch[row_idx][col_idx], places=5,
                                       msg='Row {0}, column {1}.'.format(row_idx, col_idx))

    def test_generate_new_batch_positive05(self):
        X = np.vstack(
            (
                np.random.uniform(0.0, 1.0, (3, 7)),  # 0 3
                np.random.uniform(-1.5, 0.5, (4, 7)),  # 3 7
                np.random.uniform(1.0, 2.3, (5, 7)),  # 7 12
                np.random.uniform(2.5, 3.7, (1, 7))  # 12 13
            )
        )
        y = np.concatenate(
            (
                np.full((3,), 1, dtype=np.int32),  # 0 3
                np.full((4,), 0, dtype=np.int32),  # 3 7
                np.full((5,), 2, dtype=np.int32),  # 7 12
                np.full((1,), 3, dtype=np.int32),  # 12 13
            )
        )
        self.cls = SequenceClassifier(num_recurrent_units=(10, 3), max_seq_length=50, batch_size=4)
        true_batch = (X[4:8], y[4:8])
        calculated_batch = self.cls.generate_new_batch(4, 8, X, y, True)
        self.assertIsInstance(calculated_batch, tuple)
        self.assertEqual(len(calculated_batch), 2)
        self.assertIsInstance(calculated_batch[0], np.ndarray)
        self.assertIsInstance(calculated_batch[1], np.ndarray)
        self.assertEqual(calculated_batch[0].shape, true_batch[0].shape)
        self.assertEqual(calculated_batch[1].shape, true_batch[1].shape)
        self.assertEqual(calculated_batch[1].tolist(), true_batch[1].tolist())
        indices = []
        for row_idx_1 in range(true_batch[0].shape[0]):
            for row_idx_2 in range(true_batch[0].shape[0]):
                is_equal = True
                for col_idx in range(true_batch[0].shape[1]):
                    if abs(calculated_batch[0][row_idx_1][col_idx] - true_batch[0][row_idx_2][col_idx]) > 1e-5:
                        is_equal = False
                        break
                if is_equal:
                    indices.append(row_idx_2)
                    break
        self.assertEqual(len(indices), true_batch[0].shape[0])
        self.assertEqual(len(indices), len(set(indices)))
        for idx, val in enumerate(indices):
            self.assertGreaterEqual(val, 0)
            self.assertLess(val, 4)
            self.assertEqual(calculated_batch[1][val], true_batch[1][idx])

    @classmethod
    def load_data(cls):
        X_train_name = os.path.join(os.path.dirname(__file__), 'testdata', 'X_train.npy')
        X_test_name = os.path.join(os.path.dirname(__file__), 'testdata', 'X_test.npy')
        y_train_name = os.path.join(os.path.dirname(__file__), 'testdata', 'y_train.npy')
        y_test_name = os.path.join(os.path.dirname(__file__), 'testdata', 'y_test.npy')
        if os.path.isfile(X_train_name) and os.path.isfile(X_test_name) and os.path.isfile(y_train_name) and \
                os.path.isfile(y_test_name):
            cls.X_train = np.load(X_train_name)
            cls.X_test = np.load(X_test_name)
            cls.y_train = np.load(y_train_name)
            cls.y_test = np.load(y_test_name)
        else:
            dictionary, vectors = cls.load_glove()
            data = fetch_20newsgroups(data_home=os.path.join(os.path.dirname(__file__), 'testdata'), subset='train')
            if data is None:
                raise ValueError('Data for training and testing cannot be downloaded!')
            tokenizer = NISTTokenizer()
            tokenized_train_texts = []
            cls.y_train = np.copy(data['target'])
            for cur_text in data['data']:
                tokenized_train_texts.append(
                    list(filter(
                        lambda it2: (len(it2) > 0) and (it2 in dictionary),
                        map(lambda it1: it1.strip().lower(), tokenizer.international_tokenize(cur_text))
                    ))
                )
            text_lengths = sorted([len(cur_text) for cur_text in tokenized_train_texts])
            max_seq_len = text_lengths[int(round(0.8 * (len(text_lengths) - 1)))]
            train_indices = sorted(list(
                filter(
                    lambda idx: text_lengths[idx] < max_seq_len,
                    range(len(tokenized_train_texts))
                )
            ))
            del text_lengths
            del data
            data = fetch_20newsgroups(data_home=os.path.join(os.path.dirname(__file__), 'testdata'), subset='test')
            if data is None:
                raise ValueError('Data for training and testing cannot be downloaded!')
            tokenized_test_texts = []
            cls.y_test = np.copy(data['target'])
            for cur_text in data['data']:
                tokenized_test_texts.append(
                    list(filter(
                        lambda it2: (len(it2) > 0) and (it2 in dictionary),
                        map(lambda it1: it1.strip().lower(), tokenizer.international_tokenize(cur_text))
                    ))
                )
            del data
            test_indices = sorted(list(
                filter(
                    lambda idx: len(tokenized_test_texts[idx]) < max_seq_len,
                    range(len(tokenized_test_texts))
                )
            ))
            feature_vector_size = vectors.shape[1]
            cls.X_train = np.zeros((len(train_indices), max_seq_len, feature_vector_size), dtype=np.float32)
            cls.X_test = np.zeros((len(test_indices), max_seq_len, feature_vector_size), dtype=np.float32)
            text_idx_ = 0
            for text_idx in train_indices:
                for token_idx in range(min(len(tokenized_train_texts[text_idx]), max_seq_len)):
                    cls.X_train[text_idx_][token_idx] = vectors[dictionary[tokenized_train_texts[text_idx][token_idx]]]
                text_idx_ += 1
            text_idx_ = 0
            for text_idx in test_indices:
                for token_idx in range(min(len(tokenized_test_texts[text_idx]), max_seq_len)):
                    cls.X_test[text_idx_][token_idx] = vectors[dictionary[tokenized_test_texts[text_idx][token_idx]]]
                text_idx_ += 1
            del vectors, dictionary
            cls.y_train = cls.y_train[train_indices]
            cls.y_test = cls.y_test[test_indices]
            train_classes_distr = dict()
            for class_idx in cls.y_train:
                train_classes_distr[class_idx] = train_classes_distr.get(class_idx, 0) + 1
            test_classes_distr = dict()
            for class_idx in cls.y_test:
                test_classes_distr[class_idx] = test_classes_distr.get(class_idx, 0) + 1
            possible_classes = set()
            for class_idx in set(train_classes_distr.keys()) | set(test_classes_distr.keys()):
                if (class_idx in train_classes_distr) and (class_idx in test_classes_distr):
                    if (train_classes_distr[class_idx] > 10) and (test_classes_distr[class_idx] > 10):
                        possible_classes.add(class_idx)
            del train_indices
            train_indices = list(filter(
                lambda sample_idx: cls.y_train[sample_idx] in possible_classes,
                range(len(cls.y_train))
            ))
            del test_indices
            test_indices = list(filter(
                lambda sample_idx: cls.y_train[sample_idx] in possible_classes,
                range(len(cls.y_test))
            ))
            cls.X_train = cls.X_train[train_indices]
            cls.y_train = cls.y_train[train_indices]
            cls.X_test = cls.X_test[test_indices]
            cls.y_test = cls.y_test[test_indices]
            possible_classes = sorted(list(possible_classes))
            classes_prj = dict()
            for idx, val in enumerate(possible_classes):
                classes_prj[val] = idx
            for sample_idx in range(len(cls.y_train)):
                cls.y_train[sample_idx] = classes_prj[cls.y_train[sample_idx]]
            for sample_idx in range(len(cls.y_test)):
                cls.y_test[sample_idx] = classes_prj[cls.y_test[sample_idx]]
            del classes_prj, possible_classes, train_indices, test_indices
            assert cls.X_train.shape[0] == len(cls.y_train)
            assert cls.X_test.shape[0] == len(cls.y_test)
            assert len(set(cls.y_train.tolist())) == (cls.y_train.max() + 1)
            assert len(set(cls.y_test.tolist())) == (cls.y_test.max() + 1)
            if cls.X_train.shape[0] > 1000:
                train_part = min(1000.0 / float(cls.X_train.shape[0]), 0.7)
                sss = StratifiedShuffleSplit(n_splits=1, train_size=train_part)
                train_index, _ = next(sss.split(cls.X_train, cls.y_train))
                cls.X_train = cls.X_train[train_index]
                cls.y_train = cls.y_train[train_index]
                del sss, train_index
            if cls.X_test.shape[0] > 500:
                train_part = min(500.0 / float(cls.X_test.shape[0]), 0.7)
                sss = StratifiedShuffleSplit(n_splits=1, train_size=train_part)
                train_index, _ = next(sss.split(cls.X_test, cls.y_test))
                cls.X_test = cls.X_test[train_index]
                cls.y_test = cls.y_test[train_index]
                del sss, train_index
            np.save(X_train_name[:-4], cls.X_train, allow_pickle=False)
            np.save(X_test_name[:-4], cls.X_test, allow_pickle=False)
            np.save(y_train_name[:-4], cls.y_train, allow_pickle=False)
            np.save(y_test_name[:-4], cls.y_test, allow_pickle=False)
            gc.collect()

    @classmethod
    def load_glove(cls) -> Tuple[Dict[str, int], np.ndarray]:
        glove_model_name = os.path.join(os.path.dirname(__file__), 'testdata', 'glove.6B.50d.txt')
        glove_archive_name = os.path.join(os.path.dirname(__file__), 'testdata', 'glove.6B.zip')
        if not os.path.isfile(glove_model_name):
            if not os.path.isfile(glove_archive_name):
                url = 'http://nlp.stanford.edu/data/wordvecs/glove.6B.zip'
                with open(glove_archive_name, 'wb') as f_out:
                    ufr = requests.get(url)
                    f_out.write(ufr.content)
            with zipfile.ZipFile(glove_archive_name, 'r') as f_in:
                f_in.extract(member='glove.6B.50d.txt', path=os.path.join(os.path.dirname(__file__), 'testdata'))
            os.remove(glove_archive_name)
        word_idx = 0
        dictionary = dict()
        vectors = []
        with codecs.open(glove_model_name, mode='r', encoding='utf-8', errors='ignore') as fp:
            cur_line = fp.readline()
            while len(cur_line) > 0:
                prep_line = cur_line.strip()
                if len(prep_line) > 0:
                    line_parts = prep_line.split()
                    if len(line_parts) == 51:
                        new_word = line_parts[0]
                        new_vector = np.reshape(np.array([float(cur) for cur in line_parts[1:]], dtype=np.float32),
                                                newshape=(1, 50))
                        if new_word not in dictionary:
                            dictionary[new_word] = word_idx
                            vectors.append(new_vector)
                            word_idx += 1
                cur_line = fp.readline()
        return dictionary, np.vstack(vectors)


if __name__ == '__main__':
    unittest.main(verbosity=2)
