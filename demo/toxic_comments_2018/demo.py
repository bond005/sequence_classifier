from argparse import ArgumentParser
import codecs
import os
import pickle
import requests
import sys
from typing import Dict, List, Tuple, Union
import zipfile

import numpy as np
from sklearn.metrics import roc_auc_score

try:
    from sequence_classifier.sequence_classifier import SequenceClassifier
    from sequence_classifier.utils import load_trainset_for_toxic_comments_2017
    from sequence_classifier.utils import load_testset_for_toxic_comments_2017
    from sequence_classifier.utils import print_classes_distribution
    from sequence_classifier.utils import train_test_split
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from sequence_classifier.utils import load_trainset_for_toxic_comments_2017
    from sequence_classifier.utils import load_testset_for_toxic_comments_2017
    from sequence_classifier.utils import print_classes_distribution
    from sequence_classifier.utils import train_test_split


class WordSequenceClassifier(SequenceClassifier):
    def __init__(self, dictionary: Dict[str, int], vectors: np.ndarray, num_recurrent_units: Union[List[int], tuple],
                 max_seq_length: int, batch_size: int = 32, learning_rate: float = 1e-3, l2_reg: float = 1e-5,
                 clipnorm: float = 1.0, max_epochs: int = 1000, patience: int = 5, gpu_memory_frac: float = 0.9,
                 multioutput: bool = False, warm_start: bool = False, verbose: bool = False,
                 random_seed: Union[int, None] = None):
        super().__init__(num_recurrent_units, max_seq_length, batch_size, learning_rate, l2_reg, clipnorm, max_epochs,
                         patience, gpu_memory_frac, multioutput, warm_start, verbose, random_seed)
        self.dictionary = dictionary
        self.vectors = vectors

    def get_feature_vector_size(self, X: Union[list, tuple, np.ndarray]) -> int:
        return self.vectors.shape[1] + 6

    def get_data_input(self, X: Union[list, tuple, np.ndarray], idx: int, training_phase: bool) -> np.ndarray:
        res = np.zeros((self.max_seq_length, self.vectors.shape[1] + 6), dtype=np.float32)
        for token_idx, token_text in enumerate(filter(lambda it: it in self.dictionary, X[idx])):
            if token_idx >= self.max_seq_length:
                break
            if token_idx in self.dictionary:
                res[token_idx][0:self.vectors.shape[1]] = self.vectors[self.dictionary[token_text]]
            else:
                if token_text.isalpha():
                    res[token_idx][self.vectors.shape[1]] = 1.0
                elif token_text.isdigit():
                    res[token_idx][self.vectors.shape[1] + 1] = 1.0
                else:
                    is_digit = any(map(lambda it: it.isdigit(), token_text))
                    is_alpha = any(map(lambda it: it.isalpha(), token_text))
                    if is_digit and is_alpha:
                        res[token_idx][self.vectors.shape[1] + 2] = 1.0
                    elif is_digit:
                        res[token_idx][self.vectors.shape[1] + 3] = 1.0
                    elif is_alpha:
                        res[token_idx][self.vectors.shape[1] + 4] = 1.0
                    else:
                        res[token_idx][self.vectors.shape[1] + 5] = 1.0
        return res

    def check_X_item(self, item) -> bool:
        ok = hasattr(item, '__len__') and hasattr(item, '__getitem__')
        if not ok:
            return False
        try:
            ok = all(map(
                lambda token: hasattr(token, '__len__') and hasattr(token, '__getitem__') and hasattr(token, 'split'),
                item
            ))
        except:
            ok = False
        return ok


def str_to_layers(src: str) -> List[int]:
    values = list(map(
        lambda it3: int(it3),
        filter(lambda it2: len(it2) > 0, map(lambda it1: it1.strip(), src.split('-')))
    ))
    if len(values) == 0:
        raise ValueError('{0} is wrong description of layer structure! This description is empty!'.format(src))
    if any(map(lambda it: it < 1, values)):
        raise ValueError('{0} is wrong description of layer structure! All layer sizes must be a '
                         'positive values!'.format(src))
    return values


def load_glove() -> Tuple[Dict[str, int], np.ndarray]:
    glove_model_name = os.path.join(os.path.dirname(__file__), 'data', 'glove.6B.300d.txt')
    glove_archive_name = os.path.join(os.path.dirname(__file__), 'data', 'glove.6B.zip')
    if not os.path.isfile(glove_model_name):
        if not os.path.isfile(glove_archive_name):
            url = 'http://nlp.stanford.edu/data/wordvecs/glove.6B.zip'
            with open(glove_archive_name, 'wb') as f_out:
                ufr = requests.get(url)
                f_out.write(ufr.content)
        with zipfile.ZipFile(glove_archive_name, 'r') as f_in:
            f_in.extract(member='glove.6B.300d.txt', path=os.path.join(os.path.dirname(__file__), 'data'))
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
                if len(line_parts) == 301:
                    new_word = line_parts[0]
                    new_vector = np.reshape(np.array([float(cur) for cur in line_parts[1:]], dtype=np.float32),
                                            newshape=(1, 300))
                    if new_word not in dictionary:
                        dictionary[new_word] = word_idx
                        vector_norm = np.linalg.norm(new_vector)
                        if vector_norm > 0.0:
                            vectors.append(new_vector / vector_norm)
                            word_idx += 1
            cur_line = fp.readline()
    return dictionary, np.vstack(vectors)


def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, default=None, required=False,
                        help='File name of neural model.')
    parser.add_argument('--struct', dest='structure_of_layers', type=str, required=False, default='100-50',
                        help='The structure of recurrent layers (hyphenated layer sizes).')
    parser.add_argument('--reg', dest='l2_reg', type=float, required=False, default=1e-6,
                        help='L2 regularization coefficient for output layer.')
    parser.add_argument('--lr', dest='learning_rate', type=float, required=False, default=1e-3,
                        help='The learning rate parameter.')
    parser.add_argument('--batch', dest='batch_size', type=int, required=False, default=16, help='Size of mini-batch.')
    parser.add_argument('--max', dest='max_epochs', type=int, required=False, default=1000,
                        help='Maximal number of training epochs.')
    args = parser.parse_args()

    model_name = args.model_name
    if model_name is not None:
        model_name = os.path.normpath(model_name)
    dictionary, vectors = load_glove()
    print('Vocabulary size is {0}.'.format(vectors.shape[0]))
    train_texts, train_labels, train_classes = load_trainset_for_toxic_comments_2017(
        os.path.join(os.path.dirname(__file__), 'data', 'train.csv')
    )
    test_texts, test_labels, test_classes = load_testset_for_toxic_comments_2017(
        os.path.join(os.path.dirname(__file__), 'data', 'test.csv'),
        os.path.join(os.path.dirname(__file__), 'data', 'test_labels.csv')
    )
    assert train_classes == test_classes, 'Set of classes for training does not correspond to set of classes for ' \
                                          'testing!'
    if (model_name is None) or (not os.path.isfile(model_name)):
        train_texts, train_labels, valid_texts, valid_labels = train_test_split(train_texts, train_labels, 0.1, 42)
        print('')
        print('Classes distribution in data for training:')
        print_classes_distribution(train_labels, train_classes)
        print('')
        print('Classes distribution in data for validation:')
        print_classes_distribution(valid_labels, train_classes)
        print('')
        lengths_of_texts = sorted([len(cur) for cur in train_texts])
        max_seq_length = lengths_of_texts[int(round(0.98 * (len(lengths_of_texts) - 1)))]
        print('Maximal token number in text is {0}.'.format(max_seq_length))
        print('')
        cls = WordSequenceClassifier(
            dictionary=dictionary, vectors=vectors, num_recurrent_units=str_to_layers(args.structure_of_layers),
            max_seq_length=max_seq_length, batch_size=args.batch_size, learning_rate=args.learning_rate,
            l2_reg=args.l2_reg,
            max_epochs=args.max_epochs, patience=5, clipnorm=10.0, gpu_memory_frac=0.9, multioutput=True,
            warm_start=False,
            verbose=True, random_seed=42
        )
        cls.fit(X=train_texts, y=train_labels, validation_data=(valid_texts, valid_labels))
        print('')
        if model_name is not None:
            with open(model_name, 'wb') as fp:
                pickle.dump(cls, fp)
    else:
        print('')
        print('Classes distribution in data for training:')
        print_classes_distribution(train_labels, train_classes)
        print('')
        with open(model_name, 'rb') as fp:
            cls = pickle.load(fp)
        cls.dictionary = dictionary
        cls.vectors = vectors
    print('Classes distribution in data for testing:')
    print_classes_distribution(test_labels, test_classes)
    print('')
    probabilities = cls.predict_proba(test_texts)
    y_true = np.zeros(probabilities.shape, dtype=np.int32)
    for sample_idx in range(len(test_labels)):
        for class_idx in test_labels[sample_idx]:
            y_true[sample_idx][class_idx] = 1
    roc_auc = [roc_auc_score(y_true[:, class_idx], probabilities[:, class_idx])
               for class_idx in range(len(test_classes) - 1)]
    print('Mean ROC-AUC score is {0:.9f}.'.format(np.mean(roc_auc)))
    print('By classes:')
    for class_idx in range(len(roc_auc)):
        print('  class `{0}`:'.format(test_classes[class_idx]))
        print('    ROC-AUC = {0:.9f};'.format(roc_auc[class_idx]))


if __name__ == '__main__':
    main()
