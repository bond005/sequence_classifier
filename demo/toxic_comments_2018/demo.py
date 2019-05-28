from argparse import ArgumentParser
import codecs
import copy
import csv
import os
import random
import requests
import sys
from typing import Dict, List, Set, Tuple, Union
import zipfile

from nltk.tokenize.nist import NISTTokenizer
import numpy as np
from sklearn.metrics import roc_auc_score

try:
    from sequence_classifier.sequence_classifier import SequenceClassifier
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from sequence_classifier.sequence_classifier import SequenceClassifier


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
        return self.vectors.shape[1]

    def get_data_input(self, X: Union[list, tuple, np.ndarray], idx: int, training_phase: bool) -> np.ndarray:
        res = np.zeros((self.max_seq_length, self.vectors.shape[1]), dtype=np.float32)
        for token_idx, token_text in enumerate(filter(lambda it: it in self.dictionary, X[idx])):
            if token_idx >= self.max_seq_length:
                break
            res[token_idx] = self.vectors[self.dictionary[token_text]]
            if training_phase:
                noise = np.random.uniform(-1.0, 1.0, size=(self.vectors.shape[1],))
                vector_norm = np.linalg.norm(noise)
                if vector_norm > 0.0:
                    noise /= (vector_norm * 20.0)
                    res[token_idx] += noise
                    vector_norm = np.linalg.norm(res[token_idx])
                    if vector_norm > 0.0:
                        res[token_idx] /= vector_norm
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


def load_trainset(dictionary: Dict[str, int]) -> Tuple[List[tuple], List[Set[int]], List[str]]:
    tokenizer = NISTTokenizer()
    file_name = os.path.join(os.path.dirname(__file__), 'data', 'train.csv')
    assert os.path.isfile(file_name), 'The training data file `{0}` does not exist!'.format(file_name)
    header = None
    line_idx = 1
    all_texts = []
    all_labels = []
    classes_list = []
    with codecs.open(file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        reader = csv.reader(fp, quotechar='"', delimiter=',')
        for row in reader:
            if len(row) > 0:
                if header is None:
                    if len(row) < 3:
                        raise ValueError('File `{0}`, line {1}: header must be consists of 3 columns, '
                                         'at least.'.format(file_name, line_idx))
                    if row[0] != 'id':
                        raise ValueError('File `{0}`, line {1}: first column of table must be an identifier, '
                                         'i.e. `id`.'.format(file_name, line_idx))
                    if row[1] != 'comment_text':
                        raise ValueError('File `{0}`, line {1}: second column of table must be a source text, '
                                         'i.e. `comment_text`.'.format(file_name, line_idx))
                    header = copy.copy(row)
                    classes_list = copy.copy(row[2:])
                    if len(classes_list) != len(set(classes_list)):
                        raise ValueError('File `{0}`, line {1}: names of classes are duplicated.'.format(
                            file_name, line_idx))
                    if 'normal' in classes_list:
                        raise ValueError('File `{0}`, line {1}: `normal` is inadmissible class name.'.format(
                            file_name, line_idx))
                    classes_list.append('normal')
                else:
                    if len(row) != len(header):
                        raise ValueError('File `{0}`, line {1}: this line does not correspond to the header.'.format(
                            file_name, line_idx))
                    new_text = ' '.join(
                        list(filter(lambda it2: len(it2) > 0, map(lambda it1: it1.strip(), row[1].split())))
                    )
                    all_texts.append(
                        tuple(filter(
                            lambda it2: (len(it2) > 0) and it2 in dictionary,
                            map(lambda it1: it1.strip().lower(), tokenizer.international_tokenize(text=new_text))
                        ))
                    )
                    if not all(map(lambda it: it in {'0', '1'}, row[2:])):
                        raise ValueError('File `{0}`, line {1}: all labels must be 0 or 1.'.format(file_name, line_idx))
                    new_label = set(filter(lambda idx: row[2 + idx] == '1', range(len(classes_list) - 1)))
                    if len(new_label) == 0:
                        new_label.add(classes_list.index('normal'))
                    all_labels.append(new_label)
            line_idx += 1
    print('Size of the training set is {0}.'.format(len(all_texts)))
    return all_texts, all_labels, classes_list


def load_testset(dictionary: Dict[str, int]) -> Tuple[List[tuple], List[Set[int]], List[str]]:
    tokenizer = NISTTokenizer()
    file_name = os.path.join(os.path.dirname(__file__), 'data', 'test.csv')
    assert os.path.isfile(file_name), 'The test file `{0}` does not exist!'.format(file_name)
    header = None
    line_idx = 1
    all_texts = []
    with codecs.open(file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        reader = csv.reader(fp, quotechar='"', delimiter=',')
        for row in reader:
            if len(row) > 0:
                if header is None:
                    if len(row) != 2:
                        raise ValueError('File `{0}`, line {1}: header must be consists of 2 columns.'.format(
                            file_name, line_idx))
                    if row[0] != 'id':
                        raise ValueError('File `{0}`, line {1}: first column of table must be an identifier, '
                                         'i.e. `id`.'.format(file_name, line_idx))
                    if row[1] != 'comment_text':
                        raise ValueError('File `{0}`, line {1}: second column of table must be a source text, '
                                         'i.e. `comment_text`.'.format(file_name, line_idx))
                    header = copy.copy(row)
                else:
                    if len(row) != len(header):
                        raise ValueError('File `{0}`, line {1}: this line does not correspond to the header.'.format(
                            file_name, line_idx))
                    new_text = ' '.join(
                        list(filter(lambda it2: len(it2) > 0, map(lambda it1: it1.strip(), row[1].split())))
                    )
                    all_texts.append(
                        tuple(filter(
                            lambda it2: (len(it2) > 0) and (it2 in dictionary),
                            map(lambda it1: it1.strip().lower(), tokenizer.international_tokenize(text=new_text))
                        ))
                    )
            line_idx += 1
    del header
    file_name = os.path.join(os.path.dirname(__file__), 'data', 'test_labels.csv')
    assert os.path.isfile(file_name), 'The test file `{0}` does not exist!'.format(file_name)
    header = None
    line_idx = 1
    all_labels = []
    classes_list = []
    with codecs.open(file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        reader = csv.reader(fp, quotechar='"', delimiter=',')
        for row in reader:
            if len(row) > 0:
                if header is None:
                    if len(row) < 2:
                        raise ValueError('File `{0}`, line {1}: header must be consists of 2 columns, '
                                         'at least.'.format(file_name, line_idx))
                    if row[0] != 'id':
                        raise ValueError('File `{0}`, line {1}: first column of table must be an identifier, '
                                         'i.e. `id`.'.format(file_name, line_idx))
                    header = copy.copy(row)
                    classes_list = copy.copy(row[1:])
                    if len(classes_list) != len(set(classes_list)):
                        raise ValueError('File `{0}`, line {1}: names of classes are duplicated.'.format(
                            file_name, line_idx))
                    if 'normal' in classes_list:
                        raise ValueError('File `{0}`, line {1}: `normal` is inadmissible class name.'.format(
                            file_name, line_idx))
                    classes_list.append('normal')
                else:
                    if len(row) != len(header):
                        raise ValueError('File `{0}`, line {1}: this line does not correspond to the header.'.format(
                            file_name, line_idx))
                    if not all(map(lambda it: it in {'0', '1', '-1'}, row[1:])):
                        raise ValueError('File `{0}`, line {1}: all labels must be 0, 1 or -1.'.format(
                            file_name, line_idx))
                    new_label = set(filter(lambda idx: row[1 + idx] == '1', range(len(classes_list) - 1)))
                    if len(new_label) == 0:
                        if '-1' not in row[1:]:
                            new_label.add(classes_list.index('normal'))
                    all_labels.append(new_label)
    if len(all_texts) != len(all_labels):
        raise ValueError('Number of texts is not equal to number of labels! {0} != {1}'.format(
            len(all_texts), len(all_labels)))
    print('Size of source dataset for final testing is {0}.'.format(len(all_texts)))
    indices = list(filter(lambda idx: len(all_labels[idx]) > 0, range(len(all_labels))))
    print('Size of dataset for final testing after its filtering is {0}.'.format(len(indices)))
    return [all_texts[idx] for idx in indices], [all_labels[idx] for idx in indices], classes_list


def train_test_split(texts: List[tuple], labels: List[Set[int]], test_part: float=0.1,
                     random_seed: int=None) -> Tuple[List[tuple], List[Set[int]], List[tuple], List[Set[int]]]:
    indices = list(range(len(texts)))
    n = int(round(test_part * len(indices)))
    if n < 1:
        raise ValueError('{0} is too small value for the `test_part` argument!'.format(test_part))
    if n >= len(indices):
        raise ValueError('{0} is too large value for the `test_part` argument!'.format(test_part))
    random.seed(random_seed)
    random.shuffle(indices)
    train_classes = set()
    test_classes = set()
    for idx in indices[:n]:
        test_classes |= labels[idx]
    for idx in indices[n:]:
        train_classes |= labels[idx]
    if train_classes != test_classes:
        for repeat in range(10):
            random.shuffle(indices)
            train_classes = set()
            test_classes = set()
            for idx in indices[:n]:
                test_classes |= labels[idx]
            for idx in indices[n:]:
                train_classes |= labels[idx]
            if train_classes == test_classes:
                break
    if train_classes != test_classes:
        raise ValueError('Source dataset cannot be splitted!')
    return [texts[idx] for idx in indices[n:]], [labels[idx] for idx in indices[n:]], \
           [texts[idx] for idx in indices[:n]], [labels[idx] for idx in indices[:n]]


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
    glove_model_name = os.path.join(os.path.dirname(__file__), 'data', 'glove.6B.100d.txt')
    glove_archive_name = os.path.join(os.path.dirname(__file__), 'data', 'glove.6B.zip')
    if not os.path.isfile(glove_model_name):
        if not os.path.isfile(glove_archive_name):
            url = 'http://nlp.stanford.edu/data/wordvecs/glove.6B.zip'
            with open(glove_archive_name, 'wb') as f_out:
                ufr = requests.get(url)
                f_out.write(ufr.content)
        with zipfile.ZipFile(glove_archive_name, 'r') as f_in:
            f_in.extract(member='glove.6B.100d.txt', path=os.path.join(os.path.dirname(__file__), 'data'))
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
                if len(line_parts) == 101:
                    new_word = line_parts[0]
                    new_vector = np.reshape(np.array([float(cur) for cur in line_parts[1:]], dtype=np.float32),
                                            newshape=(1, 100))
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

    dictionary, vectors = load_glove()
    print('Vocabulary size is {0}.'.format(vectors.shape[0]))
    train_texts, train_labels, train_classes = load_trainset(dictionary)
    test_texts, test_labels, test_classes = load_testset(dictionary)
    assert train_classes == test_classes, 'Set of classes for training does not correspond to set of classes for ' \
                                          'testing!'
    train_texts, train_labels, valid_texts, valid_labels = train_test_split(train_texts, train_labels, 0.1, 42)
    lengths_of_texts = sorted([len(cur) for cur in train_texts])
    max_seq_length = lengths_of_texts[int(round(0.95 * (len(lengths_of_texts) - 1)))]
    print('Maximal token number in text is {0}.'.format(max_seq_length))
    print('')
    cls = WordSequenceClassifier(
        dictionary=dictionary, vectors=vectors, num_recurrent_units=str_to_layers(args.structure_of_layers),
        max_seq_length=max_seq_length, batch_size=args.batch_size, learning_rate=args.learning_rate, l2_reg=args.l2_reg,
        max_epochs=args.max_epochs, patience=5, clipnorm=10.0, gpu_memory_frac=0.9, multioutput=True, warm_start=False,
        verbose=True, random_seed=42
    )
    cls.fit(X=train_texts, y=train_labels, validation_data=(valid_texts, valid_labels))
    print('')
    probabilities = cls.predict(test_texts)
    y_pred = np.zeros((probabilities.shape[0], probabilities.shape[1] - 1), dtype=np.float64)
    y_true = np.zeros(y_pred.shape, dtype=np.int32)
    for sample_idx in range(len(test_labels)):
        if probabilities[sample_idx].argmax() < (len(test_classes) - 1):
            y_pred[sample_idx] = probabilities[sample_idx][0:(len(test_classes) - 1)]
        for class_idx in test_labels[sample_idx]:
            if class_idx != (len(test_classes) - 1):
                y_true[sample_idx][class_idx] = 1
    roc_auc = [roc_auc_score(y_true[:, class_idx], y_pred[:, class_idx]) for class_idx in range(len(test_classes) - 1)]
    print('Mean ROC-AUC score is {0:.9f}.'.format(np.mean(roc_auc)))
    print('By classes:')
    for class_idx in range(len(roc_auc)):
        print('  class `{0}`:'.format(test_classes[class_idx]))
        print('    ROC-AUC = {0:.9f};'.format(roc_auc[class_idx]))


if __name__ == '__main__':
    main()
