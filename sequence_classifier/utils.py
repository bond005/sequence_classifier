import codecs
import copy
import csv
import os
import random
from typing import Dict, List, Set, Tuple

from nltk.tokenize.nist import NISTTokenizer


def load_trainset_for_toxic_comments_2017(file_name: str) -> \
        Tuple[List[tuple], List[Set[int]], List[str]]:
    tokenizer = NISTTokenizer()
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
                else:
                    if len(row) != len(header):
                        raise ValueError('File `{0}`, line {1}: this line does not correspond to the header.'.format(
                            file_name, line_idx))
                    new_text = ' '.join(
                        list(filter(lambda it2: len(it2) > 0, map(lambda it1: it1.strip(), row[1].split())))
                    )
                    all_texts.append(
                        tuple(filter(
                            lambda it2: len(it2) > 0,
                            map(lambda it1: it1.strip().lower(), tokenizer.international_tokenize(text=new_text))
                        ))
                    )
                    if not all(map(lambda it: it in {'0', '1'}, row[2:])):
                        raise ValueError('File `{0}`, line {1}: all labels must be 0 or 1.'.format(file_name, line_idx))
                    new_label = set(filter(lambda idx: row[2 + idx] == '1', range(len(classes_list))))
                    all_labels.append(new_label)
            line_idx += 1
    print('Size of the training set is {0}.'.format(len(all_texts)))
    return all_texts, all_labels, classes_list


def load_testset_for_toxic_comments_2017(texts_file_name: str, labels_file_name: str) -> \
        Tuple[List[tuple], List[Set[int]], List[str]]:
    tokenizer = NISTTokenizer()
    header = None
    line_idx = 1
    all_texts = []
    with codecs.open(texts_file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        reader = csv.reader(fp, quotechar='"', delimiter=',')
        for row in reader:
            if len(row) > 0:
                if header is None:
                    if len(row) != 2:
                        raise ValueError('File `{0}`, line {1}: header must be consists of 2 columns.'.format(
                            texts_file_name, line_idx))
                    if row[0] != 'id':
                        raise ValueError('File `{0}`, line {1}: first column of table must be an identifier, '
                                         'i.e. `id`.'.format(texts_file_name, line_idx))
                    if row[1] != 'comment_text':
                        raise ValueError('File `{0}`, line {1}: second column of table must be a source text, '
                                         'i.e. `comment_text`.'.format(texts_file_name, line_idx))
                    header = copy.copy(row)
                else:
                    if len(row) != len(header):
                        raise ValueError('File `{0}`, line {1}: this line does not correspond to the header.'.format(
                            texts_file_name, line_idx))
                    new_text = ' '.join(
                        list(filter(lambda it2: len(it2) > 0, map(lambda it1: it1.strip(), row[1].split())))
                    )
                    all_texts.append(
                        tuple(filter(
                            lambda it2: len(it2) > 0,
                            map(lambda it1: it1.strip().lower(), tokenizer.international_tokenize(text=new_text))
                        ))
                    )
            line_idx += 1
    del header
    header = None
    line_idx = 1
    all_labels = []
    classes_list = []
    indices = []
    with codecs.open(labels_file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        reader = csv.reader(fp, quotechar='"', delimiter=',')
        for row in reader:
            if len(row) > 0:
                if header is None:
                    if len(row) < 2:
                        raise ValueError('File `{0}`, line {1}: header must be consists of 2 columns, '
                                         'at least.'.format(labels_file_name, line_idx))
                    if row[0] != 'id':
                        raise ValueError('File `{0}`, line {1}: first column of table must be an identifier, '
                                         'i.e. `id`.'.format(labels_file_name, line_idx))
                    header = copy.copy(row)
                    classes_list = copy.copy(row[1:])
                    if len(classes_list) != len(set(classes_list)):
                        raise ValueError('File `{0}`, line {1}: names of classes are duplicated.'.format(
                            labels_file_name, line_idx))
                else:
                    if len(row) != len(header):
                        raise ValueError('File `{0}`, line {1}: this line does not correspond to the header.'.format(
                            labels_file_name, line_idx))
                    if not all(map(lambda it: it in {'0', '1', '-1'}, row[1:])):
                        raise ValueError('File `{0}`, line {1}: all labels must be 0, 1 or -1.'.format(
                            labels_file_name, line_idx))
                    new_label = set(filter(lambda idx: row[1 + idx] == '1', range(len(classes_list))))
                    if len(new_label) == 0:
                        if '-1' not in row[1:]:
                            indices.append(len(all_labels))
                    else:
                        indices.append(len(all_labels))
                    all_labels.append(new_label)
    if len(all_texts) != len(all_labels):
        raise ValueError('Number of texts is not equal to number of labels! {0} != {1}'.format(
            len(all_texts), len(all_labels)))
    print('Size of source dataset for final testing is {0}.'.format(len(all_texts)))
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


def print_classes_distribution(labels: List[Set[int]], all_classes: List[str]):
    distr = [0 for _ in range(len(all_classes))]
    for cur_label in labels:
        for class_idx in cur_label:
            distr[class_idx] += 1
    max_freq_width = max([len(str(distr[class_idx])) for class_idx in range(len(all_classes))])
    max_name_width = max([len(all_classes[class_idx]) for class_idx in range(len(all_classes))])
    for class_idx in range(len(all_classes)):
        print('{0:>{1}} {2:>{3}}'.format(all_classes[class_idx], max_name_width, distr[class_idx], max_freq_width))