import codecs
import os
import requests
import sys
from typing import Dict
import unittest
import zipfile


try:
    from sequence_classifier.utils import load_testset_for_toxic_comments_2017
    from sequence_classifier.utils import load_trainset_for_toxic_comments_2017
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from sequence_classifier.utils import load_testset_for_toxic_comments_2017
    from sequence_classifier.utils import load_trainset_for_toxic_comments_2017


class TestUtils(unittest.TestCase):
    def test_load_trainset_for_toxic_comments_2017(self):
        file_name = os.path.join(os.path.dirname(__file__), 'testdata', 'toxic_comments_2017_train.csv')
        loaded_texts, loaded_labels, loaded_classes_list = load_trainset_for_toxic_comments_2017(
            file_name)
        true_texts = [
            ('explanation', 'why', 'the', 'edits', 'made', 'under', 'my', 'username', 'hardcore', 'metallica', 'fan',
             'were', 'reverted', '?', 'they', 'weren', '\'', 't', 'vandalisms', ',', 'just', 'closure', 'on', 'some',
             'gas', 'after', 'i', 'voted', 'at', 'new', 'york', 'dolls', 'fac', '.', 'and', 'please', 'don', '\'', 't',
             'remove', 'the', 'template', 'from', 'the', 'talk', 'page', 'since', 'i', '\'', 'm', 'retired', 'now',
             '.', '89.205.38.27'),
            ('d', '\'', 'aww', '!', 'he', 'matches', 'this', 'background', 'colour', 'i', '\'', 'm', 'seemingly',
             'stuck', 'with', '.', 'thanks', '.', '(', 'talk', ')', '21:51,', 'january', '11,', '2016', '(', 'utc',
             ')'),
            ('hey', 'man', ',', 'i', '\'', 'm', 'really', 'not', 'trying', 'to', 'edit', 'war', '.', 'it', '\'', 's',
             'just', 'that', 'this', 'guy', 'is', 'constantly', 'removing', 'relevant', 'information', 'and', 'talking',
             'to', 'me', 'through', 'edits', 'instead', 'of', 'my', 'talk', 'page', '.', 'he', 'seems', 'to', 'care',
             'more', 'about', 'the', 'formatting', 'than', 'the', 'actual', 'info', '.'),
            ('"', 'more', 'i', 'can', '\'', 't', 'make', 'any', 'real', 'suggestions', 'on', 'improvement', '-', 'i',
             'wondered', 'if', 'the', 'section', 'statistics', 'should', 'be', 'later', 'on', ',', 'or', 'a',
             'subsection', 'of', '"', '"', 'types', 'of', 'accidents', '"', '"', '-', 'i', 'think', 'the', 'references',
             'may', 'need', 'tidying', 'so', 'that', 'they', 'are', 'all', 'in', 'the', 'exact', 'same', 'format', 'ie',
             'date', 'format', 'etc', '.', 'i', 'can', 'do', 'that', 'later', 'on', ',', 'if', 'no', '-', 'one', 'else',
             'does', 'first', '-', 'if', 'you', 'have', 'any', 'preferences', 'for', 'formatting', 'style', 'on',
             'references', 'or', 'want', 'to', 'do', 'it', 'yourself', 'please', 'let', 'me', 'know', '.', 'there',
             'appears', 'to', 'be', 'a', 'backlog', 'on', 'articles', 'for', 'review', 'so', 'i', 'guess', 'there',
             'may', 'be', 'a', 'delay', 'until', 'a', 'reviewer', 'turns', 'up', '.', 'it', '\'', 's', 'listed', 'in',
             'the', 'relevant', 'form', 'eg', 'wikipedia', ':', 'good', '_', 'article', '_', 'nominations', '#',
             'transport', '"'),
            ('cocksucker', 'before', 'you', 'piss', 'around', 'on', 'my', 'work'),
            ('hey', '.', '.', '.', 'what', 'is', 'it', '.', '.', '@', '|', 'talk', '.', 'what', 'is', 'it', '.', '.',
             '.', 'an', 'exclusive', 'group', 'of', 'some', 'wp', 'talibans', '.', '.', '.', 'who', 'are', 'good', 'at',
             'destroying', ',', 'self', '-', 'appointed', 'purist', 'who', 'gang', 'up', 'any', 'one', 'who', 'asks',
             'them', 'questions', 'abt', 'their', 'anti', '-', 'social', 'and', 'destructive', '(', 'non', ')', '-',
             'contribution', 'at', 'wp', '?', 'ask', 'sityush', 'to', 'clean', 'up', 'his', 'behavior', 'than', 'issue',
             'me', 'nonsensical', 'warnings', '.', '.', '.'),
            ('fuck', 'your', 'filthy', 'mother', 'in', 'the', 'ass', ',', 'dry', '!'),
            ('get', 'fucked', 'up', '.', 'get', 'fuckeeed', 'up', '.', 'got', 'a', 'drink', 'that', 'you', 'cant',
             'put', 'down', '?', '?', '?', '/', 'get', 'fuck', 'up', 'get', 'fucked', 'up', '.', 'i', '\'', 'm',
             'fucked', 'up', 'right', 'now', '!'),
            ('are', 'you', 'threatening', 'me', 'for', 'disputing', 'neutrality', '?', 'i', 'know', 'in', 'your',
             'country', 'it', '\'', 's', 'quite', 'common', 'to', 'bully', 'your', 'way', 'through', 'a', 'discussion',
             'and', 'push', 'outcomes', 'you', 'want', '.', 'but', 'this', 'is', 'not', 'russia', '.'),
            ('stupid', 'peace', 'of', 'shit', 'stop', 'deleting', 'my', 'stuff', 'asshole', 'go', 'die', 'and', 'fall',
             'in', 'a', 'hole', 'go', 'to', 'hell', '!'),
            ('=', 'tony', 'sidaway', 'is', 'obviously', 'a', 'fistfuckee', '.', 'he', 'loves', 'an', 'arm', 'up', 'his',
             'ass', '.'),
            ('locking', 'this', 'page', 'would', 'also', 'violate', 'wp', ':', 'newbies', '.', 'whether', 'you', 'like',
             'it', 'or', 'not', ',', 'conservatives', 'are', 'wikipedians', 'too', '.'),
            ('a', 'bisexual', ',', 'like', 'a', 'homosexual', 'or', 'a', 'heterosexual', ',', 'is', 'not', 'defined',
             'by', 'sexual', 'activity', '.', '(', 'much', 'like', 'a', '15', 'year', 'old', 'boy', 'who', 'is',
             'attracted', 'to', 'a', 'girl', 'sexually', 'but', 'has', 'never', 'had', 'sex', 'is', 'still', 'straight',
             ')', '.', 'a', 'person', 'who', 'is', 'actually', 'sexually', 'attracted', '/', 'aroused', 'by', 'the',
             'same', 'sex', 'as', 'well', 'as', 'the', 'opposite', 'sex', 'is', 'bisexual', '.'),
            ('redirect', 'talk', ':', 'frank', 'herbert', 'mason'),
            ('a', 'pair', 'of', 'jew', '-', 'hating', 'weiner', 'nazi', 'schmucks', '.'),
            ('wouldn', '\'', 't', 'be', 'the', 'first', 'time', 'bitch', '.', 'fuck', 'you', 'i', '\'', 'll', 'find',
             'out', 'where', 'you', 'live', ',', 'sodomize', 'your', 'wife', 'and', 'then', 'burn', 'your', 'house',
             'down', '.', 'fuck', 'you', 'you', 'fucking', 'queer', '.'),
            ('please', 'stop', '.', 'if', 'you', 'continue', 'to', 'ignore', 'our', 'policies', 'by', 'introducing',
             'inappropriate', 'pages', 'to', 'wikipedia', ',', 'you', 'will', 'be', 'blocked', '.')
        ]
        true_labels = [
            set(),
            set(),
            set(),
            set(),
            {0, 1, 2, 4},
            {0},
            {0, 2, 4},
            {0, 2},
            set(),
            {0, 1, 2, 4},
            {0, 2, 4},
            set(),
            set(),
            set(),
            {0, 2, 4, 5},
            {0, 1, 2, 3, 4, 5},
            {3}
        ]
        true_classes_list = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        self.assertIsInstance(loaded_texts, list)
        self.assertIsInstance(loaded_labels, list)
        self.assertIsInstance(loaded_classes_list, list)
        self.assertEqual(true_texts, loaded_texts)
        self.assertEqual(true_labels, loaded_labels)
        self.assertEqual(true_classes_list, loaded_classes_list)

    def test_load_testset_for_toxic_comments_2017(self):
        texts_file_name = os.path.join(os.path.dirname(__file__), 'testdata', 'toxic_comments_2017_test.csv')
        labels_file_name = os.path.join(os.path.dirname(__file__), 'testdata', 'toxic_comments_2017_test_labels.csv')
        loaded_texts, loaded_labels, loaded_classes_list = load_testset_for_toxic_comments_2017(
            texts_file_name, labels_file_name)
        true_texts = [
            ('thank', 'you', 'for', 'understanding', '.', 'i', 'think', 'very', 'highly', 'of', 'you', 'and', 'would',
             'not', 'revert', 'without', 'discussion', '.'),
            ('=', '=', 'arabs', 'are', 'committing', 'genocide', 'in', 'iraq', ',', 'but', 'no', 'protests', 'in',
             'europe', '.', '=', '=', 'may', 'europe', 'also', 'burn', 'in', 'hell', '.'),
            ('please', 'stop', '.', 'if', 'you', 'continue', 'to', 'vandalize', 'wikipedia', ',', 'as', 'you', 'did',
             'to', 'homosexuality', ',', 'you', 'will', 'be', 'blocked', 'from', 'editing', '.'),
            ('but', 'this', 'is', 'not', 'the', 'article', 'about', 'government', 'position', 'but', 'about', 'the',
             'reaction', '.', 'add', 'positions', 'to', '2008', 'kosovo', 'declaration', 'of', 'independence', 'or',
             'foreign', 'relations', 'of', 'kosovo', '.'),
            ('dj', 'robinson', 'is', 'gay', 'as', 'hell', '!', 'he', 'sucks', 'his', 'dick', 'so', 'much', '!', '!',
             '!', '!', '!'),
            ('how', 'dare', 'you', 'vandalize', 'that', 'page', 'about', 'the', 'hms', 'beagle', '!', 'don', '\'', 't',
             'vandalize', 'again', ',', 'demon', '!'),
            (':', ':', ':', ':', 'well', ',', 'it', 'sucks', 'to', 'have', 'a', 'university', 'to', 'be',
             'nicknameless', '.', 'and', 'it', '\'', 's', 'the', 'first', 'time', 'in', 'ncaa', 'history', 'that', 'it',
             'has', 'happened', '.', '/'),
            ('go', 'to', 'court', '?', 'shit', ',', 'cause', 'the', 'judge', 'look', 'just', 'like', 'that',
             'motherfucker', 'put', 'our', 'ass', 'on', 'the', 'boat', 'and', 'made', 'me', 'a', 'nigga', '[', 'outro',
             ']', 'mamma', 'rag', 'is', 'a', 'nigga', 'du', '-', 'rag', 'is', 'a', 'nigga', 'ballistics', 'is', 'a',
             'nigga', 'and', 'i', '\'', 'm', 'uretha', '\'', 's', 'nigga', 'nigga'),
            (':', 'i', 'second', 'the', 'motion', '.'),
            ('im', 'gonna', 'kill', 'u', 'and', 'ur', 'family', '!', 'peace', 'out', 'd')
        ]
        true_labels = [
            set(),
            {0},
            set(),
            set(),
            {0, 2, 4, 5},
            {0},
            {0, 2},
            {0, 1, 2, 4, 5},
            set(),
            {0, 1, 2, 3}
        ]
        true_classes_list = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        self.assertIsInstance(loaded_texts, list)
        self.assertIsInstance(loaded_labels, list)
        self.assertIsInstance(loaded_classes_list, list)
        self.assertEqual(true_texts, loaded_texts)
        self.assertEqual(true_labels, loaded_labels)
        self.assertEqual(true_classes_list, loaded_classes_list)


if __name__ == '__main__':
    unittest.main(verbosity=2)
