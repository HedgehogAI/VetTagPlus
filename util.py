# coding=utf-8

"""
Adapted from
https://github.com/facebookresearch/InferSent/blob/master/mutils.py
https://github.com/openai/finetune-transformer-lm/blob/master/text_utils.py
https://github.com/openai/finetune-transformer-lm/blob/master/utils.py
"""

import os
import inspect
from torch import optim
import re
import ftfy
import json
import time
import spacy
from tqdm import tqdm
import numpy as np

from preprocessing.cfg import EN_FIVE_DISCOURSE_MARKERS, EN_EIGHT_DISCOURSE_MARKERS, \
    EN_DISCOURSE_MARKERS, EN_OLD_FIVE_DISCOURSE_MARKERS, CH_FIVE_DISCOURSE_MARKERS, SP_FIVE_DISCOURSE_MARKERS


def get_labels(corpus):
    if corpus == "books_5":
        labels = EN_FIVE_DISCOURSE_MARKERS
    elif corpus == "books_old_5":
        labels = EN_OLD_FIVE_DISCOURSE_MARKERS
    elif corpus == "books_8":
        labels = EN_EIGHT_DISCOURSE_MARKERS
    elif corpus == "books_all" or corpus == "books_perfectly_balanced" or corpus == "books_mostly_balanced":
        labels = EN_DISCOURSE_MARKERS
    elif corpus == "gw_cn_5":
        labels = CH_FIVE_DISCOURSE_MARKERS
    elif corpus == "gw_es_5":
        labels = SP_FIVE_DISCOURSE_MARKERS
    elif corpus == "gw_es_1M_5":
        labels = SP_FIVE_DISCOURSE_MARKERS
    elif corpus == 'dat':
        labels = ['entail', 'contradict']
    else:
        raise Exception("corpus not found")

    return labels


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('…', '...')
    text = text.replace('´', "'")
    text = re.sub('''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
    text = re.sub('\s*\n\s*', ' \n ', text)
    text = re.sub('[^\S\n]+', ' ', text)
    return text.strip()


class TextEncoder(object):
    """
    mostly a wrapper for a public python bpe tokenizer
    """

    def __init__(self, encoder_path, bpe_path):
        self.nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])
        self.encoder = json.load(open(encoder_path))
        self.decoder = {v: k for k, v in self.encoder.items()}
        merges = open(bpe_path).read().split('\n')[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

    def bpe(self, token):
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token + '</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        if word == '\n  </w>':
            word = '\n</w>'
        self.cache[token] = word
        return word

    def encode(self, texts, verbose=True, lazy=False):
        # lazy: not using ftfy, SpaCy, or regex. DisSent is processed.
        texts_tokens = []
        if verbose:
            for text in tqdm(texts, ncols=80, leave=False):
                text = self.nlp(text_standardize(ftfy.fix_text(text))) if not lazy else text.split()
                text_tokens = []
                for token in text:
                    token_text = token.text if not lazy else token
                    text_tokens.extend([self.encoder.get(t, 0) for t in self.bpe(token_text.lower()).split(' ')])
                texts_tokens.append(text_tokens)
        else:
            for text in texts:
                text = self.nlp(text_standardize(ftfy.fix_text(text))) if not lazy else text.split()
                text_tokens = []
                for token in text:
                    token_text = token.text if not lazy else token
                    text_tokens.extend([self.encoder.get(t, 0) for t in self.bpe(token_text.lower()).split(' ')])
                texts_tokens.append(text_tokens)
        return texts_tokens


def np_softmax(x, t=1):
    x = x / t
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True)


def make_path(f):
    d = os.path.dirname(f)
    if d and not os.path.exists(d):
        os.makedirs(d)
    return f


class ResultLogger(object):
    def __init__(self, path, *args, **kwargs):
        if 'time' not in kwargs:
            kwargs['time'] = time.asctime()
        self.f_log = open(make_path(path), 'w')
        self.f_log.write(json.dumps(kwargs) + '\n')

    def log(self, **kwargs):
        if 'time' not in kwargs:
            kwargs['time'] = time.asctime()
        self.f_log.write(json.dumps(kwargs) + '\n')
        self.f_log.flush()

    def close(self):
        self.f_log.close()
