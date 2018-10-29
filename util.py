#coding=utf-8
import os
import inspect
from torch import optim
import re
import json
import time
from tqdm import tqdm
import numpy as np


def get_labels(corpus):
    if corpus == "csu" or corpus == "pp":
        labels = ["Obesity (disorder)", "Hyperproteinemia (disorder)", "Angioedema and/or urticaria (disorder)", "Nutritional deficiency associated condition (disorder)", "Spontaneous hemorrhage (disorder)", "Hereditary disease (disorder)", "Disorder of fetus or newborn (disorder)", "Disorder of labor / delivery (disorder)", "Disorder caused by exposure to ionizing radiation (disorder)", "Disorder of pregnancy (disorder)", "Disorder of pigmentation (disorder)", "Nutritional disorder (disorder)", "Disease caused by Arthropod (disorder)", "Disease caused by parasite (disorder)", "Mental disorder (disorder)", "Vomiting (disorder)", "Poisoning (disorder)", "Disorder of immune function (disorder)", "Anemia (disorder)", "Autoimmune disease (disorder)", "Disorder of hemostatic system (disorder)", "Disorder of cellular component of blood (disorder)", "Congenital disease (disorder)", "Propensity to adverse reactions (disorder)", "Metabolic disease (disorder)", "Disorder of auditory system (disorder)", "Hypersensitivity condition (disorder)", "Disorder of endocrine system (disorder)", "Disorder of hematopoietic cell proliferation (disorder)", "Disorder of nervous system (disorder)", "Disorder of cardiovascular system (disorder)", "Disorder of the genitourinary system (disorder)", "Traumatic AND/OR non-traumatic injury (disorder)", "Visual system disorder (disorder)", "Infectious disease (disorder)", "Disorder of respiratory system (disorder)", "Disorder of connective tissue (disorder)", "Disorder of musculoskeletal system (disorder)", "Disorder of integument (disorder)", "Disorder of digestive system (disorder)", "Neoplasm and/or hamartoma (disorder)", "Clinical finding (finding)"]
    elif corpus == "sage":
        labels = ['NO_LABEL']
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


def build_vocab(sents, specials):
    word2id = {}
    id2word = {}
    for sent in sents:
        words = sent.strip().split()
        for word in words:
            if word not in word2id:
                word2id[word] = len(word2id)
                id2word[len(id2word)] = word
    for word in specials:
        word2id[word] = len(word2id)
        id2word[len(id2word)] = word
    return word2id, id2word


def batchify(data, bsz):
    nbatch = data.shape[0] // bsz 
    data = data[:nbatch*bsz].reshape(bsz, nbatch)
    return data   


def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


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
