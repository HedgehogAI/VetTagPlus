import os
import numpy as np
import torch
from torch.autograd import Variable
import logging
from collections import defaultdict
from os.path import join as pjoin
from util import get_labels
import json


def embed_batch(batch, word_embeddings, ctx_embeddings, n_embeds=768):
    # comes out (bsize, max_len, word_dim)
    # original order is preserved
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((len(batch), max_len, n_embeds))

    for i in range(len(batch)):
        for j in range(len(batch[i])):
            embed[i, j, :] = word_embeddings[batch[i][j]] + ctx_embeddings[j]  # we sum them

    return torch.from_numpy(embed).float(), lengths


def pad_batch(batch, pad_id):
    # just build a numpy array that's padded

    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    padded_batch = np.full((len(batch), max_len), pad_id)  # fill in pad_id

    for i in range(len(batch)):
        for j in range(len(batch[i])):
            padded_batch[i, j] = batch[i][j]

    return padded_batch


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


# TODO: Maybe we need to tune dtype but right now it's fine...
def np_to_var(np_obj, gpu_id=-1, requires_grad=False):
    if gpu_id == -1:
        return Variable(torch.from_numpy(np_obj), requires_grad=requires_grad)
    else:
        return Variable(torch.from_numpy(np_obj), requires_grad=requires_grad).cuda(gpu_id)

def to_cuda(obj, gpu_id):
    if gpu_id == -1:
        return obj
    else:
        return obj.cuda(gpu_id)

def frequent_encoding(s, vocab_size=57235): # <TODO>
    all = []
    for item in s:
        one = [0 for _ in range(vocab_size)]
        for number in item:
            one[number] += 1
        all.append(one)
    return np.array(all, dtype=np.float32)


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, s1, label, metamap, pad_id, gpu_id=-1):
        # require everything passed in to be in Numpy!
        # also none of them is in GPU! we can use data here to pick out correct
        # last hidden states

        self.s1_lengths = (s1[:, :-1] != pad_id).sum(axis=1)
        self.s1 = np_to_var(s1[:, :-1], gpu_id)
        self.s1_y = np_to_var(s1[:, 1:], gpu_id)
        self.s1_mask = self.make_std_mask(self.s1, pad_id)
        # this is total number of tokens
        self.s1_ntokens = (self.s1_y != pad_id).data.sum()  # used for loss computing
        self.s1_loss_mask = to_cuda((self.s1_y != pad_id).type(torch.float), gpu_id)  # need to mask loss

        self.label = np_to_var(label, gpu_id)
        self.metamap = np_to_var(frequent_encoding(metamap), gpu_id) if len(metamap) != 0 else None

    @staticmethod
    def make_std_mask(tgt, pad_id):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad_id).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def list_to_map(dis_label):
    dis_map = {}
    for i, l in enumerate(dis_label):
        dis_map[l] = i
    return dis_map


def get_dis(data_dir, prefix, discourse_tag, cut_down_len, metamap=False):
    # we are not padding anything in here, this is just repeating
    s1 = {}
    target = {}
    meta = {}
    dis_map = list_to_map(get_labels(discourse_tag))
    logging.info(dis_map)

    for data_type in ['train', 'valid', 'test']:
        s1[data_type], meta[data_type], target[data_type] = [], [], []
        text_path = pjoin(data_dir, prefix + "_" + data_type + ".tsv")
        if not os.path.exists(text_path): continue
        with open(text_path, 'r') as f:
            for line in f:
                columns = line.strip().split('\t')
                if len(columns[0].split()) > cut_down_len: columns[0] = ' '.join([token for token in columns[0].split()[:cut_down_len]]) #<TODO>:remove
                s1[data_type].append(columns[0].lower()) # Lower
                if discourse_tag == 'csu' or discourse_tag == 'pp':
                    multi_label = np.zeros(len(dis_map), dtype='float32')
                    if len(columns) == 2:
                        for number in map(int, columns[1].split()):
                            multi_label[number] = 1
                    target[data_type].append(multi_label)
                else:
                    target[data_type].append(0) # No label
        if metamap:
            metamap_word2id = '/home/yuhuiz/Transformer/data/meta2id.json' # <TODO>
            metamap_path = pjoin(data_dir, prefix + "_metamap_" + data_type + ".tsv") # <TODO>
            meta2id = json.load(open(metamap_word2id))
            with open(metamap_path, 'r') as f:
                for line in f:
                    tokens = line.strip().split()
                    meta_id = []
                    for token in tokens:
                        if token in meta2id:
                            meta_id.append(meta2id[token])
                    meta[data_type].append(meta_id)
            assert len(s1[data_type]) == len(meta[data_type])
        assert len(s1[data_type]) == len(target[data_type])
        target[data_type] = np.array(target[data_type])
        
        logging.info('** {0} DATA : Found {1} pairs of {2} sentences.'.format(
            data_type.upper(), len(s1[data_type]), data_type))

    train = {'s1': s1['train'], 'label': target['train'], 'metamap': np.array(meta['train'])}
    dev = {'s1': s1['valid'], 'label': target['valid'], 'metamap': np.array(meta['valid'])}
    test = {'s1': s1['test'], 'label': target['test'], 'metamap': np.array(meta['test'])}
    return train, dev, test


def get_metamap(data_dir, prefix, meta2id):
    # we are not padding anything in here, this is just repeating
    s1 = {}
    meta2id = json.load(open(meta2id))
    for data_type in ['train', 'valid', 'test']:
        s1[data_type] = []
        text_path = pjoin(data_dir, prefix + "_" + data_type + ".tsv")
        if not os.path.exists(text_path): continue
        with open(text_path, 'r') as f:
            cnt = 0
            for line in f:
                cnt += 1
                if cnt % 1000 == 0: print cnt
                label = [0 for _ in range(len(meta2id))]
                tokens = line.strip().split()
                for token in tokens: 
                    if token in meta2id:
                        label[meta2id[token]] += 1
                label = np.array(label)
                s1[data_type].append(label)
        s1[data_type] = np.array(s1[data_type])
        logging.info('** {0} DATA : Found {1} pairs of {2} sentences.'.format(
            data_type.upper(), len(s1[data_type]), data_type))

    train = {'s1': s1['train']}
    dev = {'s1': s1['valid']}
    test = {'s1': s1['test']}
    return train, dev, test

if __name__ == '__main__':
    # train, dev, test = get_metamap('/home/yuhuiz/Transformer/data/csu/', 'csu_metamap', '/home/yuhuiz/Transformer/data/meta2id.json')
    # np.save('train.npy', train['s1'])
    # np.save('valid.npy', dev['s1'])
    # np.save('test.npy', test['s1'])
    train, dev, test = get_dis('/home/yuhuiz/Transformer/data/csu/', 'csu', 'csu', '600', 'csu_metamap', '/home/yuhuiz/Transformer/data/meta2id.json')
    print dev['meta'][0]
    print('ok')
    # np.save('trainY.npy', train['label'])
    # np.save('validY.npy', dev['label'])
    # np.save('testY.npy', test['label'])