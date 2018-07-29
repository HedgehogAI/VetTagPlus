import os
import numpy as np
import torch
from torch.autograd import Variable
import logging
from collections import defaultdict
from os.path import join as pjoin
from util import get_labels


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

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, s1, label, pad_id, gpu_id=-1):
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


def get_dis(data_dir, prefix, discourse_tag, cut_down_len):
    # we are not padding anything in here, this is just repeating
    s1 = {}
    target = {}
    dis_map = list_to_map(get_labels(discourse_tag))
    logging.info(dis_map)

    for data_type in ['train', 'valid', 'test']:
        s1[data_type], target[data_type] = [], []
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
        assert len(s1[data_type]) == len(target[data_type])
        target[data_type] = np.array(target[data_type])
        logging.info('** {0} DATA : Found {1} pairs of {2} sentences.'.format(
            data_type.upper(), len(s1[data_type]), data_type))

    train = {'s1': s1['train'], 'label': target['train']}
    dev = {'s1': s1['valid'], 'label': target['valid']}
    test = {'s1': s1['test'], 'label': target['test']}
    return train, dev, test

