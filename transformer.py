import sys
import copy
import torch.nn as nn
import torch
import math
import random
import time
import os
import json
from collections import defaultdict
import logging
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn import metrics
import numpy as np
from os.path import join as pjoin


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab_size, np_word_embedding=None, word_embedding_weight=None):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        # this has the advantage that we don't need an embedding matrix actually...
        # only need this one...
        if np_word_embedding is not None:
            self.proj.weight.data.copy_(torch.from_numpy(np_word_embedding))
            self.proj.weight.requires_grad = False
        if word_embedding_weight is not None:
            self.proj.weight = word_embedding_weight  # tied-weights

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, tgt_mask):
        for layer in self.layers:
            x = layer(x, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, tgt_mask):
        "Follow Figure 1 (right) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        return self.sublayer[1](x, self.feed_forward)


class DisSentT(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, config, decoder, tgt_embed, generator, projection_layer):
        super(DisSentT, self).__init__()
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.config = config

        self.classifier = nn.Sequential(
            nn.Linear(config['d_model'] * config['proj_head'], config['fc_dim']),
            nn.Linear(config['fc_dim'], config['fc_dim']),
            nn.Linear(config['fc_dim'], config['n_classes'])
        )

        self.projection_layer = projection_layer

        self.ce_loss = nn.CrossEntropyLoss(reduce=False)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.meta_loss = MetaLoss(config)
        self.cluster_loss = ClusterLoss(config)
        self.cooccur_loss = CoOccurenceLoss(config)

    def encode(self, tgt, tgt_mask):
        # tgt, tgt_mask need to be on CUDA before being put in here
        return self.decoder(self.tgt_embed(tgt), tgt_mask)

    def pick_h(self, h, lengths):
        # batch_size, lengths
        corr_h = []
        for i, j in enumerate(lengths):
            corr_h.append(h[i, j-1, :])
        corr_h = torch.stack(corr_h, dim=0)
        return corr_h

    def pick_mask(self, mask, lengths):
        corr_mask = []
        for i, j in enumerate(lengths):
            corr_mask.append(mask[i, j-1, :])
        corr_mask = torch.stack(corr_mask, dim=0)
        return corr_mask

    def forward(self, batch, clf=True, lm=True):
        "Take in and process masked src and target sequences."
        # this computes LM targets!! before the Generator
        u_h = self.encode(batch.s1, batch.s1_mask)

        if clf:
            # u_h, v_h: (batch_size, time_step, d_model) (which is n_embed)
            if self.config['pick_hid']:
                u = self.pick_h(u_h, batch.s1_lengths)
            else:
                u = u_h[:, -1, :] # last hidden state

            # self.project(u_h) -- u_h: (batch_size, time_step, d_model)
            # --> u = self.project(u_h), u: (batch_size, d_model * n_head) n_head = 4
            if self.config['proj_head'] != 1:
                picked_s1_mask = self.pick_mask(batch.s1_mask, batch.s1_lengths)
                u = self.projection_layer(u, u_h, u_h, picked_s1_mask)

            clf_output = self.classifier(u)
        
        # compute LM
        if lm:
            s1_y = self.generator(u_h)
        
        if clf and lm:
            return clf_output, s1_y
        elif clf:   
            return clf_output
        elif lm:
            return s1_y

    def compute_clf_loss(self, logits, labels):
        loss = self.bce_loss(logits, labels)
        if self.config['meta_param'] != 0.0:
            loss += self.meta_loss(logits, labels)
        elif self.config['cluster_param'] != [0.0, 0.0, 0.0]: 
            loss += self.cluster_loss(self.classifier[-1].weight, len(labels)) 
        elif self.config['cooccur_param'] != 0.0: 
            loss += self.cooccur_loss(self.classifier[-1].weight) 
        return loss

    def compute_lm_loss(self, s_h, s_y, s_loss_mask):
        # get the ingredients...compute loss
        seq_loss = self.ce_loss(s_h.contiguous().view(-1, self.config['n_words']),
                                s_y.view(-1)).view(s_h.size(0), -1)
        seq_loss *= s_loss_mask  # mask sequence loss
        return seq_loss.mean()


class LSTMEncoder(nn.Module):
    def __init__(self, config, decoder, tgt_embed, generator, projection_layer):
        super(LSTMEncoder, self).__init__()
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.config = config

        self.classifier = nn.Sequential(
            nn.Linear(config['d_model'] * config['proj_head'], config['fc_dim']),
            nn.Linear(config['fc_dim'], config['fc_dim']),
            nn.Linear(config['fc_dim'], config['n_classes'])
        )
        self.projection_layer = projection_layer

        self.ce_loss = nn.CrossEntropyLoss(reduce=False)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.meta_loss = MetaLoss(config)
        self.cluster_loss = ClusterLoss(config)
    
    def encode(self, tgt, lengths):
        return self.autolen_rnn(self.tgt_embed(tgt), lengths)
    
    def autolen_rnn(self, inputs, lengths):
        idx = np.argsort(-lengths)
        revidx = np.argsort(idx)
        packed_emb = nn.utils.rnn.pack_padded_sequence(inputs[idx, :, :], lengths[idx], batch_first=True)
        # self.encoder.flatten_parameters()
        output, (h, c) = self.decoder(packed_emb)
        output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]
        output = output[revidx, :, :]
        return output

    def pick_h(self, h, lengths):
        # batch_size, lengths
        corr_h = []
        for i, j in enumerate(lengths):
            corr_h.append(h[i, j-1, :])
        corr_h = torch.stack(corr_h, dim=0)
        return corr_h

    def pick_mask(self, mask, lengths):
        corr_mask = []
        for i, j in enumerate(lengths):
            corr_mask.append(mask[i, j-1, :])
        corr_mask = torch.stack(corr_mask, dim=0)
        return corr_mask
    
    def forward(self, batch, clf=True, lm=True):
        "Take in and process masked src and target sequences."
        # this computes LM targets!! before the Generator
        u_h = self.encode(batch.s1, batch.s1_lengths)
        if clf:
            if self.config['pick_hid']:
                u = self.pick_h(u_h, batch.s1_lengths)
            else:
                u = u_h[:, -1, :]
            if self.config['proj_head'] != 1:
                picked_s1_mask = self.pick_mask(batch.s1_mask, batch.s1_lengths)
                u = self.projection_layer(u, u_h, u_h, picked_s1_mask)
            clf_output = self.classifier(u)
        if lm:
            s1_y = self.generator(u_h)
        if clf and lm:
            return clf_output, s1_y
        elif clf:
            return clf_output
        elif lm:
            return s1_y
    
    def compute_clf_loss(self, logits, labels):
        loss = self.bce_loss(logits, labels).mean()
        if self.config['meta_param'] != 0.0:
            loss += self.meta_loss(logits, labels)
        if self.config['cluster_param'] != [0.0, 0.0, 0.0]: 
            loss += self.cluster_loss(self.classifier[-1].weight, len(labels)) # <TODO> softmax weight?
        return loss
    
    def compute_lm_loss(self, s_h, s_y, s_loss_mask):
        # get the ingredients...compute loss
        seq_loss = self.ce_loss(s_h.contiguous().view(-1, self.config['n_words']),
                                s_y.view(-1)).view(s_h.size(0), -1)
        seq_loss *= s_loss_mask  # mask sequence loss
        return seq_loss.mean()


class MetamapModel(nn.Module):
    def __init__(self, config):
        super(MetamapModel, self).__init__()
        self.classifier = nn.Linear(config['n_metamap'], config['n_classes'])
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, batch):
        clf_output = self.classifier(batch.metamap)
        return clf_output

    def compute_clf_loss(self, logits, labels):
        return self.bce_loss(logits, labels).mean()


def make_model(encoder, config, word_embeddings=None):
    # encoder: dictionary, for vocab
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(config['n_heads'], config['d_model'])
    attn_projection = MultiHeadedAttentionProjection(config['proj_head'], config['d_model'], config['proj_type'])
    ff = PositionwiseFeedForward(config['d_model'], config['d_ff'], config['dpout'])
    position = PositionalEncoding(config) # ctx_embeddings

    embedding_layer = Embeddings(encoder, config, word_embeddings)

    if config['tied']:
        if config['train_emb']:
            generator = Generator(config['d_model'], len(encoder), word_embedding_weight=embedding_layer.lut.weight)
        else:
            generator = Generator(config['d_model'], len(encoder), np_word_embedding=word_embeddings)
    else:
        generator = Generator(config['d_model'], len(encoder))

    model = DisSentT(
        config,
        Decoder(
            DecoderLayer(config['d_model'], c(attn), c(ff), config['dpout']),
            config['n_layers']),
        nn.Sequential(embedding_layer, c(position)),
        generator,
        attn_projection
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        # we won't update anything that has fixed parameters!
        if p.dim() > 1 and p.requires_grad is True:
            # if p.shape[0] == 50004: continue # <ZYH>: VERY UGLY WAY TO SOLVE BUG
            nn.init.xavier_uniform(p)
    return model


def make_lstm_model(encoder, config, word_embeddings=None): # , ctx_embeddings=None
    # encoder: dictionary, for vocab
    "Helper: Construct a model from hyperparameters."
    position = PositionalEncoding(config) # ctx_embeddings
    tgt_embed = nn.Sequential(Embeddings(encoder, config, word_embeddings), position)
    projection = MultiHeadedAttentionProjection(config['proj_head'], config['d_model'], config['proj_type'])

    decoder = nn.LSTM(
        config['d_model'], # config.emb_dim
        config['d_model'], # config.hidden_size
        1,
        batch_first=True
    )
    if config['tied']:
        if config['train_emb']:
            generator = Generator(config['d_model'], len(encoder), word_embedding_weight=tgt_embed[0].lut.weight)
        else:
            generator = Generator(config['d_model'], len(encoder), np_word_embedding=word_embeddings)
    else:
        generator = Generator(config['d_model'], len(encoder))
        
    model = LSTMEncoder(
        config,
        decoder,
        tgt_embed,
        generator,
        projection
    )
    logging.info(model.tgt_embed[0].lut.weight.data.norm())
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        # we won't update anything that has fixed parameters!
        # if p.shape[0] == 50004: continue # <ZYH>: VERY UGLY WAY TO SOLVE BUG
        if p.dim() > 1 and p.requires_grad is True:
            nn.init.xavier_uniform(p)
    logging.info(model.tgt_embed[0].lut.weight.data.norm())
    return model


def make_metamap_model(config):
    model = MetamapModel(config)
    return model


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.-
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class MultiHeadedAttentionProjection(nn.Module):
    def __init__(self, h, d_model, proj_type=1, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttentionProjection, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_model = d_model
        self.d_k = d_model // h
        self.h = h
        if proj_type == 1:
            self.linear = nn.Linear(d_model, d_model * h)
            self.linear_out = nn.Linear(d_model * h, d_model * h)
        elif proj_type == 2:
            self.linear = nn.Linear(d_model, d_model)
            self.linear_out = nn.Linear(d_model, d_model * h)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.proj_type = proj_type

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        nbatches = query.size(0)
        if mask is not None:
            # Same mask applied to all h heads.-
            mask = mask.view(nbatches, 1, 1, -1)
        nd = self.d_model if self.proj_type == 1 else self.d_k
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.linear(query).view(nbatches, -1, self.h, nd).transpose(1, 2)
        key = key.repeat(1, 1, self.h).view(nbatches, -1, self.h, nd).transpose(1, 2)
        value = value.repeat(1, 1, self.h).view(nbatches, -1, self.h, nd).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, self.h * nd)
        return self.linear_out(x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # can consider changing this non-linearity!
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# we will learn and predict via our own embeddings
class Embeddings(nn.Module):
    def __init__(self, encoder, config, word_embeddings=None):
        # encoder is the dictionary, not text_encoder
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(len(encoder), config['d_model'])
        self.d_model = config['d_model']
        if config['init_emb']:
            assert word_embeddings is not None
            logging.info('copy embeddings...')
            logging.info('2-norm %f' % (np.linalg.norm(word_embeddings)))
            self.lut.weight.data.copy_(torch.from_numpy(word_embeddings))
        if not config['train_emb']:
            self.lut.weight.requires_grad = False

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, config, max_len=5000):
        # ctx_embeddings: (max_len, n_embed)
        # we don't need to define new, just use the same...

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=config['dpout'])

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, config['d_model'])
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config['d_model'], 2) *
                             -(math.log(10000.0) / config['d_model']))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = torch.from_numpy(ctx_embeddings)
        pe = pe.unsqueeze(0)  # add one dimension to beginning (1, time, n_embed)
        self.register_buffer('pe', pe)  # this will add pe to self

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


class MetaLoss(nn.Module):
    def __init__(self, config, cluster_path='./data/csu/snomed_label_to_meta_grouping.json',
                 label_to_meta_map_path='./data/csu/snomed_label_to_meta_map.json'):
        super(MetaLoss, self).__init__()

        with open(cluster_path, 'rb') as f:
            self.label_grouping = json.load(f)

        with open(label_to_meta_map_path, 'rb') as f:
            self.meta_label_mapping = json.load(f)

        self.meta_label_size = len(self.label_grouping)
        self.config = config

        # your original classifier did this wrong...found a bug
        self.bce_loss = nn.BCELoss()  # this takes in probability (after sigmoid)

    # now that this becomes somewhat independent...maybe you can examine this more closely?
    def generate_meta_y(self, indices, meta_label_size, batch_size):
        a = np.array([[0.] * meta_label_size for _ in range(batch_size)], dtype=np.float32)
        matched = defaultdict(set)
        for b, l in indices:
            if b not in matched:
                a[b, self.meta_label_mapping[str(l)]] = 1.
                matched[b].add(self.meta_label_mapping[str(l)])
            elif self.meta_label_mapping[str(l)] not in matched[b]:
                a[b, self.meta_label_mapping[str(l)]] = 1.
                matched[b].add(self.meta_label_mapping[str(l)])
        assert np.sum(a <= 1) == a.size
        return a

    def forward(self, logits, true_y):
        batch_size = logits.size(0)
        y_hat = torch.sigmoid(logits)
        meta_probs = []
        for i in range(self.meta_label_size):
            # 1 - (1 - p_1)(...)(1 - p_n)
            meta_prob = (1 - y_hat[:, self.label_grouping[str(i)]]).prod(1)
            meta_probs.append(meta_prob)  # in this version we don't do threshold....(originally we did)

        meta_probs = torch.stack(meta_probs, dim=1)
        assert meta_probs.size(1) == self.meta_label_size

        # generate meta-label
        y_indices = true_y.nonzero()
        meta_y = self.generate_meta_y(y_indices.data.cpu().numpy().tolist(), self.meta_label_size,
                                      batch_size)
        meta_y = Variable(torch.from_numpy(meta_y)).cuda()

        meta_loss = self.bce_loss(meta_probs, meta_y) * self.config['meta_param']
        return meta_loss


# compute loss
class ClusterLoss(nn.Module):
    def __init__(self, config, cluster_path='./data/csu/snomed_label_to_meta_grouping.json'):
        super(ClusterLoss, self).__init__()

        with open(cluster_path, 'rb') as f:
            label_grouping = json.load(f)

        self.meta_category_groups = label_grouping.values()
        self.config = config

    def forward(self, softmax_weight, batch_size):
        w_bar = softmax_weight.sum(1) / self.config['n_classes']  # w_bar

        omega_mean = softmax_weight.pow(2).sum()
        omega_between = 0.
        omega_within = 0.

        for c in xrange(len(self.meta_category_groups)):
            m_c = len(self.meta_category_groups[c])
            w_c_bar = softmax_weight[:, self.meta_category_groups[c]].sum(1) / m_c
            omega_between += m_c * (w_c_bar - w_bar).pow(2).sum()
            for i in self.meta_category_groups[c]:
                # this value will be 0 for singleton group
                omega_within += (softmax_weight[:, i] - w_c_bar).pow(2).sum()

        aux_loss = omega_mean * self.config['cluster_param'][0] + (omega_between * self.config['cluster_param'][1] +
                                                       omega_within * self.config['cluster_param'][2]) / batch_size

        return aux_loss


def log_of_array_ignoring_zeros(M):
    """Returns an array containing the logs of the nonzero
    elements of M. Zeros are left alone since log(0) isn't
    defined.
    """
    log_M = M.copy()
    mask = log_M > 0
    log_M[mask] = np.log(log_M[mask])
    return log_M


def observed_over_expected(df):
    col_totals = df.sum(axis=0)
    total = col_totals.sum()
    row_totals = df.sum(axis=1)
    expected = np.outer(row_totals, col_totals) / total
    oe = df / expected
    return oe


def pmi(df, positive=True):
    df = observed_over_expected(df)
    # Silence distracting warnings about log(0):
    with np.errstate(divide='ignore'):
        df = np.log(df)
    df[np.isnan(df)] = 0.0  # log(0) = 0
    df[np.isinf(df)] = 0.0
    if positive:
        df[df < 0] = 0.0
    return df


class CoOccurenceLoss(nn.Module):
    def __init__(self, config,
                 use_csu=True,
                 glove=False,
                 x_max=100,
                 alpha=0.75,
                 ppmi=False,
                 csu_path='./data/csu/label_co_matrix.npy',
                 pp_path='./data/csu/pp_combined_label_co_matrix.npy',
                 device=-1):
        super(CoOccurenceLoss, self).__init__()
        self.co_mat_path = csu_path if use_csu else pp_path
        self.co_mat = np.load(self.co_mat_path)
        self.X = self.co_mat
        self.glove = glove

        logging.info("using co_matrix {}".format(self.co_mat_path))
        self.n = config['fc_dim']  # N-dim rep
        self.m = config['n_classes']

        if self.glove:
            self.C = torch.empty(self.m, self.n)
            self.C = Variable(self.C.uniform_(-0.5, 0.5)).cuda(device)
            self.B = torch.empty(2, self.m)
            self.B = Variable(self.B.uniform_(-0.5, 0.5)).cuda(device)

            self.indices = list(range(self.m))  # label_size

            # Precomputable GloVe values:
            self.X_log = log_of_array_ignoring_zeros(self.X)
            self.X_weights = (np.minimum(self.X, xmax) / xmax) ** alpha  # eq. (9)

            # iterate on the upper triangular matrix, off-diagonal
            self.iu1 = np.triu_indices(41, 1)  # 820 iterations
        else:

            self.X = pmi(self.X, positive=ppmi)
            self.X_mask = (self.X != 0.).astype(np.float32)
            iu1 = np.triu_indices(42, 1) # <ZYH> (42, 1)
            self.X_mask[iu1] = 0.

            self.X = Variable(torch.FloatTensor(self.X), requires_grad=False).cuda(device)
            # self.X = torch.clamp(self.X, max=4.)  # this is data specific...
            self.X_mask = Variable(torch.FloatTensor(self.X_mask), requires_grad=False).cuda(device)
            self.X_final = self.X * self.X_mask

            self.mse = nn.MSELoss()

    def forward(self, softmax_weight):
        # this computes a straight-through pass of the GloVE objective
        # similar to "Auxiliary" training
        # return the loss
        # softmax_weight: [d, |Y|]
        if self.glove:
            loss = 0.
            for i, j in zip(self.iu1[0], self.iu1[1]):
                if self.X[i, j] > 0.0:
                    # Cost is J' based on eq. (8) in the paper:
                    # (1, |Y|) dot (1, |Y|)
                    diff = softmax_weight[:, i].dot(self.C[j]) + self.B[0, i] + self.B[1, j] - self.X_log[i, j]
                    loss += self.X_weights[i, j] * diff   # f(X_ij) * (w_i w_j + b_i + b_j - log X_ij)
                    # this is the summation, not average
        else:
            # softmax_weight: (d, m)
            # (m, d) (d, m)
            a = torch.matmul(softmax_weight, torch.transpose(softmax_weight, 1, 0))
            a = a * self.X_mask
            
            loss = self.mse(a, self.X_final) # <ZYH> mask
        return loss
            