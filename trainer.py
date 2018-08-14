import os
import sys
import csv
import time
import json
import argparse
from os.path import join as pjoin
from itertools import izip
import numpy as np
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
from data import get_dis, pad_batch, Batch
from util import get_labels, TextEncoder, batchify
from transformer import NoamOpt, make_model, make_lstm_model, make_metamap_model
import logging
from sklearn import metrics


parser = argparse.ArgumentParser(description='Clinical Dataset')
# paths
parser.add_argument("--corpus", type=str, default='sage', help="sage|csu|pp")
parser.add_argument("--hypes", type=str, default='hypes/default.json', help="load in a hyperparameter file")
parser.add_argument("--outputdir", type=str, default='exp/', help="Output directory")
parser.add_argument("--inputdir", type=str, default='', help="Input model dir")
parser.add_argument("--outputmodelname", type=str, default='model')
parser.add_argument("--cut_down_len", type=int, default="2147483647", help="sentence will be cut down if tokens num greater than this")
# training
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--cur_epochs", type=int, default=1)
parser.add_argument("--cur_valid", type=float, default=-1e10, help="must set this otherwise resumed model will be saved by default")
parser.add_argument("--bptt_size", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--dpout", type=float, default=0.1, help="residual, embedding, attention dropout") # 3 dropouts
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--maxlr", type=float, default=2.5e-4, help="this is not used...")
parser.add_argument("--warmup_steps", type=int, default=8000, help="OpenNMT uses steps") # TransformerLM uses 0.2% of training data as warmup step, that's 5785 for DisSent5/8, and 8471 for DisSent-All
parser.add_argument("--factor", type=float, default=1.0, help="learning rate scaling factor")
parser.add_argument("--l2", type=float, default=0.01, help="on non-bias non-gain weights")
parser.add_argument("--max_norm", type=float, default=2., help="max norm (grad clipping). Original paper uses 1.")
parser.add_argument("--log_interval", type=int, default=100, help="how many batches to log once")
parser.add_argument('--lm_coef', type=float, default=0.5)
parser.add_argument("--train_emb", default=False, action='store_true', help="Allow to learn embedding, default to False")
parser.add_argument("--init_emb", default=False, action='store_true', help="Initialize embedding randomly, default to False")
parser.add_argument("--pick_hid", default=True, action='store_true', help="Pick correct hidden states")
parser.add_argument("--tied", default=True, action='store_true', help="Tie weights to embedding, should be always flagged True")
parser.add_argument("--proj_head", type=int, default=1, help="last docoder layer head number")
parser.add_argument("--proj_type", type=int, default=1, help="last decoder layer blow up type, 1 for initial linear transformation, 2 for final linear transformation")
parser.add_argument("--model_type", type=str, default="transformer", help="transformer|lstm")
parser.add_argument("--meta_param", type=float, default=0.0, help="meta param")
parser.add_argument("--cluster_param_a", type=float, default=0.0, help="cluster param a")
parser.add_argument("--cluster_param_b", type=float, default=0.0, help="cluster param b")
parser.add_argument("--cluster_param_c", type=float, default=0.0, help="cluster param c")
# for now we fix non-linearity to whatever PyTorch provides...could be SELU
# model
parser.add_argument("--d_ff", type=int, default=2048, help="decoder nhid dimension")
parser.add_argument("--d_model", type=int, default=768, help="decoder nhid dimension")
parser.add_argument("--n_heads", type=int, default=8, help="number of attention heads")
parser.add_argument("--n_layers", type=int, default=6, help="decoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
# parser.add_argument("--pool_type", type=str, default='max', help="flag if we do max pooling, which hasn't been done before")
parser.add_argument("--reload_val", action='store_true', help="Reload the previous best epoch on validation, should be used with tied weights")
parser.add_argument("--no_stop", default=True, action='store_true', help="no early stopping")
parser.add_argument("--metamap", default=False, action='store_true', help="use meta map")
# gpu
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--seed", type=int, default=1234, help="seed")

params, _ = parser.parse_known_args()

# set gpu device
torch.cuda.set_device(params.gpu_id)

"""
SEED
"""
random.seed(params.seed)
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)

"""
Logging
"""
logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)

if not os.path.exists(params.outputdir):
    os.makedirs(params.outputdir)
file_handler = logging.FileHandler("{0}/log.txt".format(params.outputdir))
formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p')
file_handler.setFormatter(formatter)
logging.getLogger().addHandler(file_handler)
# print parameters passed, and all parameters
logger.info('\ntogrep : {0}\n'.format(sys.argv[1:]))
logger.info(params)

"""
Default json file loading
"""
json_config = json.load(open(params.hypes, 'rb'))
data_dir = json_config['data_dir']
prefix = json_config[params.corpus]
bpe_encoder_path = json_config['bpe_encoder_path']
bpe_vocab_path = json_config['bpe_vocab_path']
if params.init_emb: wordvec_path = json_config['wordvec_path']

"""
BPE encoder
"""
text_encoder = TextEncoder(bpe_encoder_path, bpe_vocab_path)
encoder = text_encoder.encoder
# add special token
encoder['_pad_'] = len(encoder)
encoder['_start_'] = len(encoder)
encoder['_end_'] = len(encoder)
encoder['_unk_'] = len(encoder)
n_special = 4

"""
DATA
"""
train, valid, test = get_dis(data_dir, prefix, params.corpus, params.cut_down_len, params.metamap)  # this stays the same
# If this is slow...we can speed it up
# Numericalization; No padding here
# Also, Batch class from OpenNMT will take care of target generation
max_len = 0.
for split in ['s1']:
    for data_type in ['train', 'valid', 'test']:
        num_sents = []
        y_sents = []
        for sent in eval(data_type)[split]:
            if params.corpus == 'sage':
                num_sent = text_encoder.encode([sent], lazy=True, bpe=False)[0]
                num_sents.append(num_sent)
            else:
                num_sent = text_encoder.encode([sent], lazy=True, bpe=False)[0]
                num_sents.append([encoder['_start_']] + num_sent + [encoder['_end_']])
            max_len = max_len if max_len > len(num_sent) else len(num_sent)
        if params.corpus == 'sage':
            eval(data_type)[split] = batchify(np.array(num_sents[0]), params.batch_size)
        else:
            eval(data_type)[split] = np.array(num_sents)
logger.info(max_len)
logger.info(train['s1'].shape)
logger.info(valid['s1'].shape)
logger.info(test['s1'].shape)

"""
Params
"""
if params.init_emb:
    word_embeddings = np.concatenate([np.load(wordvec_path).astype(np.float32),
                                      np.zeros((1, params.d_model), np.float32), # pad, zero-value!
                                      (np.random.randn(n_special - 1, params.d_model) * 0.02).astype(np.float32)], 0)
else:                                                          
    word_embeddings = (np.random.randn(len(encoder), params.d_model) * 0.02).astype(np.float32)
dis_labels = get_labels('csu')
label_size = len(dis_labels)

"""
MODEL
"""
# model config
config_dis_model = {
    'n_words': len(encoder),
    'd_model': params.d_model, # same as word embedding size
    'd_ff': params.d_ff, # this is the bottleneck blowup dimension
    'n_layers': params.n_layers,
    'dpout': params.dpout,
    'dpout_fc': params.dpout_fc,
    'fc_dim': params.fc_dim,
    'bsize': params.batch_size,
    'n_classes': label_size,
    'n_heads': params.n_heads,
    'gpu_id': params.gpu_id,
    'train_emb': params.train_emb,
    'init_emb': params.init_emb,
    'pick_hid': params.pick_hid,
    'tied': params.tied,
    'proj_head': params.proj_head,
    'proj_type': params.proj_type,
    'meta_param': params.meta_param,
    'cluster_param': [params.cluster_param_a, params.cluster_param_b, params.cluster_param_c],
    'metamap': params.metamap,
    'n_metamap': len(json.load(open('/home/yuhuiz/Transformer/data/meta2id.json')))
}
if params.cur_epochs == 1:
    if params.metamap: # <TODO>
        logger.info('model metamap')
        dis_net = make_metamap_model(config_dis_model)
    elif params.model_type == "lstm":
        logger.info('model lstm')
        dis_net = make_lstm_model(encoder, config_dis_model, word_embeddings) # ctx_embeddings
    else:
        logger.info('model transformer')
        dis_net = make_model(encoder, config_dis_model, word_embeddings)
    logger.info(dis_net)
else:
    # if starting epoch is not 1, we resume training
    # 1. load in model
    # 2. resume with the previous learning rate
    model_path = pjoin(params.outputdir, params.outputmodelname + ".pickle")  # this is the best model
    # this might have conflicts with gpu_idx...
    dis_net = torch.load(model_path)

need_grad = lambda x: x.requires_grad
model_opt = NoamOpt(params.d_model, params.factor, params.warmup_steps,
            torch.optim.Adam(filter(need_grad, dis_net.parameters()), lr=0, betas=(0.9, 0.98), eps=1e-9))
if params.cur_epochs != 1:
    # now we need to set the correct learning rate
    prev_steps = (len(train['s1']) // params.batch_size) * (params.cur_epochs - 1)
    model_opt._step = prev_steps  # now we start with correct learning rate
if params.gpu_id != -1:
    dis_net.cuda(params.gpu_id)


"""
TRAIN
"""
val_acc_best = -1e10 if params.cur_epochs == 1 else params.cur_valid
adam_stop = False
stop_training = False

def train_epoch_csu(epoch):
    # initialize
    logger.info('\nTRAINING : Epoch ' + str(epoch))
    dis_net.train()
    all_costs, all_em, all_micro_p, all_micro_r, all_micro_f1, all_macro_p, all_macro_r, all_macro_f1 = [], [], [], [], [], [], [], []

    # shuffle the data
    permutation = np.random.permutation(len(train['s1']))
    s1 = train['s1'][permutation]
    target = train['label'][permutation]
    metamap = train['metamap'][permutation] if params.metamap else []

    for stidx in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch = pad_batch(s1[stidx: stidx + params.batch_size], encoder['_pad_'])
        label_batch = target[stidx:stidx + params.batch_size]
        metamap_batch = metamap[stidx:stidx + params.batch_size] if params.metamap else []
        b = Batch(s1_batch, label_batch, metamap_batch, encoder['_pad_'], gpu_id=params.gpu_id)

        # model forward
        if params.metamap:
            clf_output = dis_net(b)
        else:
            clf_output, s1_y_hat = dis_net(b, clf=True, lm=True)

        # evaluation
        pred = (torch.sigmoid(clf_output) > 0.5).data.cpu().numpy().astype(float)
        em = metrics.accuracy_score(label_batch, pred)
        p, r, f1, s = metrics.precision_recall_fscore_support(label_batch, pred, average=None)
        try:
            micro_p, micro_r, micro_f1 = np.average(p, weights=s), np.average(r, weights=s), np.average(f1, weights=s)
            macro_p, macro_r, macro_f1 = np.average(p[p.nonzero()]), np.average(r[r.nonzero()]), np.average(f1[f1.nonzero()])
        except:
            continue
        all_em.append(em)
        all_micro_p.append(micro_p)
        all_micro_r.append(micro_r)
        all_micro_f1.append(micro_f1)
        all_macro_p.append(macro_p)
        all_macro_r.append(macro_r)
        all_macro_f1.append(macro_f1)
        # assert len(p) == len(s1[stidx:stidx + params.batch_size])

        # loss
        if params.metamap:
            clf_loss = dis_net.compute_clf_loss(clf_output, b.label)
            loss = clf_loss
        else:
            clf_loss = dis_net.compute_clf_loss(clf_output, b.label)
            s1_lm_loss = dis_net.compute_lm_loss(s1_y_hat, b.s1_y, b.s1_loss_mask)
            loss = clf_loss + params.lm_coef * s1_lm_loss

        all_costs.append(loss.data.item())
        
        # backward
        model_opt.optimizer.zero_grad()
        loss.backward()

        # optimizer step
        model_opt.step()

        # log and reset 
        if len(all_costs) == params.log_interval:
            logger.info('{0}; loss {1}; em {2}; p {3}; r {4}; f1 {5}; lr: {6}; embed_norm: {7}'.format(
                stidx, 
                round(np.mean(all_costs), 2),
                round(np.mean(all_em), 3),
                (round(np.mean(all_micro_p), 3), round(np.mean(all_macro_p), 3)),
                (round(np.mean(all_micro_r), 3), round(np.mean(all_macro_r), 3)),
                (round(np.mean(all_micro_f1), 3), round(np.mean(all_macro_f1), 3)),
                model_opt.rate(),
                0 if params.metamap else dis_net.tgt_embed[0].lut.weight.data.norm()))
            all_costs, all_em, all_micro_p, all_micro_r, all_micro_f1, all_macro_p, all_macro_r, all_macro_f1 = [], [], [], [], [], [], [], []

    # save
    torch.save(dis_net, os.path.join(params.outputdir, params.outputmodelname + "-" + str(epoch) + ".pickle"))


def evaluate_epoch_csu(epoch, eval_type='valid'):
    # initialize
    if eval_type == 'valid': logger.info('\nVALIDATION : Epoch {0}'.format(epoch))
    else: logger.info('\nTEST : Epoch {0}'.format(epoch))
    global dis_net, val_acc_best, lr, stop_training, adam_stop
    dis_net.eval()
    
    # data without shuffle
    s1 = valid['s1'] if eval_type == 'valid' else test['s1']
    target = valid['label'] if eval_type == 'valid' else test['label']
    if params.metamap: metamap = valid['metamap'] if eval_type == 'valid' else test['metamap']
    else: metamap = []
    valid_preds, valid_labels = [], []

    for stidx in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch = pad_batch(s1[stidx: stidx + params.batch_size], encoder['_pad_'])
        label_batch = target[stidx:stidx + params.batch_size]
        metamap_batch = metamap[stidx:stidx + params.batch_size] if params.metamap else []
        b = Batch(s1_batch, label_batch, metamap_batch, encoder['_pad_'], gpu_id=params.gpu_id)

        # model forward
        if params.metamap:
            clf_output = dis_net(b)
        else:
            clf_output = dis_net(b, clf=True, lm=False)

        # evaluation
        pred = (torch.sigmoid(clf_output) > 0.5).data.cpu().numpy().astype(float)
        valid_preds.extend(pred.tolist())
        valid_labels.extend(label_batch.tolist())

    valid_preds, valid_labels = np.array(valid_preds), np.array(valid_labels)
    if eval_type == 'test': np.save('%s/preds-%s.npy' % (params.outputdir, str(epoch)), valid_preds)
    em = metrics.accuracy_score(valid_labels, valid_preds)
    p, r, f1, s = metrics.precision_recall_fscore_support(valid_labels, valid_preds, average=None)
    micro_p, micro_r, micro_f1 = np.average(p, weights=s), np.average(r, weights=s), np.average(f1, weights=s)
    macro_p, macro_r, macro_f1 = np.average(p[p.nonzero()]), np.average(r[r.nonzero()]), np.average(f1[f1.nonzero()])

    logger.info('{0}; em {1}; p {2}; r {3}; f1 {4}'.format(
        epoch, 
        round(em, 3),
        (round(micro_p, 3), round(macro_p, 3)),
        (round(micro_r, 3), round(macro_r, 3)),
        (round(micro_f1, 3), round(macro_f1, 3))))


def train_epoch_sage(epoch):
    logger.info('\nTRAINING : Epoch ' + str(epoch))
    dis_net.train()
    all_costs = []
    s1 = train['s1']

    for stidx in range(0, s1.shape[1], params.bptt_size):
        # prepare batch
        pad_start = np.ones([params.batch_size, 1]) * encoder['_start_']
        pad_end = np.ones([params.batch_size, 1]) * encoder['_end_']
        s1_batch = np.concatenate([pad_start, s1[:, stidx: stidx + params.bptt_size], pad_end], 1).astype(np.int64)
        label_batch = np.ones(params.batch_size)
        b = Batch(s1_batch, label_batch, encoder['_pad_'], gpu_id=params.gpu_id)

        # model forward
        s1_y_hat = dis_net(b, clf=False, lm=True)

        # loss
        s1_lm_loss = dis_net.compute_lm_loss(s1_y_hat, b.s1_y, b.s1_loss_mask)
        loss = s1_lm_loss
        all_costs.append(loss.data.item())

        # backward
        model_opt.optimizer.zero_grad()
        loss.backward()

        # optimizer step
        model_opt.step()

        if len(all_costs) == params.log_interval:
            logger.info('{0} ; loss {1} ; perplexity: {2} ; lr: {3} ; embed_norm: {4}'.format(
                stidx, 
                round(np.mean(all_costs), 2),
                round(np.exp(np.mean(all_costs)), 2),
                model_opt.rate(),
                dis_net.tgt_embed[0].lut.weight.data.norm()))
            all_costs = []
    torch.save(dis_net, os.path.join(params.outputdir, params.outputmodelname + "-" + str(epoch) + ".pickle"))


def evaluate_epoch_sage(epoch, eval_type='valid'):
    global dis_net, val_acc_best, lr, stop_training, adam_stop
    if eval_type == 'valid':
        logger.info('\nVALIDATION : Epoch {0}'.format(epoch))
    else:
        logger.info('\nTEST : Epoch {0}'.format(epoch))
    dis_net.eval()
    all_costs = []

    s1 = valid['s1'] if eval_type == 'valid' else test['s1']

    for stidx in range(0, s1.shape[1], params.bptt_size):
        # prepare batch
        pad_start = np.ones([s1.shape[0], 1]) * encoder['_start_']
        pad_end = np.ones([s1.shape[0], 1]) * encoder['_end_']
        s1_batch = np.concatenate([pad_start, s1[:, stidx: stidx + params.bptt_size], pad_end], 1).astype(np.int64)
        label_batch = np.ones(s1.shape[0])
        b = Batch(s1_batch, label_batch, encoder['_pad_'], gpu_id=params.gpu_id)

        # model forward
        s1_y_hat = dis_net(b, clf=False, lm=True)
        s1_lm_loss = dis_net.compute_lm_loss(s1_y_hat, b.s1_y, b.s1_loss_mask)
        loss = s1_lm_loss
        all_costs.append(loss.data.item())

    logger.info(('loss {0} ; perplexity: {1} ; embed_norm: {2}'.format(
                round(np.mean(all_costs), 2),
                round(np.exp(np.mean(all_costs)), 2),
                dis_net.tgt_embed[0].lut.weight.data.norm())))


"""
Train model on Discourse Classification task
"""
if __name__ == '__main__':
    epoch = params.cur_epochs  # start at 1

    if params.corpus == 'pp':
        del dis_net
        dis_net = torch.load(params.inputdir)
        evaluate_epoch_csu(epoch, eval_type='test')
    # elif params.corpus == 'csu':
    #     del dis_net
    #     dis_net = torch.load(params.inputdir)
    #     evaluate_epoch_csu(epoch, eval_type='test')
    elif params.corpus == 'csu':
        if len(params.inputdir) != 0:
            logger.info('Load Model from %s' % (params.inputdir))
            dis_net.load_state_dict(torch.load(params.inputdir).state_dict())
        # evaluate_epoch_csu(epoch)
        # evaluate_epoch_csu(epoch, eval_type='test')
        while not stop_training and epoch <= params.n_epochs:
            train_epoch_csu(epoch)
            evaluate_epoch_csu(epoch)
            evaluate_epoch_csu(epoch, eval_type='test')
            epoch += 1
    elif params.corpus == 'sage':
        while not stop_training and epoch <= params.n_epochs:
            train_epoch_sage(epoch)
            evaluate_epoch_sage(epoch)
            evaluate_epoch_sage(epoch, eval_type='test')
            epoch += 1

