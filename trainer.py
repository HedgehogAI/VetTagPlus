"""
Code adapted from
https://github.com/facebookresearch/InferSent/blob/master/train_nli.py

with minor modifications
"""

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
from transformer import NoamOpt, make_model, make_lstm_model

import logging

parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--corpus", type=str, default='sage', help="sage|csu|pp")
parser.add_argument("--hypes", type=str, default='hypes/default.json', help="load in a hyperparameter file")
parser.add_argument("--outputdir", type=str, default='sandbox/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='dis-model')

# training
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--cur_epochs", type=int, default=1)
parser.add_argument("--cur_valid", type=float, default=-1e10, help="must set this otherwise resumed model will be saved by default")
parser.add_argument("--bptt_size", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dpout", type=float, default=0.1, help="residual, embedding, attention dropout") # 3 dropouts
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--maxlr", type=float, default=2.5e-4, help="this is not used...")
parser.add_argument("--warmup_steps", type=int, default=8000, help="OpenNMT uses steps")
# TransformerLM uses 0.2% of training data as warmup step, that's 5785 for DisSent5/8, and 8471 for DisSent-All
parser.add_argument("--factor", type=float, default=1.0, help="learning rate scaling factor")
parser.add_argument("--l2", type=float, default=0.01, help="on non-bias non-gain weights")
parser.add_argument("--max_norm", type=float, default=2., help="max norm (grad clipping). Original paper uses 1.")
parser.add_argument("--log_interval", type=int, default=100, help="how many batches to log once")
parser.add_argument('--lm_coef', type=float, default=0.5)
parser.add_argument("--train_emb", action='store_true', help="Initialize embedding randomly, and then learn it, default to False")
parser.add_argument("--pick_hid", action='store_true', help="Pick correct hidden states")
parser.add_argument("--tied", action='store_true', help="Tie weights to embedding, should be always flagged True")
parser.add_argument("--proj_head", type=int, default=1, help="last docoder layer head number")
parser.add_argument("--proj_type", type=int, default=1, help="last decoder layer blow up type, 1 for initial linear transformation, 2 for final linear transformation")
# for now we fix non-linearity to whatever PyTorch provides...could be SELU

# model
parser.add_argument("--d_ff", type=int, default=3072, help="decoder nhid dimension")
parser.add_argument("--d_model", type=int, default=768, help="decoder nhid dimension")
parser.add_argument("--n_heads", type=int, default=12, help="number of attention heads")
parser.add_argument("--n_layers", type=int, default=8, help="decoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
# parser.add_argument("--pool_type", type=str, default='max', help="flag if we do max pooling, which hasn't been done before")
parser.add_argument("--reload_val", action='store_true', help="Reload the previous best epoch on validation, should "
                                                              "be used with tied weights")
parser.add_argument("--no_stop", action='store_true', help="no early stopping")

# gpu
parser.add_argument("--gpu_id", type=int, default=-1, help="GPU ID")
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
logging.getLogger().addHandler(file_handler)

# print parameters passed, and all parameters
logger.info('\ntogrep : {0}\n'.format(sys.argv[1:]))
logger.info(params)

"""
Default json file loading
"""
with open(params.hypes, 'rb') as f:
    json_config = json.load(f)

data_dir = json_config['data_dir']
prefix = json_config[params.corpus]
bpe_encoder_path = json_config['bpe_encoder_path']
bpe_vocab_path = json_config['bpe_vocab_path']
params_path = json_config['params_path']

"""
BPE encoder
"""
text_encoder = TextEncoder(bpe_encoder_path, bpe_vocab_path)
encoder = text_encoder.encoder

# add special token
encoder['_pad_'] = len(encoder)
encoder['_start_'] = len(encoder)
encoder['_end_'] = len(encoder)

"""
DATA
1. build vocab through BPE
"""
train, valid, test = get_dis(data_dir, prefix, params.corpus)  # this stays the same

# word_vec = build_vocab(train['s1'] + valid['s1'] + test['s1'], glove_path)
# batching function needs to be different:
# 1). return s1, s2, y_label

# If this is slow...we can speed it up
# Numericalization; No padding here
# Also, Batch class from OpenNMT will take care of target generation
max_len = 0.
for split in ['s1']:
    for data_type in ['train', 'valid', 'test']:
        num_sents = []
        y_sents = []
        for sent in eval(data_type)[split]:
            num_sent = text_encoder.encode([sent], verbose=False, lazy=True, bpe=False)[0]
            # num_sents.append([encoder['_start_']] + num_sent + [encoder['_end_']])
            num_sents.append(num_sent)
            # y_sents.append(num_sent + [encoder['_end_']])
            max_len = max_len if max_len > len(num_sent) + 1 else len(num_sent) + 1
        # print num_sents[0]
        eval(data_type)[split] = batchify(np.array(num_sents[0]), params.batch_size)
        # eval(data_type)['y_'+split] = np.array(y_sents)
print max_len
print train['s1'].shape
print valid['s1'].shape
print test['s1'].shape
"""
Params
2. Load in parameters (word embeddings)
"""

# shapes = json.load(open(pjoin(params_path, 'params_shapes.json')))
# offsets = np.cumsum([np.prod(shape) for shape in shapes])
# init_params = [np.load(pjoin(params_path, 'params_{}.npy'.format(n))) for n in range(3)]
# init_params = np.split(np.concatenate(init_params, 0), offsets[:2])[:-1]
# init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes[:2])]

# n_special = 3  # <s>, </s>, <pad>
# n_ctx = 1024

# init_params[0] = init_params[0][:n_ctx]
# word_embeddings = np.concatenate([init_params[1],
#                                    np.zeros((1, params.d_model), np.float32), # pad, zero-value!
#                                   (np.random.randn(n_special-1, params.d_model)*0.02).astype(np.float32)], 0)
# ctx_embeddings = init_params[0]
# del init_params[1]

word_embeddings = (np.random.randn(len(encoder), params.d_model)*0.02).astype(np.float32)

dis_labels = get_labels(params.corpus)
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
    # 'pool_type': params.pool_type,
    'n_heads': params.n_heads,
    'gpu_id': params.gpu_id,
    'train_emb': params.train_emb,
    'pick_hid': params.pick_hid,
    'tied': params.tied,
    'proj_head': params.proj_head,
    'proj_type': params.proj_type
}

# TODO: reload model in here...
if params.cur_epochs == 1:
    dis_net = make_lstm_model(encoder, config_dis_model, word_embeddings) # ctx_embeddings
    logger.info(dis_net)
else:
    # if starting epoch is not 1, we resume training
    # 1. load in model
    # 2. resume with the previous learning rate
    model_path = pjoin(params.outputdir, params.outputmodelname + ".pickle")  # this is the best model
    # this might have conflicts with gpu_idx...
    dis_net = torch.load(model_path)

# warmup_steps: 8000
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

def trainepoch(epoch):
    logger.info('\nTRAINING : Epoch ' + str(epoch))
    dis_net.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    # correct = 0.
    # shuffle the data
    # permutation = np.random.permutation(len(train['s1']))

    # s1 = train['s1'][permutation]
    # target = train['label'][permutation]
    s1 = train['s1']

    if model_opt._step == 0:
        logger.info('Current learning rate : {0}'.format(model_opt.rate(1)))
    else:
        logger.info('Current learning rate : {0}'.format(model_opt.rate()))

    # for stidx in range(0, len(s1), params.batch_size):
    for stidx in range(0, s1.shape[1], params.bptt_size):
        # prepare batch
        # s1_batch = pad_batch(s1[stidx:stidx + params.batch_size], encoder['_pad_'])
        # label_batch = target[stidx:stidx + params.batch_size]
        pad_start = np.ones([s1.shape[0], 1]) * encoder['_start_']
        pad_end = np.ones([s1.shape[0], 1]) * encoder['_end_']
        s1_batch = np.concatenate([pad_start, s1[:, stidx: stidx + params.bptt_size], pad_end], 1).astype(np.int64)
        # print 's1_batch_shape', s1_batch.shape
        label_batch = np.ones(s1.shape[0])
        b = Batch(s1_batch, label_batch, encoder['_pad_'], gpu_id=params.gpu_id)

        # s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        # tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size])).cuda()
        k = s1_batch.shape[0]  # actual batch size

        # model forward
        # clf_output, s1_y_hat, s2_y_hat = dis_net(b)
        s1_y_hat = dis_net(b, clf=False, lm=True)

        # pred = clf_output.data.max(1)[1]
        # correct += pred.long().eq(b.label.data.long()).cpu().sum()
        # assert len(pred) == len(s1[stidx:stidx + params.batch_size])

        # loss
        # clf_loss = dis_net.compute_clf_loss(clf_output, b.label)
        s1_lm_loss = dis_net.compute_lm_loss(s1_y_hat, b.s1_y, b.s1_loss_mask)

        # loss = clf_loss + params.lm_coef * s1_lm_loss + params.lm_coef * s2_lm_loss
        loss = s1_lm_loss

        all_costs.append(loss.data[0])
        words_count += s1_batch.size / params.d_model

        # backward
        model_opt.optimizer.zero_grad()
        loss.backward()

        # optimizer step
        model_opt.step()

        if len(all_costs) == params.log_interval:
            logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; perplexity: {4} ; lr: {5} ; embed_norm: {6}'.format(
                stidx, round(np.mean(all_costs), 2),
                int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                int(words_count * 1.0 / (time.time() - last_time)),
                round(np.exp(np.mean(all_costs)), 2), #acc: round(100. * correct / (stidx + k), 2),
                model_opt.rate(),
                dis_net.tgt_embed[0].lut.weight.data.norm()))
            logger.info(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []
    torch.save(dis_net, os.path.join(params.outputdir, params.outputmodelname + "-" + str(epoch) + ".pickle"))

    # train_acc = round(100 * correct / len(s1), 2)
    # logger.info('results : epoch {0} ; mean accuracy train : {1}'
                # .format(epoch, train_acc))
    # return train_acc

def get_multiclass_recall(preds, y_label):
    # preds: (label_size), y_label; (label_size)
    label_cat = range(label_size)
    labels_accu = {}

    for la in label_cat:
        # for each label, we get the index of the correct labels
        idx_of_cat = y_label == la
        cat_preds = preds[idx_of_cat]
        if cat_preds.size != 0:
            accu = np.mean(cat_preds == la)
            labels_accu[la] = [accu]
        else:
            labels_accu[la] = []

    return labels_accu


def get_multiclass_prec(preds, y_label):
    label_cat = range(label_size)
    labels_accu = {}

    for la in label_cat:
        # for each label, we get the index of predictions
        idx_of_cat = preds == la
        cat_preds = y_label[idx_of_cat]  # ground truth
        if cat_preds.size != 0:
            accu = np.mean(cat_preds == la)
            labels_accu[la] = [accu]
        else:
            labels_accu[la] = []

    return labels_accu

def evaluate(epoch, eval_type='valid', final_eval=False, save_confusion=False):
    global dis_net

    dis_net.eval()
    all_costs = []
    # correct = 0.
    global val_acc_best, lr, stop_training, adam_stop

    if eval_type == 'valid':
        logger.info('\nVALIDATION : Epoch {0}'.format(epoch))

    s1 = valid['s1'] if eval_type == 'valid' else test['s1']
    # target = valid['label'] if eval_type == 'valid' else test['label']
    # valid_preds, valid_labels = [], []

    for stidx in range(0, s1.shape[1], params.bptt_size):
        # prepare batch
        # s1_batch = pad_batch(s1[stidx:stidx + params.batch_size], encoder['_pad_'])
        # label_batch = target[stidx:stidx + params.batch_size]
        pad_start = np.ones([s1.shape[0], 1]) * encoder['_start_']
        pad_end = np.ones([s1.shape[0], 1]) * encoder['_end_']
        s1_batch = np.concatenate([pad_start, s1[:, stidx: stidx + params.bptt_size], pad_end], 1).astype(np.int64)
        # print 's1_batch_shape', s1_batch.shape
        label_batch = np.ones(s1.shape[0])
        b = Batch(s1_batch, label_batch, encoder['_pad_'], gpu_id=params.gpu_id)

        s1_y_hat = dis_net(b, clf=False, lm=True)
        s1_lm_loss = dis_net.compute_lm_loss(s1_y_hat, b.s1_y, b.s1_loss_mask)
        loss = s1_lm_loss
        all_costs.append(loss.data[0])

    logger.info(('loss {0} ; perplexity: {1} ; embed_norm: {2}'.format(
                round(np.mean(all_costs), 2),
                round(np.exp(np.mean(all_costs)), 2), #acc: round(100. * correct / (stidx + k), 2),
                dis_net.tgt_embed[0].lut.weight.data.norm())))
        # model forward
        # clf_output = dis_net(b, clf=True, lm=False)

        # pred = clf_output.data.max(1)[1]
        # correct += pred.long().eq(b.label.data.long()).cpu().sum()

        # we collect samples
        # labels = target[stidx:stidx + params.batch_size]
        # preds = pred.cpu().numpy()

        # valid_preds.extend(preds.tolist())
        # valid_labels.extend(labels.tolist())

    # mean_multi_recall = get_multiclass_recall(np.array(valid_preds), np.array(valid_labels))
    # mean_multi_prec = get_multiclass_prec(np.array(valid_preds), np.array(valid_labels))

    # multiclass_recall_msg = 'Multiclass Recall - '
    # for k, v in mean_multi_recall.iteritems():
        # multiclass_recall_msg += dis_labels[k] + ": " + str(v[0]) + " "

    # multiclass_prec_msg = 'Multiclass Precision - '
    # for k, v in mean_multi_prec.iteritems():
        # if len(v) == 0:
            # v = [0.]
        # multiclass_prec_msg += dis_labels[k] + ": " + str(v[0]) + " "

    # logger.info(multiclass_recall_msg)
    # logger.info(multiclass_prec_msg)

    # if params.corpus == "gw_cn_5" or params.corpus == "gw_es_5":
    #     print(multiclass_recall_msg)
    #     print(multiclass_prec_msg)

    # save model
    # eval_acc = round(100 * correct / len(s1), 2)
    # if final_eval:
        # logger.info('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    # else:
        # logger.info('togrep : results : epoch {0} ; mean accuracy {1} :\
              # {2}'.format(epoch, eval_type, eval_acc))
        # print out multi-class recall and precision

    # if save_confusion:
        # with open(pjoin(params.outputdir, 'confusion_test.csv'), 'wb') as csvfile:
            # fieldnames = ['preds', 'labels']
            # writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # writer.writeheader()

            # for pair in izip(valid_preds, valid_labels):
                # writer.writerow({'preds': pair[0], 'labels': pair[1]})

    # there is no re-loading of previous best model btw
    # if eval_type == 'valid' and epoch <= params.n_epochs:
    #     if eval_acc > val_acc_best:
    #         logger.info('saving model at epoch {0}'.format(epoch))
    #         if not os.path.exists(params.outputdir):
    #             os.makedirs(params.outputdir)
    #         torch.save(dis_net, os.path.join(params.outputdir,
    #                                          params.outputmodelname + ".pickle"))
    #         # monitor memory usage
    #         try:
    #             torch.save(dis_net, os.path.join(params.outputdir,
    #                                              params.outputmodelname + "-" + str(epoch) + ".pickle"))
    #         except:
    #             print("saving by epoch error, maybe due to disk space limit")

    #         val_acc_best = eval_acc

    #         if adam_stop is True:
    #             # we reset both when there's an improvement
    #             adam_stop = False
    #             stop_training = False
    #     else:
    #         # early stopping (at 2nd decrease in accuracy)
    #         stop_training = adam_stop
    #         adam_stop = True if not params.no_stop else False

    #         # now we finished annealing, we can reload
    #         if params.reload_val:
    #             del dis_net
    #             dis_net = torch.load(os.path.join(params.outputdir, params.outputmodelname + ".pickle"))
    #             logger.info("Load in previous best epoch")

    # return eval_acc

"""
Train model on Discourse Classification task
"""
if __name__ == '__main__':
    global dis_net
    epoch = params.cur_epochs  # start at 1

    # del dis_net
    # dis_net = torch.load('/home/yuhuiz/Transformer/exp/sage_lm/dis-model-1.pickle')
    # evaluate(epoch, eval_type='test')

    while not stop_training and epoch <= params.n_epochs:
        trainepoch(epoch)
        evaluate(epoch)
        # train_acc = trainepoch(epoch)
        # eval_acc = evaluate(epoch, 'valid')
        epoch += 1

    # Run best model on test set.
    # del dis_net
    # dis_net = torch.load(os.path.join(params.outputdir, params.outputmodelname + ".pickle"))

    # logger.info('\nTEST : Epoch {0}'.format(epoch))
    # evaluate(1e6, 'valid', True)
    # evaluate(0, 'test', True, True)  # save confusion results on test data