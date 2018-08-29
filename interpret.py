import torch
import torch.nn as nn
from torch.autograd import Variable
from transformer import *
from util import *
from data import *
from collections import defaultdict

train, valid, test = get_dis('/home/yuhuiz/Transformer/data/csu/', 'csu_bpe', 'csu', 600, False)
text_encoder = TextEncoder('/home/yuhuiz/Transformer/data/sage/encoder_bpe_50000.json', '/home/yuhuiz/Transformer/data/sage/vocab_50000.bpe')
encoder = text_encoder.encoder
encoder['_pad_'] = len(encoder)
encoder['_start_'] = len(encoder)
encoder['_end_'] = len(encoder)
encoder['_unk_'] = len(encoder)
decoder = {v: k for k, v in encoder.items()}

for split in ['s1']:
    for data_type in ['train', 'valid', 'test']:
        num_sents = []
        y_sents = []
        for sent in eval(data_type)[split]:
            num_sent = text_encoder.encode([sent], lazy=True, bpe=False)[0]
            num_sents.append([encoder['_start_']] + num_sent + [encoder['_end_']])
        eval(data_type)[split] = np.array(num_sents)



stat_pos = {}
stat_neg = {}
for i in range(42):
    stat_pos[i] = defaultdict(int)
    stat_neg[i] = defaultdict(int)
# model = torch.load('/home/yuhuiz/Transformer/exp/bpe/transformer_auxiliary_pretrain/model-8.pickle')
model = torch.load('/home/yuhuiz/Transformer/exp/bpe/lstm_auxiliary_pretrain/model-10.pickle', map_location='cpu')
model.eval()
s1 = test['s1'] # test or valid
target = test['label'] # test or valid
for i in range(len(test['label'])):
    if i % 100 == 0: 
        print(i)
        json.dump(stat_pos, open('stat_pos.json', 'w'))
        json.dump(stat_neg, open('stat_neg.json', 'w'))
    # data
    s1_batch = pad_batch(s1[i:i+1], encoder['_pad_'])
    label_batch = target[i:i+1]
    b = Batch(s1_batch, label_batch, [], encoder['_pad_'])

    # interpret
    x = model.tgt_embed[0](b.s1)
    xx = model.tgt_embed[1](x)
    # u_h = model.decoder(xx, b.s1_mask)
    u_h = model.autolen_rnn(xx, b.s1_lengths)
    u = model.pick_h(u_h, b.s1_lengths)
    picked_s1_mask = model.pick_mask(b.s1_mask, b.s1_lengths)
    u = model.projection_layer(u, u_h, u_h, picked_s1_mask)
    clf_output = model.classifier(u)
    pred = (torch.sigmoid(clf_output) > 0.5)
    pred = pred[0].nonzero().view(-1).cpu().numpy().tolist()

    text_id = s1_batch[0].tolist()
    text = [decoder[i] for i in text_id]
    
    for pred_idx in pred:
        if pred_idx == 41: continue
        if b.label[0][pred_idx] == 1.0:
            y = clf_output[0][pred_idx]
            model.zero_grad()
            grads = x * torch.autograd.grad(y, x, retain_graph=True)[0]
            grads = grads.sum(-1).data.view(-1).cpu().numpy().tolist()
            for idx, grad in enumerate(grads):
                if grad > 0.2: stat_pos[pred_idx][text[idx]] += 1
                elif grad < -0.2: stat_neg[pred_idx][text[idx]] += 1
    
json.dump(stat_pos, open('stat_pos.json', 'w'))
json.dump(stat_neg, open('stat_neg.json', 'w'))

