from RNNModel import RNNModel
import torch
from torch import nn
from config import config
from torch.autograd import Variable
from util import to_scalar, argmax, log_sum_exp
from torch.nn import functional as F
import numpy as np
import os

START_TAG = 4
END_TAG = 5

class BiLSTM_CRF(nn.Module):
    def __init__(self, n_token):
        super(BiLSTM_CRF, self).__init__()
        self.LSTM = RNNModel(n_token)
        self.LSTM.load_state_dict(torch.load(os.path.join(config['data_dir'], 'RNNmodel')))

        # tag_size == 6
        # B M E S <beg_tag> <end_tag>
        self.transitions = nn.Parameter(torch.randn(6, 6))
        self.transitions.data[:,:] = 0

#        self.transitions.data[START_TAG, :] = -1e3
#        self.transitions.data[:, END_TAG] = -1e3

        self.repack = True

    def get_lstm_features(self, seq):
        lstm_features = self.LSTM(seq)
        lstm_features = lstm_features.permute(1, 0, 2)
        #lstm_features = F.sigmoid(lstm_features) * 5 - 10
        seq_size, batch_size, _ = lstm_features.size()
        zero_padding = Variable(torch.FloatTensor(seq_size, batch_size, 2).fill_(-100).cuda())
        lstm_features = torch.cat([lstm_features, zero_padding], dim=2)
        if self.repack:
            lstm_features = Variable(lstm_features.data, requires_grad=False)
        return lstm_features

    def _norm(self, feats):
        seq_len, batch_size, tag_size = feats.size()

        alpha = feats.data.new(batch_size, tag_size).fill_(-10000)
        alpha[:, START_TAG] = 0
        alpha = Variable(alpha)

        for feat in feats: # batch_size x tag_size
            # [batch_size] * tag_size
            feat_exp = feat.unsqueeze(-1).expand(batch_size, tag_size, tag_size)
            alpha_exp = alpha.unsqueeze(-1).expand(batch_size, tag_size, tag_size)
            trans_exp = self.transitions.unsqueeze(0).expand(batch_size, tag_size, tag_size)
            mat = trans_exp + alpha_exp + feat_exp
            alpha = log_sum_exp(mat, 2)

        trn_exp = self.transitions[END_TAG].unsqueeze(0).expand(batch_size, tag_size)
        alpha = alpha + trn_exp
        norm = log_sum_exp(alpha, 1).squeeze(-1)

        return norm

    def transition_score(self, labels):
        # labels: batch_size, seq_len
        batch_size, seq_len = labels.size()
        tag_size, _ = self.transitions.size()

        labels_exp = Variable(labels.data.new(batch_size, seq_len+2))
        labels_exp[:, 0] = START_TAG
        labels_exp[:, 1:-1] = labels
        labels_exp[:, -1] = END_TAG

        trn_exp = self.transitions.unsqueeze(0).expand(batch_size, tag_size, tag_size)
        lbl_r = labels_exp[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), tag_size)
        # batch_size, tag_size-1, tag_size
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)

        lbl_l = labels_exp[:, :-1]
        lbl_lexp = lbl_l.unsqueeze(-1)
        trn_score = torch.gather(trn_row, 2, lbl_lexp)
        trn_score = trn_score.sum(1)
        trn_score = trn_score.squeeze(-1)
        return trn_score

    def _viterbi_decode(self, feats):
        # features: seq_size, batch_size, tag_size

        # seq_size, batch_size, tag_size
        pointers = []

        # seq_size, batch_size, tag_size
        seq_len, batch_size, tag_size = feats.size()

        vit = feats.data.new(batch_size, tag_size).fill_(-1e5)
        vit[:,START_TAG] = 0

        # vit: batch_size, tag_size
        vit = Variable(vit.cuda())
        # feat: batch_size, tag_size
        for feat in feats:
            # batch_size, tag_size
            vit_exp = vit.unsqueeze(1).expand(batch_size, tag_size, tag_size)
            trn_exp = self.transitions.unsqueeze(0).expand(batch_size, tag_size, tag_size)
            vit_trn_sum = vit_exp + trn_exp

            # the state of the previous state which maximize the vit value
            # vt_max: batch_size, tag_size
            # vt_argmax: batch_size, tag_size
            vt_max, vt_argmax = vit_trn_sum.max(2)

            vt_max = vt_max.squeeze(-1)
            vit = vt_max + feat
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))
        vit += self.transitions[END_TAG].unsqueeze(0).expand(batch_size, tag_size)

        # seq_len, batch_size, tag_size
        pointers = torch.cat(pointers)
        scores, idx = vit.max(1)
        # idx: batch_size
        idx = idx.squeeze(-1)
        paths = [idx.unsqueeze(1)]
        for argmax in reversed(pointers):
            # idx_exp: batch_size, 1
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)
            paths.insert(0, idx.unsqueeze(1))
        paths = torch.cat(paths[1:], 1)
        return scores, paths

    def _bilstm_score(self, logits, y):
        y_exp = y.unsqueeze(-1)
        logits = logits.transpose(1, 0)
        scores = torch.gather(logits, 2, y_exp).squeeze(-1)
        return scores.sum(1).squeeze(-1)

    def score(self, xs, y, logits=None):
        if logits is None:
            logits = self.get_lstm_features(xs)
        trn_score = self.transition_score(y)
        if self.repack:
            return trn_score
        bilstm_score = self._bilstm_score(logits, y)
        score = trn_score + bilstm_score
        return score

    def loglik(self, xs, y):
        logits = self.get_lstm_features(xs)
        norm_score = self._norm(logits)
        seq_score = self.score(xs, y, logits=logits)
        loglik = seq_score - norm_score
        return loglik, logits
