import torch
import time
from torch import nn
from torch.autograd import Variable
from config import config
import numpy as np

def train(model,corpus,ahead=1000):
    model.train()
    since = time.time()
    batch_id = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    gen = corpus.gen(train=True)
    total_loss = 0
    print_period = 100
    for xs, ys in gen:
        xs, ys = Variable(xs.cuda()), Variable(ys.cuda())
        #output = model(xs).view(-1, 4)
        #ys = ys.view(-1)
        loglik, _ = model.loglik(xs, ys)
        loglik = loglik.mean()
        loss = -loglik
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.data[0]
        batch_id += 1
        if batch_id % print_period == 0:
            print('time: %.3f s, loss: %.3f' % (time.time() - since,
                                                total_loss / print_period))
            total_loss = 0
            since = time.time()
        if batch_id == ahead * print_period:
            break

def evaluate(model, corpus):
    model.eval()
    since = time.time()
    gen = corpus.gen(train=False, batch_size=config['test_batch_size'])
    total_loss = 0
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    batch_counter = 0
    mark_counter = 0
    for xs, ys in gen:
        xs, ys = Variable(xs.cuda()), Variable(ys.cuda())
        loglik, logits = model.loglik(xs, ys)
        scores, prediction = model._viterbi_decode(logits)
        neg_loglik = -loglik.mean().data[0]
        total_loss += neg_loglik
        xs, ys, prediction = xs, ys>1, prediction.cuda()>1
        xs, ys, prediction = xs.data, ys.data, prediction.data

        prediction[xs == 0] = 1
        before = torch.ByteTensor(xs.size(0), xs.size(1)).cuda()
        before[:,:xs.size(1)-1] = (xs == 0)[:,1:]
        before[:,xs.size(1)-1] = 0
        prediction[before] = 1

        mark_counter += (xs == 0).sum()
        true_pos += (ys * prediction).sum()
        false_neg += (ys * (1-prediction)).sum()
        false_pos += ((1-ys) * prediction).sum()
        true_neg += ((1-ys) *(1-prediction)).sum()
        batch_counter += 1
    print('-' * 15)
    f1 = print_info(true_pos, false_pos, true_neg, false_neg, mark_counter, corpus.testMarkCounter, since, 0, batch_counter)
    print('-' * 15)
    return f1

def train_LSTM(model, corpus):
    model.train()
    since = time.time()
    batch_id = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    gen = corpus.gen(train=True)
    total_loss = 0
    print_period = 100
    for xs, ys in gen:
        xs, ys = Variable(xs.cuda()), Variable(ys.cuda())
        output = model(xs).view(-1, 4)
        ys = ys.view(-1)
        loss = criterion(output, ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.data[0]
        batch_id += 1
        if batch_id % print_period == 0:
            print('time: %.3f s, loss: %.3f' % (time.time() - since,
                                                total_loss / print_period))
            total_loss = 0
            since = time.time()

def evaluate_LSTM(model, corpus):
    model.eval()
    since = time.time()
    gen = corpus.gen(train=False, batch_size=config['test_batch_size'])
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    batch_counter = 0
    mark_counter = 0
    for xs, ys in gen:
        xs, ys = Variable(xs.cuda()), Variable(ys.cuda())
        prediction = model(xs)
        total_loss += criterion(prediction.view(-1, 4), ys.view(-1)).data[0]
        xs, ys, prediction = xs.data, ys.data>1, prediction.data.max(2)[1]>1

        prediction[xs == 0] = 1
        before = torch.ByteTensor(xs.size(0), xs.size(1)).cuda()
        before[:,:xs.size(1)-1] = (xs == 0)[:,1:]
        before[:,xs.size(1)-1] = 0
        prediction[before] = 1

        mark_counter += (xs == 0).sum()
        true_pos += (ys * prediction).sum()
        false_neg += (ys * (1-prediction)).sum()
        false_pos += ((1-ys) * prediction).sum()
        true_neg += ((1-ys) *(1-prediction)).sum()
        batch_counter += 1
    return print_info(true_pos, false_pos, true_neg, false_neg,
        mark_counter, corpus.testMarkCounter, since, total_loss, batch_counter)

def print_info(true_pos, false_pos, true_neg, false_neg,
               mark_counter, testMarkCounter, since, total_loss, batch_counter):
    print('-' * 15)
    print('TP: %d\nTN: %d\nFP: %d\nFN: %d\nmark_counter: %d\n' %
          (true_pos-mark_counter, true_neg, false_pos, false_neg, testMarkCounter))
    true_pos += testMarkCounter - mark_counter
    print('-' * 15)
    print('time: %.3f s\nloss: %.4f\nprecision: %.4f\nrecall: %.4f\nf score: %.4f\naccuracy: %.4f'
          % (
              time.time() - since,
              total_loss / batch_counter,
              true_pos / (true_pos + false_pos + 1e-6),
              true_pos / (true_pos + false_neg + 1e-6),
              2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6),
              (true_pos + true_neg) / (true_neg + true_pos + false_neg + false_pos)
          ))
    print('-' * 15)
    return 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
