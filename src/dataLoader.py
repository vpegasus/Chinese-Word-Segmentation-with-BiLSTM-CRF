import os
import torch
import re
import pickle
from functools import reduce
import operator
import numpy as np
from config import config

digit_pattern = re.compile(r'[１２３４５６７８９０.％∶]+')
letter_pattern = re.compile(r'[ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ]+')
mark_pattern = re.compile(r'[，。、（）：；＊！？－\-“”《》『』＇｀〈〉…／]+')


def replace(s):
    for digit in digit_pattern.findall(s):
        if len(digit) == 1:
            s = s.replace(digit, 'a')
        elif len(digit) == 2:
            s = s.replace(digit, 'b')
        else:
            s = s.replace(digit, 'c')

    for letter in letter_pattern.findall(s):
        if len(letter) == 1:
            s = s.replace(letter, 'z')
        else:
            s = s.replace(letter, 'y')

    for mark in mark_pattern.findall(s):
        s = s.replace(mark, 'm')

    return s


def longReplace(s):
    ss = s.split('\n')
    ss = [replace(s) for s in ss]
    return '\n'.join(ss)


class Corpus(object):

    def __init__(self, dataDir):

        trainPath = os.path.join(dataDir, 'train.txt')
        testPath = os.path.join(dataDir, 'test.answer.txt')
        self.vocabulary = Vocabulary()
        self.buildVocabulary(trainPath)
        self.buildVocabulary(testPath)
        self.vocabulary.trim(5000)
        self.trainData, self.trainMarkCounter = self.tokenize(trainPath)
        self.testData, self.testMarkCounter = self.tokenize(testPath)

        self.n_token = len(self.vocabulary.idx2word)

    def tokenize(self, fileName, separate=True):
        retSeq = []
        retLabel = []
        with open(fileName, 'r') as f:
            texts = longReplace(f.read())
            mark_counter = len(re.findall('m', texts))
            if separate:
                texts = texts.replace('m', '\n')
            lines = ['A' + _ + 'A' for _ in texts.split('\n') if len(_) > 0]
            for line in lines:
                curr_token = []
                curr_label = []
                line = line.replace('  ', 'A').replace(' ', 'A')
                line = line.replace('AAA', 'A').replace('AA', 'A').replace('AA', 'A')
                assert 'AA' not in line
                for idx, w in enumerate(line):
                    if w == 'A':
                        continue
                    curr_token.append(self.vocabulary[w])
                    if line[idx-1] == 'A' and line[idx+1] == 'A':
                        curr_label.append(3)
                    elif line[idx-1] == 'A' and not line[idx+1] == 'A':
                        curr_label.append(0)
                    elif not line[idx-1] == 'A' and line[idx+1] == 'A':
                        curr_label.append(2)
                    else:
                        curr_label.append(1)

                retSeq.append(curr_token)
                retLabel.append(curr_label)
        return [retSeq, retLabel], mark_counter

    def buildVocabulary(self, filePath):
        with open(filePath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = replace(line)
                for w in line:
                    if w not in ' \n':
                        self.vocabulary.add(w)

    def gen(self,
            window_size=config['window_size'],
            batch_size=config['batch_size'],
            train=True):
        if train:
            seq, labels = self.trainData
        else:
            seq, labels = self.testData
        m = self.vocabulary['m']
        xs = torch.LongTensor(batch_size, window_size)
        ys = torch.LongTensor(batch_size, window_size)
        _ = 0
        for idx in range(len(seq)):
            curr_seq = [m]*(window_size//2) + seq[idx] + [m]*(window_size//2)
            curr_label = [3]*(window_size//2) + labels[idx] + [3]*(window_size//2)
            for i in range(0, len(curr_label)-config['window_size'], config['window_step']):
                xs[_] = torch.IntTensor(curr_seq[i: i+window_size])
                ys[_] = torch.IntTensor(curr_label[i: i+window_size])
                _ += 1
                if _ == batch_size:
                    yield xs, ys
                    _ = 0

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.word_counter = {}

    def add(self, other):
        if other in self.word2idx:
            self.word_counter[other] += 1
        else:
            self.word_counter[other] = 1
            self.word2idx[other] = len(self.idx2word)
            self.idx2word.append(other)

    def trim(self, size):
        self.idx2word.sort(key=lambda x: -self.word_counter[x])
        for idx, word in enumerate(self.idx2word):
            self.word2idx[word] = idx
        if size > len(self.idx2word):
            return
        for w in self.idx2word[size:]:
            del self.word2idx[w]
        self.idx2word = self.idx2word[:size]

    def __getitem__(self, item):
        if item in self.word2idx:
            return self.word2idx[item]
        else:
            return len(self.word2idx) - 1
