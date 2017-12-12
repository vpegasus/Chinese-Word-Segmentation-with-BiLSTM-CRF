import os
import re
import numpy as np
from config import config
import pickle


digit_pattern = re.compile(r'[１２３４５６７８９０.％∶]{1}')
letter_pattern = re.compile(r'[ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ]{1}')
mark_pattern = re.compile(r'[，。、（）：；＊！？－\-“”《》『』＇｀〈〉…／]{1}')


def replace(s):
    for digit in digit_pattern.findall(s):
        s = s.replace(digit, 'd')

    for letter in letter_pattern.findall(s):
        s = s.replace(letter, 'l')

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

        single_voc_file = os.path.join(config['data_dir'], 'single_voc.pkl')
        double_voc_file = os.path.join(config['data_dir'], 'double_voc.pkl')
        if os.path.exists(single_voc_file):
            self.single_voc = Vocabulary(single_voc_file)
            self.double_voc = Vocabulary(double_voc_file)
        else:
            self.single_voc = Vocabulary()
            self.double_voc = Vocabulary()
            self.buildVocabulary(trainPath)
            self.buildVocabulary(testPath)
            self.single_voc.save(single_voc_file)
            self.double_voc.save(double_voc_file)

        self.single_voc.trim(config['n_unigram'])
        self.double_voc.trim(config['n_bigram'])

        self.trainSet = self.getDataSet(trainPath)
        self.testSet = self.getDataSet(testPath)

    def getDataSet(self, filePath):
        rst = []
        with open(filePath, 'r', encoding='utf8') as f:
            lines = f.readlines()
        for line in lines:
            assert '\n' in line
            tokens, labels = self.tokenize(line)
            rst.append([tokens, labels])
        return rst

    def tokenize(self, s):

        labels = []
        s = s.replace('  ', 'A').replace(' ', 'A')
        raw_s = s.replace('A', '')
        s = 'A' + s + 'A'
        s = s.replace('AAA', 'A').replace('AA', 'A').replace('AA', 'A')
        assert 'AA' not in s
        for idx, w in enumerate(s):
            if w == 'A':
                continue
            if s[idx-1] == 'A' and s[idx+1] == 'A':
                labels.append(3)
            elif s[idx-1] == 'A' and not s[idx+1] == 'A':
                labels.append(0)
            elif not s[idx-1] == 'A' and s[idx+1] == 'A':
                labels.append(2)
            else:
                labels.append(1)
        assert len(labels) == len(raw_s)

        N = len(raw_s)
        tokens = np.zeros(shape=(N, 4), dtype=np.int32)

        raw_s = 'p' + raw_s + 'p'
        for i in range(N):
            tokens[i][0] = self.single_voc[raw_s[i+1]]
            tokens[i][1] = self.double_voc[raw_s[i:i+2]]
            tokens[i][2] = self.double_voc[raw_s[i+1:i+3]]
            tokens[i][3] = self.double_voc[raw_s[i]+raw_s[i+2]]

        return tokens, np.array(labels)

    def buildVocabulary(self, filePath):
        with open(filePath, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                line = replace(line)
                line = line.replace(' ', '').replace('\n', '')
                line = 'p' + line + 'p'
                for w in line[1:-1]:
                    self.single_voc.add(w)
                for idx in range(len(line)-1):
                    self.double_voc.add(line[idx:idx+2])


class Vocabulary(object):
    def __init__(self, fileName=None):
        self.word2idx = {}
        self.idx2word = []
        self.word_counter = {}
        if fileName is not None:
            self.word2idx, self.idx2word, self.word_counter = \
            pickle.load(open(fileName, 'rb'))

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

    def save(self, fileName):
        pickle.dump([self.word2idx, self.idx2word,
                     self.word_counter],
                    open(fileName, 'wb'))

if __name__ == '__main__':
    cor = Corpus(dataDir='./dataset/')
