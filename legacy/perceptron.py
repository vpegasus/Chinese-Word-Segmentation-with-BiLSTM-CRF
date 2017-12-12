from config import config
from dataLoader import Corpus
import numpy as np


class perceptron(object):
    def __init__(self):
        self.n_features = config['n_unigram'] + config['n_bigram']*3
        self.theta = np.random.rand(self.n_features, 4)

    def convert(self, xs):
        xs = xs.copy()
        xs[:,1] += config['n_unigram']
        xs[:,2] += config['n_unigram'] + config['n_bigram']
        xs[:,3] += config['n_unigram'] + config['n_bigram'] * 2
        return xs

    def pred(self, xs):
        N, _ = xs.shape
        xs = self.convert(xs)
        label = np.zeros(N)
        for idx, x in enumerate(xs):
            logit = self.theta[x,:]
            label[idx] = np.argmax(logit.sum(axis=0))
        return label

    def train(self, xs, ys):
        N, _ = xs.shape
        xs = self.convert(xs)
        for x, y in zip(xs, ys):
            logit = self.theta[x,:]
            pred = np.argmax(logit.sum(axis=0))
            if pred == y:
                continue
            self.theta[x, y] += config['smooth']
            self.theta[x, pred] -= config['smooth']

