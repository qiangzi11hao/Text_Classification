# -*- coding: utf-8 -*-

import codecs
import numpy as np
import word2vec
import os
import pickle
import re
import random
import collections
import re
import tensorflow as tf
import math
import argparse
# from compiler.ast import flatten
import time
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.feature_extraction.text import  TfidfTransformer


PAD_ID = 0

from tflearn.data_utils import pad_sequences

_GO = "_GO"
_END = "_END"
_PAD = "_PAD"


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub("<sssss>"," thisisendsign", string)
    #string = re.sub("<sssss>"," ", string)
    
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def read_corpus(path, clean=True, TREC=False):
    data = []
    labels = []
    with open(path) as fin:
        for line in fin:
            label, sep, text = line.partition(' ')
            label = int(label)-1
            text = clean_str(text.strip()) if clean else text.strip()
            labels.append(label)
            texts = text.split()
            data.append(texts)
    return data, labels
    
def read_MR(path, seed=1234):
    file_path = os.path.join(path, "rt-polarity.all")
    data, labels = read_corpus(file_path)
    random.seed(seed)
    perm = range(len(data))
    random.shuffle(perm)
    data = [ data[i] for i in perm ]
    labels = [ labels[i] for i in perm ]
    return data, labels

def read_SUBJ(path, seed=1234):
    file_path = os.path.join(path, "subj.all")
    data, labels = read_corpus(file_path)
    random.seed(seed)
    perm = range(len(data))
    random.shuffle(perm)
    data = [ data[i] for i in perm ]
    labels = [ labels[i] for i in perm ]
    return data, labels

def read_CR(path, seed=1234):
    file_path = os.path.join(path, "custrev.all")
    data, labels = read_corpus(file_path)
    random.seed(seed)
    perm = range(len(data))
    random.shuffle(perm)
    data = [ data[i] for i in perm ]
    labels = [ labels[i] for i in perm ]
    return data, labels

def read_MPQA(path, seed=1234):
    file_path = os.path.join(path, "mpqa.all")
    data, labels = read_corpus(file_path)
    random.seed(seed)
    perm = range(len(data))
    random.shuffle(perm)
    data = [ data[i] for i in perm ]
    labels = [ labels[i] for i in perm ]
    return data, labels

def read_TREC(path, seed=1234):
    train_path = os.path.join(path, "TREC.train.all")
    test_path = os.path.join(path, "TREC.test.all")
    train_x, train_y = read_corpus(train_path, TREC=True)
    test_x, test_y = read_corpus(test_path, TREC=True)
    random.seed(seed)
    perm = range(len(train_x))
    random.shuffle(perm)
    train_x = [ train_x[i] for i in perm ]
    train_y = [ train_y[i] for i in perm ]
    return train_x, train_y, test_x, test_y

def read_yelp(path=os.getcwd(), seed=4321 , year= 2013):
    train_path = os.path.join(path, "yelp-%s-train.txt.ss"%year)
    valid_path = os.path.join(path, "yelp-%s-dev.txt.ss"%year)
    test_path = os.path.join(path, "yelp-%s-test.txt.ss"%year)
    train_x, train_y = read_corpus(train_path)
    valid_x, valid_y = read_corpus(valid_path)
    test_x, test_y = read_corpus(test_path)
    random.seed(seed)
    perm = list(range(len(train_x)))
    random.shuffle(perm)
    train_x = [ train_x[i] for i in perm ]
    train_y = [ train_y[i] for i in perm ]
    #data = zip(train_x,train_y)
    #data = sorted(data,key = lambda x:len(x[0]))
    #train_x = [ i[0] for i in data ]
    #train_y = [ i[1] for i in data ]
    maxw = 0
    for para in train_x+valid_x+test_x:
        maxw = max(maxw,len(para))
    print ("maxw=",maxw)
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def read_SST(path=os.getcwd(), seed=1234):
    train_path = os.path.join(path, "stsa.binary.train")
    valid_path = os.path.join(path, "stsa.binary.dev")
    test_path = os.path.join(path, "stsa.binary.test")
    train_x, train_y = read_corpus(train_path)
    valid_x, valid_y = read_corpus(valid_path)
    test_x, test_y = read_corpus(test_path)
    random.seed(seed)
    perm = range(len(train_x))
    random.shuffle(perm)
    train_x = [ train_x[i] for i in perm ]
    train_y = [ train_y[i] for i in perm ]

    maxw = 0
    for para in train_x+valid_x+test_x:
        maxw = max(maxw,len(para))
    print ("maxw=",maxw)
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def read_IMDB(path=os.getcwd(), seed=1234):
    train_path = os.path.join(path, "imdb-train.txt.ss")
    valid_path = os.path.join(path, "imdb-dev.txt.ss")
    test_path = os.path.join(path, "imdb-test.txt.ss")
    train_x, train_y = read_corpus(train_path)
    valid_x, valid_y = read_corpus(valid_path)
    test_x, test_y = read_corpus(test_path)
    random.seed(seed)
    perm = range(len(train_x))
    random.shuffle(perm)
    train_x = [ train_x[i] for i in perm ]
    train_y = [ train_y[i] for i in perm ]
    
    maxw = 0
    for para in train_x+valid_x+test_x:
        maxw = max(maxw,len(para))
    print ("maxw=",maxw)
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def tf_idf(content):
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(content))

    feature_words = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    res = []
    f = open('tf_idf.txt', 'a')
    for i, sentence in enumerate(content):
        tmp = []
        for word in sentence:
            if word in feature_words:
                j = feature_words.index(word)
                f.write(str(weight[i][j]) + ' ')
                tmp.append(weight[i],[j])
            else:
                tmp.append(0)
                print(str(word))
                f.write(str(0))
        res.append(tmp)
        f.write('\n')
    f.close()
    return  np.array(res)


def read(tfidf_path):
    f = open(tfidf_path)
    lines = f.readlines()
    res = []
    for line in lines:
        tfidf = map(float, line.strip())
        res.append(tfidf)
    return np.array(res)