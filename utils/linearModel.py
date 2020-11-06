#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import sklearn as skl
import pandas as pd
from sklearn import linear_model
import joblib

from sklearn.model_selection import KFold
import csv

import configparser
import argparse


import sys
sys.path.append('.')

from utils.io_helper import IOHelper
from utils.data_helper import DataHelper

CONFIG_PATH = 'utils/model.config'


def embed2featureWords(sentence, emb, emb_dim):
    #for each sentence compute mean of all word vectors
    featVec = np.zeros(emb_dim)
    count = 0
    for word in sentence:
        try:
            featVec += emb[word]
            count += 1
        except KeyError:
            count = count

    if count == 0:
        return featVec
    else:
        return featVec/(count)


def score(ys_true: np.ndarray, ys_pred: np.ndarray):
    return skl.metrics.accuracy_score(ys_true, ys_pred)


if __name__ == "__main__":
    """
    Train linear model with constant parameters given in config file and adaptable parameters passed as arguments.
    Model can be cross validated via a KFold as well as trained on the entire test data set. 
    """
    
    model_name = "linear"

    #parse all arguments
    parser = argparse.ArgumentParser()
     #size of twitter dataset
    parser.add_argument('--small', dest='size', action='store_const', const='small')
    parser.add_argument('--full', dest='size', action='store_const', const='full')
    #type of embedding
    parser.add_argument('--keras', dest='emb_type', action='store_const', const='keras')
    parser.add_argument('--word2vec', dest='emb_type', action='store_const', const='word2vec')
    parser.add_argument('--fasttext', dest='emb_type', action='store_const', const='fasttext')
    #type of preprocessing
    parser.add_argument('--enslp', dest='preprocess', action='store_const', const='enslp')
    parser.add_argument('--enrs', dest='preprocess', action='store_const', const='enrs')
    parser.add_argument('--num', dest='preprocess', action='store_const', const='num')
    parser.add_argument('--none', dest='preprocess', action='store_const', const='none')
    #boolean for cross validation
    parser.add_argument('--0', dest='cross_valid', action='store_false')
    parser.add_argument('--1', dest='cross_valid', action='store_true')

    args = parser.parse_args()

    #parse config file
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    print("Config file read")

    #load embedding, read data and tokenize
    if args.emb_type == 'keras':
        raise NotImplementedError("Keras Embedding is not applicable to linear model.",
                                              self.emb_type)
    
    io = IOHelper(size = args.size, emb_type = args.emb_type, preprocess = args.preprocess)

    dh = DataHelper(io, 
                ln=(int)(config['MODEL']['LENGTH']), 
                nr_words=(int)(config['MODEL']['NUM_WORDS']), 
                dim=(int)(config['MODEL']['DIM']))
    #read data
    X_raw, Y_raw, X_test_raw = io.readData()

    #prepare data for training
    emb, emb_dim = dh.getEmbedding()

    #compute feature vectors
    X_feat = []

    X_feat = [embed2featureWords(row, emb, emb_dim) for row in X_raw]
    
    X_feat = pd.DataFrame(X_feat)
    y = pd.DataFrame(Y_raw)

    if args.cross_valid:
        #--------------------------------------------------------------------------
        #           cross validation
        #--------------------------------------------------------------------------
        cv = KFold(n_splits=  (int)(config['DEFAULT']['KFOLD']), random_state=7)
        Acc_scores = []
        counter = 0

        for train_index, test_index in cv.split(X_feat, y):
            print("Kfold iteration: {}".format(counter))
            counter = counter + 1

            X_train, X_test = X_feat.iloc[train_index], X_feat.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            print("Training set size {} | Validation set size {}".format(X_train.shape, X_test.shape))

            #create Model
            svm = linear_model.SGDClassifier(loss='hinge')

            #train Model
            svm.fit(X_train, y_train.values.ravel())
    
            #test Model
            test_ys_pred = svm.predict(X_test)
            print('fitting score:', score( y_test, test_ys_pred))
            Acc_scores.append(score( y_test, test_ys_pred))


        io.writeAccuracyFile(Acc_scores, model_name)
        #--------------------------------------------------------------------------
        

    #linear SVM
    svm = linear_model.SGDClassifier(loss='hinge')
    svm.fit(X_feat, y.values.ravel())
    
    #predict test file
    X_test = [embed2featureWords(row, emb, emb_dim) for row in X_test_raw]
    pred = svm.predict(X_test)

    print("TestData predicted")
    io.writePredictionFile(pred, model_name)