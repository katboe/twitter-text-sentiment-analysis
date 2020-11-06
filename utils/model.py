import sys
import csv
import numpy as np
import pandas as pd

import configparser
import argparse

import keras
import keras.layers as layers
from keras.models import Sequential


import sys
sys.path.append('.')

from utils.io_helper import IOHelper
from utils.data_helper import DataHelper

from utils.Attention_layer import Attention1
from utils.Attention_layer import Attention2

CONFIG_PATH = 'utils/model.config'


class Model():
    def __init__(self, io, dh, arch_name = 'LSTM', epochs = 2, ln = 32, batch_size = 32):
        """Initialization of parameters
        
        :param io: instance of IOHelper (for input/output)
        :param dh: instance of DataHelper (for keras specific data manipulation)
        :param arch_name: type of architecture
        :param epochs: number of training epochs
        :param ln: length of considered sequence
        :param batch_size: batchsize for training
        """

        self.io = io
        self.dh = dh

        self.arch_name = arch_name
        self.epochs = epochs
        self.ln = ln
        self.batch_size = batch_size
        
        #initialization of model and model name
        self.model = None
        self.model_name = f'{self.arch_name}_epochs{self.epochs}_ln{self.ln}'

    def build_model(self, embedding_layer):
        """
        Builds the keras model corresponding to specified type of architecture
        :param embedding_layer: preloaded embedding layer for keras model
        :return: keras model
        """

        model = Sequential()
        model.add(embedding_layer)

        #3-layered bidirectional LSTM
        if self.arch_name == "NN":
            model.add(layers.Flatten())
            model.add(layers.Dense(1024, activation='relu'))
            model.add(layers.Dense(2, activation='softmax'))

        elif self.arch_name == "BiLSTM":
            model.add(layers.Bidirectional(layers.LSTM(100, return_sequences=True, dropout=0.1)))
            model.add(layers.Bidirectional(layers.LSTM(100, return_sequences=True)))
            model.add(layers.Bidirectional(layers.LSTM(100)))
            model.add(layers.Dense(50, activation='relu'))
            model.add(layers.Dense(2, activation='softmax'))
            
        #two-layered forward LSTM
        elif self.arch_name == "LSTM":
            model.add(layers.LSTM(200, return_sequences=True))
            model.add(layers.LSTM(100, return_sequences=False))
            model.add(layers.Dense(1000, activation='relu'))
            model.add(layers.Dense(2, activation='softmax'))

        #CNN model with attention layer
        elif self.arch_name == "CNN":
            model.add(layers.Conv1D(100, 7, activation='relu', padding='same'))
            model.add(layers.MaxPooling1D(2))
            model.add(layers.Conv1D(64, 7, activation='relu', padding='same'))
            model.add(Attention1())
            model.add(layers.Dense(2, activation='sigmoid'))

        #single-layered bidirectional GRU with attention layer
        elif self.arch_name == "GRU":
            model.add(layers.Bidirectional(layers.GRU(200, return_sequences=True, dropout=0.1)))
            model.add(Attention2(self.ln))
            model.add(layers.Dense(200, activation='relu'))
            model.add(layers.Dense(2, activation='softmax'))

        #Stacked CNN and LSTM with attention layer
        elif self.arch_name == "CNN_LSTM":
            model.add(layers.Conv1D(128, 3, activation='relu', padding='valid')) # filters: 100, kernel_size = 7 -> output = 100 
            model.add(layers.MaxPooling1D(2))
            model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
            model.add(Attention1())
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(2, activation='softmax'))

        else:
            raise NotImplementedError("Specified architecture is not implemented:",
                                              self.arch_name)
        return model



    def fit(self, X_train, Y_train, X_test, Y_test=None, kfold = None):
        """
        Fits the model to the given training data and evaluated if test-labels are given

        :param X_train: tokenized training features
        :param Y_train: training labels as categorical variables
        :param X_test: tokenized test features
        :param Y_test: test labels as categorical variables, None for official test data

        :return: test accuracy if model evaluated, otherwise None
        """

        self.model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
        
        test_acc = None

        #on official test data, fit without validation data
        if Y_test is None:
            self.model.fit(X_train, Y_train, epochs=self.epochs)

        #during cross validation, fit and evaluate on validation data
        else:
            self.model.fit(X_train, Y_train, epochs=self.epochs, validation_data=(X_test, Y_test), batch_size = self.batch_size)
            test_loss, test_acc = self.model.evaluate(X_test, Y_test, verbose=2)

        #save model when trained on full dataset
        if Y_test is None:
            mode_path_h5, model_path_json = self.io.getModelPaths(self.model_name)
            try:
                self.model.save(mode_path_h5)
                model_json = self.model.to_json()
                with open(model_path_json, "w") as json_file:
                    json_file.write(model_json)

            except:
                 print(f"Model could not be saved.")

        return test_acc

    def  crossValidate(self, kfold = 5):
        """
        Compute cross validation: repeat training on deterministic kfold (precomputed)

        :param kfold: number of folds
        """

        Acc_scores = []

        #for each fold, compute training
        for k in range(kfold):
            print(f"-------------- Starting cross validation {k+1} of {kfold} --------------")
            test_acc = self.train(k)
            Acc_scores.append(test_acc)

        #save accuracies for later analysis
        self.io.writeAccuracyFile(Acc_scores, self.model_name)


    def train(self, k = None):
        """
        Train the model and predict labels for test set

        :param k: current kfold-iteration, None for training on full training dataset
        """

        # load both train and test data
        if k == None:
            #load full training set and official test set without labels
            X_train_raw, Y_train_raw, X_test_raw = self.io.readData()
            Y_test_raw = None
        else:
            #load crossvalidation training and test sets
            X_train_raw, Y_train_raw, X_test_raw, Y_test_raw  = self.io.readCrossValidationData(k)

        #manipulate data for model training
        X_train, Y_train, X_test, Y_test = self.dh.prepareData(X_train_raw, Y_train_raw, X_test_raw, Y_test_raw)
        
        #load specified embedding layer and build model with it
        emb_layer = self.dh.getEmbeddingLayer()
        self.model = self.build_model(emb_layer)
        
        #print model architecture for first run
        if k == None or k == 0:
            self.model.summary()

        test_acc = self.fit(X_train, Y_train, X_test, Y_test)

        #save predictions (probability output)
        pred = self.model.predict(X_test)
        self.io.writeProbabilityPredictionFile(pred, self.model_name, Y_test_raw, k)

        print("Training finished")

        #free memory
        del self.model

        return test_acc


if __name__ == "__main__":
    """
    Train model with constant parameters given in config file and adaptable parameters passed as arguments.
    Model can be cross validated via a KFold as well as trained on the entire test data set. 
    """

    #parse all arguments
    parser = argparse.ArgumentParser()
     #size of twitter dataset
    parser.add_argument('--small', dest='size', action='store_const', const='small')
    parser.add_argument('--full', dest='size', action='store_const', const='full')
    #type of embedding
    parser.add_argument('--keras', dest='emb_type', action='store_const', const='keras')
    parser.add_argument('--word2vec', dest='emb_type', action='store_const', const='word2vec')
    parser.add_argument('--fasttext', dest='emb_type', action='store_const', const='fasttext')
    #type of architecture
    parser.add_argument('--lstm', dest='arch_name', action='store_const', const='LSTM')
    parser.add_argument('--bilstm', dest='arch_name', action='store_const', const='BiLSTM')
    parser.add_argument('--cnn', dest='arch_name', action='store_const', const='CNN')
    parser.add_argument('--cnnlstm', dest='arch_name', action='store_const', const='CNN_LSTM')
    parser.add_argument('--gru', dest='arch_name', action='store_const', const='GRU')
    parser.add_argument('--nn', dest='arch_name', action='store_const', const='NN')
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

    #initialize IO- and Data-Helper
    io = IOHelper(size=args.size , 
                emb_type =args.emb_type,
                 preprocess = args.preprocess)

    dh = DataHelper(io, 
                ln=(int)(config['MODEL']['LENGTH']), 
                nr_words=(int)(config['MODEL']['NUM_WORDS']), 
                dim=(int)(config['MODEL']['DIM']))

    #initialize model class for training
    model = Model(io, dh, arch_name=args.arch_name, 
                epochs=(int)(config['MODEL']['EPOCHS']), 
                ln=(int)(config['MODEL']['LENGTH']),
                batch_size=(int)(config['MODEL']['BATCH_SIZE']))

    if args.cross_valid:
        #print("Cross Validation")
        model.crossValidate(kfold = (int)(config['DEFAULT']['KFOLD']))

    print("Train on full dataset")
    model.train()