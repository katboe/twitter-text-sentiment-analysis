import os
import configparser
import argparse
import numpy as np
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense
from keras.layers import Dropout
import sys
sys.path.append('.')

from utils.io_helper import IOHelper

CONFIG_PATH = 'utils/model.config'

if __name__ == "__main__":
    """
    This script trains a simple neural network classifier on prediction files of specified models. 
    The prediction files have to be precomputed.
    """

    print("Computing Neural Network Ensemble Classifier")

    #read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default=None, nargs="?",
                        help='Path to list of models')
    parser.add_argument('--0', dest='cross_valid', action='store_false')
    parser.add_argument('--1', dest='cross_valid', action='store_true')

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    if args.model_file == None:
         raise Exception("Specify filepath to list of modelnames.") 

    io = IOHelper()

    #read list of models

    modelNames = io.readModels(args.model_file)

    #read probabilitys data and true labels
    X, Y = io.readProbabilityData(modelNames, (int)(config['DEFAULT']['KFOLD']))

    # cross validation
    if args.cross_valid:
        scores = []

        for i in range((int)(config['DEFAULT']['KFOLD'])):
            print(f"Cross validation {i+1} out of {(int)(config['DEFAULT']['KFOLD'])}")
            #concatenate predictions but one
            X_train = []
            Y_train = []
            for j in range((int)(config['DEFAULT']['KFOLD'])):
                if (i != j):
                    X_train = X_train + X[j]
                    Y_train = Y_train + Y[j]

            #modify data for input in classifier
            Y_train = to_categorical(Y_train)
            X_train = np.array(X_train)
            X_test = np.array(X[i])
            Y_test = to_categorical(Y[i])

            #build and fit classifier
            model = Sequential()
            model.add(Dense(len(modelNames)*2, activation='relu', input_dim=len(modelNames)))
            model.add(Dense(len(modelNames)*2, activation='relu',))
            model.add(Dropout(0.1))
            model.add(Dense(2, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X_train, Y_train, epochs=1, validation_data=(X_test, Y_test))
            #evaluate classifier
            test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
            scores.append(test_acc)
            
            del model

        print(f"score: {np.mean(scores)}")

    print("predict testfile")
    #read probabilitys data and true labels
    X_test = io.readProbabilityData(modelNames)

    #concatenate all predictions of cross validation
    X_train = []
    Y_train = []
    for j in range((int)(config['DEFAULT']['KFOLD'])):
        X_train = X_train + X[j]
        Y_train = Y_train + Y[j]

    #modify data for input in classifier
    Y_train = to_categorical(Y_train)
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    #build and fit classifier
    model = Sequential()
    model.add(Dense(len(modelNames)*2, activation='relu', input_dim=len(modelNames)))
    model.add(Dense(len(modelNames)*2, activation='relu',))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=1)

    #predict labels of test file and write prediction file
    pred = model.predict(X_test)
    model_name = "neural_network"
    io.writePredictionFile(pred, model_name)