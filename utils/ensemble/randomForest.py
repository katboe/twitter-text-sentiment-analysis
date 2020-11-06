import os
import configparser
import argparse
import numpy as np

from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.append('.')

from utils.io_helper import IOHelper

CONFIG_PATH = 'utils/model.config'

if __name__ == "__main__":
    """
    This script trains a random forest classifier on prediction files of specified models. 
    The prediction files have to be precomputed.
    """
    
    print("Computing Random Forest Ensemble Classifier")

    #read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default=None, nargs="?",
                        help='Path to list of models')
    parser.add_argument('--0', dest='cross_valid', action='store_false')
    parser.add_argument('--1', dest='cross_valid', action='store_true')

    args = parser.parse_args()

    #read model config file
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
            #train RandomForest Classifier
            clf = RandomForestClassifier(n_estimators=(int)(config['RANDOMFOREST']['N_ESTIMATORS']),
                                        max_depth=(int)(config['RANDOMFOREST']['MAX_DEPTH']), 
                                        random_state=(int)(config['RANDOMFOREST']['RANDOM_STATE']))
            clf.fit(X_train, Y_train)
            #evaluate model
            pred = clf.score(X[i], Y[i])
            scores.append(pred)

        print(f"score: {np.mean(scores)}")

    #read probabilitys data and true labels
    X_test = io.readProbabilityData(modelNames)

    #concatenate all predictions of cross validation
    X_train = []
    Y_train = []
    for j in range((int)(config['DEFAULT']['KFOLD'])):
        X_train = X_train + X[j]
        Y_train = Y_train + Y[j]
    #train RandomForest Classifier on full dataset
    clf = RandomForestClassifier(n_estimators=(int)(config['RANDOMFOREST']['N_ESTIMATORS']),
                                max_depth=(int)(config['RANDOMFOREST']['MAX_DEPTH']), 
                                random_state=(int)(config['RANDOMFOREST']['RANDOM_STATE']))
    clf.fit(X_train, Y_train)

    #predict and write test file
    pred = clf.predict(X_test)
    model_name = "random_forest"
    io.writePredictionFile(pred, model_name)