import os
import configparser
import argparse
import numpy as np

from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.append('.')

from utils.io_helper import IOHelper

CONFIG_PATH = 'utils/model.config'

def majorityPrediction(X_test, Y_test=None, weighted=False):
    #compute majority voting
    pred = []
    for row in X_test:  
        if args.weighted:
            vote = np.zeros(2)
            vote[0] = len(row) - np.sum(row)
            vote[1] = np.sum(row)
            pred.append(vote)
            
        else:
            vote = np.zeros(2)
            for entry in row:
                if entry > 0.5:
                    vote[1] += 1
                else:
                    vote[0] += 1
            pred.append(vote)
         
    if Y_test is None:
        return pred

    else:
        #evaluate on Y_test
        score = 0
        for i in range(len(Y_test)):
            if (Y_test[i] == 0):
                if (pred[i][0] >= pred[i][1]):
                    score += 1
            else:
                if (pred[i][0] < pred[i][1]):
                    score += 1

        return float(score)/len(Y_test)

if __name__ == "__main__":
    """
    Computes a simple or weighted majority Vote for prediction files of specified models. 
    The prediction files have to be precomputed.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default=None, nargs="?",
                        help='Path to list of models')
    parser.add_argument('--0', dest='cross_valid', action='store_false')
    parser.add_argument('--1', dest='cross_valid', action='store_true')
    parser.add_argument('--weighted', dest='weighted', action='store_true')
    parser.add_argument('--simple', dest='weighted', action='store_false')

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
            X_val, Y_val = X[i], Y[i]
            score = majorityPrediction(X_val, Y_val, weighted=args.weighted)
            scores.append(score)

        print(f"score: {np.mean(scores)}")

    #read probabilitys data and true labels
    X_test = io.readProbabilityData(modelNames)

    pred = majorityPrediction(X_test, weighted=args.weighted)
    #write final prediction to file
    if args.weighted:
        model_name = "maj_prediction_weighted"
    else:
        model_name = "maj_prediction"


    io.writePredictionFile(pred, model_name)