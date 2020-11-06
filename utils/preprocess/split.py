import sys
import csv
import pandas as pd

import argparse

from sklearn.model_selection import KFold

if __name__ == "__main__":
    """ 
    This script splits the training data into several crossvalidation training and validation files
    """

    #parse all arguments
    parser = argparse.ArgumentParser()
    #path to config file
    parser.add_argument('--dataDir', type=str, default='data', nargs="?", help='Path to config file for model')
    #size of twitter dataset
    parser.add_argument('--small', dest='size', action='store_const', const='small')
    parser.add_argument('--full', dest='size', action='store_const', const='full')
    #type of preprocessing
    parser.add_argument('--enslp', dest='preprocess', action='store_const', const='enslp')
    parser.add_argument('--enrs', dest='preprocess', action='store_const', const='enrs')
    parser.add_argument('--num', dest='preprocess', action='store_const', const='num')
    parser.add_argument('--none', dest='preprocess', action='store_const', const='none')

    args = parser.parse_args()
    
    #open preprocessed training file
    f_tr = open(f"{args.dataDir}/train_{args.size}_{args.preprocess}.txt", encoding='utf-8')
    
    # Preprocess training data
    text_train = f_tr.read()
    text_train = text_train.split("\n")

    cv = KFold(n_splits=5, random_state=7)

    df_train = pd.DataFrame(text_train)
    counter = 0
    for train_index, test_index in cv.split(df_train):

        X_train, X_test = df_train.iloc[train_index], df_train.iloc[test_index]

        f_cv_train = f"{args.dataDir}/cv_train_{args.size}_{args.preprocess}_{counter+1}.txt"
        f_cv_test = f"{args.dataDir}/cv_test_{args.size}_{args.preprocess}_{counter+1}.txt"

        #save CV data
        X_train.to_csv(f_cv_train, index=False, header=False)

        X_test.to_csv(f_cv_test, index=False, header=False)
       
        counter += 1

    

