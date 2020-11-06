#!/usr/bin/env python3
import numpy as np
import pickle
import random
import sys
from gensim.models import FastText

import argparse
import configparser

sys.path.insert(0,'.')

CONFIG_PATH = 'utils/embeddings/fasttext/fasttext.config'

if __name__ == "__main__":
    #parse all arguments
    parser = argparse.ArgumentParser()
    #path to data directory
    parser.add_argument('--dataDir', type=str, default='data', nargs="?", help='Path to config file for model')
    #path to embedding directory file
    parser.add_argument('--embDir', type=str, default='data/embeddings', nargs="?", help='Path to config file for model')
    #size of twitter dataset
    parser.add_argument('--small', dest='size', action='store_const', const='small')
    parser.add_argument('--full', dest='size', action='store_const', const='full')
    #type of preprocessing
    parser.add_argument('--enslp', dest='preprocess', action='store_const', const='enslp')
    parser.add_argument('--enrs', dest='preprocess', action='store_const', const='enrs')
    parser.add_argument('--num', dest='preprocess', action='store_const', const='num')
    parser.add_argument('--none', dest='preprocess', action='store_const', const='none')
  
    args = parser.parse_args()

    #read data
    sentences = []
    with open(f'{args.dataDir}/train_{args.size}_{args.preprocess}.txt', "r") as f:
            content = [line.strip().split(" ") for line in f.readlines()]
            sentences_train = [s[1:] for s in content if len(s) > 1]
    
    with open(f'{args.dataDir}/test_{args.preprocess}.txt', "r") as f:
            content = [line.strip().split(" ") for line in f.readlines()]
            sentences_test = [s[1:] for s in content if len(s) > 1]

    sentences = sentences_train + sentences_test

    #parse config file
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    print("config file read")

    #train FastText model
    model = FastText(size=(int)(config['DEFAULT']['DIM']), workers=(int)(config['DEFAULT']['NUM_WORKERS']))
    model.build_vocab(sentences)

    print("training FastText model...")
    model.train(sentences, total_examples=model.corpus_count, 
                epochs=(int)(config['DEFAULT']['EPO']), 
                model='cbow', 
                min_n= (int)(config['DEFAULT']['MIN_N']), 
                max_n=(int)(config['DEFAULT']['MAX_N']))

    #save FastText model
    model.save(f"{args.embDir}/fasttext_{args.size}_{args.preprocess}_{(int)(config['DEFAULT']['DIM'])}.model")