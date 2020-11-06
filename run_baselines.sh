#!/bin/bash

#run baselines
echo "train linear model"
if ! [[ -f "results/predictions/linear_small_enslp_word2vec.csv" ]]; then
	./run_classifier.sh -s full -p enslp -e word2vec -c linear -d 1
fi

echo "train simple neural networks"

if ! [[ -f "results/predictions/NN_epochs2_ln32_small_enslp_keras.csv" ]]; then
	./run_classifier.sh -s full -p enslp -e keras -c nn -d 1
fi
if ! [[ -f "results/predictions/NN_epochs2_ln32_small_enslp_word2vec.csv" ]]; then
	./run_classifier.sh -s full -p enslp -e word2vec -c nn -d 1
fi