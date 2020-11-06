# Twitter Text Sentiment Analysis
Authors: Judith Beestermöller, Katharina Börsig, Zuzana Frankovská and Katja Möhring

This repository contains the implementation for training an ensemble method of complex deep neural networks in order to perform text classification on twitter data. The project was performed as part of the Computer Intelligence Lab 2020 at ETH Zurich. For further details on the preprocessing, embeddings and models, please refer to our project report provided in the 'report' folder.

### How to run the scripts:

1. Setup the environment by installing all requirements: `pip install -r requirements.txt`.
2. Copy the given twitter data files (5 files in total) into the directory 'twitterData'. The files are supposed to have the following names:  

	- `train_neg.txt`, `train_pos.txt`	(full dataset, labeled)
	- `train_neg_full.txt`, `train_pos_full.txt`	(small dataset, labeled)
	- `test_data.txt`	(test dataset, unlabeled)


3. Make the run scripts executable: `chmod +x run_*`.

Prediction files computed after training the classifiers can be found in the 'results/predictions'-directory.

### Final Model and Baselines
For computing the final model as well as the baseline models run scripts have been constructed containing all necessary parameters. 

- For the final model: `./run_finalModel.sh`.  
The resulting prediction file can be found here: 'results/predictions/neural_network.csv'.

- For the baseline models: `./run_baselines.sh`.

### Preprocessing & Word Embeddings

For preprocessing the data and training word embeddings execute: `./run_setup.sh`.  
The possible choices for preprocessing are:
- num: remove numbers
- enrs: remove numbers, handle emojis, stemming
- enslp: remove numbers, handle emojis, stemming, lemmatization

The following parameters can be applied:
```python
./run_setup.sh  
        --size small | full  
        --preprocess none | num | enrs | enslp
        --embedding word2vec | fasttext
```
### Classifiers

For training a classifier on the twitter data execute: `./run_classifier.sh`.  
The following parameters can be applied:  
```python
./run_classifier.sh  
        --size small | full
        --preprocess none | num | enrs | enslp
        --embedding word2vec | fasttext
        --classifier linear | nn | lstm | bilstm | gru | cnn | cnnlstm
        --debug 0 | 1 (cross validation boolean)
```
### Ensemble Methods

For training an ensemble method on a set of precomputed models: `./run_ensemble.sh`.  
The following parameters can be applied:
```python
./run_ensemble.sh  
        --modelfile {relative path to file containing list of models}
        --ensemble Majority | WeightedMajority | RandomForest | NN
        --debug 0 | 1 (cross validation boolean)
```
