from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
from sklearn.model_selection import KFold
from sklearn import linear_model
import sklearn
import numpy as np
import pandas as pd
import csv
import sys

#################################################
#           gensim word2vec
#################################################

name = sys.argv[1]

w2vsize = 50
nr_epochs = 10
gw2vmodel = Word2Vec( min_count = 15, workers=4, iter=1, size=w2vsize)

sentences = []
with open(f"data/train_small_{name}.txt") as f:
    for line in f:
         sentences.append(line.strip().split()[1:])

gw2vmodel.build_vocab(sentences)
gw2vmodel.train(sentences, total_examples=gw2vmodel.corpus_count, epochs=nr_epochs)

###########################################################
####            training with cross validation          ###
###########################################################

wv = gw2vmodel.wv

cv = KFold(n_splits=10)
Acc_scores = []
Acc_scores2 = []
counter = 0
X,Y = [],[]

fp = open(f"data/train_small_{name}.txt")
for line in fp:
    tokens = line.strip().split()
    sentiment = tokens[0] # get the sentiment and then ignore first value
    summing = np.zeros(w2vsize)
    ctr = 0
    for t in tokens[1:] :
        try:
            vec = wv[t]
            summing += vec
            ctr += 1
        except KeyError:
            ctr = ctr
    if ctr > 0 :
        summing = [float(i)/ctr for i in summing]
    X.append(summing)
    Y.append(int(sentiment))

Xpd = pd.DataFrame(X)
Ypd = pd.DataFrame(Y)

for train_index, test_index in cv.split(Xpd, Ypd):
    #training the classifier
    X_train, X_test = Xpd.iloc[train_index], Xpd.iloc[test_index]
    Y_train = Ypd.iloc[train_index]
    Y_test = Ypd.iloc[test_index]
    ref = linear_model.LinearRegression()
    ref.fit(X_train,Y_train)
    #validation
    predictions = ref.predict(X_test)
    predictions = list(map(lambda pred: -1 if pred < 0 else 1 , predictions))
    score = sklearn.metrics.balanced_accuracy_score(Y_test, predictions)
    Acc_scores.append(score)
print(f"score : {np.mean(Acc_scores)}")
