import numpy as np

import keras
from keras.layers import Embedding
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from gensim.models import Word2Vec
from gensim.models import FastText

from utils.io_helper import IOHelper


class DataHelper():
    """Data Manager
    
    This class prepares the given data for training.
    """

    def __init__(self, io, ln=32, nr_words=10000, dim=200):
        """Initialization of parameters

        Additionally loads Word2Vec file if specified emb_type is word2vec.
        
        :param io: instance of io-helper for loading embedding
        :param ln: length of considered sequence
        :param nr_words: number of considered words (if keras embedding)
        :param dim: dimension of embedding (if keras embedding)
        """

        self.emb_type = io.emb_type
        self.ln = ln
        self.nr_words = nr_words
        self.dim = dim
        self.vocabulary = None
        
        #load_embedding file
        if self.emb_type != 'keras':
            emb_path = io.getEmbPath(self.dim)
            try:
                if self.emb_type == 'word2vec': 
                    embmodel = Word2Vec.load(emb_path)
                    print("Word2Vec File loaded")
                elif self.emb_type == 'fasttext':
                    embmodel = FastText.load(emb_path)
                    print("FastText File loaded")

                self.wv = embmodel.wv
            except:
                 print(f"Embedding File could not be loaded from {emb_path}. Recompute the embedding.")

        
    def prepareData(self, X_train, Y_train, X_test, Y_test = None):
        """Tokenize and pad the data
        
        :param X_train: training data, list of sentences
        :param Y_train: training labels
        :param X_test: test data, list of sentences
        :param Y_test: test labels (optional)

        :return X_train: tokenized and padded training data
        :return Y_train: training labels as categorical variables 
        :return X_test: tokenized and padded test data
        :return Y_test: test labels as categorical variable (optional)
        """

        t = None

        #construct Tokenizer
        if self.emb_type == 'keras':
            t = Tokenizer(num_words=self.nr_words)
            t.fit_on_texts(X_train)

        elif self.emb_type == 'word2vec' or self.emb_type == 'fasttext':
            self.vocabulary = {word: vector.index for word, vector in self.wv.vocab.items()} 
            t = Tokenizer(num_words=len(self.vocabulary))
            t.word_index = self.vocabulary

        else:
            raise NotImplementedError("Type of Embedding is not implemented:",
                                              self.emb_type)

        #tokenize and pad data
        X_train = t.texts_to_sequences(X_train)
        X_train = pad_sequences(X_train, maxlen=self.ln, padding = 'pre')
        

        X_test = t.texts_to_sequences(X_test)
        X_test = pad_sequences(X_test, maxlen=self.ln, padding = 'pre')


        print("Data ready for training")

        #transform labels to categorical variables and return
        if Y_test is None:
            return X_train, to_categorical(Y_train), X_test, None

        else:
            return X_train, to_categorical(Y_train), X_test, to_categorical(Y_test)
        

    def getEmbedding(self):
        """return embedding and embedding dimension
        
        :return self.wv: embedding as directoy
        :return self.dim: dimension of embedding
        """
        if self.emb_type == 'keras':
             raise NotImplementedError("Keras Embedding can't be returned as simple embedding")
        else:
            return self.wv, self.dim

    def getEmbeddingLayer(self):
        """Construct Embedding layer
        
        :return emb_layer: embedding layer for keras model
        """
        
        if self.emb_type == 'keras':
            #return simple keras embedding layer (randomly initialized and optimized during training)
                emb_layer = Embedding(input_dim=self.nr_words, output_dim=self.dim, input_length=self.ln)
             
        elif self.emb_type == 'word2vec' or self.emb_type == 'fasttext':
            # compute weights for embedding layer based on word2vec embedding
            embedding_matrix = np.zeros((len(self.vocabulary), self.dim))

            # generate embedding matrix
            for word in self.vocabulary:
                i = self.vocabulary[word]
                try:
                    embedding_vector = self.wv[word]
                    embedding_matrix[i] = embedding_vector
                except KeyError:
                    embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),self.dim)

            #return keras embedding layer initialized with word2vec weights
            emb_layer = Embedding(len(self.vocabulary), output_dim=self.dim, weights = [embedding_matrix], input_length=self.ln, trainable=True)

        else:
            raise NotImplementedError("Type of Embedding is not implemented:",
                                              self.emb_type)
        return emb_layer