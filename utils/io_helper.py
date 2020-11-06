import sys
import csv
import numpy as np
import pandas as pd
import configparser


CONFIG_PATH = 'structure.config'

class IOHelper():
    """Input/Output Manager
    
    This class is responsible for all input/output related functions. It constructs correct
    filepaths and reads/writes the specified data
    """

    def __init__(self, size = 'full', emb_type = 'keras', preprocess = 'enslp'):
        """Initialization of parameters
        
        :param size: size of training set 
        :param emb_type: type of embedding
        :param preprocess: type of preprocessing
        """

        self.size = size
        self.emb_type = emb_type
        self.preprocess = preprocess

        #load config file for construction of path names
        self.config = configparser.ConfigParser()
        self.config.read(CONFIG_PATH)
        print("Config file read")


        self.train_file = f"train_{self.size}_{self.preprocess}"
        self.test_file_size = f"test_{self.size}_{self.preprocess}"
        self.test_file = f"test_{self.preprocess}"

    def readTxtFile(self, file_name, label = True):
        """Read text file with twitter data
        
        :param file_name: name of file 
        :param label: read labeled or unlabeled data (train- or test-dataset)
        :return X_train, Y_train: list of tweets in X_train, list of labels in Y_train 
                                (if label = False only X_train is returned)
        """


        path_name = f"{self.config['DEFAULT']['DATA']}/{file_name}.txt"
        try:
            #if data labeled return tweets split into words and labels modified to 0 & 1 labels
            if label:
                X_train,Y_train = [],[]
                f_train = open(path_name)

                for line in f_train:
                    tokens = line.strip('\"').split()
                    if len(tokens) == 0:
                        continue
                    sentiment = int(tokens[0])
                    if sentiment == -1:
                        sentiment = 0
                    Y_train.append(sentiment)
                    X_train.append(tokens[1:])
            
                return X_train,  Y_train

            #if data not labeled return tweets split into words
            else:
                X_train = []
                f_train = open(path_name)
                for line in f_train:
                    tokens = line.strip('\"').split()
                    if len(tokens) == 0:
                        continue
                    X_train.append(tokens[1:]) #first token is a comma for all tweets, so start at second token
            
                return X_train
        except:
            print(f"Textfile {path_name} could not be loaded.")

    def readCrossValidationData(self, k = 0):
        """Read cross validation data of specified iteration
        
        :param k: iteration of crossvalidation
        """

        X_train, Y_train = self.readTxtFile(f"cv_{self.train_file}_{k+1}", label = True)
        X_test, Y_test = self.readTxtFile(f"cv_{self.test_file_size}_{k+1}", label = True)

        print("Data loaded")
        return X_train, Y_train, X_test, Y_test

    def readData(self):
        """Read twitter data for final model, i.e. read full training and test set
    
        """

        X_train, Y_train = self.readTxtFile(self.train_file, label = True)
        X_test = self.readTxtFile(self.test_file, label = False)

        print("Data loaded")
        return X_train, Y_train, X_test

    def writeProbabilityPredictionFile(self, pred, model_name, labels = None, k=None):
        """Write prediction to file

        :param pred: List of predictions (probabilities)
        :param model_name: name of model for filename
        :param k: iteration of cross validation, if None no number is added at end of filename
        """

        if k == None:
            path_name = f"{self.config['DEFAULT']['PRED']}/{model_name}_{self.size}_{self.preprocess}_{self.emb_type}.csv"
        else:
            path_name = f"{self.config['DEFAULT']['PRED']}/{model_name}_{self.size}_{self.preprocess}_{self.emb_type}_kfold{k+1}.csv"

        try:
            f = open(path_name, 'w')
            with f:
                if labels == None:
                    fnames = ['Id', '-1', '1']
                    writer = csv.DictWriter(f, fieldnames=fnames)    
                    writer.writeheader()  
                    for j, row in enumerate(pred):
                        writer.writerow({'Id' : j+1, '-1' : row[0], '1' : row[1]})

                else:
                    fnames = ['Id', '-1', '1', 'trueLabel']
                    writer = csv.DictWriter(f, fieldnames=fnames)    
                    writer.writeheader()
                    for j, row in enumerate(pred):
                        writer.writerow({'Id' : j+1, '-1' : row[0], '1' : row[1], 'trueLabel' : labels[j]})
                                
            print("Predictions written to file.")
        except:
            print(f"Prediction file could not be written to {path_name}.")
 

    def writePredictionFile(self, pred, model_name):
        path_name = f"{self.config['DEFAULT']['PRED']}/{model_name}.csv"

        if model_name == "random_forest":
            f = open(path_name, 'w')
            with f:
                fnames = ['Id', 'Prediction']
                writer = csv.DictWriter(f, fieldnames=fnames)    
                writer.writeheader()  
                for i, row in enumerate(pred):
                    if ((float)(row) < 0.5):
                        writer.writerow({'Id' : i+1, 'Prediction': '-1'})
                    else:
                        writer.writerow({'Id' : i+1, 'Prediction': '1'})

        elif model_name == "linear":
            model_name = f"{model_name}_{self.size}_{self.preprocess}_{self.emb_type}"
            path_name = f"{self.config['DEFAULT']['PRED']}/{model_name}.csv"
            f = open(path_name, 'w')
            with f:
                fnames = ['Id', 'Prediction']
                writer = csv.DictWriter(f, fieldnames=fnames)    
                writer.writeheader()  
                for i,row in enumerate(pred):
                        writer.writerow({'Id' : i+1, 'Prediction': str(row)})

        else:    
            f = open(path_name, 'w')
            with f:
                fnames = ['Id', 'Prediction']
                writer = csv.DictWriter(f, fieldnames=fnames)    
                writer.writeheader()  
                for i,row in enumerate(pred):
                    if (row[0] > row[1]):
                        writer.writerow({'Id' : i+1, 'Prediction': '-1'})
                    else:
                        writer.writerow({'Id' : i+1, 'Prediction': '1'}) 



    def writeAccuracyFile(self, acc_scores, model_name):
        """Write accuracy to file

        :param acc_scores: List of computed accuracies
        :param model_name: name of model for filename
        """

        path_name = f"{self.config['DEFAULT']['ACC']}/{model_name}_{self.size}_{self.preprocess}_{self.emb_type}.csv"
        
        try:
            print(f"mean score : {np.mean(acc_scores)} +/- {np.std(acc_scores)}")
            with open(path_name, "w") as f:
                for acc in acc_scores:
                    f.write(str(acc) + "\n")

            print("Accuracies written to file.")

        except:
            print(f"Accuracy file could not be written to {path_name}.")


    def getModelPaths(self, model_name):
        """get path for model saving

        :param model_name: name of model for filename
        """

        path_h5 = f"{self.config['DEFAULT']['MODEL']}/{model_name}_{self.size}_{self.preprocess}_{self.emb_type}.h5"
        path_json =  f"{self.config['DEFAULT']['MODEL']}/{model_name}_{self.size}_{self.preprocess}_{self.emb_type}.json"

        return path_h5, path_json

    def getEmbPath(self, dim):
        """get path for loading embedding

        :param dim: dimension of embedding
        """


        emb_path = f"{self.config['DEFAULT']['EMB']}/{self.emb_type}_{self.size}_{self.preprocess}_{dim}.model"

        return emb_path


    def readModels(self, filename_list):
        """read file with list of models

        :param filename_list: path to file with list of models
        :return modelNames: list of models
        """

        try:
            fp = open(filename_list, 'r')
            modelNames = fp.read()
            modelNames = modelNames.split("\n")

            print(f"List of models: {modelNames}")
            return modelNames
        except:
            print(f"File can't be loaded. Check filepath: {filename_list}.")

    def readProbabilityData(self, model_names, kfold = None):
        """read predictions of specified models

        :param model_names: list of models
        :param kfold: number of folds, None if not called in cross validation
        :return X, Y: list for each datapoint containing all positive prediction probabilities of specified models
        """

        if kfold == None:
            #read test file predictions
            X = []
            for k in range(len(model_names)):
                #add one feature per model
                X.append([])

            for k, name in enumerate(model_names):
                with open(f"{self.config['DEFAULT']['PRED']}/{name}.csv") as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    header = next(csv_reader)
                    for i,neg,pos in csv_reader:
                        #only append positive value since positive $ negative correlated (together 1)
                        X[k].append(float(pos))

            return (np.asarray(X)).transpose().tolist()

        else:
            #read crossvalidation predictions
            X_all, Y_all = [], []
            for j in range(kfold):
                X, Y = [], []
                for k in range(len(model_names)):
                    #add one feature per model
                    X.append([])
                    

                for k, name in enumerate(model_names):
                    with open(f"{self.config['DEFAULT']['PRED']}/{name}_kfold{j+1}.csv") as csv_file:
                        csv_reader = csv.reader(csv_file, delimiter=',')
                        header = next(csv_reader)
                        if k == 0:
                            for i,neg,pos,label in csv_reader:
                                #only append positive value since positive $ negative correlated (together 1)
                                X[k].append(float(pos))
                                #read true labels
                                Y.append(label)
                        else:
                            for i,neg,pos,label in csv_reader:
                                #only append positive value since positive $ negative correlated (together 1)
                                X[k].append(float(pos))

                X_all.append((np.asarray(X)).transpose().tolist())
                Y_all.append(Y)

            return X_all, Y_all


