import string
import sys
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet') 

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
import random

import argparse

# smileys
sad_smileys = {":(", ":-(", "):", ")-:", "=(", ")=", ":'(", ":,(", ":/", ":-/", ";/", ":'/", ";'/", "/:", "/;", "/-:", "=/", ":|", ":-|", "|:", "|-:", ":'|", ":\\", ":'\\", ":-\\", "=\\", ":[", ":-[" "=[", ":'[", "=[", ":o(", ":o/"}
happy_smileys = {":)", ":-)", "(:", "(-:", "=)", "(=", ":')", ";)", ";-)", "(;", "(-;", ":3", "=3", ":o)", ":]", ":-]"}
laughting_smileys = {":d", ":-d", "xd", "x-d", ";d", ";-d", ":p", ":-p", ";p", ";-p", ":'d", ":'P", "=d", "=p"}

# stopwords
sw = ['i', 'im', 'ill', 'id', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "youre", "youve", "youll", "youd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'hes', 'him', 'his', 'himself', 'she', "shes", 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'theyre', 'theyll', 'them', 'their', 'theirs', 'themselves', 'what', 'whats', 'which', 'who', 'whom', 'this', 'that', 'thats', "thatll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', "shouldve", 'now', "could", 'would', 'u', "not", "isnt", "arent", "wasnt", "werent","havent","hasnt","hadnt","wont","wouldnt", "dont", "doesnt","didnt","cant","couldnt","shouldnt","mightnt","mustnt", "aint", "neednt" ]

class Preprocessor():
    """Data Preprocessor
    
    This class preprocesses the given twitter data.
    """

    def __init__(self, preprocess_type='none'):
        """Initialization of parameters
        
        :param preprocess_type: type of preprocessing (what preprocessing steps should be made)
        """
        self.p_type = preprocess_type

    def process(self, text):
        """Preprocess the text
        
        :param text: text that is to be preprocessed
        :return processed_tweets: preprocessed split into tweets
        """

        #only split sentences
        if self.p_type == 'none': 
            tweets = text.split("\n")
    
            return tweets

        #remove numbers
        elif self.p_type == 'num':
            tweets = text.split("\n")
            processed_tweets = []
            for tweet in tweets:

                # Remove numbers
                tweet = tweet.translate(str.maketrans('','',string.digits))
                
                processed_tweets.append(tweet)

            return processed_tweets
        
        #remove punctuation
        elif self.p_type == 'pun':
            tweets = text.split("\n")
            processed_tweets = []
            for tweet in tweets:

                # Remove punctuation
                tweet = tweet.translate(str.maketrans('','',string.punctuation))
        
                processed_tweets.append(tweet)

            return processed_tweets

        #remove repeated letters
        elif self.p_type == 'rep':
            tweets = text.split("\n")
            processed_tweets = []
            for tweet in tweets:
                tokens = word_tokenize(tweet)
                    
                # Remove repeated letters
                pattern = re.compile(r"(.)\1{2,}")
                tokens = [pattern.sub(r"\1\1\1", i) for i in tokens]
                            
                tweet = (" ").join(tokens)
                processed_tweets.append(tweet)

            return processed_tweets
            
        #remove stopwords
        elif self.p_type == 'sw':
            tweets = text.split("\n")
            processed_tweets = []
            for tweet in tweets:
                tokens = word_tokenize(tweet)
            
                # Remove stopwords
                tokens = [word for word in tokens if not word in sw]
                        
                tweet = (" ").join(tokens)
                processed_tweets.append(tweet)

            return processed_tweets
            
        #stemming
        elif self.p_type == 'stem':
            ps = PorterStemmer()
            
            tweets = text.split("\n")
            processed_tweets = []
            for tweet in tweets:
                tokens = word_tokenize(tweet)
            
                # Remove stemming
                tokens = [ps.stem(i) for i in tokens]
                
                tweet = (" ").join(tokens)
                processed_tweets.append(tweet)

            return processed_tweets
            
        #lemmatization
        elif self.p_type == 'lem':
            wn = WordNetLemmatizer()
            
            tweets = text.split("\n")
            processed_tweets = []
            for tweet in tweets:
                tokens = word_tokenize(tweet)
            
                # Remove lemmatization 
                tokens = [wn.lemmatize(i) for i in tokens]
                        
                tweet = (" ").join(tokens)
                processed_tweets.append(tweet)

            return processed_tweets
        
        #handle emojis
        elif self.p_type == 'emo':
            tweets = text.split("\n")
            processed_tweets = []
            for tweet in tweets:

                # Handle smileys
                for x in sad_smileys: tweet = tweet.replace(x,'SADSMILEY')
                for x in happy_smileys: tweet = tweet.replace(x,'HAPPYSMILEY')
                for x in laughting_smileys: tweet = tweet.replace(x,'LAUGHTINGSMILEY')
                tweet = tweet.replace('<3','HEARTEMOJI')
        
                processed_tweets.append(tweet)

            return processed_tweets

        #remove numbers and the punctuation '<>' , shorten repeated letters, remove stemming, handle smileys, remove lemmatization
        elif self.p_type == 'enslp':
            ps = PorterStemmer()
            wn = WordNetLemmatizer()
            
            tweets = text.split("\n")
            processed_tweets = []
            for tweet in tweets:

                # Handle smileys
                for x in sad_smileys: tweet = tweet.replace(x,'SADSMILEY')
                for x in happy_smileys: tweet = tweet.replace(x,'HAPPYSMILEY')
                for x in laughting_smileys: tweet = tweet.replace(x,'LAUGHTINGSMILEY')
                tweet = tweet.replace('<3','HEARTEMOJI')

                # Remove numbers
                tweet = tweet.translate(str.maketrans('','',string.digits))
                
                # Remove some punctuation
                tweet = tweet.translate(str.maketrans('','',"<>"))

                tokens = word_tokenize(tweet)
                    
                # Remove stemming
                tokens = [ps.stem(i) for i in tokens]
                
                # Remove lemmatization
                tokens = [wn.lemmatize(i) for i in tokens]
                                
                tweet = (" ").join(tokens)
                processed_tweets.append(tweet)

            return processed_tweets

if __name__ == "__main__":
    """ 
    This script preprocessed the tweetData from the specified src directory 
    and saves the processed tweets in the specified destination Directory
    """

    #parse all arguments
    parser = argparse.ArgumentParser()
    #path to config file
    parser.add_argument('--srcDir', type=str, default='data', nargs="?", help='Directory for raw files')
    parser.add_argument('--destDir', type=str, default='data', nargs="?", help='Directory for preprocessed files')
    #size of twitter dataset
    parser.add_argument('--small', dest='size', action='store_const', const='small')
    parser.add_argument('--full', dest='size', action='store_const', const='full')
    #type of preprocessing
    parser.add_argument('--none', dest='preprocess', action='store_const', const='none')
    parser.add_argument('--num', dest='preprocess', action='store_const', const='num')
    parser.add_argument('--pun', dest='preprocess', action='store_const', const='pun')
    parser.add_argument('--rep', dest='preprocess', action='store_const', const='rep')
    parser.add_argument('--sw', dest='preprocess', action='store_const', const='sw')
    parser.add_argument('--stem', dest='preprocess', action='store_const', const='stem')
    parser.add_argument('--lem', dest='preprocess', action='store_const', const='lem')
    parser.add_argument('--emo', dest='preprocess', action='store_const', const='emo')
    parser.add_argument('--enslp', dest='preprocess', action='store_const', const='enslp')

    args = parser.parse_args()
    
    #load data files
    if args.size == "full":
        f_p = open(f"{args.srcDir}/train_pos_full.txt",encoding='utf-8')
        f_n = open(f"{args.srcDir}/train_neg_full.txt",encoding='utf-8')
    else:
        f_p = open(f"{args.srcDir}/train_pos.txt",encoding='utf-8')
        f_n = open(f"{args.srcDir}/train_neg.txt",encoding='utf-8')
    
    f_t = open(f"{args.srcDir}/test_data.txt",encoding='utf-8')

    p = Preprocessor(args.preprocess)
  
    # Preprocess training data
    text_p = f_p.read()
    tweets_p = p.process(text_p)
    #f_new_p.write(("\n").join(tweets_p))
    text_n = f_n.read()
    tweets_n = p.process(text_n)
    #f_new_n.write(("\n").join(tweets_n))
    
    # Merge training datasets and shuffle it
    train = list(zip([1]*len(tweets_p), tweets_p))
    train += list(zip([-1]*len(tweets_n), tweets_n))
    random.shuffle(train)

    # Preprocess test data
    text_t = f_t.read()

    #destination files
    f_new_t = open(f"{args.destDir}/test_{args.preprocess}.txt", "w")
    f_tr = open(f"{args.destDir}/train_{args.size}_{args.preprocess}.txt", "w")

        
    #save all training data in one file
    for label, text in train:
        f_tr.write(str(label) + " " + text + "\n")

    #save test data
    f_new_t.write(("\n").join(p.process(text_t)))
    

