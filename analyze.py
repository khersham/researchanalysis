"""
Created on Wed Mar 25 12:08:16 2015

@author: lim
"""
#from Bio import Entrez
#from Bio import Medline
import MySQLdb as mdb
import sys
import requests
#import urllib2
import string
import time
#from bs4 import BeautifulSoup 
import re
import datetime
import pandas as pd
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer #Import stemmer to group words that have similar stem
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer #Import vectorizer
from sklearn.cluster import KMeans #Import clustering tools

class medID:
    def __init__(self, pmid, Abstract):
        self.name = pmid
        self.abstract = Abstract
  
"""
Open a connection to the database
"""        
con = mdb.connect(host = 'localhost', user = 'genuser', passwd = 'genocide', db = 'Articles')
#con.set_character_set('utf8')
db = con.cursor()   
#db.execute('SET CHARACTER SET utf8;')
#db.execute('SET character_set_connection=utf8;')     
"""
Database is now connected
"""                  
                  
                  
def query_search(db, text):
    try:
        db.execute(text)
        return db.fetchall()
    except:
        print("Query problem. Please check the query again.")


class TFIDFmember:
    
    #Define the necessary stopwords and stems
    
    def __init__(self,dataset):
        self.token_dict = {}
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.stemmer = SnowballStemmer("english")
        self.vtext = self.vectorize_text(dataset)
        self.vectorizer = TfidfVectorizer(max_df=0.8, min_df=0.2, stop_words='english',use_idf=True,tokenizer=self.tokenize_and_stem,ngram_range=(1,3))   
        self.tfidf = self.vectorizer.fit_transform(self.token_dict.values())        

    #define a tokenizer which filters out stopwords and group the words by stem
    def tokenize_and_stem(self,text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        try:
            tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
            # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
            filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]
            stems = [self.stemmer.stem(t) for t in filtered_tokens if t not in self.stopwords]
            return stems
        except:
            print("Incorrect input for tokenization.")      
        
        #define vectorizer parameters
    def vectorize_text(self,texts):
        for elem in texts:
            if elem[1] is None:
                textcopy = "None"
            else:      
                textcopy = elem[1].lower() #We convert all the texts into lower case
            no_punctuation = textcopy.translate(None, string.punctuation) #We get rid of the punctuation
            identifier = str(elem[0])
            self.token_dict[identifier] = no_punctuation
           
exampleText = """SELECT Id, Abstract FROM Article WHERE Id BETWEEN 1415898 AND 1415927"""        

def kmeanfit(number_cluster): 
    return KMeans(n_clusters=number_cluster) #Now choose the number of clusters in K-mean