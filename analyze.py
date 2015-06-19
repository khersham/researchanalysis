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
import time
from bs4 import BeautifulSoup 
import re
import datetime
import pandas as pd
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer #Import stemmer to group words that have similar stem
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer #Import vectorizer

class medID:
    def __init__(self, pmid, Abstract):
        self.name = pmid
        self.abstract = Abstract
  
"""
Open a connection to the database
"""        
con = mdb.connect(host = 'localhost', user = 'genuser', passwd = 'genocide', db = 'pubmedRetrieve')
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

"""
Define the necessary stopwords and stems
"""                
stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")

#define a tokenizer which filters out stopwords and group the words by stem
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    try:
        tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]
        stems = [stemmer.stem(t) for t in filtered_tokens if t not in stopwords]
        return stems
    except:
        print("Incorrect input for tokenization.")      
        
#define vectorizer parameters
#vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,min_df=0.2, stop_words='english',use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))            
vectorizer = TfidfVectorizer(max_df=0.8, min_df=0.2, stop_words='english',use_idf=True)   

exampleText = """SELECT Id, Abstract FROM Article WHERE Id BETWEEN 1415898 AND 1415927"""        