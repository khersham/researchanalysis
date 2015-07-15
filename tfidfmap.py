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
from collections import Counter #Tabulate dictionary
from collections import namedtuple #To put TDIDF into tuple dictionary
import math


#Import Stopwords
temp = pd.read_csv('/home/lim/Documents/mapreduce/common.txt',delimiter=",")
temp3 = [i[0] for i in temp.values.tolist()]
temp = pd.read_csv('/home/lim/Documents/mapreduce/common1.txt',delimiter=",")
temp2 = [i[0] for i in temp.values.tolist()]
stopwords = nltk.corpus.stopwords.words('english')
stopwords = set(stopwords) | set(temp2) | set(temp3)
stopwords = list(stopwords)
stopwords = [word.decode('utf8') for word in stopwords]
stemmer = SnowballStemmer("english")
#End of import


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
        return db.fetchall() #convert tuple to list
    except:
        print("Query problem. Please check the query again.")
       
exampleText = """SELECT Id, Abstract FROM Article WHERE Id BETWEEN 1415898 AND 1418427""" 
       
#Start of Mapper function         
#First we tokenize the words for unigram
def tokenize_and_stem(text):
# first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    try:
        tokens = [word.decode('utf8') for word in nltk.word_tokenize(text)]
        #tokens = [word.decode('utf8') for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation). Careful as numbers are eliminated
        filtered_tokens = [token.lower() for token in tokens if re.search('[a-zA-Z]', token)] #Lowercase for every word
        stems = [stemmer.stem(t) for t in filtered_tokens if t not in stopwords] #Select out words not in stopword list
        return stems 
    except:
        print("Incorrect input for tokenization.")  
        
#Based on the word collection tokenized we create a dictionary        
def create_dict(word_collection,ID):
    try:
        #article_length = len(word_collection) #Obtain the length of the abstract
        dkey = namedtuple("dkey", ["dWord","dId"])
        dict1 = Counter(word_collection) #A dictionary based on number of terms appeared is created
        article_length = max(dict1.values()) #For Tf we require max term in a document
        #dict2 = {word:[ID,dict1[word],article_length,1] for word in dict1} #Pack the word, abstarct number and length into tuple
        dict2 = {dkey(dWord = word, dId = ID):[dict1[word],article_length,1,0] for word in dict1} #To select, use [dict[word] for word in dict if word.dict_Word=='XXX']
        return dict2
    except:
        print("Unable to create dictionary, please check your file.")
        

#Now we sum up the document frequency
def docFrequency(dictfile):
    try:
        counter_list = Counter([word.dWord for word in dictfile])
        for word in dictfile:
            for elem in counter_list:
                if word.dWord==elem:
                    dictfile[word].pop(2)
                    dictfile[word].insert(2,counter_list[elem])
        
    except:
        print("Error in summing up the document frequency.")

#TF function can be defined accordingly
def TFfunction(term_freq,term_max):
    return 0.5 + (0.5*float(term_freq))/float(term_max)
    
#IDF function can be defined accordingly
def IDFfunction(total_num,occurence):
    try:
        result = math.log((1+total_num)/float((1+occurence)))
        return result
    except:
        print("Invalid function")
                
        
#Now let us define a TFIDF function
def TFIDFfunction(dictfile):
    try:
        total_num = len(set([word.dId for word in dictfile]))
        for word in dictfile:
            dictfile[word].pop()
            TFIDFf = TFfunction(dictfile[word][0],dictfile[word][1])*IDFfunction(total_num,dictfile[word][2])
            dictfile[word].append(TFIDFf)
    except:
        print("Error in TFIDF function, check your file or function")
        
    
 


                        