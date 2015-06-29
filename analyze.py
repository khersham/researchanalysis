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


class TFIDFmember:

    def __init__(self,dataset):
        self.token_dict = {}
        self.totalvocab_stemmed = []
        self.totalvocab_tokenized = []
        self.stopwords = nltk.corpus.stopwords.words('english')#Define the necessary stopwords and stems
        self.stemmer = SnowballStemmer("english")
        self.datalist = self.textTrim(dataset)
        self.vtext = self.vectorize_text(self.datalist)
        self.vectorizer = TfidfVectorizer(max_df=1.0, min_df=0.1, stop_words='english',use_idf=True,tokenizer=self.tokenize_and_stem,ngram_range=(1,3))   
        self.tfidf = self.vectorizer.fit_transform(self.token_dict.values())
        self.terms = self.vectorizer.get_feature_names()
        self.vocabulary = self.wordCollect(zip(*self.datalist)[1])        

    def textTrim(self,texts):
        tmplist = []
        for elem in texts:
            if elem[1] is None:
                elm1 = "None"
            else:      
                elm1 = elem[1].lower() #We convert all the texts into lower case
            tmplist.append([elem[0],elm1])
        return tmplist        

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
            
    def tokenize(self,text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        try:
            tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
            # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
            return [token for token in tokens if re.search('[a-zA-Z]', token)]   
        except:
            print("Incorrect input for tokenization.")                      
        
        #define vectorizer parameters
    def vectorize_text(self,texts):
        for elem in texts:
            #if elem[1] is None:
            #    textcopy = "None"
            #else:      
            #    textcopy = elem[1].lower() #We convert all the texts into lower case
            no_punctuation = elem[1].translate(None, string.punctuation) #We get rid of the punctuation
            identifier = str(elem[0])
            self.token_dict[identifier] = no_punctuation
            
    def wordCollect(self,texts):
        for i in texts:
            self.totalvocab_stemmed.extend(self.tokenize_and_stem(i)) #extend the 'totalvocab_stemmed' list
            self.totalvocab_tokenized.extend(self.tokenize(i))  
        return pd.DataFrame({'words': self.totalvocab_stemmed}, index = self.totalvocab_stemmed) #Experiment with token         
            
           
exampleText = """SELECT Id, Abstract FROM Article WHERE Id BETWEEN 1415898 AND 1416427"""        

class KMeanCluster:
    def __init__(self,dataset,tfidf_matrix,number_cluster):
        self.num_cluster = number_cluster
        self.TFIDF_vocab = tfidf_matrix.vocabulary
        self.TFIDF_terms = tfidf_matrix.terms
        self.meanfit = KMeans(n_clusters = number_cluster) #Now choose the number of clusters in K-mean
        self.fitted = (self.meanfit).fit(tfidf_matrix.tfidf) #Clustering applied to the sorted TFIDF matrix
        self.clusters = self.meanfit.labels_.tolist() #The cluster numbers are labelled accordingly to the data
        self.dict = {'Id':zip(*dataset)[0], 'Abstract':zip(*dataset)[1], 'Cluster':self.clusters} #We glue everything together
        self.pdframe = pd.DataFrame(self.dict, index = [self.clusters], columns = ['Id','Abstract','Cluster'])  
        
    def printAll(self):
        order_centroids = self.meanfit.cluster_centers_.argsort()[:, ::-1] 
        for i in range(self.num_cluster):
            print("Cluster %d:" %i)
            
            for ind in order_centroids[i, :10]:
                print(' %s' % self.TFIDF_vocab.ix[self.TFIDF_terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'))
            print()    
      
                  
        