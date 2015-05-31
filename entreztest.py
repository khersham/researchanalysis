# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 12:08:16 2015

@author: lim
"""
from Bio import Entrez
from Bio import Medline
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
#import threading

Entrez.email = "khersham.lim@mpi-hd.mpg.de"


class meddata:
    def __init__(self,pmid):
        self.name = pmid
        self.comdata = self.fetcher()
        self.abstract = self.getdata('AB')
        self.affiliation = self.getdata('AD')
        self.title = self.getdata('TI')
        self.publishdate = self.getdata('DEP')
        self.journaltitle = self.getdata('JT')
    
    def fetcher(self):
        handle = Entrez.efetch(db='pubmed', id=self.name, retmode='text', rettype='medline')
        return Medline.read(handle)
        
    def getdata(self, key):    
        try:
            return self.comdata[key]
        except:
            return None
#def fetcher(pmid):
#    try:
#        handle = Entrez.efetch(db='pubmed', id=pmid, retmode='text', rettype='medline')
#        return Medline.read(handle)
#    except:
#        return None

# AB for abstract, AD for affiliation, TI for article title, DEP for published date, JT for journal title 
#def fetch_Abstract(text_data):
#    try:
#        abstract = text_data['AB']
#        return abstract
#    except:
#        return None
        

        
def query_Search(start_date, end_date, number_per_search):
    try:
        urlsite = """http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&datetype=pdat&mindate="""+ start_date +"""&maxdate=""" + end_date
        urllink = requests.get(urlsite)
        soup = BeautifulSoup(urllink.text)
        get_count = int(soup.find_all("count")[0].text)
        repeater = get_count/number_per_search
        #rest = get_count % number_per_search    
        link_list = []
        
        for start_point in range(repeater + 1): 
            website  = urlsite+"""&retstart="""+ str(start_point * number_per_search) +"""&retmax="""+ str(number_per_search)  
            urllink = requests.get(website)
            soup = BeautifulSoup(urllink.text)
        
            for link in soup.find_all("id"):
                link_list.append(str(link.text))
        
        return link_list
    except:
        print("Query problem from "+ start_date +" till "+ end_date)    
        
        
        
def SavePubMedIDs(pubmedIds):
    ## connect to the database
    ## you have create it first
    try:
        con = mdb.connect(host = 'localhost', user = 'genuser', passwd = 'genocide', db = 'pubmedRetrieve')
        con.set_character_set('utf8')
        cur = con.cursor()
        cur.execute('SET NAMES utf8;') 
        cur.execute('SET CHARACTER SET utf8;')
        cur.execute('SET character_set_connection=utf8;')
        for pubmedId in pubmedIds:
            full_text = meddata(pubmedId)
            cur.execute("INSERT INTO Article(Name, Title, Abstract, Journal, Date_Published, Affiliation) VALUES(%s, %s, %s, %s, %s, %s)", (pubmedId, full_text.title, full_text.abstract, full_text.journaltitle, full_text.publishdate, full_text.affiliation))
            con.commit()
        cur.close()
        con.close()
        return
    except:
        print("Saving PubMed ID error")  
        
        
def get_all(start_date, end_date):
    try:
        start = pd.Series(pd.date_range(start_date, end_date, freq='MS'))
        end = pd.Series(pd.date_range(start_date, end_date, freq='M'))  
        period_start = start.map(lambda x: x.strftime('%Y/%m/%d'))
        period_end = end.map(lambda x: x.strftime('%Y/%m/%d'))
        
        for i in range(len(period_start)):
            temp_list = query_Search(period_start[i], period_end[i], 1000)
            SavePubMedIDs(temp_list)
        #    time.sleep(1)
        #return [period_start,period_end]
        
    except:
        print('Wrong date time')
        
#CREATE TABLE IF NOT EXISTS Article(Id INT(10) NOT NULL AUTO_INCREMENT, Name VARCHAR(50) CHARACTER SET utf8 COLLATE utf8_unicode_ci NOT NULL , Title VARCHAR(5000) CHARACTER SET utf8 COLLATE utf8_unicode_ci NULL, Abstract VARCHAR(10000) CHARACTER SET utf8 COLLATE utf8_unicode_ci NULL, Journal VARCHAR(500) CHARACTER SET utf8 COLLATE utf8_unicode_ci NULL, Date_Published VARCHAR(100) CHARACTER SET utf8 COLLATE utf8_unicode_ci NULL, Affiliation VARCHAR(1500) CHARACTER SET utf8 COLLATE utf8_unicode_ci NULL, Time_stamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (Id));        