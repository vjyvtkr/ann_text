# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:59:20 2016
@author: Vijay Yevatkar
"""

import pandas as pd
import nltk

def word_list(p):
#p = "C:/Users/u505123/Desktop/Final_Achuth.csv"
    df = pd.read_csv(p)
    
    all_words2 = [[]]
    ind = 0
    for i in df["Phrase"]:
        if df["Indicator"].ix[ind]=="Yes":    
            x=[]
            x = nltk.word_tokenize(i)        
            
            if len(x):
                #print "yo",
                for word in x:
                    if word in nltk.corpus.stopwords.words('english'): 
                        x.remove(word)
            all_words2.append(x)
        ind+=1
    return all_words2