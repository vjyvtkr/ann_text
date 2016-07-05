Enter file contents here# -*- coding: utf-8 -*-
"""
Created on Tue Jul 05 17:27:24 2016

@author: u505123
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("C:\\Users\\u505123\\Desktop\\Final_SVM.csv")
test_df = df[0:1874]

sw_file = open("C:/Users/u505123/Documents/Project/ann_text-master/stopwords.txt")    
stop_words = sw_file.readlines()

for i in range(0,len(stop_words)):
    stop_words[i] = stop_words[i].strip()

sw_file.close()

yes_df = df.loc[df["Indicator"]=="Yes"]
no_df = df.loc[df["Indicator"]=="No"]

yes_sent = CountVectorizer(stop_words=stop_words,max_features=50)  
yes_sample = yes_sent.fit_transform(yes_df["Phrase"])  
no_sent = CountVectorizer(stop_words=stop_words,max_features=200)
no_sample = no_sent.fit_transform(no_df["Phrase"])

yes_words = yes_sent.get_feature_names()
no_words = no_sent.get_feature_names()
feats = yes_words + no_words
final_feat = list(set(feats))
