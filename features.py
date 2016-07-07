# -*- coding: utf-8 -*-
"""
Created on Tue Jul 05 17:27:24 2016

@author: u505123
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re

#cleanPhrse removes all the unnessary data from the sentences.
def cleanPhrase(s):
    s = re.sub(r"[^A-Za-z\']", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r" [A-Za-z] ", " ", s)
    return s.lower().strip()

#Returns a random dataframe after cleaning the data
def clean(data):
    phrases = data["Phrase"]
    new_phrases = []    
    for i in phrases:
        x = cleanPhrase(i)
        new_phrases.append(x)
    data["Phrase"] = new_phrases
    return data

#path = "C:\\Users\\u505123\\Desktop\\Final_SVM.csv"
path = "/home/vjyvtkr/WF/Project/"
df = pd.read_csv(path+"Final_Achuth.csv")
test_df = df[0:1874]

sw_file = open(path+"stopwords.txt")    
stop_words = sw_file.readlines()

for i in range(0,len(stop_words)):
    stop_words[i] = stop_words[i].strip()

sw_file.close()

yes_df = clean(df.loc[df["Indicator"]=="Yes"])
no_df = clean(df.loc[df["Indicator"]=="No"])

yes_sent = CountVectorizer(stop_words=stop_words,max_features=100)  
#yes_sample = yes_sent.fit_transform(yes_df["Phrase"])  
no_sent = CountVectorizer(stop_words=stop_words,max_features=100)
#no_sample = no_sent.fit_transform(no_df["Phrase"])

yes_words = yes_sent.get_feature_names()
no_words = no_sent.get_feature_names()
feats = yes_words + no_words
final_feat = list(set(feats))
obj=open("/home/vjyvtkr/WF/Project/vocab_vj.txt","w+")
for i in final_feat:
    obj.write(i)
    obj.write(",")
obj.close()
