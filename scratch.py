# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 22:36:52 2016

@author: vjyvtkr
"""

import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import fsgd
import test_sgd

def run(feat,inp_df):
    #feat = 208    
    path = "C:\\Users\\u505123\\Documents\\Project\\ann_text-master\\"
    sw_file = open(path+"stopwords.txt")    
    stopwords = sw_file.readlines()
    for i in range(0,len(stopwords)):
        stopwords[i] = stopwords[i].strip()
    sw_file.close()
    
    def cleanPhrase(s):
        s = re.sub(r"[^A-Za-z\']", " ", s)
        s = re.sub(r"\s{2,}", " ", s)
        s = re.sub(r" [A-Za-z] ", " ", s)
        return s.lower().strip()
    
    def clean(data):
        phrases = data["Phrase"]
        new_phrases = []    
        for i in phrases:
            x = cleanPhrase(i)
            new_phrases.append(x)
        data["Phrase"] = new_phrases
        return data.reindex(np.random.permutation(data.index))
    
    '''
    def getLabelledFeatures(data):
        cv = CountVectorizer(stop_words=stopwords,max_features=240)
        feat = cv.fit(data["Phrase"]).get_feature_names()
        feat_yes_count = {}
        feat_no_count = {}
        for j in feat:
            yc=0
            nc=0
            for i in range(0,len(data.index)):
                c = data["Phrase"].ix[i].count(j)
                if data["Indicator"].ix[i]=="Yes":
                    yc+=c                
                else:
                    nc+=c
            feat_yes_count[j]=yc
            feat_no_count[j]=nc
        fin_feat = []
        fin_feat_out = []    
        yc=0
        nc=0
        for i in feat:
            if feat_yes_count[i]>feat_no_count[i] and yc<50:
                fin_feat_out.append(1)
                fin_feat.append(i)
                yc+=1
            elif nc<50:
                fin_feat_out.append(0)
                fin_feat.append(i)
                nc+=1
        
    
        return fin_feat,fin_feat_out
    '''  
    
    #inp_doc = path+"Final_Achuth.csv"  
    #inp_df = pd.read_csv(inp_doc)
    #inp_df = inp_df.reindex(np.random.permutation(inp_df.index))
    train_data = inp_df[0:int(0.7*len(inp_df.index))]
    test_data = inp_df[int(0.7*len(inp_df.index)):]
    
    final_train_data = clean(train_data)
    
    #features,y = getLabelledFeatures(final_train_data)
    #c = list(zip(features,y))
    #random.shuffle(c)
    #features,y = zip(*c)
    #features=list(features)
    
    y = []
    for i in final_train_data["Indicator"]:
        if i=="Yes":
            y.append(1)
        else:
            y.append(0)
    y = np.array([y]).T
    
    cv = CountVectorizer(stop_words=stopwords,max_features=feat)
    nn_inp = cv.fit_transform(final_train_data["Phrase"].values).toarray()
    ans = fsgd.two_layers(nn_inp,y,len(nn_inp[0]))
    
    final_test_data = clean(test_data)
    test_sample = cv.fit_transform(final_test_data["Phrase"].values).toarray()
    test_out = test_sgd.two_layers(test_sample,ans[0])
    
    final_comp = {}
    ind=0
    for i in final_test_data["Indicator"]:
        if i=="Yes":
            if test_out[ind]>0.5:
                final_comp[ind]=[1,1]
            else:
                final_comp[ind]=[1,0]
        else:
            if test_out[ind]>0.5:
                final_comp[ind]=[0,1]
            else:
                final_comp[ind]=[0,0]
        ind+=1
    
    tp=0
    fp=0
    fn=0
    tn=0
    for i in final_comp.keys():
        if final_comp[i]==[1,1]:
            tp+=1
        elif final_comp[i]==[1,0]:
            fn+=1
        elif final_comp[i]==[0,1]:
            fp+=1
        else:
            tn+=1
            
    precision = float(tp)/(tp+fp)
    
    sensitivity = float(tp)/(tp+fn)
    
    acc = float(tp+tn)/(tp+tn+fp+fn)
    
    return precision,sensitivity,acc
