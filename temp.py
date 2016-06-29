# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 10:51:43 2016

@author: u505123
"""

import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import fsgd
import test_sgd
import sklearn.metrics
import matplotlib.pyplot as plt

thresh = 0.1
feat = 208    
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

inp_doc = path+"Final_Achuth.csv"  
inp_df = pd.read_csv(inp_doc)
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
        if test_out[ind]>thresh:
            final_comp[ind]=[1,1]
        else:
            final_comp[ind]=[1,0]
    else:
        if test_out[ind]>thresh:
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

accuracy = float(tp+tn)/(tp+tn+fp+fn)

act = []

predicted = []
ind = 0
for i in final_comp.keys():
    act.append(final_comp[i][0])    
    predicted.append(test_out[i])
    ind+=1

fpr, tpr,thresholds = sklearn.metrics.roc_curve(act,predicted)
for i in range(0,len(thresholds)):
    if thresholds[i]>1.0:
        thresholds[i]=1.0
roc_auc = sklearn.metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.show()
#PLOTTING SENSITIVITY VS. SPECIFICITY CURVE
plt.plot(tpr,thresholds, 'b', label="Spec/Sens")
plt.plot(1-fpr,thresholds,'r',label="Spec/Sens")
plt.show()

gain_model = [0]
gain_random = [0]
random_cumul = 0.0
model_cumul = 0.0
dec = len(final_test_data)/10
#df_test = df_test.sample(n = len(df_test))
final_test_data['Probability'] = test_out
final_test_data.sort(columns = 'Probability', inplace = True, ascending = False)
final_test_data = final_test_data.reset_index(drop = True)
df_GL = pd.DataFrame([],columns  = ['Decile', 'Random', 'Model'])

'''Dividing the data set into deciles and calculating gain'''    

for i in range(10):
    df_GL.loc[i, 'Decile'] = str(i + 1)
    random = 0.0
    model = 0.0
    for k in range(i*dec, (i*dec) + dec):
        if final_test_data.loc[k, 'Indicator'] == 'Yes':
            model = model + 1
        if final_test_data.loc[k, 'Indicator'] == 'Yes':
            random = random + 1
    random_cumul = random_cumul + random
    model_cumul = model_cumul + model
    df_GL.loc[i, 'Random'] = str(random_cumul)
    df_GL.loc[i, 'Model'] = str(model_cumul)
    gain_model.append(((model_cumul)/(tp + fn))*100.00)
    gain_random.append(((random_cumul)/(tp + fn))*100.00)

'''Plotting the cumulative gain chart'''

percent_pop = range(0, 110, 10)

plt.clf()
plt.plot(percent_pop, gain_model, label='Model')
plt.plot(percent_pop, gain_random, label='Random')
plt.xlim([0.0, 100.0])
plt.ylim([0.0, 100.0])
plt.xlabel('Percentage of Data Set')
plt.ylabel('Percentage of Positive Cases')
plt.title('Cumulative Gain Chart')
plt.legend(loc='lower right')
plt.show()




