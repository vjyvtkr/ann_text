# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 15:30:55 2016
@author: u505123
"""

import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
import fsgd

'''

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    
    string = re.sub(r"[^A-Za-z0-9(),%!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string =  string.strip().lower()
    for i in stopwords:
        if i in string:
            string = re.sub(r" (%s) "%i," ",string)
    return string
p = "C:\\Users\\u505123\\Desktop\\Final_Achuth.csv"
tdf = pd.read_csv(p)
df = tdf[0:int(0.7*len(tdf))]
yesdf = pd.DataFrame(df["Phrase"].loc[df['Indicator'] == "Yes"],columns=["Phrase","Class"])
nodf = pd.DataFrame(df["Phrase"].loc[df['Indicator'] == "No"],columns=["Phrase","Class"])
yesdf.index = [i for i in range(0,len(yesdf.index))]
nodf.index = [i for i in range(0,len(nodf.index))]
yesdf["Class"] = [1 for i in range(0,len(yesdf.index))]
nodf["Class"] = [0 for i in range(0,len(nodf.index))]
data = yesdf.append(nodf)
data.index = [i for i in range(0,len(data.index))]
final_phrase_list = []
for i in data["Phrase"]:
    x = clean_str(i)
    final_phrase_list.append(x)
data["Phrase"] = [i for i in final_phrase_list]
data = data.reindex(np.random.permutation(data.index))
cv = CountVectorizer(stop_words=stopwords,max_features=500)
x = cv.fit(data["Phrase"])
feat = x.get_feature_names()
fin_feat=[]
pat = re.compile(r'[0-9]+')
for i in feat:
    if not pat.search(i):
        fin_feat.append(i)
feat_yes_count = {}
feat_no_count = {}
for j in fin_feat:
    yc=0
    nc=0
    for i in range(0,len(data.index)):
        c = data["Phrase"].ix[i].count(j)
        if data["Class"].ix[i]==1:
            yc+=c                
        else:
            nc+=c
    feat_yes_count[j]=yc
    feat_no_count[j]=nc
feat_out = []
for i in fin_feat:
    if feat_yes_count[i]>feat_no_count[i]:
        feat_out.append(1)
    else:
        feat_out.append(0)
'''        
chk = cv.fit_transform(fin_feat).toarray()

chk_y = np.array([feat_out]).T

ans = fsgd.two_layers(chk,chk_y)

'''
sample = count_vectorizer.fit_transform(data['Phrase'].values).toarray()
#tfidf = TfidfVectorizer()
#features = tfidf.fit_transform(data['Phrase'].values).todense()
#X = [sample,features]
#y = np.array([i for i in data["Class"]]).T
#clf = MLPClassifier()
#clf.fit(X, y)
y = np.array([[i for i in data['Class']]]).T
for i in range(0,len(data.index)):
    ans = sgd.two_layers(np.array([sample[i]]).T,y[i])
X = [sample,features]
Y = [i for i in data["Class"]]
clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
counts = count_vectorizer.fit_transform(data['Phrase'].values)
classifier = MultinomialNB()
targets = data['Class'].values
classifier.fit(counts, targets)
example = ["Customer A accounted for 15% of net sales.", "Major contribution from the geographical area was from Venezuela"]
example_counts = count_vectorizer.transform(example)
predictions = classifier.predict(example_counts)
predictions
#example = ["Customer A accounted for 15% of net sales.", "Major contribution from the geographical area was from Venezuela"]
example = []
test_df = tdf[int(0.7*len(tdf)):]
fin_test1 = test_df.loc[test_df["Indicator"]=="Yes"]
fin_test2 = test_df.loc[test_df["Indicator"]=="No"]
fin_test = pd.DataFrame(fin_test1["Phrase"].append(fin_test2["Phrase"]),columns = ["Phrase"])
fin_test.index = [i for i in range(0,len(fin_test.index))]
for i in fin_test["Phrase"]:
    x = clean_str(i)
    example.append(x)
fin_test["Phrase"] = [i for i in example]
cv_test = CountVectorizer(stop_words=stopwords,max_features=500)
test = cv_test.fit(fin_test["Phrase"])
test_feat = test.get_feature_names()
fin_test_feat=[]
#pat = re.compile(r'[0-9]+')
for i in test_feat:
    if not pat.search(i):
        fin_test_feat.append(i)
pipeline = Pipeline([
    ('vectorizer',  CountVectorizer()),
    ('classifier',  MultinomialNB()) ])
pipeline.fit(data['Phrase'].values, data['Class'].values)
ans = pipeline.predict(example)
j=0
prec_count = 0
for i in test_df["Indicator"]:
    if i=="Indicator":
        continue
    if i=="Yes" and ans[j]==1:
        prec_count+=1
    j+=1
k_fold = KFold(n=len(data), n_folds=6)
scores = []
confusion = numpy.array([[0, 0], [0, 0]])
for train_indices, test_indices in k_fold:
    train_text = data.iloc[train_indices]['Phrase'].values
    train_y = data.iloc[train_indices]['Class'].values
    test_text = data.iloc[test_indices]['Phrase'].values
    test_y = data.iloc[test_indices]['Class'].values
    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)
    confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions, pos_label=1)
    scores.append(score)
print('Total emails classified:', len(data))
print('Score:', sum(scores)/len(scores))
print('Confusion matrix:')
print(confusion)
'''