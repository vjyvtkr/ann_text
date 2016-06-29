# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 16:04:02 2016

@author: u505123
"""
import model
import re
import pandas as pd

#run model.py, then
#p = "C:/Users/u505123/Desktop/Final_Achuth.csv"
p = "/home/vjyvtkr/WF/Project/Final_Achuth.csv"
all_words2 = model.word_list(p)

flattened2 = [val for sublist in all_words2 for val in sublist]

final_words2 = list(set(flattened2))
pat = re.compile(r'[0-9]')
while True:
    f = len(final_words2)
    for i in range(0,len(final_words2)):
        try:
            if pat.search(final_words2[i]) or "-" in final_words2[i] or "/" in final_words2[i] or "$" in final_words2[i] or "_" in final_words2[i] or "(" in final_words2[i] or ")" in final_words2[i] or "%" in final_words2[i] or "*" in final_words2[i] or "`" in final_words2[i] or "," in final_words2[i] or len(final_words2[i])<=2 or "." in final_words2[i]:
                final_words2.remove(final_words2[i])
        except:
            break
    l = len(final_words2)
    if f==l:
        break

ffinal_words2 = []
for i in final_words2:
    ffinal_words2.append(i.lower())
    

final_words2 = []

for i in ffinal_words2:
    try:
        int(i)
    except:
        final_words2.append(i)

ffinal_words2 = list(set(final_words2))

word_dict = {}
for i in ffinal_words2:
    val = 0
    for j in df["Phrase"]:
        if i in j:
            val+=1
    word_dict[i] = val

cons_words = []    
for i in word_dict.keys():
    if word_dict[i]>20:
        cons_words.append(i)