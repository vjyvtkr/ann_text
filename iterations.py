# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 01:41:58 2016

@author: vjyvtkr
"""

import scratch as sc
import pandas as pd
import numpy as np

p=0.0
s=0.0
a=0.0

feat = 30
path = "C:\\Users\\u505123\\Documents\\Project\\ann_text-master\\"
obj = open(path+"log_shuffle.txt","w+")
inp_doc = path+"Final_Achuth.csv"
inp_df = pd.read_csv(inp_doc)
inp_df = inp_df.reindex(np.random.permutation(inp_df.index))
for i in range(30,2000):
    print "Running iter with f = %s\n"%(str(i))
    obj.write("Running iter with f = %s\n"%(str(i)))
    t1,t2,t3 = sc.run(i,inp_df)
    if t2>s and t3>a:
        p=t1
        s=t2
        a=t3
        feat = i
    obj.write("Result in iter with feat = %s\n"%(str(i)))
    obj.write("Precision = %s\n"%(str(t1)))
    obj.write("Sensitivity = %s\n"%(str(t2)))
    obj.write("Accuracy = %s\n\n\n"%(str(t3)))
    
obj.write("Best result in iter with feat = %s\n"%(str(feat)))
obj.write("Precision = %s\n"%(str(p)))
obj.write("Sensitivity = %s\n"%(str(s)))
obj.write("Accuracy = %s\n\n\n"%(str(a)))

obj.close()

'''
reasonable f:
58 
60
208 (best)

'''
