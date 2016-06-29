# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 01:41:58 2016

@author: vjyvtkr
"""

import scratch as sc

p=0.0
s=0.0
a=0.0

feat = 50

obj = open("/home/vjyvtkr/WF/Project/log.txt","w+")

for i in range(50,2000):
    print "Running iter with f = %s\n"%(str(i))
    obj.write("Running iter with f = %s\n"%(str(i)))
    t1,t2,t3 = sc.run(i)
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