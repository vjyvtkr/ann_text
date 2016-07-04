# -*- coding: utf-8 -*-
"""
Created on Fri Jul 01 13:15:33 2016
@author: Vijay Yevatkar
"""
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import sklearn.metrics
import pandas as pd
import numpy as np
import re

class ANN(object):
    
    # compute sigmoid nonlinearity
    def sigmoid(self,x):
        output = 1/(1+np.exp(-x))
        return output
    
    # convert output of sigmoid function to its derivative
    def sigmoid_output_to_derivative(self,output):
        return output*(1-output)    
    
    def trainNN(self,X,y,alpha=0.1, max_iter = 5000):
    
        dim = len(X[0])
        #print "\nTraining With Alpha:" + str(alpha)
        np.random.seed(1)
    
        # randomly initialize our weights with mean 0
        synapse_0 = 2*np.random.random((dim,1)) - 1

        j=-1
        while j<max_iter:
            j+=1

            #Feed forward through layers 1 and 2
            layer_0 = X
            layer_1 = self.sigmoid(np.dot(layer_0,synapse_0))
    
            #Calculate the error in the layer
            layer_1_error = layer_1 - y
    
            #Absolute error
            err = np.mean(np.abs(layer_1_error))
            
            #Check if error is minimizing after every 100 iterations            
            #if not j%100:
            #    print "Error after "+str(j)+" iterations:" + str(err)
            
            #if error goes below 0.1, break            
            if err < 0.1:
                break
    
            #In what direction is the target value?
            #Check delta
            layer_1_delta = layer_1_error*self.sigmoid_output_to_derivative(layer_1)
    
            #Find the final weights
            synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta))  
                   
        print "Total Iterations =",j
        print "Error is",err
        return synapse_0
        
    def testNN(self,X,w):
        layer_0 = X
        layer_1 = self.sigmoid(np.dot(layer_0,w))
        return layer_1

class WF(object):
    
    #sw_path = "C:\\Users\\u505123\\Documents\\Project\\ann_text-master\\stopwords.txt"            
    
    def __init__(self, file_path,sw_path = "/home/vjyvtkr/WF/Project/stopwords.txt",feat=208):
        self.file_is = file_path
        self.inp_df = pd.read_csv(file_path)
        self.train_data = self.clean(self.inp_df[0:int(0.5*len(self.inp_df.index))])
        self.test_data = self.clean(self.inp_df[int(0.5*len(self.inp_df.index)):])

        sw_file = open(sw_path)    
        self.stop_words = sw_file.readlines()
        
        for i in range(0,len(self.stop_words)):
            self.stop_words[i] = self.stop_words[i].strip()

        sw_file.close()
    
        self.cv = CountVectorizer(stop_words=self.stop_words,max_features=feat)

    def cleanPhrase(self,s):
        s = re.sub(r"[^A-Za-z\']", " ", s)
        s = re.sub(r"\s{2,}", " ", s)
        s = re.sub(r" [A-Za-z] ", " ", s)
        return s.lower().strip()
    
    def clean(self,data):
        phrases = data["Phrase"]
        new_phrases = []    
        for i in phrases:
            x = self.cleanPhrase(i)
            new_phrases.append(x)
        data["Phrase"] = new_phrases
        return data#reindex(np.random.permutation(data.index))    
    
    def trainData(self,y):
        
        nn_inp = self.cv.fit_transform(self.train_data["Phrase"].values).toarray()
        neural_net = ANN()
        ans = neural_net.trainNN(nn_inp,y)
        return ans
    
    def testData(self,w):
        neural_net = ANN()
        test_sample = self.cv.fit_transform(self.test_data["Phrase"].values).toarray()
        test_out = neural_net.testNN(test_sample,w)    
        return test_out
    
    def plotCurves(self,fpr,tpr,thresholds):
        
        #ROC Curve
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
        plt.show()
        
        #Sensitivity vs. Specificity
        plt.plot(tpr,thresholds, 'b', label="Spec/Sens")
        plt.plot(1-fpr,thresholds,'r',label="Spec/Sens")
        plt.show()
        
        #Lift Curve
        #TP=self.ranks['tp']
        #FN=self.ranks['fn']
        plot_test_data=self.test_data.reset_index()
        del plot_test_data['index']
        plot_test_data['pred_scores'] = [i for j in self.predicted_prob.tolist() for i in j]
        plot_test_data = plot_test_data.sort(columns=["pred_scores"],ascending=False)
        plot_test_data=plot_test_data.reset_index()
        del plot_test_data['index']
        decile = [i for i in range(1,11)]
                
        #gain_model = [0]
        #model_cumul = 0.0
        dec = len(plot_test_data)/10
        vals = [dec for i in range(1,10)]
        dec1 = len(plot_test_data)%10
        vals.append(dec1)
        responses = []
        
        #plot_test_data = plot_test_data.sample(n = len(self.test_data))
        res_count=0 
        total_yes_count = 0
        for i in range(0,len(plot_test_data)):
            if not i%dec and i>0:
                responses.append(res_count)
                res_count=0
            if plot_test_data['pred_scores'][i]>=self.optimal_thresh:
                res_count+=1
                total_yes_count+=1
        
        cumm_response = [responses[0]]
        for i in range(1,len(responses)):
            cumm_response.append(responses[i]+cumm_response[i-1])
                    
        percent = []
        for i in responses:
            per = (float(i)/total_yes_count)*100
            percent.append("%.2f"%per)
        
        gains = []
        for i in cumm_response:
            per = (float(i)/total_yes_count)*100
            gains.append("%.2f" % per)
        percent_pop = range(10, 110, 10)        
        lift = []
        for i in range(0,10):
            x = float(gains[i])/percent_pop[i]
            lift.append(x)        

        fig3=plt.figure()
        pl_gains = fig3.add_subplot(111)
        pl_gains.plot(percent_pop, gains, 'b')
        x=[10,20,30,40,50,60,70,90,100]
        pl_gains.plot(x,x,'r')
        #fig4 = plt.figure()
        #pl_lift = fig4.add_subplot(111)
        #pl_lift.plot(percent_pop,lift, 'y')
        
        
        
    
    
    def run(self):
        self.y_train_scores = []
        self.y_test_scores = []        
        for i in self.train_data["Indicator"]:
            if i=="Yes":
               self.y_train_scores.append(1)
            else:
                self.y_train_scores.append(0)
        
        for i in self.test_data["Indicator"]:
            if i=="Yes":
                self.y_test_scores.append(1)
            else:
                self.y_test_scores.append(0)        
        
        valid_scores = np.array([self.y_train_scores]).T
        ans = self.trainData(valid_scores)
        self.predicted_prob = self.testData(ans)
        fpr,tpr,thresholds = sklearn.metrics.roc_curve(self.y_test_scores,self.predicted_prob)
        thresholds[0]=1.0
        max_sum = 0
        self.optimal_thresh = 0.5
        for i in range(0,len(tpr)):
            temp = tpr[i]+(1.0-fpr[i])
            if temp>max_sum:
                max_sum = temp
                self.optimal_thresh = thresholds[i]


        self.ranks = {'tp':0,'fp':0,'fn':0,'tn':0}
        ind=0
        self.final_scores = []
        for i in self.y_test_scores:
            if i==1:
                if self.predicted_prob[ind]>self.optimal_thresh:
                    self.ranks['tp'] = self.ranks['tp']+1
                    self.final_scores.append(1)
                else:
                    self.ranks['fn'] = self.ranks['fn']+1
                    self.final_scores.append(0)
            else:
                if self.predicted_prob[ind]>self.optimal_thresh:
                    self.ranks['fp'] = self.ranks['fp']+1
                    self.final_scores.append(1)
                else:
                    self.ranks['tn'] = self.ranks['tn']+1
                    self.final_scores.append(0)
            ind+=1
            
        precision = float(self.ranks['tp'])/(self.ranks['tp']+self.ranks['fp'])
        sensitivity = float(self.ranks['tp'])/(self.ranks['tp']+self.ranks['fn'])    
        accuracy = float(self.ranks['tp']+self.ranks['tn'])/(self.ranks['tp']+self.ranks['tn']+self.ranks['fp']+self.ranks['fn'])
        
        self.plotCurves(fpr,tpr,thresholds)        
        
        return [precision, sensitivity, accuracy]
