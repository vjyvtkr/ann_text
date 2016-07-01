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
    
    def trainNN(self,X,y,dim):
    
        alpha = 0.1
        
        print "\nTraining With Alpha:" + str(alpha)
        np.random.seed(1)
    
        # randomly initialize our weights with mean 0
        synapse_0 = 2*np.random.random((dim,1)) - 1

        j=-1
        while j<5000:
            j+=1

            #Feed forward through layers 1 and 2
            layer_0 = X
            layer_1 = self.sigmoid(np.dot(layer_0,synapse_0))
    
            #Calculate the error in the layer
            layer_1_error = layer_1 - y
    
            #Absolute error
            err = np.mean(np.abs(layer_1_error))
            
            #Check if error is minimizing after every 100 iterations            
            if not j%100:
                print "Error after "+str(j)+" iterations:" + str(err)
            
            #if error goes below 0.1, break            
            if err < 0.1:
                break
    
            #In what direction is the target value?
            #Check delta
            layer_1_delta = layer_1_error*self.sigmoid_output_to_derivative(layer_1)
    
            #Find the final weights
            synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta))  
                   
        print "Total Iterations =",j
        return [synapse_0]
        
    def testNN(self,X,w):
        layer_0 = X
        layer_1 = self.sigmoid(np.dot(layer_0,w))
        return layer_1

class WF(object):
        
    sw_path = "C:\\Users\\u505123\\Documents\\Project\\ann_text-master\\stopwords.txt"            
    feat = 208    
    
    def __init__(self, file_path):
        self.file_is = file_path
        self.inp_df = pd.read_csv(file_path)
        
        self.train_data = self.clean(self.inp_df[0:int(0.7*len(self.inp_df.index))])
        self.test_data = self.clean(self.inp_df[int(0.7*len(self.inp_df.index)):])

        sw_file = open(self.sw_path)    
        self.stop_words = sw_file.readlines()
        
        for i in range(0,len(self.stop_words)):
            self.stop_words[i] = self.stop_words[i].strip()

        sw_file.close()
    
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
        return data.reindex(np.random.permutation(data.index))    
    
    def trainTestData(self,y):
        cv = CountVectorizer(stop_words=self.stop_words,max_features=self.feat)
        nn_inp = cv.fit_transform(self.train_data["Phrase"].values).toarray()
        neural_net = ANN()
        ans = neural_net.trainNN(nn_inp,y,len(nn_inp[0]))
    
        test_sample = cv.fit_transform(self.test_data["Phrase"].values).toarray()
        test_out = neural_net.testNN(test_sample,ans[0])    
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
        TP=self.ranks['tp']
        FN=self.ranks['fn']
        self.test_data=self.test_data.reset_index()
        gain_model = [0]
        model_cumul = 0.0
        dec = len(self.test_data)/10
        dec1 = len(self.test_data)/10
        self.test_data = self.test_data.sample(n = len(self.test_data))
        df1 = pd.DataFrame(columns=["Indicator","prob","true"])
        df1["Indicator"] = self.y_test_scores
        df1["prob"] = [i for j in self.predicted_prob.tolist() for i in j]
        df1["true"] = self.final_scores
        df1.sort(columns = 'prob', inplace = True, ascending = False)
        df1 = df1.reset_index(drop = True)
        df_GL = pd.DataFrame([],columns  = ['Decile', 'Random', 'Model'])
        
        for i in range(10):
            df_GL.loc[i, 'Decile'] = str(i + 1)
            model = 0.0
            if (i==9):
                dec1=len(self.test_data)%10
            for k in range(i*dec, (i*dec) + dec1):
                    if df1['true'][k] == 1:
                        model = model + 1
            model_cumul = model_cumul + model
            df_GL.loc[i, 'Model'] = str(model_cumul)
            val = ((model_cumul)/(TP + FN))*100.00
            if val>100.0:
                val=100.0
            gain_model.append(val)
        
        
        percent_pop = range(0, 110, 10)
        fig3=plt.figure()
        pl= fig3.add_subplot(111)
        pl.plot(percent_pop, gain_model, 'b')
        x=[0,10,20,30,40,50,70,90,100]
        pl.plot(x,x,'r')
        
    
    
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
        
        self.predicted_prob = self.trainTestData(valid_scores)
        fpr,tpr,thresholds = sklearn.metrics.roc_curve(self.y_test_scores,self.predicted_prob)
        
        for i in range(0,len(thresholds)):
            if thresholds[i]>1.0:
                thresholds[i]=1.0        
        
        idx = np.argwhere(np.isclose((1-fpr)*1000,tpr*1000, atol=10)).reshape(-1)
        optimal_thresh = round(thresholds[idx[idx.__len__()/2]],3)

        self.ranks = {'tp':0,'fp':0,'fn':0,'tn':0}
        ind=0
        self.final_scores = []
        for i in self.y_test_scores:
            if i==1:
                if self.predicted_prob[ind]>optimal_thresh:
                    self.ranks['tp'] = self.ranks['tp']+1
                    self.final_scores.append(1)
                else:
                    self.ranks['fn'] = self.ranks['fn']+1
                    self.final_scores.append(0)
            else:
                if self.predicted_prob[ind]>optimal_thresh:
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
        