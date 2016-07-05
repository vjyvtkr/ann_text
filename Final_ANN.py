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
import warnings
warnings.filterwarnings("ignore")
'''
This is the ANN class which implements the Artificial Neural Network.
For training the data it takes following inputs:
    1. X - Feature set
    2. y - Labelled Data
    3. alpha - This is the learning rate, default is 0.1
    4. max_iter - Maximum iterations to see the error minimized, default = 5000
For testing, it just takes the input data abd the calculated weights

Two methods of this calss are to calculate the activation function used in the NN.
1 - sigmoid -> the sigmoid function
2 - sigmoid_output_to derivative - calculates the derivative of the sigmoid function and returns it.
'''
class ANN(object):
    
    # compute sigmoid nonlinearity
    def sigmoid(self,x):
        output = 1/(1+np.exp(-x))
        return output
    
    # convert output of sigmoid function to its derivative
    def sigmoid_output_to_derivative(self,output):
        return output*(1-output)    
    
    #Train the data
    def trainNN(self,X,y,alpha=0.1,max_iter=5000):
            
        dim = len(X[0])
        #print "\nTraining With Alpha:" + str(alpha)
        np.random.seed(1)
    
        # randomly initialize our weights with mean 0
        synapse_0 = 2*np.random.random((dim,1)) - 1

        j=-1
        while j<max_iter:
            j+=1

            #Feed forward through layers 0 and 1
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
                   
        #print "Total Iterations =",j
        return synapse_0
    
    #Test the data
    def testNN(self,X,w):
        layer_0 = X
        layer_1 = self.sigmoid(np.dot(layer_0,w))
        return layer_1

'''
This is the WF class, to run the training and testing datasets and to get the required output.
Input while instantiation of the WF object:
    1 - A file/panda Dataframe: You can either input the path of the file or the dataframe itself. 
        Please refer readme for the structure of the Dataframs.
    2 - feature count: The default feature count is 208, user can enter required feature count.
    3 - sw: This is the stop_words text file. Default sw file is in the project folder.

'''

class WF(object):   
    
    #instantiate the WF object
    def __init__(self,file_or_df,feat=208,sw="stopwords.txt"):

        self.feat = feat
        self.sw_path = sw        

        if type(file_or_df) is str:
            self.file_is = file_or_df
            self.inp_df = pd.read_csv(file_or_df)
        else:
            self.inp_df = file_or_df
        
        #Divide the training and testing data. 70% = Training, 30% = Testing.
        self.train_data = self.clean(self.inp_df[0:1874])
        self.test_data = self.clean(self.inp_df[1874:])

        sw_file = open(self.sw_path)    
        self.stop_words = sw_file.readlines()
        
        for i in range(0,len(self.stop_words)):
            self.stop_words[i] = self.stop_words[i].strip()

        sw_file.close()
    
        #This is the count vectorizer to convert the sentence data to features matrix
        vocab = open("SVMVocab.txt")
        data = vocab.read()
        self.vocab = data.split(",")
        self.cv = CountVectorizer(stop_words=self.stop_words,vocabulary=self.vocab)    
    #cleanPhrse removes all the unnessary data from the sentences.
    def cleanPhrase(self,s):
        s = re.sub(r"[^A-Za-z\']", " ", s)
        s = re.sub(r"\s{2,}", " ", s)
        s = re.sub(r" [A-Za-z] ", " ", s)
        return s.lower().strip()
    
    #Returns a random dataframe after cleaning the data
    def clean(self,data):
        phrases = data["Phrase"]
        new_phrases = []    
        for i in phrases:
            x = self.cleanPhrase(i)
            new_phrases.append(x)
        data["Phrase"] = new_phrases
        return data
    
    #train the given data with the ANN
    def trainData(self,y):
        nn_inp = self.cv.fit_transform(self.train_data["Phrase"].values).toarray()
        neural_net = ANN()
        ans = neural_net.trainNN(nn_inp,y)
        return ans
        
    #Test the given data with the ANN
    def testData(self,ans):
        test_sample = self.cv.fit_transform(self.test_data["Phrase"].values).toarray()
        neural_net = ANN()        
        test_out = neural_net.testNN(test_sample,ans)    
        return test_out
    
    #Plot the various modelling curves
    def plotCurves(self,fpr,tpr,thresholds):
        
        #ROC Curve
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
        plt.legend(shadow=True, fancybox=True, loc=4)
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.xlabel('False Positive Rate (1-Specificity)')
        plt.title("ROC Curve")
        plt.show()
        
        #Sensitivity vs. Specificity
        fig, ax1 = plt.subplots()
        ax1.plot(tpr,thresholds, 'b')
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Sensitivity', color='b')
        for tl in ax1.get_yticklabels():
            tl.set_color('b')
        ax2 = ax1.twinx()
        ax2.plot(1-fpr,thresholds,'r')
        ax2.set_ylabel('Specificity', color='r')
        for tl in ax2.get_yticklabels():
            tl.set_color('r')
        plt.title("Specificity-Sensitivity Curve")
        plt.show()
        
        #Gain Curve
        plot_test_data=self.test_data.reset_index()
        del plot_test_data['index']
        plot_test_data['pred_scores'] = [i for j in self.predicted_prob.tolist() for i in j]
        plot_test_data = plot_test_data.sort(columns=["pred_scores"],ascending=False)
        plot_test_data=plot_test_data.reset_index()
        del plot_test_data['index']
        
        dec = len(plot_test_data)/10
                
        vals = [0] + [dec for i in range(1,10)]
        dec1 = len(plot_test_data)%10
        vals.append(dec1)
        responses = [0]        
        
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
            percent.append(float("%.2f"%per))
        
        self.gains = []
        for i in cumm_response:
            per = (float(i)/total_yes_count)*100
            self.gains.append(float("%.2f" % per))
        
        percent_pop = range(0, 110, 10)        
        
        '''self.lift = []
        for i in range(0,len(self.gains)):
            if i==0:
                x=0
            else:
                x = self.gains[i]/percent_pop[i]
            self.lift.append(x)        
        '''
        x = [0,10,20,30,40,50,60,70,90,100]
        plt.plot(percent_pop, self.gains, 'b', label="Model")
        plt.plot(x,x,'r',label="Random")
        plt.legend(shadow=True, fancybox=True,loc=4)        
        plt.ylabel('Target Percentage')
        plt.xlabel('Population Percentage')
        plt.title("Gains Chart")
        plt.show()
        
        '''
        plt.plot(percent_pop,self.lift, 'y', label="Lift")
        plt.legend(shadow=True, fancybox=True,loc=1)        
        plt.ylabel('Cummulative Lift')
        plt.xlabel('Population Percentage')
        plt.title("Lift Chart")
        plt.show()        
        '''
    #Couple of getters
    def print_confusion_matrix(self):
        print "----------------------------------------------"
        print "|                       |       Model        |"
        print "----------------------------------------------"
        print "----------------------------------------------"
        print "|                       |True     |  False   |"
        print "----------------------------------------------"
        print "|                   |Yes|  %s    |    %s    |"%(self.ranks['tp'],self.ranks['fp'])
        print "  Actual            |---|---------|-----------"
        print "|                   |No |  %s     |    %s   |"%(self.ranks['fn'],self.ranks['tn'])
        print "----------------------------------------------"
        
    def get_confusion_matrix(self):
        return self.ranks
    
    #This is the run function.
    #Arguments are whether to print the curves or not. Default is False
    def run(self,print_curves=False):
        
        #Labelled O/P        
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
        
        #Train the data and get the weight matrix
        valid_scores = np.array([self.y_train_scores]).T
        ans = self.trainData(valid_scores)
        
        #Get the probabilities after testing the data on the obtained weights
        self.predicted_prob = self.testData(ans)
        
        #calculate the false positive rate, true positive rate and threshold matrix
        self.fpr,self.tpr,self.thresholds = sklearn.metrics.roc_curve(self.y_test_scores,self.predicted_prob)
        self.thresholds[0]=1.0
        
        #Get the optimal threshold
        max_sum = 0
        self.__optimal_thresh = 0.5
        for i in range(0,len(self.tpr)):
            temp = self.tpr[i]+(1.0-self.fpr[i])
            if temp>max_sum:
                max_sum = temp
                self.optimal_thresh = self.thresholds[i]
        

        #Keep track of all the false positives:fp, true positives:tp, false negatives:fn, true negatives:tn
        
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
        
        #Return the precision, sensitivity and accuracy            
        precision = float(self.ranks['tp'])/(self.ranks['tp']+self.ranks['fp'])
        sensitivity = float(self.ranks['tp'])/(self.ranks['tp']+self.ranks['fn'])    
        accuracy = float(self.ranks['tp']+self.ranks['tn'])/(self.ranks['tp']+self.ranks['tn']+self.ranks['fp']+self.ranks['fn'])
        
        if print_curves:
            self.plotCurves(self.fpr,self.tpr,self.thresholds)        
        
        return ["Precision = %s"%(precision), "Sensitivity = %s"%(sensitivity), "Accuracy = %s"%(accuracy)]
