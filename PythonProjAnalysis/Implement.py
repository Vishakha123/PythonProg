'''this code performs random forest classification on  train and test dataset '''
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import matplotlib as plt

def getNaiveBayesClassifier():
   
    global survivedGrp ,dframe,dframe1
    
    colArrTrain =  list(dframe.columns)
    colArrTest = list(dframe1.columns)
    
    
    
    nvByTrain = GaussianNB()
    nvByTrain.fit(dframe,survivedGrp)
     
    output_file_NB=nvByTrain.predict(dframe1)
    sg = survivedGrp[401:819]
    #dframe1['Survived'] = output_file_NB
    #output_file_NB = dframe1.copy()
    opdfNB  = dframe1.copy()
    opdfNB['Survived'] = output_file_NB
    
    #print  "Accuracy with Naive Bayesian:" +accuracy_score(sg,output_file_NB)
    if opdfNB.to_csv("OutputPredictionNB.csv"):
        print " File Saved Successfully!"
    
    print "Accuracy with Naive Bayesian: ",    nvByTrain.score(dframe,survivedGrp)
    
    
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def getRandomForestClassifier():
    '''create the random forest obj , and train it on  training datafile'''
    global dframe 
    global dframe1,survivedGrp
    
    randForTrain = RandomForestRegressor(n_estimators= 100,n_jobs =1,oob_score = True,random_state=42) #initialize
    opdfRF  = dframe1.copy()
    '''This function initializes Random Forest Regressor '''
    randForTrain.fit(dframe,survivedGrp)
    
    #predict result
    output_file_rf = randForTrain.predict(dframe1)
    #print output_file_rf
    
    opdfRF['Survived'] = output_file_rf
     
    if opdfRF.to_csv("OutputPredictionRF.csv") : 
        print " File Saved Successfully!"
    
    print "Out of bag score for Random Forest : " + str(randForTrain.oob_score_)
    surv_oob_score = randForTrain.oob_prediction_
    print "c-stat  for Random Forest Classifier: ",roc_auc_score(survivedGrp,surv_oob_score)
    #return roc_auc_score(survivedGrp,surv_oob_score)

    feature_series  = pd.Series(randForTrain.feature_importances_,index = dframe1.columns)
    feature_series.sort_values(inplace =  True)
    feature_series.plot(kind= "barh",figsize=(7,6))
    
   
 #_______________________________________________________________________________________________________________________   


def  getLogisticRegression():
    global dframe,dframe1,survivedGrp 
    
    modelLR  = LogisticRegression(random_state=42)
    opdfLR  = dframe1.copy()
    modelLR.fit(dframe,survivedGrp)
    varLR = modelLR.score(dframe,survivedGrp)
    print "correctly trained : ", varLR
    sg = survivedGrp[401:819]
    output_file_LR = modelLR.predict(dframe1)
    opdfLR['Survived'] = output_file_LR
        
    print  "accuracy :" , accuracy_score(sg,output_file_LR)
    if opdfLR.to_csv("OutputPredictionLR.csv"):
        print " File Saved Successfully!"
     

#______________________________________________________________________________________________________________

global dframe,dframe1,survivedGrp, var1, var2,var3
    #read the cleaned test and train data 
dframe = pd.read_csv("./CleanDataTrain.csv",index_col = 0)
dframe1 = pd.read_csv("./CleanDataTest.csv",index_col = 0)
   
survivedGrp = dframe. pop('Survived')
