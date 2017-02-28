
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np 
import matplotlib as plt

#______________________________________________________________________________________________________
def cleanTrainData():

    global dframe
    print dframe.info()
    # Survived : 0 indicates  no and 1 = yes
    
    dframe.Survived.value_counts()

    dframe.Survived.value_counts(1)

    '''Handling missing values  : Conisdering all attributes '''
    print dframe[dframe.Survived.isnull()]
    print dframe[dframe.Pclass.isnull()]
    print dframe[dframe.Age.isnull()]  
    '''Pclass and Survived aren't having missing values  ,AGE Column has missing values ,best way is to handle age is by 
    taking  average'''

    avgAge= dframe.Age.mean()
    dframe.Age = dframe.Age.fillna(avgAge)
    dframe[dframe.Age.isnull()]

    dframe.Age = dframe.Age.astype("int64")
    print dframe
    
    dframe.Fare = dframe.Fare.astype("float64")
    #for col in ['Fare']:
      #  dframe[col] = pd.convert_objects(dframe[col],convert_numeric = "True")
    
    dframe['Fare'] = dframe['Fare'].fillna(value  =dframe.Fare.median())
    dframe['Embarked'] = dframe['Embarked'].fillna(value= "None")
    
    dframe['Cabin']=dframe.Cabin.apply(handleCabinVal)
    handleCatValTrain()
    
    dframe.drop(dframe.index.get_duplicates())
    #saving the test  data for further use
    dframe.to_csv("./CleanDataTrain.csv") 
#______________________________________________________________________________________________________    
'''The test dataset is to be predicted by algorithm which is trained on Train.csv and predict how fit the model 
   is  for given dataset'''
def cleanTestData():
    #The Age attribute has float values, and some blank values

    avgAgeTest = dataFrameTest.Age.mean()
    print avgAgeTest  #so this comes to about : 30
    dataFrameTest.Age = dataFrameTest.Age.fillna(value=avgAgeTest)
    print dataFrameTest[dataFrameTest.Age.isnull()]

    dataFrameTest.Age = dataFrameTest['Age'].astype("int64")
    print dataFrameTest

    dataFrameTest['Fare'] = dataFrameTest['Fare'].fillna(value  = dataFrameTest.Fare.median())

    dataFrameTest['Cabin'] = dataFrameTest.Cabin.apply(handleCabinVal)
    handleCatValTest()
    
    dataFrameTest.drop(dataFrameTest.index.get_duplicates())
    dataFrameTest.to_csv("./CleanDataTest.csv") #saving data in csv format for further analysis
    
#______________________________________________________________________________________________________
''' handle the categorical data in both train and test set'''    
def handleCatValTest():
    '''Not considering PId ,Name , ticket , filling blank values with NA '''
    global dataFrameTest
    dataFrameTest.drop(['Name','PassengerId','Ticket'],axis = 1, inplace = "True")  #drop axis = 1 : column
    cat_var= ['Sex','Embarked','Cabin']
    for var in cat_var :
        dataFrameTest[var].fillna(value = "None",inplace="True")
        #create dummy variables
        dummy_var = pd.get_dummies(dataFrameTest[var],prefix= "Variables")
    
    #update the dataframe by appending it
        dataFrameTest = pd.concat([dataFrameTest,dummy_var],axis= 1) 
    #delete the main categories
        dataFrameTest.drop([var],axis=1,inplace="True")
#____________________________________________________________________________________________________   

def handleCatValTrain():
    '''Not considering PId ,Name , ticket , filling blank values with NA '''
    global dframe
    dframe.drop(['Name','PassengerId','Ticket'],axis = 1, inplace = "True")  #drop axis = 1 : column
    cat_var= ['Sex','Embarked','Cabin']
    for var in cat_var :
        dframe[var].fillna(value = "None",inplace="True")
     
        #create dummy variables
        dummy_var = pd.get_dummies(dframe[var],prefix= "Variables")
    
        #update the dataframe by appending it
        dframe = pd.concat([dframe,dummy_var],axis= 1) 
        #delete the main categories
        dframe.drop([var],axis=1,inplace="True")
        
#____________________________________________________________________________________________________   


def handleCabinVal(var):
  
    # Cabin column has value  :C32,c23  , so just taking first letter and  
    try : 
        return var[0]
    except TypeError:
        return "None"

#______________________________________________________________________________________________________
 
global dframe, dataFrameTest
dframe = pd.read_csv("./train.csv",sep=",")
dataFrameTest =  pd.read_csv("./test.csv",sep=",")









