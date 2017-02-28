import argparse
import os
import pandas as pd

#from Sample3 import __dframe__
#from Sample3 import dataFrameTest

parser = argparse.ArgumentParser(description = " This is program for analysing ")
parser.add_argument('-c','--clean',help="Clean the dataset",action = "store_true")
parser.add_argument('-a','--analyse',help="Implement classifier",action="store_true")
send_args=parser.parse_args()



if (send_args.clean) :
    import Clean
    Clean.cleanTestData()
    Clean.cleanTrainData()
    print " Cleaning Task Successful"
elif  (send_args.analyse) : 
    import Implement
    print " The analysis : is :\n"
    Implement.getRandomForestClassifier()
    Implement.getNaiveBayesClassifier()
    Implement.getLogisticRegression()
else :
    print "Run with parameters "
