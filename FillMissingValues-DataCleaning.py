# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 13:48:58 2018

@author: Venkat Rao Vangalapudi
"""

import numpy as np

inputData = np.array([4,5,6,4,4,"4",-1,6,5,4,4,5,45,5,5,5,6,5,6,4,99,4,6,"?",6,6,None,4,5,4,6,6,6,6,4,5,4,6,4,6,4,5,5,5])

FlagMissingDigit = [(inputData == "?") | (inputData == " ") | (inputData == None)]


#FlagBad = ~ FlagDigit
inputData[FlagMissingDigit] = 0
 
#Fills Missing Values 
def fillMissingValues(inputData):
    "Find Missing Digit"
    FlagMissingDigit = [(inputData == "?") | (inputData == " ") | (inputData == None)]

    #Set missing data to Zero
    inputData[FlagMissingDigit] = 0
    
    return inputData;
   

def replaceOutLiers(inputData):
    
    #Clean The Data
    # Find elements that are numbers
    FlagDigit = [str(element).isdigit() for element in inputData]
    inputData = inputData[FlagDigit]
    
    print("Input Data After Cleanup",inputData)
    
    
    #Convert String Type to Integer
    inputData = inputData.astype(np.int)
    # calculate the limits for values that are not outliers. 
    LimitHi = np.mean(inputData) + 2*np.std(inputData)
    LimitLo = np.mean(inputData) - 2*np.std(inputData)

    ########################
    # Create Flag for values outside of limits
    FlagBad = (inputData < LimitLo) | (inputData > LimitHi)

    # present the flag
    FlagBad
    ########################

    # Replace outlieres with mean of the whole array
    inputData[FlagBad] = np.mean(inputData)
    
    # FlagGood is the complement of FlagBad
    FlagGood = ~FlagBad
    
    # Replace outleiers with the mean of non-outliers
    inputData[FlagBad] = np.mean(inputData[FlagGood])
    return inputData


# Removes Outliers from an Array
def removeOutliers( inputData ):
    
    #Clean The Data
    # Find elements that are numbers
    FlagDigit = [str(element).isdigit() for element in inputData]
    inputData = inputData[FlagDigit]
    
    print("Input Data After Cleanup",inputData)
    
    
    #Convert String Type to Integer
    inputData = inputData.astype(np.int)
    
    # Calculate the limits for values that are not outliers. 
    LimitHi = np.mean(inputData) + 2*np.std(inputData)
    LimitLo = np.mean(inputData) - 2*np.std(inputData)
    
    print("LimitHi:" ,LimitHi)
    print("LimitLo:", LimitLo)
    
    
    
    #Create Flag for values outside of limits
    FlagGood = (inputData <= LimitHi) & (inputData >= LimitLo)
    
    #replace the original array with new outcome    
    inputData = inputData[FlagGood]
    
    return inputData

inputData = np.array([4,5,6,4,4,"4",-1,6,5,4,4,5,45,5,5,5,6,5,6,4,99,4,6,"?",6,6,None,4,5,4,6,6,6,6,4,5,4,6,4,6,4,5,5,5])


print ("Result After Removing Outliers: ", removeOutliers(inputData))

print ("***************")

print ("Result After Replacing Outliers: ", replaceOutLiers(inputData))

print ("***************")

print ("Result After Replacing Missing Data: ", fillMissingValues(inputData))