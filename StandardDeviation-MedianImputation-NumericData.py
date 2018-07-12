"""
# Data Science
# Import statements
#Load the dataset to a data frame named ILPD from:  https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv (Links to an external site.)Links to an external site.
#Assign reasonable column names, like "Age", "Gender", and "TB" based on the data set description
#Histogram all variables.  Use plt.show() after each histogram
#Create a scatterplot.  Use plt.show() after the scatterplot
#Determine the standard deviation of all numeric variables.  Use print() for each standard deviation
#Median imputation of the missing numeric values
#Outlier replacement for column 2 (TB)########################
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

# Origin of data:
# https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv"

# Inform read_csv that the data contain no column headers
PatientDataset = pd.read_csv(url, header=None)
PatientDataset.head()
##################

#Add Column headers
#age, gender, total Bilirubin, direct Bilirubin, total proteins, albumin, A/G ratio, SGPT, SGOT and Alkphos
PatientDataset.columns = ["Age", "Gender", "Total Bilirubin", "Direct Bilirubin", "Total Proteins", "Albumin", "AG Ratio",
                "SGPT","SGOT", "Alkphos","Count"]

#display sample data
PatientDataset.head()

#find Columns with Null Values
nullRows= PatientDataset.isnull().sum()

#for each Column in the data set
#loop through all the columns and draw histograms
for index, value in nullRows.items(): 
    #check if the column has any null/Nan values 
    if (value>0) & (index != "Gender"): #Gender is not a numeric so omit Gender
        #Find all nulls
        HasNan = np.isnan(PatientDataset.loc[:,index])
        #print(HasNan)
        #replace Nan values with median
        PatientDataset.loc[HasNan, index] = np.nanmedian(PatientDataset.loc[:,index])
    #Create Histogram   
    if index != "Gender":
        plt.hist(PatientDataset.loc[:,index])        
        plt.show()
        print('Column: ', index)

# Plot all the numeric columns against each other
for index, value in nullRows.items(): 
    if index != "Gender":
        s = scatter_matrix(PatientDataset, c=PatientDataset.loc[:,index],  figsize=[11,11], s=1000)
        print('Column: ', index)
##############

#Find standard deviation for each column 
for index, value in nullRows.items(): 
    if index != "Gender":
        print('Standard Deviation of ',index, ':' , np.std(PatientDataset.loc[:,index]))

#Remove outliers
LimitHi = np.mean(PatientDataset.loc[:,"Total Bilirubin"]) + 2*np.std(PatientDataset.loc[:,"Total Bilirubin"])
LimitLo = np.mean(PatientDataset.loc[:,"Total Bilirubin"]) - 2*np.std(PatientDataset.loc[:,"Total Bilirubin"])
print('LimitHi',LimitHi)
print('LimitLo',LimitLo)
########################

# Create Flag for values outside of limits
FlagBad = (PatientDataset.loc[:,"Total Bilirubin"] < LimitLo) | (PatientDataset.loc[:,"Total Bilirubin"] > LimitHi)

#replace outliers
PatientDataset.loc[FlagBad, "Total Bilirubin"] = np.mean(PatientDataset.loc[:,"Total Bilirubin"])
PatientDataset.head()
plt.hist(PatientDataset.loc[:,"Total Bilirubin"])   
  

