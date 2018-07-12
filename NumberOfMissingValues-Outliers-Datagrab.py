"""
1 Import statements
2 URL of the data set and header names (if applicable)
3 Code the prints the first 5 rows of the data set
4 Code that assigns the column names
5 Code that histograms all numeric data
6 Code that histograms or bar charts all categorical data
7 Code that creates a scatter plot 
8 Code that determines the number of missing values for some attribute with missing values
9 Code that identifies potential outliers
Comments explaining the code blocks.
"""

#1 Import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
import time

#2 URL of the data set and header names (if applicable)
# Origin of data:
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

# Inform read_csv that the data contain no column headers
AdultDataset = pd.read_csv(url, header=None)
AdultDataset.head()
##################

#3 Code that assigns the column names
#Add Column headers
#age, gender, total Bilirubin, direct Bilirubin, total proteins, albumin, A/G ratio, SGPT, SGOT and Alkphos
AdultDataset.columns = ["Age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                "relationship","race", "sex","capital-gain","capital-loss","hours-per-week", "native-country","Income"]

#4 display sample data
AdultDataset.head()

ColumnTypes = AdultDataset.dtypes

#8 find Columns with Null Values
nullRows= AdultDataset.isnull().sum()

#5Code that histograms all numeric data
#for each Column in the data set
#loop through all the columns and draw histograms
for index, value in nullRows.items(): 
    #check if the column has any null/Nan values 
    if value == "int64": # draw historam for only numeric data
        #Find all nulls
        HasNan = np.isnan(AdultDataset.loc[:,index])
        #print(HasNan)
        #replace Nan values with median
        AdultDataset.loc[HasNan, index] = np.nanmedian(AdultDataset.loc[:,index])
  
        
for index, value in ColumnTypes.items(): 
    #Create Histogram   
    if value == "int64":
        plt.hist(AdultDataset.loc[:,index])        
        plt.show()
        print('Column: ', index)
        
        

#6 Code that histograms or bar charts all categorical data :Income
df = AdultDataset.replace(np.nan,0)
#loop through all numeric data
for index, value in ColumnTypes.items(): 
    #Create Histogram   
    if value == "int64":
        #loop through all categories
       for category, value1 in ColumnTypes.items(): 
           #Create Histogram   
           if value1 != "int64":
               print(category,index)
               df.groupby([category])[index].count().plot.bar()
               plt.show()
               time.sleep(2)
               #print(df1)
               #df1.plot.bar()

       print('Column: ', index)
        

print("Plotting Scatter Matrix..at the end .Please wait!!")

# Plot all the numeric columns against each other
for index, value in ColumnTypes.items(): 
    if value == "int64":
        s = scatter_matrix(AdultDataset, c=AdultDataset.loc[:,index],  figsize=[11,11], s=1000)
        plt.show()
        print('Column: ', index)
##############


#Find standard deviation for each column 
for index, value in ColumnTypes.items(): 
    if value == "int64":
        print('Standard Deviation of ',index, ':' , np.std(AdultDataset.loc[:,index]))

#9 Find outliers
LimitHi = np.mean(AdultDataset.loc[:,"fnlwgt"]) + 2*np.std(AdultDataset.loc[:,"fnlwgt"])
LimitLo = np.mean(AdultDataset.loc[:,"fnlwgt"]) - 2*np.std(AdultDataset.loc[:,"fnlwgt"])
print('LimitHi',LimitHi)
print('LimitLo',LimitLo)
########################

# Create Flag for values outside of limits
Outliers = (AdultDataset.loc[:,"fnlwgt"] < LimitLo) | (AdultDataset.loc[:,"fnlwgt"] > LimitHi)

#Print outliers
print(AdultDataset.loc[Outliers, "fnlwgt"])
AdultDataset.head()
plt.hist(AdultDataset.loc[:,"fnlwgt"])   
plt.show()
  

