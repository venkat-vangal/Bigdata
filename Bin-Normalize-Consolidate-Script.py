"""
1 Account for aberrant data (missing and outlier values).
2 Normalize numeric values (at least 1 column).
3 Bin numeric variables (at least 1 column).
4 Consolidate categorical data (at least 1 column).
5 Remove obsolete columns.
"""

# import packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import *
import matplotlib.pyplot as plt
###################

# Download the data
# https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
Adult = pd.read_csv(url, header=None)
Adult.columns = ["Age", "workclass", "fnlwgt", "education", "educationnum",
             "marital-status","occupation","relationship","race",
             "sex", "capital-gain","capital-loss","hours-per-week","native-country","income"] 
##################

#1 Account for aberrant data (missing and outlier values).
#find education Column with Null Values
nullRows= Adult.isnull().sum()

Adult.loc[:,"educationnum"].value_counts()

x = Adult.loc[:,"education"] == "Some-college"

# Do not allow specific texts
MissingValue = (Adult.loc[:,"education"] == "?") | (Adult.loc[:,"education"] == " ") | (Adult.loc[:,"education"] == "Some-college")

# Impute missing values
Adult.loc[Adult.loc[:,"education"] == "Some-college", "education"] = "noeducation"

###################################################
#2 Normalize numeric values (at least 1 column).
###################################################

d = np.array(Adult.loc[:,"educationnum"])


dz = (d - np.mean(d))/np.std(d)
dmm = (d - np.min(d))/(np.max(d) - np.min(d))

# 1st min-max normalization then Z-normalization 
dmmz = (dmm - np.mean(dmm))/np.std(dmm)

# 1st Z-normalization then Z-normalization 
dzz = (dz - np.mean(dz))/np.std(dz)

# 1st Z-normalization then min-max normalization
dzmm = (dz - np.min(dz))/(np.max(dz) - np.min(dz))

# 1st min-max normalization then min-max normalization
dmmmm = (dmm - np.min(dmm))/(np.max(dmm) - np.min(dmm))

binsval = 16

plt.hist(d, bins = binsval, color=[0, 0, 0, 1])
plt.title("Education Number Original Distribution")
plt.show()

plt.hist(dz, bins = binsval, color=[1, 1, 0, 1])
plt.title("Education Number Z-normalization")
plt.show()

plt.hist(dmm, bins = binsval, color=[0, 0, 1, 1])
plt.title("Education Number  min-max-normalization")
plt.show()

# Compare the 3 distributions!  Pay attention to the scale on the coordinate!

plt.hist(dmmz, bins = binsval, color=[1, 1, 0, 1])
plt.title("Education Number 1st Min-Max then Z-normalization")
plt.show()

plt.hist(dzz, bins = binsval, color=[1, 1, 0, 1])
plt.title("Education Number 1st  Z then Z-normalization")
plt.show()

plt.hist(dzmm, bins = binsval, color=[0, 0, 1, 1])
plt.title("Education Number 1st  Z then Min-Max-normalization")
plt.show()

plt.hist(dmmmm, bins = binsval, color=[0, 0, 1, 1])
plt.title("Education Number 1st  Min-Max then Min-Max-normalization")
plt.show()

###################################################
#3 Bin numeric variables (at least 1 column).
###################################################


NB = 3 # number of bins

############
# freq, bounds = np.histogram(x, NB) # one way of obtaining the boundaries of the bins
bounds = np.linspace(np.min(d), np.max(d), NB + 1) # more straight-forward way for obtaining the boundaries of the bins
############



def bin(x, b): # x = data array, b = boundaries array
    nb = len(b)
    N = len(x)
    y = np.empty(N, int) # empty integer array to store the bin numbers (output)
    
    for i in range(1, nb): # repeat for each pair of bin boundaries
        y[(x >= bounds[i-1])&(x < bounds[i])] = i
    
    y[x == bounds[-1]] = nb - 1 # ensure that the borderline cases are also binned appropriately
    return y
#############
    

#############

""" BINNING """
bx = bin(d, bounds)
print ("\n\nBinned variable educationnumber, for ", NB, "bins\n")
print ("Bin boundaries: ", bounds)
print ("Binned variable: ", bx)

#############

# Equal-width Binning using numpy
NumberOfBins = 3
BinWidth = (max(d) - min(d))/NumberOfBins
MaxBin2 = min(x) + 2 * BinWidth
MaxBin3 = min(x) + 3 * BinWidth
MaxBin4 = min(x) + 4 * BinWidth

print("Bin 1 ends at",BinWidth)
print("Bin 2 ends at",MaxBin2)
print("Bin 3 ends at",MaxBin3)

# Equal-frequency Binning
BinCount=len(d)/NumberOfBins
print("Each Bin contains",BinCount,"elements.")

##############

##############
#4 Consolidate categorical data (at least 1 column).
##############

Adult.loc[:,"educationnum"]

Adult.loc[ Adult.loc[:, "educationnum"] <= 5, "educationnum"] = 5
Adult.loc[ (Adult.loc[:, "educationnum"] > 5) & (Adult.loc[:, "educationnum"] <= 8), "educationnum"] = 8
Adult.loc[ (Adult.loc[:, "educationnum"] > 9) & (Adult.loc[:, "educationnum"] <= 12), "educationnum"] = 12

# Create 6 new columns, one for each state in "educationnumber"
Adult.loc[:, "Primary"] = (Adult.loc[:, "educationnum"] == 5).astype(int)
Adult.loc[:, "Middle"] = (Adult.loc[:, "educationnum"] == 8).astype(int)
Adult.loc[:, "High"] = (Adult.loc[:, "educationnum"] == 12).astype(int)
Adult.loc[:, "Bachelors"] = (Adult.loc[:, "educationnum"] == 13).astype(int)
Adult.loc[:, "Masters"] = (Adult.loc[:, "educationnum"] == 14).astype(int)
Adult.loc[:, "Doctorate"] = (Adult.loc[:, "educationnum"] == 16).astype(int)

#################################
#5 Remove obsolete columns.
Adult = Adult.drop("educationnum", axis=1)
Adult.head()
##############

# Write a local copy of the file. index=False does not create a new column for the indices
Adult.to_csv('VenkatRaoVangalapudi-M02-Dataset.csv', sep=",", index=False)

# Where is the file located?
import os
os.getcwd()
os.listdir()

# Check the file displays the same dataframe as before
Adult2=pd.read_csv('VenkatRaoVangalapudi-M02-Dataset.csv')
Adult2.head()

###################
