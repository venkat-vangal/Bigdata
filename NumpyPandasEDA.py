"""
# Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
##############

# Download the data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data"
Mamm = pd.read_csv(url, header=None) 
Mamm.columns = ["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"]

# Check the data types
Mamm.dtypes
##############

# Corece to numeric and impute medians for BI-RADS column
Mamm.loc[:, "BI-RADS"] = pd.to_numeric(Mamm.loc[:, "BI-RADS"], errors='coerce')
HasNan = np.isnan(Mamm.loc[:,"BI-RADS"])
Mamm.loc[HasNan, "BI-RADS"] = np.nanmedian(Mamm.loc[:,"BI-RADS"])
##############

# Check the distribution of the "BI-RADS" column
plt.hist(Mamm.loc[:, "BI-RADS"])
##############

# Replace outlier
TooHigh = Mamm.loc[:, "BI-RADS"] > 6
Mamm.loc[TooHigh, "BI-RADS"] = 6

# Check the distribution of the "BI-RADS" column
plt.hist(Mamm.loc[:, "BI-RADS"])
##############

# Corece to numeric and impute medians for Age column
Mamm.loc[:, "Age"] = pd.to_numeric(Mamm.loc[:, "Age"], errors='coerce')
HasNan = np.isnan(Mamm.loc[:,"Age"]) 
Mamm.loc[HasNan, "Age"] = np.nanmedian(Mamm.loc[:,"Age"])

# Check the distribution of the "Age" column
plt.hist(Mamm.loc[:, "Age"])
##############

# The next ordinal or numeric column is >Density<. 

# Corece to numeric and impute medians for Density column
Mamm.loc[:, "Density"] = pd.to_numeric(Mamm.loc[:, "Density"], errors='coerce')
HasNan = np.isnan(Mamm.loc[:,"Density"])
Mamm.loc[HasNan, "Density"] =  np.nanmedian(Mamm.loc[:,"Density"])

# Check the distribution of the "Density" column
plt.hist(Mamm.loc[:, "Density"]) 
##############

# Check the distribution of the "Severity" column
plt.hist(Mamm.loc[:, "Severity"]) 

# Check the data types
Mamm.dtypes
#############
# Plot all the numeric columns against each other
scatter_matrix(Mamm) 
#############
_ = scatter_matrix(Mamm, c=Mamm.loc[:,"Severity"], figsize=[8,8], s=1000)

##############