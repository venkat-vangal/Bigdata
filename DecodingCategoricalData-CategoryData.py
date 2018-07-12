# -*- coding: utf-8 -*-
"""
Create a new Python script that includes the following:
1 Import statements
2 Loading your dataset
3 Decoding categorical data
4 Imputing missing values
5 Consolidating categories if applicable
6 One-hot encoding (dummy variables) for a categorical column with more than 2 categories
7 New columns created, obsolete deleted if applicable
8 Plots for 1 or more categories
9 Comments explaining the code blocks
10 Summary comment block on how the categorical data has been treated: decoded, imputed, consolidated, dummy variables created.
"""
# import package
import pandas as pd

# Download the data
# http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data"
Mamm = pd.read_csv(url, header=None)
Mamm.columns = ["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"] 
##################

# The category columns are decoded and missing values are imputed
Mamm.loc[ Mamm.loc[:, "Shape"] == "1", "Shape"] = "round"
Mamm.loc[Mamm.loc[:, "Shape"] == "2", "Shape"] = "oval"
Mamm.loc[Mamm.loc[:, "Shape"] == "3", "Shape"] = "lobular"
Mamm.loc[Mamm.loc[:, "Shape"] == "4", "Shape"] = "irregular"
Mamm.loc[Mamm.loc[:, "Shape"] == "?", "Shape"] = "irregular"
Mamm.loc[Mamm.loc[:, "Margin"] == "1", "Margin"] = "circumscribed"
Mamm.loc[Mamm.loc[:, "Margin"] == "2", "Margin"] = "microlobulated"
Mamm.loc[Mamm.loc[:, "Margin"] == "3", "Margin"] = "obscured"
Mamm.loc[Mamm.loc[:, "Margin"] == "4", "Margin"] = "ill-defined"
Mamm.loc[Mamm.loc[:, "Margin"] == "5", "Margin"] = "spiculated"
Mamm.loc[Mamm.loc[:, "Margin"] == "?", "Margin"] = "circumscribed"
####################

# Check the first rows of the data frame
Mamm.head()
####################

# Get the counts for each value
Mamm.loc[:,"Shape"].value_counts()
Mamm.loc[:,"Margin"].value_counts()
####################

# Plot the counts for each category
Mamm.loc[:,"Shape"].value_counts().plot(kind='bar')
####################

# Simplify Shape by consolidating oval and round
Mamm.loc[Mamm.loc[:, "Shape"] == "round", "Shape"] = "oval"

# Plot the counts for each category
Mamm.loc[:,"Shape"].value_counts().plot(kind='bar')
####################

# Plot the counts for each category
Mamm.loc[:,"Margin"].value_counts().plot(kind='bar')
####################

# Simplify Margin by consolidating ill-defined, microlobulated, and obscured
Mamm.loc[Mamm.loc[:, "Margin"] == "microlobulated", "Margin"] = "ill-defined"
Mamm.loc[Mamm.loc[:, "Margin"] == "obscured", "Margin"] = "ill-defined"

# Plot the counts for each category
Mamm.loc[:,"Margin"].value_counts().plot(kind='bar')

#####################

# hot encoding 

# Create 3 new columns, one for each state in "Shape"
Mamm.loc[:, "oval"] = (Mamm.loc[:, "Shape"] == "oval").astype(int)
Mamm.loc[:, "lobul"] = (Mamm.loc[:, "Shape"] == "lobular").astype(int)
Mamm.loc[:, "irreg"] = (Mamm.loc[:, "Shape"] == "irregular").astype(int)
##############

# Remove obsolete column
Mamm = Mamm.drop("Shape", axis=1)
Mamm.loc[:,"oval"].value_counts().plot(kind='bar')
Mamm.loc[:,"lobul"].value_counts().plot(kind='bar')
Mamm.loc[:,"irreg"].value_counts().plot(kind='bar')
##############
