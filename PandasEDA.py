"""
# Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
Cars = pd.read_csv(url, header=None)
Cars.columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "evaluation"]
####################

Cars.head()
####################
Cars.shape
####################

import matplotlib.pyplot as plt

# Try to create a histogram
plt.hist(Cars.loc[:,"doors"])
####################

# Find what values are in the doors column
Cars.loc[:,"doors"].unique()
####################
Cars.dtypes
####################

# determine the locations of "5more" in "doors"
FiveMore = Cars.loc[:,"doors"] == "5more"
#####################

# Assign the value 5 to wherever "5more" was found
Cars.loc[FiveMore,"doors"] = 5

# Find what values are in the doors column
print(Cars.loc[:,"doors"].unique())
#####################

# Try again to create a histogram.  This attempt should fail too!
plt.hist(Cars.loc[:,"doors"])
#####################

# Cast "doors" to integer
Cars.loc[:,"doors"] = Cars.loc[:,"doors"].astype(int)
print(Cars.loc[:,"doors"].unique())
#####################

# Try again to create a histogram.  This attempt should succeed.
plt.hist(Cars.loc[:,"doors"])
# What does the the histogram tell you?

######################

# What values are in the persons column?
print(Cars.loc[:,"persons"].unique()) 
######################

# Assign the value 6 to wherever "more" was found
Cars.loc[Cars.loc[:,"persons"] == "more","persons"] = 6
# Cast "persons" to integer
Cars.loc[:,"persons"] = Cars.loc[:,"persons"].astype(int)
# View the distribution of "persons"
plt.hist(Cars.loc[:,"persons"])

#####################

Cars.dtypes

# View the distribution of "buying"
Cars.groupby('buying').count().plot(kind='bar')

xc = Cars.groupby('buying').count()
type(xc)
xb = Cars.groupby('buying').size()
type(xb)
xc.plot(kind= 'bar')
xb.plot(kind= 'bar')


#####################