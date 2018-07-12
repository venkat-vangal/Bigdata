# -*- coding: utf-8 -*-
"""
Created on Sat May  5 21:27:51 2018

@author: v-venva
"""

NB = 4 # number of bins
x = np.array([1.0,12.0,6.0,2.0,15.0,3.0,5.0,9.0,8.0,8.0,2.0,5.0,7.0,3.0,6.0,20.0])
X = pd.DataFrame(x)

NB = 3
import numpy as np
x = np.array([81, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 
              6, 6, 7, 7, 7, 7, 8, 8, 9, 12, 24, 24, 25])
X = pd.DataFrame(x)
bounds = np.linspace(np.min(x), np.max(x), NB + 1) # mo

def bin(x, b): # x = data array, b = boundaries array
    nb = len(b)
    N = len(x)
    y = np.empty(N, int) # empty integer array to store the bin numbers (output)
    
    for i in range(1, nb): # repeat for each pair of bin boundaries
        y[(x >= bounds[i-1])&(x < bounds[i])] = i
    
    y[x == bounds[-1]] = nb - 1 # ensure that the borderline cases are also binned appropriately
    return y

bx = bin(x, bounds)
minmax_scale = MinMaxScaler().fit(X)
standardization_scale = StandardScaler().fit(X)
y = minmax_scale.transform(X)
z = standardization_scale.transform(X)
print ("\nScaled variable x using MinMax and Standardized scaling\n")
print (np.hstack((np.reshape(x, (28,1)), y, z)))



import numpy as np
x = np.array([81, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 
              6, 6, 7, 7, 7, 7, 8, 8, 9, 12, 24, 24, 25])
MaxBin1 = 5.5
MaxBin2 = 7.5
labeled = np.empty(28, dtype=str)     
labeled[(x > -float("inf")) & (x <= MaxBin1)]      = "Bin1"
labeled[(x > MaxBin1)       & (x <= MaxBin2)]      = "Bin2"
labeled[(x > MaxBin2)       & (x <= float("inf"))] = "Bin3"

import numpy as np
x = np.array([81, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 
              6, 6, 7, 7, 7, 7, 8, 8, 9, 12, 24, 24, 25])
MaxBin1 = 29
MaxBin2 = 55
labeled = np.empty(28, dtype=str)     
labeled[(x > -float("inf")) & (x <= MaxBin1)]      = "Bin1"
labeled[(x > MaxBin1)       & (x <= MaxBin2)]      = "Bin2"
labeled[(x > MaxBin2)       & (x <= float("inf"))] = "Bin3"
print(labeled)

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
d = np.array([1., -1, -1, 1, 1, 17, -3, 1, 1, 3])
dmm = (d - np.min(d))/(np.max(d) - np.min(d))
np.mean(dmm)
np.min(dmm)
dz = (d - np.mean(d))/np.std(d)
np.mean(dz)
np.min(dz)

import pandas as pd

# Download the data
# http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data"
Mamm = pd.read_csv(url, header=None)
Mamm.columns = ["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"] 
##################
Mamm = Maam.isnull
d = np.array(Mamm["BI-RADS"])
y = d.astype(np.float)
dz = (d - np.mean(d))/np.std(d)


DeviceTypes = [
"Cell Phone", "Dish Washer", "Laptop", "Phone", "Refrigerator", "Server",
"Oven", "Computer", "Drill", "Server", "Saw", "Computer", "Nail Gun",
"Screw Driver", "Drill", "Saw", "Saw", "Laptop", "Oven", "Dish Washer",
"Oven", "Server", "Mobile Phone", "Cell Phone", "Server", "Phone"]
Devices = pd.DataFrame(DeviceTypes, columns=["Names"])

Devices.loc[:,"Names"].value_counts()
Devices.loc[Devices.loc[:, "Names"] == "Laptop", "Consolidated_name"] = "Computer"

len(Devices.loc[:,"Names"])

len(Devices.loc[:,"Names"].unique())

Devices.loc[:,"Names"].value_counts().plot(kind='bar')


import pandas as pd
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
Auto = pd.read_csv(url, delim_whitespace=True, header=None)
Auto.columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name"]
Auto.loc[:,"weight"].value_counts().plot(kind='bar')
plt.hist(Auto)
Auto.groupby("weight").size().plot(kind='bar')
plt.hist(Auto.loc[:, "weight"])

import numpy as np
x = np.array(["WA", "Washington", "Wash", "UT", "Utah", "Utah", "UT", "Utah", "IO"])
WA = x[x == "Washington"] WA = x[x == "Wash"]UT = x[x == "Utah"]

x[x == "Washington"] = "WA"
x[x == "Wash"] = "WA"
x[x == "Utah"] = "UT"


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
Auto = pd.read_csv(url, delim_whitespace=True, header=None)
Auto.columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", 
               "acceleration", "model_year", "origin", "car_name"]
Auto.head()

AB = ["B", "A", "B", "B", "B", "A"]
Test = pd.DataFrame(AB, columns=["AB"])
Test.loc[:, "isA"] = (Test.loc[:, "AB"] == "A").astype(int)


import pandas as pd
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
Auto = pd.read_csv(url, delim_whitespace=True, header=None)
Auto.columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name"]
plt.hist(Auto.loc[:, "weight"])
Auto.groupby("weight").size().plot(kind='bar')