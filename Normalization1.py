# L06-A-3-Normalization.py

import numpy as np
import random as rn
import matplotlib.pyplot as plt

# Normalization continued
# Compounded linear normalizations

sigma = 1
mu2a = 3
mu2b = 7
d = [15]
d = d + [rn.normalvariate(mu2a, sigma) for ii in range(100)]
d = d + [rn.normalvariate(mu2b, sigma) for ii in range(50)]
d = np.array(d)

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

plt.hist(d, bins = 20, color=[0, 0, 0, 1])
plt.title("Original Distribution")
plt.show()

plt.hist(dz, bins = 20, color=[1, 1, 0, 1])
plt.title("Z-normalization")
plt.show()

plt.hist(dmm, bins = 20, color=[0, 0, 1, 1])
plt.title("min-max-normalization")
plt.show()

# Compare the 3 distributions!  Pay attention to the scale on the coordinate!

plt.hist(dmmz, bins = 20, color=[1, 1, 0, 1])
plt.title("1st Min-Max then Z-normalization")
plt.show()

plt.hist(dzz, bins = 20, color=[1, 1, 0, 1])
plt.title("1st  Z then Z-normalization")
plt.show()

plt.hist(dzmm, bins = 20, color=[0, 0, 1, 1])
plt.title("1st  Z then Min-Max-normalization")
plt.show()

plt.hist(dmmmm, bins = 20, color=[0, 0, 1, 1])
plt.title("1st  Min-Max then Min-Max-normalization")
plt.show()

# Compare the distributions!  Pay attention to the scale on the coordinate!

# What is the effect of compounding linear normalizations?
