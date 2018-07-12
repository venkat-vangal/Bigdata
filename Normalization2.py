# L06-A-4-Normalization.py

import numpy as np
import random as rn
import matplotlib.pyplot as plt

# Normalization continued
# De-normalize linear normalizations

sigma = 1
mu2a = 3
mu2b = 7
d = [15]
d = d + [rn.normalvariate(mu2a, sigma) for ii in range(100)]
d = d + [rn.normalvariate(mu2b, sigma) for ii in range(50)]
d = np.array(d)

dz = (d - np.mean(d))/np.std(d)

# N = (X - o)/s
# Where:
# X is distribuition
# o is offset
# s is spread

# Get Denormalization with some algebra:
# X = N*s + o
d0 = dz*np.std(dz) + np.mean(dz)

plt.hist(d, bins = 20, color=[0, 0, 0, 1])
plt.title("Original Distribution")
plt.show()

plt.hist(dz, bins = 20, color=[1, 1, 0, 1])
plt.title("Z-normalization")
plt.show()

plt.hist(d0, bins = 20, color=[0, 0, 0, 1])
plt.title("De-normalized")
plt.show()
