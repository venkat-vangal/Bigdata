# L06-A-2-Normalization.py

import numpy as np
import random as rn
import matplotlib.pyplot as plt


# Distribution 1
sigma = 0.3
mu1a = 8.9
mu1b = 10.1
d1 = [6.5]
d1 = d1 + [rn.normalvariate(mu1a, sigma) for ii in range(100)]
d1 = d1 + [rn.normalvariate(mu1b, sigma) for ii in range(50)]
d1 = np.array(d1)

# Distribution 2
sigma = 1
mu2a = 3
mu2b = 7
d2 = [15]
d2 = d2 + [rn.normalvariate(mu2a, sigma) for ii in range(100)]
d2 = d2 + [rn.normalvariate(mu2b, sigma) for ii in range(50)]
d2 = np.array(d2)

# Compare the distributions by overlaying histograms
plt.hist(d1, bins = 20, color=[0, 0, 1, 0.5])
plt.hist(d2, bins = 20, color=[1, 1, 0, 0.5])
plt.title("Compare Distributions without normalization")
plt.show()
# Are the distributions different?  If so, how are they different?

# Min Max Normalization or Feature Scaling
d1mm = (d1 - np.min(d1))/(np.max(d1) - np.min(d1))
d2mm = (d2 - np.min(d2))/(np.max(d2) - np.min(d2))
# Compare the distributions by overlaying histograms
plt.hist(d1mm, bins = 20, color=[0, 0, 1, 0.5])
plt.hist(d2mm, bins = 20, color=[1, 1, 0, 0.5])
plt.title("Compare Distributions after Min-Max Normalization")
plt.show()
# Are the distributions different?  If so, how are they different?

# Standard Score or Z-Normalization
d1z = (d1 - np.mean(d1))/np.std(d1)
d2z = (d2 - np.mean(d2))/np.std(d2)
# Compare the distributions by overlaying histograms
plt.hist(d1z, bins = 20, color=[0, 0, 1, 0.5])
plt.hist(d2z, bins = 20, color=[1, 1, 0, 0.5])
plt.title("Compare Distributions after Z-normalization")
plt.show()
# Are the distributions different?  If so, how are they different?
