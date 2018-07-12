# L06-A-5-Normalization.py

# Categorical data represents categories
# The datatype that represents categories is string

# Binning is important to convert numerical data into categories
# (categorical data) when numerical data is not well tolerated.

import numpy as np
############
x = np.array((81, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 9, 12, 24, 24, 25))
############

# Equal-width Binning using numpy
NumberOfBins = 3
BinWidth = (max(x) - min(x))/NumberOfBins
MinBin1 = float('-inf')
MaxBin1 = min(x) + 1 * BinWidth
MaxBin2 = min(x) + 2 * BinWidth
MaxBin3 = float('inf')

print("\n########\n\n Bin 1 is from ", MinBin1, " to ", MaxBin1)
print(" Bin 2 is greater than ", MaxBin1, " up to ", MaxBin2)
print(" Bin 3 is greater than ", MaxBin2, " up to ", MaxBin3)

Binned_EqW = np.array([" "]*len(x)) # Empty starting point for equal-width-binned array
Binned_EqW[(MinBin1 < x) & (x <= MaxBin1)] = "L" # Low
Binned_EqW[(MaxBin1 < x) & (x <= MaxBin2)] = "M" # Med
Binned_EqW[(MaxBin2 < x) & (x  < MaxBin3)] = "H" # High

print(" x binned into 3 equal-width bins: ")
print(Binned_EqW)

# Equal-frequency Binning
ApproxBinCount = len(x)/NumberOfBins
print("\n########\n\n Each bin should contain approximately", ApproxBinCount, "elements.")

print(np.sort(x))

# Bins with 4, 12, and 12 elements:
# 3, 3, 4, 4, | 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, | 7, 7, 7, 7, 8, 8, 9, 12, 24, 24, 25, 81

# Bins with 14, 6, and 8 elements:
# 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, | 6, 6, 7, 7, 7, 7, | 8, 8, 9, 12, 24, 24, 25, 81
MaxBin1 = 5.5
MaxBin2 = 7.5

Binned_EqF = np.array([" "]*len(x)) # Empty starting point for equal-frequency-binned array
Binned_EqF[(MinBin1 < x) & (x <= MaxBin1)] = "L" # Low
Binned_EqF[(MaxBin1 < x) & (x <= MaxBin2)] = "M" # Med
Binned_EqF[(MaxBin2 < x) & (x  < MaxBin3)] = "H" # High

print(" x binned into 3 equal-freq1uency bins: ")
print(Binned_EqF)
