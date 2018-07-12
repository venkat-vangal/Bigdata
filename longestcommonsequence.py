# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 02:21:18 2018

@author: v-venva
"""

# get the input
n, m = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# fill the table
f = [[0]*(m+1) for i in range(n+1)]
for i in range(n+1):
    for j in range(m+1):
        if i == 0 or j == 0:
            f[i][j] = 0
        elif A[i-1] == B[j-1]:
            f[i][j] = 1 + f[i-1][j-1]
        else:
            f[i][j] = max(f[i-1][j], f[i][j-1])

# define the 'reconstruct' function
def reconstruct(i, j):
    if i == 0 or j == 0:
        return []
    elif A[i-1] == B[j-1]:
        return reconstruct(i-1, j-1) + [A[i-1]]
    elif f[i][j] == f[i-1][j]:
        return reconstruct(i-1, j)
    else:
        return reconstruct(i, j-1)

# reconstruct the LCS, and print
print(*reconstruct(n, m))