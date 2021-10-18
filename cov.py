import numpy as np
from numpy import linalg as LA

A = [45,37,42,35,39]
B = [38,31,26,28,33]
C = [10,15,17,21,12]

A_norm = A/np.linalg.norm(A)
B_norm = B/np.linalg.norm(B)
C_norm = C/np.linalg.norm(C)

data = np.array([A_norm,B_norm,C_norm])

# determine covariance
covMatrix = np.cov(data,bias=True)
print (covMatrix)

# compute eigen vactors and eigenvalues
w, v = LA.eig(covMatrix)
print("eigen vectors")
print(v)

print("eigen values")
print(w)

varianceV = np.empty(3)

# calculate variances
varianceV[0] = w[0]/(w[0]+w[1]+w[2])
varianceV[1] = w[1]/(w[0]+w[1]+w[2])
varianceV[2] = w[2]/(w[0]+w[1]+w[2])


print(f' variance of v1 : {varianceV[0]}')
print(f' variance of v2 : {varianceV[1]}')
print(f' variance of v3 : {varianceV[2]}')

# calculate feature vector
v_initial = 0
featureVector = np.empty(3)
for i in range(0,3):
    if varianceV[i] > v_initial:
        featureVector = v[i]

print(f'feature vector: {featureVector}')
