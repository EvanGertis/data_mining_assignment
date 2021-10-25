import pandas as pd
import numpy as np
from numpy import linalg as LA

def main():
    s = pd.read_csv('A1-dm.csv')
    s = pca(s)

def df_to_array(df):
    A1 = df[['A1']].to_numpy()
    A2 = df[['A2']].to_numpy()
    if 'A3' in df:
        A3 = df[['A3']].to_numpy()

    df_as_matrix = np.array(A1,A2)
    if 'A3' in s:
        df_as_matrix = np.concatenate(df_as_matrix,A3)


    return df_as_matrix

def pca(s):
    # Normalize each s
    A1 = s[['A1']].to_numpy()
    A2 = s[['A2']].to_numpy()
    
    print(A1.ndim)
    if 'A3' in s:
        A3 = s[['A3']].to_numpy()
        A3_norm = A3/np.linalg.norm(A3)

    A1_norm = A1/np.linalg.norm(A1)
    A2_norm = A2/np.linalg.norm(A2)

    data = np.array([A1_norm,A2_norm])
    if 'A3' in s:
        data = np.array([A1_norm,A2_norm,A3_norm]).squeeze()

    # determine covariance
    covMatrix = np.cov(data,bias=True)
    print(covMatrix)

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
    resolved_dataset = np.matmul(np.transpose(featureVector),data)
    print(f'resolved_dataset.ndim = {resolved_dataset.ndim}')
    print(f'dataset = {resolved_dataset}')


main()