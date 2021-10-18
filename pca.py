from numpy.core.defchararray import count
import pandas as pd
import numpy as np
import numpy as np
from math import log2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plot

def main():
    s = pd.read_csv('A1-dm.csv')
    print("*******************************************************")
    print("PCA                                             STARTED")
    s = pca(s)
    print("PCA                                            COMPLETED")
    print(s)
    print("*******************************************************")

def pca(s):
    # Normalize each s
    s_normalized=(s - s.mean()) / s.std()
    pca = PCA(n_components=s.shape[1])
    pca.fit(s_normalized)

    # build the covariance matrix of the s.

    # rank eigenvectors in descending order of their eigenvalues
    # and keep the the significant eigenvectors

    # build the feature vector our of the selected eigenvectors
    
    # Reformat and view results
    loadings = pd.DataFrame(pca.components_.T,
    columns=['PC%s' % _ for _ in range(len(s_normalized.columns))],
    index=s.columns)
    print(loadings)

    plot.plot(pca.explained_variance_ratio_)
    plot.ylabel('Explained Variance')
    plot.xlabel('Components')
    plot.show()

    # TODO: return transformed data.
    s = pca.transformed_data
    return s


main()