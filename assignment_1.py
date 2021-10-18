from numpy.core.defchararray import count
import pandas as pd
import numpy as np
import numpy as np
from math import log2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plot

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def main():
    s = pd.read_csv('A1-dm.csv')
    # print("******************************************************")
    # print("Entropy Discretization                         STARTED")
    # entropy_discretization(s)
    # print("Entropy Discretization                         COMPLETED")
    # print("******************************************************")
    print("Segmentation By Natural Partitioning           STARTED")
    s = segmentation_by_natural_partitioning(s)
    print(s.head())
    print("Applying Segmentation By Natural Partitioning COMPLETED")
    print("*******************************************************")
    # print("Correlation Calculation                         STARTED")
    # calculate_correlation(s)
    # print("*******************************************************")
    # print("Correlation Calculation                       COMPLETED")
    # print("*******************************************************")
    # print("PCA                                             STARTED")
    # pca(s)
    # print("PCA                                            COMPLETED")
    # print("*******************************************************")

# This method discretizes attribute A1
# If the information gain is 0, i.e the number of 
# distinct class is 1 or
# If min f/ max f < 0.5 and the number of distinct values is floor(n/2)
# Then that partition stops splitting.
# This method discretizes s A1
# If the information gain is 0, i.e the number of 
# distinct class is 1 or
# If min f/ max f < 0.5 and the number of distinct values is floor(n/2)
# Then that partition stops splitting.
def entropy_discretization(s):

    I = {}
    i = 0
    while(uniqueValue(s)):
        # Step 1: pick a threshold
        threshold = s['A'].iloc[0]

        # Step 2: Partititon the data set into two parttitions
        s1 = s[s['A'] < threshold]
        print("s1 after spitting")
        print(s1)
        print("******************")
        s2 = s[s['A'] >= threshold]
        print("s2 after spitting")
        print(s2)
        print("******************")
            
        # Step 3: calculate the information gain.
        informationGain = information_gain(s1,s2,s)
        I.update({f'informationGain_{i}':informationGain,f'threshold_{i}': threshold})
        print(f'added informationGain_{i}: {informationGain}, threshold_{i}: {threshold}')
        s = s[s['A'] != threshold]
        i += 1

    # Step 5: calculate the min information gain
    n = int(((len(I)/2)-1))
    print("Calculating minimum threshold")
    print("*****************************")
    minInformationGain = 0
    minThreshold       = 0 
    for i in range(0, n):
        if(I[f'informationGain_{i}'] < minInformationGain):
            minInformationGain = I[f'informationGain_{i}']
            minThreshold       = I[f'threshold_{i}']

    print(f'minThreshold: {minThreshold}, minInformationGain: {minInformationGain}')

    # Step 6: keep the partitions of S based on the value of threshold_i
    minPartition(minInformationGain,minThreshold,s,s1,s2)

def uniqueValue(s):
    # are records in s the same? return true
    if s.nunique()['A'] == 1:
        return False
    # otherwise false 
    else:
        return True

def minPartition(minInformationGain,minThreshold,s,s1,s2):
    print(f'informationGain: {minInformationGain}, threshold: {minThreshold}')
    print("Best Partitions")
    print("***************")
    print(s1)
    print(s2)
    print(s)


def information_gain(s1, s2, s):
    # calculate cardinality for s1
    cardinalityS1 = len(pd.Index(s1['A']).value_counts())
    print(f'The Cardinality of s1 is: {cardinalityS1}')
    # calculate cardinality for s2
    cardinalityS2 = len(pd.Index(s2['A']).value_counts())
    print(f'The Cardinality of s2 is: {cardinalityS2}')
    # calculate cardinality of s
    cardinalityS = len(pd.Index(s['A']).value_counts())
    print(f'The Cardinality of s is: {cardinalityS}')
    # calculate informationGain
    informationGain = (cardinalityS1/cardinalityS) * entropy(s1) + (cardinalityS2/cardinalityS) * entropy(s2)
    print(f'The total informationGain is: {informationGain}')
    return informationGain



def entropy(s):
    print("calculating the entropy for s")
    print("*****************************")
    print(s)
    print("*****************************")

    # initialize ent
    ent = 0

    # calculate the number of classes in s
    numberOfClasses = s['Class'].nunique()
    print(f'Number of classes for dataset: {numberOfClasses}')
    value_counts = s['Class'].value_counts()
    p = []
    for i in range(0,numberOfClasses):
        n = s['Class'].count()
        # calculate the frequency of class_i in S1
        print(f'p{i} {value_counts.iloc[i]}/{n}')
        f = value_counts.iloc[i]
        pi = f/n
        p.append(pi)
    
    print(p)

    for pi in p:
        ent += -pi*log2(pi)

    return ent

def segmentation_by_natural_partitioning(s):
    # calculate 5th and 95th percentiles.
    s_as_array = np.array(s)
    fith_percentile = np.percentile(s_as_array, 5)
    nienty_fith_percentile = np.percentile(s_as_array, 95)
    print("*****************************")
    print(f'fith_percentile {fith_percentile}')
    print(f'nienty_fith_percentile {nienty_fith_percentile}')
    print("*****************************")
   
    # sort the data.
    s['A2'] = s['A2'].sort_values()

    n = s['A2'].count()
    print("*****************************")
    print(f'Total number of records {n}')
    print("*****************************")
    # keep the values from floor(n*0.05) to floor(n*0.95)
    print(s['A2'])
    f1 = np.math.floor(n*fith_percentile)
    f2 = np.math.floor(n*nienty_fith_percentile)
    print("*****************************")
    print(f'floor(n*0.05) {f1}')
    print(f'floor(n*0.95) {f2}')
    print("*****************************")
    s = s[(s['A2'] > f1) and (s['A2'] < f2)]

    return s

def calculate_correlation(s):
    s = s[['A1','A3']]
    correlation = s.corr().iloc[1,0]
    # if correlation > 0.6 or correlation < 0.6 remove A3
    if correlation > 0.6 or correlation < -0.6:
        s = s.drop(['A3'], axis=1)
    
    return s

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

     



main()