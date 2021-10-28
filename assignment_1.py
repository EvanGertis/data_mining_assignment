from numpy.core.defchararray import count
import pandas as pd
import numpy as np
import numpy as np
from math import ceil, floor, log2
from sklearn.decomposition import PCA
from numpy import linalg as LA

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def main():
    s = pd.read_csv('A1-dm.csv')
    print("******************************************************")
    print("Entropy Discretization                         STARTED")
    s = entropy_discretization(s)
    print("Entropy Discretization                         COMPLETED")
    print(s)
    print("******************************************************")
    print("Segmentation By Natural Partitioning           STARTED")
    s = segmentation_by_natural_partitioning(s)
    # print("Applying Segmentation By Natural Partitioning COMPLETED")
    # print(s)
    # print("*******************************************************")
    # print("Correlation Calculation                         STARTED")
    # s = calculate_correlation(s)
    # print("*******************************************************")
    # print("Correlation Calculation                       COMPLETED")
    # print(s)
    # print("*******************************************************")
    # print("PCA                                             STARTED")
    # s = pca(s)
    # print("PCA                                            COMPLETED")
    # print(s)
    # print("*******************************************************")
    # print("Writing to csv                                  STARTED")
    # compression_opts = dict(method='zip',archive_name='out.csv') 
    # s.to_csv('out.zip', index=False,compression=compression_opts) 
    # print("Writing to csv                                COMPLETED")
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
    n = s.nunique()['Class']
    s1 = pd.DataFrame()
    s2 = pd.DataFrame()
    s3 = pd.DataFrame()
    distinct_values = s['A1'].value_counts().index
    information_gain_indicies = []
    partitions = []
    print(f'The unique values for dataset s["A1"] are {distinct_values}')
    for i in distinct_values:

        # Step 1: pick a threshold
        threshold = i
        print(f'Using threshold {threshold}')

        # Step 2: Partititon the data set into two parttitions
        s1 = s[s['Class'] < threshold]
        print("s1 after spitting")
        print(s1)
        print("******************")
        s2 = s[s['Class'] >= threshold]
        print("s2 after spitting")
        print(s2)
        print("******************")

        print("******************")
        print("calculating maxf")
        print(f" maxf {maxf(s['Class'])}")
        print("******************")

        print("******************")
        print("calculating minf")
        print(f" maxf {minf(s['Class'])}")
        print("******************")

        # Step 3: calculate the information gain.
        informationGain = information_gain(s1,s2,s)
        I.update({f'informationGain_{i}':informationGain,f'threshold_{i}': threshold})
        print(f'added informationGain_{i}: {informationGain}, threshold_{i}: {threshold}')
        information_gain_indicies.append(i)

        print(f"Checking condition a if {s1.nunique()['Class']} == {1}")
        if (s1.nunique()['Class'] == 1):
            if (s1.nunique()['Class'] == 1 and s2.nunique()['Class'] == 1):
                print("The number of distinct classes in s1 are 1.")
                print(f"Checking the condition {s1.nunique()['Class']} == {1} and {s2.nunique()['Class']} == {1}")
                break
            s3 = entropy_discretization(s2)
            break


        print(f"Checking condition b  {maxf(s1['Class'])}/{minf(s1['Class'])} < {0.5} {s1.nunique()['Class']} == {floor(n/2)}")
        if (maxf(s1['Class'])/minf(s1['Class']) < 0.5) and (s1.nunique()['Class'] == floor(n/2)):
            print(f"Condition b is met{maxf(s1['Class'])}/{minf(s1['Class'])} < {0.5} {s1.nunique()['Class']} == {floor(n/2)}")
            break


    print("Elements in I")
    print(I)
    print("*****************************")

    # Step 5: calculate the min information gain
    n = int(((len(I)/2)-1))
    print("Calculating maximum threshold")
    print("*****************************")
    maxInformationGain = 0
    maxThreshold       = 0 
    for i in information_gain_indicies:
        print(f"if({I[f'informationGain_{i}']} > {maxInformationGain})")
        if(I[f'informationGain_{i}'] > maxInformationGain):
            maxInformationGain = I[f'informationGain_{i}']
            maxThreshold       = I[f'threshold_{i}']

    print(f'maxThreshold: {maxThreshold}, maxInformationGain: {maxInformationGain}')

    # replace values
    # print(f" {s1['A1'].value_counts().index}")
    # for i in s1['A1'].value_counts().index:
    #     print(f"s1['A1'].replace({i},1)")
    #     s1['A1'] = s1['A1'].replace(i,1)

    # print(f" {s2['A1'].value_counts().index}")
    # for i in s2['A1'].value_counts().index:
    #     print(f"s2['A1'].replace({i},2)")
    #     s2['A1'] = s2['A1'].replace(i,2)

    # print(f" {s3['A1'].value_counts().index}")
    # for i in s3['A1'].value_counts().index:
    #     print(f"s3['A1'].replace({i},3)")
    #     s3['A1'] = s3['A1'].replace(i,3)

    print("s1 after replacing values")
    print(s1)
    print("******************")
    print("s2 after replacing values")
    print(s2)
    print("******************")
    print("s3 after replacing values")
    print(s3)
    print("******************")
    

    partitions = [s1,s2,s3]
    print("*************** print partitions *****************")
    print(partitions)
    print("*************** end partitions *****************")
    s = pd.concat(partitions)

    # Step 6: keep the partitions of S based on the value of threshold_i
    return s #maxPartition(maxInformationGain,maxThreshold,s,s1,s2)
    

def maxf(s):
    return s.max()

def minf(s):
    return s.min()

def uniqueValue(s):
    # are records in s the same? return true
    if s.nunique()['Class'] == 1:
        return False
    # otherwise false 
    else:
        return True

def maxPartition(maxInformationGain,maxThreshold,s,s1,s2):
    print(f'informationGain: {maxInformationGain}, threshold: {maxThreshold}')
    merged_partitions =  pd.merge(s1,s2)
    merged_partitions =  pd.merge(merged_partitions,s)
    print("Best Partition")
    print("***************")
    print(merged_partitions)
    print("***************")
    return merged_partitions




def information_gain(s1, s2, s):
    # calculate cardinality for s1
    cardinalityS1 = len(pd.Index(s1['Class']).value_counts())
    print(f'The Cardinality of s1 is: {cardinalityS1}')
    # calculate cardinality for s2
    cardinalityS2 = len(pd.Index(s2['Class']).value_counts())
    print(f'The Cardinality of s2 is: {cardinalityS2}')
    # calculate cardinality of s
    cardinalityS = len(pd.Index(s['Class']).value_counts())
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
    print(f"range [{s['A2'].max()},{s['A2'].min()}]")
    print()
    print("*****************************")
    print(f'fith_percentile {fith_percentile}')
    print(f'nienty_fith_percentile {nienty_fith_percentile}')
    print("*****************************")
   
    # sort the data.
    print(s['A2'])
    s['A2'] = s['A2'].sort_values()

    # n = s['A2'].count()
    # print("*****************************")
    # print(f'Total number of records {n}')
    # print("*****************************")
    # # keep the values from floor(n*0.05) to floor(n*0.95)
    # print("******************************")
    # print("Dataset attribute A2")
    # print(s['A2'])
    # print("******************************")
    # f1 = fith_percentile # np.math.floor(n*0.05)
    # f2 = nienty_fith_percentile # np.math.floor(n*0.95)
    # print("*****************************")
    # print(f'fith_percentile {f1}')
    # print(f'nienty_fith_percentile {f2}')
    # print("*****************************")
    # s = s[s['A2'] > f1]
    # s = s[s['A2'] < f2]

    # print(f"range after cleaning [{s['A2'].max()},{s['A2'].min()}]")

    # maximum = ceil(s['A2'].max())
    # minimum = floor(s['A2'].min())

    # print(f"[maximum,minimum] [{minimum},{maximum}]")
    # print(f"subtract {most_significant_while_floordiv(maximum)} - {most_significant_while_floordiv(minimum)}")
    # numberOfGaps =  most_significant_while_floordiv(maximum) - most_significant_while_floordiv(minimum)
    # print(f"numberOfGaps {numberOfGaps}")
    # print(f"The number of values that cover the range are [{minimum},{maximum}]")

    # return s

def most_significant_while_floordiv(i):
    while i >= 10:
        i //= 10
    return i

def calculate_correlation(s):
    s_temp = s[['A1','A3']]
    correlation = s_temp.corr().iloc[1,0]
    print("******************************")
    print(f'Correlation between A1 & A3: {correlation}')
    print("******************************")
    # if correlation > 0.6 or correlation < 0.6 remove A3
    if correlation > 0.6 or correlation < -0.6:
        s = s.drop(['A3'], axis=1)
        print(f'A3 was removed {correlation} > 0.6 or {correlation} < -0.6')
        print("******************************")
    
    return s

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
    df = pd.DataFrame(resolved_dataset, columns = ['data'])
    return df


main()