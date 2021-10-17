from numpy.core.defchararray import count
import pandas as pd
import numpy as np
import numpy as np


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def main():
    df = pd.read_csv('A1-dm.csv')
    # entropy_discretization(df['A1'])
    segmentation_by_natural_partitioning(df['A2'])

# This method discretizes attribute A1
# If the information gain is 0, i.e the number of 
# distinct class is 1 or
# If min f/ max f < 0.5 and the number of distinct values is floor(n/2)
# Then that partition stops splitting.
def entropy_discretization(attribute):
    # pick a threshold
    threshold = 6
    
    print(attribute.head())


def segmentation_by_natural_partitioning(attribute):
    print(attribute.head())
    a = np.array(attribute)

    # calculate 5th and 95th percentiles.
    fith_percentile = np.percentile(a, 5)
    nienty_fith_percentile = np.percentile(a, 95) 

    # sort the data.
    sorted_data = np.sort(a)
    n = a.size
    # keep the values from floor(n*0.05) to floor(n*0.95)
    new_a = split(a, (a > np.math.floor(n*fith_percentile)) & (a < np.math.floor(n*nienty_fith_percentile)))
    return attribute

# def calculate_correlation(attributeOne, attributeTwo):

def pca(df):
    # Normalize each attribute

    # build the covariance matrix of the attributes.

    # rank eigenvectors in descending order of their eigendvalues
    # and keep the the significant eigenvectors

    # build the feature vector our of the selected eigenvectors
     

def split(arr, cond):
    return [arr[cond], arr[~cond]]



main()