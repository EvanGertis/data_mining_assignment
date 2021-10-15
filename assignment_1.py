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
    print(attribute.head())


def segmentation_by_natural_partitioning(attribute):
    print(attribute.head())
    a = np.array(attribute)
    fith_percentile = np.percentile(a, 5) # return 5th percentile.
    nienty_fith_percentile = np.percentile(a, 95) # return 95th percentile.
    sorted_data = np.sort(a)
    n = a.size
    print(n)
    a = a.flatten()
    a.shape
    print(split(a, a > np.math.floor(a*fith_percentile)))
    # print(split(a, a < np.math.floor(a*nienty_fith_percentile)))
    
    # print(sorted_data)

def split(arr, cond):
    return [arr[cond], arr[~cond]]



main()