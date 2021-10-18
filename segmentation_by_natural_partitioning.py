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
    print("Segmentation By Natural Partitioning           STARTED")
    s = segmentation_by_natural_partitioning(s)
    print("Applying Segmentation By Natural Partitioning COMPLETED")
   

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
    print("******************************")
    print("Dataset attribute A2")
    print(s['A2'])
    print("******************************")
    f1 = fith_percentile # np.math.floor(n*0.05)
    f2 = nienty_fith_percentile # np.math.floor(n*0.95)
    print("*****************************")
    print(f'fith_percentile {f1}')
    print(f'nienty_fith_percentile {f2}')
    print("*****************************")
    s = s[s['A2'] > f1]
    s = s[s['A2'] < f2]

    return s

main()