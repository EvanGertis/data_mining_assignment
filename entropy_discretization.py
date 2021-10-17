import pandas as pd
import numpy as np
import entropy_based_binning as ebb
from math import log2

def main():
    df = pd.read_csv('A1-dm.csv')
    s = df
    s = entropy_discretization(s)

# This method discretizes s A1
# If the information gain is 0, i.e the number of 
# distinct class is 1 or
# If min f/ max f < 0.5 and the number of distinct values is floor(n/2)
# Then that partition stops splitting.
def entropy_discretization(s):

    informationGain = {}

    while(uniqueValue(s)):
        # Step 1: pick a threshold
        threshold = 6

        # Step 2: Partititon the data set into two parttitions
        s1 = split(s , (s > threshold))
        s2 = split(s , (s < threshold))
        
        # Step 3: calculate the information gain.
        informationGain = information_gain(s1,s2)

    # Step 5: calculate the max information gain
    minInformationGain = min(informationGain)

    # Step 6: keep the partitions of S based on the value of threshold_i
    s = worstPartition(minInformationGain, s)

def uniqueValue(s):
    # are records in s the same? return true

    # otherwise false 

def worstPartition(maxInformationGain):
    # determine be threshold_i
    threshold_i = 6

    return 


def information_gain(s1, s2, s):
    # calculate cardinality for s1
    cardinalityS1 = 2

    # calculate cardinality for s2
    cardinalityS2 = 10

    # calculate cardinality of s
    cardinalityS = 10

    # calculate informationGain
    informationGain = (cardinalityS1/cardinalityS) * entropy(s1) + (cardinalityS2/cardinalityS) * entropy(s2)

    return informationGain



def entropy(s):
    # calculate the number of classes in s
    numberOfClasses = 2

    # TODO calculate pi for each class.
    # calculate the frequency of class_i in S1
    p1 = 2/4
    p2 = 3/4
    ent = -(p1*log2(p2)) - (p2*log2(p2))

    return ent 

def split(arr, cond):
    return [arr[cond], arr[~cond]]
main()