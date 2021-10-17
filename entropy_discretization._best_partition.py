import pandas as pd
import numpy as np
import entropy_based_binning as ebb
from math import log2
from random import randrange, uniform

def main():
    df = pd.read_csv('S1.csv')
    s = df
    s = entropy_discretization(s)

# This method discretizes s A1
# If the information gain is 0, i.e the number of 
# distinct class is 1 or
# If min f/ max f < 0.5 and the number of distinct values is floor(n/2)
# Then that partition stops splitting.
def entropy_discretization(s):

    I = {}
    while(uniqueValue(s)):
        # Step 1: pick a threshold
        maxValue = int(np.ceil(s['A'].max()))
        minValue = int(np.floor(s['A'].min()))
        print(f'min {minValue}')
        print(f'max {maxValue}')
        
        threshold = randrange(minValue,maxValue)

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
        print(f'Calculated information gain {informationGain}')

        I.update({'informationGain':informationGain,'threshold':threshold})
        print(I)

    # Step 5: calculate the max information gain
    maxInformationGain = np.amax(informationGain)
    print(f'Calculated maximum information gain {maxInformationGain}')


    # Step 6: keep the partitions of S based on the value of threshold_i
    s = bestPartition(minInformationGain, s)

def uniqueValue(s):
    # are records in s the same? return true
    if s.nunique()['A'] == 1:
        return False
    # otherwise false 
    else:
        return True

def bestPartition(maxInformationGain):
    # determine be threshold_i
    threshold_i = 6

    return 


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

main()