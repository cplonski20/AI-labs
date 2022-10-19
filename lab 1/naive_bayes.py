# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import math
from tqdm import tqdm
from collections import Counter
import reader
import numpy as np
"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


"""
load_data calls the provided utility to load in the dataset.
You can modify the default values for stemming and lowercase, to improve performance when
    we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace= .0058, pos_prior=0.5 ,silently=False):
    print_paramter_vals(laplace,pos_prior)
    print()
    yhats = []
    
    logpriors = np.zeros((2))
    logpriors[0] = 1-pos_prior
    logpriors[1] = pos_prior
    logpriors = np.log(logpriors)
    # print(train_set)
    # train_labels = np.array(train_labels)
    # train_set = np.array([np.array(xi) for xi in train_set])
    freqPos = {}
    freqNeg = {}
    for a in range(len(train_labels)):
        if(train_labels[a] == 1):
            for b in train_set[a]:
                if(b in freqPos):
                    freqPos[b] += 1
                else:
                    freqPos[b] = 1
        else:
            for b in train_set[a]:
                if(b in freqNeg):
                    freqNeg[b] +=1
                else:
                    freqNeg[b] = 1
    
    posbase = (sum(freqPos.values())+ laplace *(len(freqPos)+ 1))
    negbase = (sum(freqNeg.values())+ laplace *(len(freqNeg)+ 1))
    posUnk = np.log(laplace/posbase)
    negUnk = np.log(laplace/negbase)
    for a in freqPos:
        freqPos[a] = np.log((freqPos[a] + laplace)/posbase)
    for a in freqNeg:
        freqNeg[a] = np.log((freqNeg[a] + laplace)/negbase)


    for doc in tqdm(dev_set,disable=silently):
        # yhats.append(-1)
        probpos = logpriors[1]
        probneg = logpriors[0]
        for word in doc:
            if(word in freqPos):
                probpos += freqPos[word]
            else:
                probpos += posUnk
            if(word in freqNeg):
                probneg += freqNeg[word]
            else:
                probneg += negUnk
        if(probpos > probneg):
            yhats.append(1)
        else:
            yhats.append(0)
    return yhats





def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

# main function for the bigrammixture model
def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.15, bigram_laplace=.0001, bigram_lambda=.13,pos_prior=0.767, silently=False):
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    yhats = []
    logpriors = np.zeros((2))
    logpriors[0] = 1-pos_prior
    logpriors[1] = pos_prior
    logpriors = np.log(logpriors)

    freqPos = {}
    freqNeg = {}
    for a in range(len(train_labels)):
        if(train_labels[a] == 1):
            for b in train_set[a]:
                if(b in freqPos):
                    freqPos[b] += 1
                else:
                    freqPos[b] = 1
        else:
            for b in train_set[a]:
                if(b in freqNeg):
                    freqNeg[b] +=1
                else:
                    freqNeg[b] = 1
    
    bifreqPos = {}
    bifreqNeg = {}
    for a in range(len(train_labels)):
        if(train_labels[a] == 1):
            for b in range(1,len(train_set[a])):
                bi = (train_set[a][b-1], train_set[a][b])
                if(bi in bifreqPos):
                    bifreqPos[bi] += 1
                else:
                    bifreqPos[bi] = 1
        else:
            for b in range(1,len(train_set[a])):
                bi = (train_set[a][b-1], train_set[a][b])
                if(bi in bifreqNeg):
                    bifreqNeg[bi] += 1
                else:
                    bifreqNeg[bi] = 1
    
    
    posbase = (sum(freqPos.values())+ unigram_laplace *(len(freqPos)+ 1))
    negbase = (sum(freqNeg.values())+ unigram_laplace *(len(freqNeg)+ 1))
    posUnk = np.log(unigram_laplace/posbase)
    negUnk = np.log(unigram_laplace/negbase)
    for a in freqPos:
        freqPos[a] = np.log((freqPos[a] + unigram_laplace)/posbase)
    for a in freqNeg:
        freqNeg[a] = np.log((freqNeg[a] + unigram_laplace)/negbase)


    biposbase = (sum(bifreqPos.values())+ bigram_laplace *(len(bifreqPos)+ 1))
    binegbase = (sum(bifreqNeg.values())+ bigram_laplace *(len(bifreqNeg)+ 1))
    biposUnk = np.log(bigram_laplace/biposbase)
    binegUnk = np.log(bigram_laplace/binegbase)
    for a in bifreqPos:
        bifreqPos[a] = np.log((bifreqPos[a] + bigram_laplace)/biposbase)
    for a in bifreqNeg:
        bifreqNeg[a] = np.log((bifreqNeg[a] + bigram_laplace)/binegbase)


    for doc in tqdm(dev_set,disable=silently):
        # yhats.append(-1)
        uniprobpos = logpriors[1]
        uniprobneg = logpriors[0]
        biprobpos = logpriors[1]
        biprobneg = logpriors[0]
        biword = ['null', doc[0]]
        first = 1
        for word in doc:
            if(word in freqPos):
                uniprobpos += freqPos[word]
            else:
                uniprobpos += posUnk
            if(word in freqNeg):
                uniprobneg += freqNeg[word]
            else:
                uniprobneg += negUnk
        
            biword[0] = biword[1]
            biword[1] = word
            biwordtuple = tuple(biword)

            if first == 1:
                first = 0
                continue
            if(biwordtuple in bifreqPos):
                biprobpos += bifreqPos[biwordtuple]
            else:
                biprobpos += biposUnk
            if(biwordtuple in bifreqNeg):
                biprobneg += bifreqNeg[biwordtuple]
            else:
                biprobneg += binegUnk
        
        
        totalpos = (1- bigram_lambda) *uniprobpos + bigram_lambda * biprobpos
        totalneg = (1- bigram_lambda) *uniprobneg + bigram_lambda *biprobneg

        if(totalpos > totalneg):
            yhats.append(1)
        else:
            yhats.append(0)
    return yhats
