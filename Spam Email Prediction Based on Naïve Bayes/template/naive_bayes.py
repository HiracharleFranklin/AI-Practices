# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader

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
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset_main(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def create_word_maps_uni(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: words 
        values: number of times the word appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word appears 
    """
    print(len(X),'X')
    pos_vocab = {}
    neg_vocab = {}
    ##TODO:
    #raise RuntimeError("Replace this line with your code!")
    # loop for the first time to create the categories
    counter = 0
    for ls in X:
        if (y[counter] == 1):
            for pos_word in ls:
                pos_vocab[pos_word] = 0
        else:
            for neg_word in ls:
                neg_vocab[neg_word] = 0
        counter = counter + 1
    
    # loop for the second time to calculate the numbers
    counter = 0
    for ls in X:
        if (y[counter] == 1):
            for pos_word in ls:
                pos_vocab[pos_word] = pos_vocab[pos_word] + 1
        else:
            for neg_word in ls:
                neg_vocab[neg_word] = neg_vocab[neg_word] + 1
        counter = counter + 1
            
    return dict(pos_vocab), dict(neg_vocab)


def create_word_maps_bi(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: pairs of words
        values: number of times the word pair appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word pair appears 
    """
    #print(len(X),'X')
    pos_vocab = {}
    neg_vocab = {}
    ##TODO:
    #raise RuntimeError("Replace this line with your code!")
        # loop for the first time to create the categories
    counter = 0
    for ls in X:
        if (y[counter] == 1):
            for i in range(len(ls)-1):
                word = ls[i] + ' ' + ls[i+1]
                pos_vocab[word] = 0
        else:
            for i in range(len(ls)-1):
                word = ls[i] + ' ' + ls[i+1]
                neg_vocab[word] = 0
        counter = counter + 1
        
    counter = 0
    for ls in X:
        if (y[counter] == 1):
            for pos_word in ls:
                pos_vocab[pos_word] = 0
        else:
            for neg_word in ls:
                neg_vocab[neg_word] = 0
        counter = counter + 1
    
    # loop for the second time to calculate the numbers
    counter = 0
    for ls in X:
        if (y[counter] == 1):
            for i in range(len(ls)-1):
                word = ls[i] + ' ' + ls[i+1]
                pos_vocab[word] = pos_vocab[word] + 1
        else:
            for i in range(len(ls)-1):
                word = ls[i] + ' ' + ls[i+1]
                neg_vocab[word] = neg_vocab[word] + 1
        counter = counter + 1
    
    counter = 0
    for ls in X:
        if (y[counter] == 1):
            for pos_word in ls:
                pos_vocab[pos_word] = pos_vocab[pos_word] + 1
        else:
            for neg_word in ls:
                neg_vocab[neg_word] = neg_vocab[neg_word] + 1
        counter = counter + 1
        
    return dict(pos_vocab), dict(neg_vocab)



# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=0.001, pos_prior=0.8, silently=False):
    '''
    Compute a naive Bayes unigram model from a training set; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    # Keep this in the provided template
    print_paramter_vals(laplace,pos_prior)

    #raise RuntimeError("Replace this line with your code!")
    
    # The result to output
    dev_labels = [] 
    # First, compute P(X=x|Y=ham) and P(X=x|Y=spam) and put them into a dictionary
    pos_vocab,neg_vocab = create_word_maps_uni(train_set, train_labels, None)
    pos_prob = {}
    neg_prob = {}
    # Calculate basic parameters
    N_pos = 0
    N_neg = 0
    kind_pos = 0
    kind_neg = 0
    for count in pos_vocab.values():
        N_pos = N_pos + count
        kind_pos = kind_pos + 1
    for count in neg_vocab.values():
        N_neg = N_neg + count
        kind_neg = kind_neg + 1
    # Calculate conditional probabilities
    for word in pos_vocab:
        pos_prob[word] = (pos_vocab[word] + laplace)/(N_pos + laplace*(1 + kind_pos))
    for word in neg_vocab:
        neg_prob[word] = (neg_vocab[word] + laplace)/(N_neg + laplace*(1 + kind_neg))
    # Calculate the posterior probabilities
    for email in dev_set:
        p_pos = math.log(pos_prior)
        p_neg = math.log(1-pos_prior)
        for word in email:
            if word in pos_prob.keys():
                p_pos = p_pos + math.log(pos_prob[word])
            else: 
                p_pos = p_pos + math.log((0 + laplace)/(N_pos + laplace*(1 + kind_pos)))
            if word in neg_prob.keys():
                p_neg = p_neg + math.log(neg_prob[word])
            else:
                p_neg = p_neg + math.log((0 + laplace)/(N_neg + laplace*(1 + kind_neg)))
        if p_pos > p_neg:
            dev_labels.append(1)
        else:
            dev_labels.append(0)
    return dev_labels


# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.001, bigram_laplace=0.005, bigram_lambda=0.5,pos_prior=0.8,silently=False):
    '''
    Compute a unigram+bigram naive Bayes model; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    unigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    bigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating bigram probs
    bigram_lambda (scalar float) = interpolation weight for the bigram model
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    max_vocab_size = None

    #raise RuntimeError("Replace this line with your code!")
    # The result to output
    dev_labels = [] 
    
    # First, compute P(X=x|Y=ham) and P(X=x|Y=spam) and put them into a dictionary
    pos_vocab,neg_vocab = create_word_maps_uni(train_set, train_labels, None)
    pos_bivocab, neg_bivocab = create_word_maps_bi(train_set, train_labels, None)
    pos_prob = {}
    neg_prob = {}
    pos_biprob = {}
    neg_biprob = {}
    
    # Calculate basic parameters
    N_pos = 0
    N_bipos = 0
    N_neg = 0
    N_bineg = 0
    kind_pos = 0
    kind_bipos = 0
    kind_neg = 0
    kind_bineg = 0
    for count in pos_vocab.values():
        N_pos = N_pos + count
        kind_pos = kind_pos + 1
    for count in neg_vocab.values():
        N_neg = N_neg + count
        kind_neg = kind_neg + 1
        
    for count in pos_bivocab.values():
        N_bipos = N_bipos + count
        kind_bipos = kind_bipos + 1
    for count in neg_bivocab.values():
        N_bineg = N_bineg + count
        kind_bineg = kind_bineg + 1 

    # Calculate conditional probabilities
    for word in pos_vocab:
        pos_prob[word] = (pos_vocab[word] + unigram_laplace)/(N_pos + unigram_laplace*(1 + kind_pos))
    for word in neg_vocab:
        neg_prob[word] = (neg_vocab[word] + unigram_laplace)/(N_neg + unigram_laplace*(1 + kind_neg))
    for word in pos_bivocab:
        pos_biprob[word] = (pos_bivocab[word] + bigram_laplace)/(N_bipos + bigram_laplace*(1 + kind_bipos))
    for word in neg_bivocab:
        neg_biprob[word] = (neg_bivocab[word] + bigram_laplace)/(N_bineg + bigram_laplace*(1 + kind_bineg))
        
    # Calculate the posterior probabilities
    for email in dev_set:
        p_pos = math.log(pos_prior/1)
        p_neg = math.log((1-pos_prior)/1)
        # unigram case
        for word in email:
            if word in pos_prob.keys():
                p_pos = p_pos + (1-bigram_lambda)*math.log(pos_prob[word])
            else: 
                p_pos = p_pos + (1-bigram_lambda)*math.log((0 + unigram_laplace)/(N_pos + unigram_laplace*(1 + kind_pos)))
            if word in neg_prob.keys():
                p_neg = p_neg + (1-bigram_lambda)*math.log(neg_prob[word])
            else:
                p_neg = p_neg + (1-bigram_lambda)*math.log((0 + unigram_laplace)/(N_neg + unigram_laplace*(1 + kind_neg)))
                
         #bigram case
        for i in range(len(email)-1):
            word = email[i]+' '+email[i+1]
            if word in pos_biprob.keys():
                p_pos = p_pos + bigram_lambda*math.log(pos_biprob[word])
            else:
                p_pos = p_pos + bigram_lambda*math.log((0 + bigram_laplace)/(N_bipos + bigram_laplace*(1 + kind_bipos)))
            if word in neg_biprob.keys():
                p_neg = p_neg + bigram_lambda*math.log(neg_biprob[word])
            else:
                p_neg = p_neg + bigram_lambda*math.log((0 + bigram_laplace)/(N_bineg + bigram_laplace*(1 + kind_bineg)))
                
        if p_pos > p_neg:
            dev_labels.append(1)
        else:
            dev_labels.append(0)

    return dev_labels
