# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:45:54 2022

@author: Lenovo
"""

import sys
import argparse
import configparser
import copy
import numpy as np

import reader
import naive_bayes as nb
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS440 MP3 Naive Bayes')
    parser.add_argument('--bigram',dest="bigram", type=bool,default=False)
    parser.add_argument('--training', dest='training_dir', type=str, default = '../data/spam_data/train',
                        help='the directory of the training data')
    parser.add_argument('--development', dest='development_dir', type=str, default = '../data/spam_data/dev',
                        help='the directory of the development data')

    # When doing final testing, reset the default values below to match your settings in naive_bayes.py
    parser.add_argument('--stemming',dest="stemming", type=bool, default=False,
                        help='Use porter stemmer')
    parser.add_argument('--lowercase',dest="lowercase", type=bool, default=False,
                        help='Convert all word to lower case')
    parser.add_argument('--laplace',dest="laplace", type=float, default = 0.001,
                        help='Laplace smoothing parameter')
    parser.add_argument('--bigram_laplace',dest="bigram_laplace", type=float, default = 0.005,
                        help='Laplace smoothing parameter for bigrams')
    parser.add_argument('--bigram_lambda',dest="bigram_lambda", type=float, default = 0.5,
                        help='Weight on bigrams vs. unigrams')
    parser.add_argument('--pos_prior',dest="pos_prior", type=float, default = 0.8,
                        help='Positive prior, i.e. percentage of test examples that are positive')

    args = parser.parse_args()
    train_set, train_labels, dev_set, dev_labels = nb.load_data(args.training_dir,args.development_dir,args.stemming,args.lowercase)
    
#pos_uni_student,neg_uni_student = nb.create_word_maps_bi(train_set, train_labels,None)
#print(pos_uni_student)
#print(neg_uni_student)
    prior = 0.0
    total = 0.0
    for i in train_labels:
        total = total + 1
        if (i==1):
            prior = prior + 1
    pos_prior = prior/total
    print(pos_prior)    
    #print_paramter_vals(laplace,pos_prior)

