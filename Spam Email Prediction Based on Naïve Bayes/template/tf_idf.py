# tf_idf_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020
# Modified by Kiran Ramnath 02/13/2021
# Modified by Mohit Goyal (mohit@illinois.edu) on 01/16/2022
"""
This is the main entry point for the Extra Credit Part of this MP. You should only modify code
within this file for the Extra Credit Part -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import math
from collections import Counter, defaultdict
import time
import operator


# def compute_tf_idf(train_set, train_labels, dev_set):
#     """
#     train_set - List of list of words corresponding with each movie review
#     example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
#     Then train_set := [['like','this','movie'], ['i','fall','asleep']]

#     train_labels - List of labels corresponding with train_set
#     example: Suppose I had two reviews, first one was positive and second one was negative.
#     Then train_labels := [1, 0]

#     dev_set - List of list of words corresponding with each review that we are testing on
#               It follows the same format as train_set

#     Return: A list containing words with the highest tf-idf value from the dev_set documents
#             Returned list should have same size as dev_set (one word from each dev_set document)
#     """



#     # TODO: Write your code here
    


#     # return list of words (should return a list, not numpy array or similar)
#     return []

def compute_tf_idf(train_set, train_labels, dev_set):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    Return: A list containing words with the highest tf-idf value from the dev_set documents
            Returned list should have same size as dev_set (one word from each dev_set document)
    """


    # TODO: Write your code here
    result=[]
    dev_dic =[]
    train_dic =[]
    
    for doc in dev_set:
        dic = {}
        
        for word in doc:
            dic[word]=0
        for word in doc:
            dic[word]= dic[word] + 1
        
        dev_dic.append(dic)
        
    for doc in train_set:
        dic = {}
        
        for word in doc:
            dic[word]=0
        for word in doc:
            dic[word]= dic[word] + 1
        
        train_dic.append(dic)
    
    for dic in dev_dic:
        word_in_doc = 0.0
        tot_in_doc = 0.0
        tot_doc = 0.0
        doc_with_word = 0.0
        subdic = {}
        for word in dic.keys():
            word_in_doc = dic[word]
            for count in dic.values():
                tot_in_doc = tot_in_doc + count
            for dictionary in train_dic:
                tot_doc = tot_doc + 1
                if word in dictionary.keys():
                   doc_with_word = doc_with_word + 1
            subdic[word] = word_in_doc/tot_in_doc*math.log(tot_doc/(1+doc_with_word))
        number = 0.0
        toptf = ''
        for compare in subdic.keys():
            if subdic[compare]>number:
                toptf = compare
                number = subdic[compare]
        result.append(toptf)


    # return list of words (should return a list, not numpy array or similar)
    return result