# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 11:36:23 2022

@author: Lenovo
"""

from extracredit_embedding import ChessDataset, initialize_weights

trainset = ChessDataset(filename='extracredit_train.txt')

print(trainset)