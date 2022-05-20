# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 07:55:39 2022

@author: Lenovo
"""
'''
train = [[('a',1),('b',2),('c',3),('d',4)],[('a',1),('b',2),('c',3)]]
tagfreq = {}            # tag frequency
tag_pair = {}       # (tag a, tag b) pairs frequency
word_tag_pair = {}  # (word,tag) pairs frequency
    
for sentence in train:
    word_num = 0
    for word_tag in sentence:
        word_num = word_num + 1
        word = word_tag[0]
        tag = word_tag[1]
            
        # count occurrence of tags
        if tag in tagfreq:
            tagfreq[tag] = tagfreq[tag] + 1
        else:
            tagfreq[tag] = 1
                
        # count occurance of (word,tag) pairs
        if word_tag in word_tag_pair:
            word_tag_pair[word_tag] = word_tag_pair[word_tag] + 1
        else:
            word_tag_pair[word_tag] = 1
                
        # count occurance of (tag a, tag b) pairs
        if word_num == 1:
            tag1 = tag
            continue
        elif word_num == 2:
            tag2 = tag
        else:
            tag1 = tag2
            tag2 = tag
        tags = (tag1,tag2)
        if tags in tag_pair:
            tag_pair[tags] = tag_pair[tags] + 1
        else:
            tag_pair[tags] = 1
print(tagfreq)
print(tag_pair)
print(word_tag_pair)
print(len(tag_pair))
print(max(tag_pair, key=tag_pair.get))
'''

tag_corpus = {1:{"a":1,"b":1,"c":1},2:{"a":2,"b":2,"c":2},3:{"a":1,"b":3,"c":1}}    
tagword = tag_corpus
hapax = {}
for tag_words in tag_corpus.items():
    tag = tag_words[0]
    print(tag)
    words = tag_words[1]
    print(words)
    for word in words:
        print(word)
        if words[word] == 1:
            if tag in hapax:
                hapax[tag] = hapax[tag] + 1
            else:
                hapax[tag] = 1
print(hapax) 
hapax={}
hapax_tot = 0             
for tag, words in tagword.items():
    for word in words.values():
        if word == 1:
            if tag not in hapax:
                hapax[tag] = 1
                hapax_tot += 1
            else:
                hapax[tag] += 1
                hapax_tot += 1            
print(hapax) 