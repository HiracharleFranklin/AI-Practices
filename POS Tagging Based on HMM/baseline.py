# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath
"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    
    corpus = {}
    tagfreq = {}
    result = []
    
    for sentence in train:
        for word_tag in sentence:
            word = word_tag[0]
            tag = word_tag[1]
            if word in corpus:
                if tag in corpus[word]:
                    corpus[word][tag] = corpus[word][tag] + 1
                else:
                    corpus[word][tag] = 1
            else:
                corpus[word] = {}
                corpus[word][tag] = 1
            
            if tag in tagfreq:
                tagfreq[tag] = tagfreq[tag] + 1
            else:
                tagfreq[tag] = 1
    
    mostoften = max(tagfreq.items(),key = lambda x:x[1])[0]
    
    for sentence in test:
        element = []
        for word in sentence:
            if word in corpus:
                element.append((word,max(corpus[word].items(),key = lambda x:x[1])[0]))
            else:
                element.append((word, mostoften))
        result.append(element)
               
    return result