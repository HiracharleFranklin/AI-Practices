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
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
Extra Credit: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""
import math

def viterbi_ec(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    
    # Count occurrences of tags, tag pairs, tag/word pairs
    corpus = {}
    tag_corpus = {}
    tagfreq = {}        # tag frequency
    tag_pair = {}       # (tag a, tag b) pairs frequency
    word_tag_pair = {}  # (word,tag) pairs frequency
    
    # Count occurrences of hapax words 
    hapax = {}
    hapax_corpus = {}
    
    # three probabilities
    initial = {}        # initial probability
    transition = {}     # transition probability
    emission = {}       # emission probability
    hapax_emission = {} # scaled emission probability due to hapax
    
    # prefix & suffix
    prefixes = ["dis","in","im","ir","il","un","non","mis","de","anti","counter","pre","ante","fore","pro","ex","post","sub","re","inter","extra","hyper","super","auto","tele","ab","para","mono","uni","bi","di","tri","multi","poly","syn","en","em"]
    suffixes = ["ly","ful","ty","ity","ing","ed","s","es","tion","sion","tive","able","ible","er","ee","en","est","tory","age","al","ality","ism","ant","ance","ancy","ary","ess","fic","fy","hood","ian","ic","ian","ish","ist","less","ness"]

    
    # result sentence with tag
    result = []
    
    for sentence in train:
        #count occurences of tag in start of a sentence
        start = sentence[1][1]
        if start not in initial:
            initial[start] = 1
        else:
            initial[start] = initial[start] + 1
            
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
                
            # count the occurance of words
            if word in corpus:
                if tag in corpus[word]:
                    corpus[word][tag] = corpus[word][tag] + 1
                else:
                    corpus[word][tag] = 1
            else:
                corpus[word] = {}
                corpus[word][tag] = 1
                
            # count the occurance of tag : words
            if tag in tag_corpus:
                if word in tag_corpus[tag]:
                    tag_corpus[tag][word] = tag_corpus[tag][word] + 1
                else:
                    tag_corpus[tag][word] = 1
            else:
                tag_corpus[tag] = {}
                tag_corpus[tag][word] = 1
                
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
                
    # count the occurance of hapax words
    N_hapax = 0 # no. of unique hapax word
    for tag_words in tag_corpus.items():
        tag = tag_words[0]
        words = tag_words[1]
        for word in words:
            if words[word] == 1:
                N_hapax = N_hapax + 1
                if tag in hapax:
                    hapax[tag] = hapax[tag] + 1
                else:
                    hapax[tag] = 1
                hapax_corpus[word] = tag
    
    # count the occurance with prefixes and suffixes
    for word in hapax_corpus:
        for suffix in suffixes:
            tag = "X-" + suffix
            if word.endswith(suffix):
                if hapax_corpus[word] != "X":
                    if tag not in tag_corpus[hapax_corpus[word]]:
                        tag_corpus[hapax_corpus[word]][tag] = 1
                    else:
                        tag_corpus[hapax_corpus[word]][tag] += 1
        for prefix in prefixes:
            tag = prefix + "-X"
            if word.startswith(prefix):
                if hapax_corpus[word] != "X":
                    if tag not in tag_corpus[hapax_corpus[word]]:
                        tag_corpus[hapax_corpus[word]][tag] = 1
                    else:
                        tag_corpus[hapax_corpus[word]][tag] += 1

            
    # Compute smoothed probabilities
    # and Take the log of each probability
    laplace = 0.00001
    scaled_laplace = laplace/N_hapax
    
    #The Initial probabilities (How often does each tag occur at the start of a sentence?)
    for start in initial:       
        initial[start] = math.log(initial[start] / len(train))
    
    # The Transition probabilities (How often does tag tb follow tag ta?)
    for tag2 in tagfreq:
        for tag1 in tagfreq:
            # calculate how often the first tag appears
            N_tag = 0
            for tags in tag_pair:
                if tags[0] == tag1:
                    N_tag = N_tag + 1
            # Apply laplace smoothing
            if (tag1, tag2) in tag_pair:
                transition[(tag1, tag2)] = math.log((tag_pair[(tag1, tag2)] + laplace) / (tagfreq[tag2] + laplace*(N_tag + 1)))
            else:
                transition[(tag1, tag2)] = math.log(laplace / (tagfreq[tag2] + laplace*(N_tag + 1)))
                
    # The Emission probabilities (How often does tag t yield word w?)           
    for sentence in train:
        for word_tag in sentence:
            word = word_tag[0]
            tag = word_tag[1]
            # we only care about the first occurance
            if word_tag not in emission:
                N = len(tag_corpus[tag])
                if tag in hapax:
                    emission[word_tag] = math.log((tag_corpus[tag][word] + scaled_laplace*hapax[tag]) / (tagfreq[tag] + scaled_laplace*hapax[tag]*(N+1)))
                else:
                    emission[word_tag] = math.log((tag_corpus[tag][word] + scaled_laplace) / (tagfreq[tag] + scaled_laplace*(N+1)))
                    
    # The Scaled Emission probabilities (How often does tag t yield hapax word w?)
    for tag in tagfreq:
        N = len(tag_corpus[tag])
        if tag in hapax:
            hapax_emission[tag] = math.log((scaled_laplace*hapax[tag]) / (tagfreq[tag] + scaled_laplace*hapax[tag]*(N + 1)))
        else:
            hapax_emission[tag] = math.log(scaled_laplace / (tagfreq[tag] + scaled_laplace*(N + 1)))
            
    # The hapax probabilities
    for tag in tagfreq: 
        N = len(tag_corpus[tag])
        for prefix in prefixes:
            word = prefix + "-X"
            if word in tag_corpus[tag]:
                emission[(word,tag)] = math.log((tag_corpus[tag][word] + scaled_laplace*hapax[tag]) / (tagfreq[tag] + scaled_laplace*hapax[tag]*(N + 1)))
        for suffix in suffixes:
            word = "X-" + suffix
            if word in tag_corpus[tag]:
                emission[(word,tag)] = math.log((tag_corpus[tag][word] + scaled_laplace*hapax[tag]) / (tagfreq[tag] + scaled_laplace*hapax[tag]*(N + 1)))
    
    # Construct the trellis
    node = {}
    tag_pointer = {}
    trellis = [node,tag_pointer]
    
    for sentence in test:
        reverse_sentence = []
        predict_sentence = []
        # skip the start word
        for position in range(1,len(sentence)):
            # for each word, we construct the emission probability of each tag
            for tag in tagfreq:
                word = sentence[position]
                candidate = {}
                # first get the emission probability
                if word in tag_corpus[tag]:
                    emission_prob = emission[(word,tag)]
                else:
                    # If not appeared in train, use hapax scaled one
                    emission_prob = hapax_emission[tag]
                    if word not in corpus:
                        for prefix in prefixes:
                            prefix_word = prefix + "-X"
                            if word.startswith(prefix) and prefix_word in tag_corpus[tag]:
                                emission_prob = emission[(prefix_word,tag)]
                        for suffix in suffixes:
                            suffix_word = "X-" + suffix
                            if word.endswith(suffix) and suffix_word in tag_corpus[tag]:
                                emission_prob = emission[(suffix_word,tag)]

                # then get the initial probability if necessary
                for tag_prev in tagfreq:
                    prob = 0
                    if position == 1:
                        if tag in initial:
                            prob = prob + initial[tag]
                        else:
                            prob = prob + 0
                    else:
                        # add the previous node prob for each tag - trellis[0][(position-1, tag_prev)]
                        prob = prob + trellis[0][(position-1, tag_prev)]
                    # add the transition probability - transition[(tag_prev,tag)]
                    # add the current emission probability - emission_prob
                    prob = prob + transition[(tag_prev,tag)] + emission_prob
                    #store the prob and choose the max one latter
                    candidate[tag_prev] = prob
                    
                # choose the largest one as the node prob
                trellis[0][(position, tag)] = max(candidate.values())
                # also set the pointer to the previous node
                trellis[1][(position, tag)] = max(candidate, key=candidate.get)
                    
        # Return the best path through the trellis
        max_prob = float("-inf")
        # start from the last node, find the one with largest prob as end node
        for tag in tagfreq:
            if (trellis[0][(len(sentence)-1, tag)] > max_prob):
                last_tag = tag
                max_prob = trellis[0][(len(sentence)-1, tag)]
        reverse_sentence.append((sentence[len(sentence)-1], last_tag))
        # iteratively get the whole path
        position = len(sentence)-1
        tag = last_tag
        while (position>1):
            reverse_sentence.append((sentence[position-1], trellis[1][(position, tag)]))
            tag = trellis[1][(position, tag)]
            position = position - 1
        # generate the prediction
        reverse_sentence.append((sentence[0], 'START'))
        predict_sentence = reverse_sentence[::-1]
        # add to result
        result.append(predict_sentence)

    return result