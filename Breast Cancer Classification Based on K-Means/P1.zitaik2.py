# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:31:56 2022

@author: Lenovo
"""
import random
import math
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
#helper functions
def distance(point1,point2):
    if(len(point1)!=len(point2)):
        return -1
    accumulate = 0
    for i in range(len(point1)):
        accumulate = accumulate + (point1[i]-point2[i])**2
    return math.sqrt(accumulate)

def check_converge(clusters1, clusters2):
    for i in clusters1.keys():
        match_flag = 0
        for j in clusters2.keys():
            if (match_flag == 1):
                break;
            cluster1 = clusters1[i]
            cluster2 = clusters2[j]
            if (len(cluster1)!=len(cluster2)):
                continue
            unmatch_flag1 = 0
            for m in cluster1:
                if m not in cluster2:
                    unmatch_flag1 = 1
                    break
            if (unmatch_flag1 == 1):
                break
            unmatch_flag2 = 0
            for n in cluster2:
                if n not in cluster1:
                    unmatch_flag2 = 1
                    break
            if (unmatch_flag2 == 1):
                break
            match_flag = 1
            
        if (match_flag == 0):
            return -1
        
    return 1

def SSE(clusters):
    loss_sum = 0
    for centroid in clusters.keys():
        for sample in clusters[centroid]:
            loss_sum = loss_sum + distance(sample,centroid)**2
    return loss_sum           
###############################################################################
# main function

# load in data
file = open('breast-cancer-wisconsin.data','r')
raw_data = file.readlines()
#print(content)
file.close()

###############################################################################

# data preprocess
data = []
sample_num = 0
pass_flag = 0
for line in raw_data:
    # convert string to list and remove '\n'
    raw_sample = line[:-1].split(',')
    # remove id and label
    raw_sample = raw_sample[1:-1]
    # ignore the sample with '?'
    # convert char to integer at the same time
    sample = []
    for attribute in raw_sample:
        if (attribute == '?'):
            pass_flag = 1
            break
        sample.append(int(attribute))
    if (pass_flag == 1):
        pass_flag = 0
        continue
    # put good sample to database
    data.append(tuple(sample))
    # increment the sample number
    sample_num = sample_num + 1
    
###############################################################################

# Basic algorithm
    
K_list = [2,3,4,5,6,7,8]

sse = []
iteration = []

for k in K_list:
    # initialize:
    
    # select k data points to be centroid
    # set seeds 
    random.seed(42)
    # choose points
    centroids = random.choices(data,weights=None, k=k)
    #print(centroids)
    # convert into tuples
    temp_centroids = []
    for centroid in centroids: 
        temp_centroids.append(tuple(centroid))
    centroids = temp_centroids
    
    # iteration
    convergence = False
    loop_time = 0
    old_clusters = {}
    for centroid in centroids:
        old_clusters[centroid] = []
    
    # loop until convergency
    while(convergence == False):
        loop_time += 1
        # set up the cluster 
        clusters = {}
        for centroid in centroids:
            clusters[centroid] = []
        
        # Step1: assign each point to the nearest centroid's cluster
        for sample in data:
            dist={}
            # calculate each distance
            for i in range(k):
                dist[centroids[i]] = distance(sample, centroids[i])
            # take the nearest one
            nearest_centroid = min(dist, key = dist.get)
            # assign it in the cluster
            samples = list(clusters[nearest_centroid])
            samples.append(sample)
            clusters[nearest_centroid] = samples
        #print(clusters)
        
        # Additional step: if you find an empty cluster
        # drop it and split the largest one into two
        cluster_num = 0
        for i in clusters.keys():
            cluster_num += 1
        diff = k - cluster_num
        if (diff > 0):
            while(diff!=0):
                diff -= 1
                # find the largest cluster
                largest_cluster = None
                largest_len = 0
                for cluster in clusters.keys():
                    length = len[clusters[cluster]]
                    if (length > largest_len):
                        largest_cluster = cluster
                        largest_len = length
                # split into two clusters
                cluster = clusters[largest_cluster]
                len1 = len[cluster]
                cluster1 = []
                temp_cluster1 = []
                cluster2 = []
                temp_cluster2 = []
                counter = 0
                for sample in cluster:
                    if (counter < int(len1/2)):
                        cluster1.append(sample)
                        temp_cluster1.append(list(sample))
                    else:
                        cluster2.append(sample)
                        temp_cluster2.append(list(sample))                    
                centroid1 = tuple(np.mean(temp_cluster1,axis=0))
                centroid2 = tuple(np.mean(temp_cluster1,axis=0))
                # add them back
                clusters[centroid1] = cluster1
                clusters[centroid2] = cluster2
                # delete original one
                del clusters[largest_cluster]
        
        # Step2: recompute the new centroids
        new_centroids = []
        #for cluster in clusters:
        for centroid in centroids:
            cluster = clusters[centroid]
            # convert back to list
            temp_cluster = []
            for sample in cluster:
                temp_cluster.append(list(sample))
            cluster = temp_cluster
            # take the mean
            new_centroid = tuple(np.mean(cluster,axis=0))
            #print(new_centroid)
            new_centroids.append(new_centroid)
        centroids = new_centroids
        
        # check for convergence
        if (check_converge(clusters, old_clusters) == 1):
            convergence = True
        else:
            old_clusters = clusters
            
    # compute SSE
    loss = SSE(clusters)
    sse.append(loss)
    iteration.append(loop_time)
    #print(clusters)
    #print(loop_time)
    #print(loss)
    #cluster_num = 0
    #for i in clusters.keys():
    #    cluster_num += 1
    #print(cluster_num)
print(sse) 
print(iteration)
  
###############################################################################
    
# plot the curve of L(k) vs. k value
plt.plot(K_list, sse)
plt.ylabel('k value')
plt.xlabel('L(k)')
plt.title("the curve of L(k) vs. k value") 

    
