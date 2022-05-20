# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 08:57:48 2022

@author: Lenovo
"""

from collections import deque
import heapq
'''
queue = deque([])
dic ={}
cells = [(1,1),(2,2),(3,3)]
for cell in cells:
    queue.append(cell)
print(queue)
queue.insert(0,(4,4))
print(queue)
for i in queue:
    print(i)




dic = {}
cell = (1,1)
dic[cell]=(2,2)
print(dic)

x=1
y=1

neighbor = [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]

print(neighbor)

if ((0,0) in dic.keys()) == False:
    print("in")
'''

class MST:
    def __init__(self, objectives):
        self.elements = {key: None for key in objectives}

        # TODO: implement some distance between two objectives
        # ... either compute the shortest path between them, or just use the manhattan distance between the objectives
        self.distances   = {
                (i, j): (abs(i[0]-j[0]) + abs(i[1]-j[1]))#DISTANCE(i, j)
                for i, j in self.cross(objectives)
            }

    # Prim's algorithm adds edges to the MST in sorted order as long as they don't create a cycle
    def compute_mst_weight(self):
        weight      = 0
        for distance, i, j in sorted((self.distances[(i, j)], i, j) for (i, j) in self.distances):
            if self.unify(i, j):
                weight += distance
        return weight
    # helper checks the root of a node, in the process flatten the path to the root
    def resolve(self, key):
        path = []
        root = key
        while self.elements[root] is not None:
            path.append(root)
            root = self.elements[root]
        for key in path:
            self.elements[key] = root
        return root

    # helper checks if the two elements have the same root they are part of the same tree
    # otherwise set the root of one to the other, connecting the trees
    def unify(self, a, b):
        ra = self.resolve(a)
        rb = self.resolve(b)
        if ra == rb:
            return False
        else:
            self.elements[rb] = ra
            return True

    # helper that gets all pairs i,j for a list of keys
    def cross(self, keys):
        return (x for y in (((i, j) for j in keys if i < j) for i in keys) for x in y)
'''
cells = []  
mst = MST(cells)
print(mst.compute_mst_weight())
'''
a = (1,(2,3))
print(tuple(tuple(a)))
