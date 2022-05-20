# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP3. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

from collections import deque
import heapq
from copy import copy

# Feel free to use the code below as you wish
# Initialize it with a list/tuple of objectives
# Call compute_mst_weight to get the weight of the MST with those objectives
# TODO: hint, you probably want to cache the MST value for sets of objectives you've already computed...
# Note that if you want to test one of your search methods, please make sure to return a blank list
#  for the other search methods otherwise the grader will not crash.
class MST:
    def __init__(self, objectives):
        self.elements = {key: None for key in objectives}

        # TODO: implement some distance between two objectives
        # ... either compute the shortest path between them, or just use the manhattan distance between the objectives
        self.distances   = {
                (i, j): (abs(i[0]-j[0]) + abs(i[1]-j[1])) #DISTANCE(i, j)
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

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # boundary of the maze
    rows    = maze.size.y
    columns = maze.size.x
    # start point
    start = maze.start
    # end point(in this part there is only one)
    waypoints = maze.waypoints
    waypoint = waypoints[0]
    # set up the queue(fifo) for bfs
    queue = deque()
    queue.append(start)
    # explore is a dictionary contain the point explored as key
    # and the parent of this point as values
    explore = {}
    explore[start]=start   #The parent of the start point is set to itself
    
    result_path = deque()
    path = []
    
    while(1):
        current_cell = queue.popleft()
        y,x = current_cell
        if current_cell == waypoint:
            break
        neighbor = maze.neighbors(y,x) #[(y-1,x),(y+1,x),(y,x-1),(y,x+1)]
        for cell in neighbor:
            y_cell, x_cell = cell
            if (y_cell >= 0 and y_cell <rows and x_cell >= 0 and x_cell < columns):
                if (maze.navigable(y_cell, x_cell)):
                    if (cell in explore.keys()) == False:
                        queue.append(cell)
                        explore[cell] = current_cell

    child = waypoint
    parent = waypoint
    result_path.appendleft(child)
    
    while(parent != start):
        parent = explore[child]
        result_path.appendleft(parent)
        child = parent
    
    for cell in result_path:
        path.append(cell)
    
    return path

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # boundary of the maze
    rows    = maze.size.y
    columns = maze.size.x
    # start point
    start = maze.start
    # end point(in this part there is only one)
    waypoints = maze.waypoints
    waypoint = waypoints[0]
    # set up the queue(fifo) for bfs
    queue = deque()
    queue.append(start)
    # explore is a dictionary contain the point explored as key
    # and the parent of this point as values
    explore = {}
    explore[start]=start   #The parent of the start point is set to itself
    distance = {}
    distance[start] = abs(start[0]-waypoint[0])+abs(start[1]-waypoint[1])
    exp_dist = {}
    exp_dist[start] = 0
    
    result_path = deque()
    path = []
    
    while(1):
        current_cell = queue.popleft()
        y,x = current_cell
        if current_cell == waypoint:
            break
        neighbor = maze.neighbors(y,x) #[(y-1,x),(y+1,x),(y,x-1),(y,x+1)]
        for cell in neighbor:
            i = 0
            insert = 0
            y_cell, x_cell = cell
            distance[cell] = abs(cell[0]-waypoint[0])+abs(cell[1]-waypoint[1])
            exp_dist[cell] = exp_dist[current_cell]+1
            if (y_cell >= 0 and y_cell <rows and x_cell >= 0 and x_cell < columns):
                if (maze.navigable(y_cell, x_cell)):
                    if (cell in explore.keys()) == False:
                        cell_dist = distance[cell] + exp_dist[cell]
                        for explored_cell in queue:
                            if cell_dist < (distance[explored_cell] + exp_dist[explored_cell]):
                                queue.insert(i,cell)
                                insert = 1
                                break
                                i = i + 1
                        if (insert == 0):
                            queue.append(cell)
                        explore[cell] = current_cell

    child = waypoint
    parent = waypoint
    result_path.appendleft(child)
    
    while(parent != start):
        parent = explore[child]
        result_path.appendleft(parent)
        child = parent
    
    for cell in result_path:
        path.append(cell)
    
    return path


def distance(cell, waypoints, weight):
    min_dist = float('inf')
    for i in waypoints:
        dist = abs(cell[0]-i[0])+(cell[1]-i[1])
        if dist < min_dist:
            min_dist = dist
    return (min_dist + weight)


def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    # boundary of the maze
    rows    = maze.size.y
    columns = maze.size.x
    # start point
    start = maze.start
    # end points
    waypoints = list(maze.waypoints)
    
    #weights
    mst = MST(waypoints)
    weight = mst.compute_mst_weight()
    exp_dist = 0    # how far we have walked
    #weights is used to store the mst length for reuse
    weights = {}
    weights[maze.waypoints] = weight
    
    # explore is a dictionary contain the point explored as key
    # and the parent of this point as values
    explore = {}
    # dist is used to store the 
    dist = {start: {maze.waypoints: (weight+exp_dist)}}
    
    # set up the queue(fifo) for bfs
    queue = []
    # initialize the queue
    heapq.heappush(queue,((weight+exp_dist),(start, waypoints)))
    
    path = []
    

    while (len(queue) != 0):
        # cost - the heuristic cost of current point
        # current_cell - the current place we are
        # current_waypoints - the current waypoints left
        cost, (current_cell, current_waypoints) = heapq.heappop(queue)
        y,x = current_cell
        waypoints = copy(current_waypoints)
        waypoints_left = tuple(waypoints)
        exp_dist = cost - weights[waypoints_left] + 1

        # if no waypoint is left, we are done!
        if len(waypoints) == 0:
            break
        
        neighbor = maze.neighbors(y,x) #[(y-1,x),(y+1,x),(y,x-1),(y,x+1)]
    
        for cell in neighbor:
            # we return to the start point for each cell
            waypoints = copy(current_waypoints)
            # if we reach a waypoints, remove it
            if cell in waypoints:
                waypoints.remove(cell)
            waypoints_left = tuple(waypoints)
            
            # update the cost for current cell
            if (waypoints_left in weights.keys()) == False:
                mst = MST(waypoints)
                weight = mst.compute_mst_weight()
                weights[waypoints_left] = weight
            else:
                weight = weights[waypoints_left]
            cost = exp_dist + weight

            
            if cell in dist.keys():
                if waypoints_left in dist[cell].keys():
                    if cost >= dist[cell][waypoints_left]:
                        # if the cell has been visited and 
                        #the new cost is higher, we ignore it
                        continue
                    # else we update the cell
                    else:
                       dist[cell][waypoints_left] = cost 
                else:
                    dist[cell][waypoints_left] = cost
            else:
                dist[cell] = {}
                dist[cell][waypoints_left] = cost
            
            
            # store the parent info
            if cell in explore.keys():
                explore[cell][waypoints_left] = (current_cell, current_waypoints)
            else:
                explore[cell] = {}
                explore[cell][waypoints_left] = (current_cell, current_waypoints)
                
            # push onto the queue
            heapq.heappush(queue, (cost, (cell, waypoints)))

    # take the last point as the start of track 
    child = current_cell
    child_waypoints = tuple(waypoints)
    path.append(current_cell)    
    #print("parent: ", explore)
    
    #retrace back to get the path
    while (child != start) or (child_waypoints != maze.waypoints):
        parent, parent_waypoints = explore[child][child_waypoints]
        #print("cp1: ",explore[child][child_waypoints])
        #print("cp2: ",parent, parent_waypoints)
        path.insert(0, parent)
        child = parent
        child_waypoints = tuple(parent_waypoints)


    return path          


def fast(maze):
    """
    Runs suboptimal search algorithm for extra credit/part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
        # boundary of the maze
    rows    = maze.size.y
    columns = maze.size.x
    # start point
    start = maze.start
    # end points
    waypoints = list(maze.waypoints)
    
    #weights
    mst = MST(waypoints)
    weight = mst.compute_mst_weight()*2.5
    exp_dist = 0    # how far we have walked
    #weights is used to store the mst length for reuse
    weights = {}
    weights[maze.waypoints] = weight
    
    # explore is a dictionary contain the point explored as key
    # and the parent of this point as values
    explore = {}
    # dist is used to store the 
    dist = {start: {maze.waypoints: (weight+exp_dist)}}
    
    # set up the queue(fifo) for bfs
    queue = []
    # initialize the queue
    heapq.heappush(queue,((weight+exp_dist),(start, waypoints)))
    
    path = []
    

    while (len(queue) != 0):
        # cost - the heuristic cost of current point
        # current_cell - the current place we are
        # current_waypoints - the current waypoints left
        cost, (current_cell, current_waypoints) = heapq.heappop(queue)
        y,x = current_cell
        waypoints = copy(current_waypoints)
        waypoints_left = tuple(waypoints)
        exp_dist = cost - weights[waypoints_left] + 1

        # if no waypoint is left, we are done!
        if len(waypoints) == 0:
            break
        
        neighbor = maze.neighbors(y,x) #[(y-1,x),(y+1,x),(y,x-1),(y,x+1)]
    
        for cell in neighbor:
            # we return to the start point for each cell
            waypoints = copy(current_waypoints)
            # if we reach a waypoints, remove it
            if cell in waypoints:
                waypoints.remove(cell)
            waypoints_left = tuple(waypoints)
            
            # update the cost for current cell
            if (waypoints_left in weights.keys()) == False:
                mst = MST(waypoints)
                weight = mst.compute_mst_weight()*2.5
                weights[waypoints_left] = weight
            else:
                weight = weights[waypoints_left]
            cost = exp_dist + weight

            
            if cell in dist.keys():
                if waypoints_left in dist[cell].keys():
                    if cost >= dist[cell][waypoints_left]:
                        # if the cell has been visited and 
                        #the new cost is higher, we ignore it
                        continue
                    # else we update the cell
                    else:
                       dist[cell][waypoints_left] = cost 
                else:
                    dist[cell][waypoints_left] = cost
            else:
                dist[cell] = {}
                dist[cell][waypoints_left] = cost
            
            
            # store the parent info
            if cell in explore.keys():
                explore[cell][waypoints_left] = (current_cell, current_waypoints)
            else:
                explore[cell] = {}
                explore[cell][waypoints_left] = (current_cell, current_waypoints)
                
            # push onto the queue
            heapq.heappush(queue, (cost, (cell, waypoints)))

    # take the last point as the start of track 
    child = current_cell
    child_waypoints = tuple(waypoints)
    path.append(current_cell)    
    #print("parent: ", explore)
    
    #retrace back to get the path
    while (child != start) or (child_waypoints != maze.waypoints):
        parent, parent_waypoints = explore[child][child_waypoints]
        #print("cp1: ",explore[child][child_waypoints])
        #print("cp2: ",parent, parent_waypoints)
        path.insert(0, parent)
        child = parent
        child_waypoints = tuple(parent_waypoints)
        
    return path  


