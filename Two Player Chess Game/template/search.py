import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(side, board, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(side, board, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return (value, [ move ], { encode(*move): {} })
    else:
        return (evaluate(board), [], {})

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.
def minimax(side, board, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    #raise NotImplementedError("you need to write this!")
    # Define Basic return variables
    value = 0   # value of the final board in the minimax-optimal move sequence
    moveList = []  # the minimax-optimal move sequence, as a list of moves
    moveTree = {}  # a tree of moves that were evaluated in the search process
    
    # Define helper variables
    best_move = None
    
    # All next possible moves
    moves = []
    for move in generateMoves(side, board, flags):
        moves.append(move)
    
    # Base: if depth = 0 or node is a terminal node then
    if (depth == 0) or (len(moves) == 0):
        # evaluate the heuristic value of node
        value = evaluate(board)
        # directly return
        return (value, moveList, moveTree)
    
    # Recursive case
    # For the maximizing Player
    if (side == False):
        # initialize value
        value = -math.inf
        # Iterate over all moves that are legal from the current board position
        for move in moves:
            # Evaluate the result of taking current step
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            # Expand from current position
            newValue, newList, newTree = minimax(newside, newboard, newflags, depth - 1)
            
            # Update variables
            # Store the newly explored move in tree
            encoded=encode(*move)
            moveTree[encoded] = newTree
            # Take the highest value
            # If gain higher score, update value and list to show we take the step
            if (newValue > value):
                best_move = move
                value = newValue
                moveList = newList
    # For the minimizing Player
    else:
        # initialize value
        value = math.inf
        # Iterate over all moves that are legal from the current board position
        for move in moves:
            # Evaluate the result of taking current step
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            # Expand from current position
            newValue, newList, newTree = minimax(newside, newboard, newflags, depth - 1)
            
            # Update variables
            # Store the newly explored move in tree
            encoded=encode(*move)
            moveTree[encoded] = newTree
            # Take the lowest value
            # If gain lower score, update value and list to show we take the step
            if (newValue < value):
                best_move = move
                value = newValue
                moveList = newList
                
    # Add the best move at the current step to the front of list
    moveList.insert(0, best_move)
    return (value, moveList, moveTree)

def alphabeta(side, board, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    #raise NotImplementedError("you need to write this!")
    # Define Basic return variables
    value = 0   # value of the final board in the minimax-optimal move sequence
    moveList = []  # the minimax-optimal move sequence, as a list of moves
    moveTree = {}  # a tree of moves that were evaluated in the search process
    
    # Define helper variables
    best_move = None
    
    # All next possible moves
    moves = []
    for move in generateMoves(side, board, flags):
        moves.append(move)
    
    # Base: if depth = 0 or node is a terminal node then
    if (depth == 0) or (len(moves) == 0):
        # evaluate the heuristic value of node
        value = evaluate(board)
        # directly return
        return (value, moveList, moveTree)
    
    # Recursive case
    # For the maximizing Player
    if (side == False):
        # initialize value
        value = -math.inf
        # Iterate over all moves that are legal from the current board position
        for move in generateMoves(side,board,flags):
            # Evaluation
            # Evaluate the result of taking current step
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            # Expand from current position
            newValue, newList, newTree = alphabeta(newside, newboard, newflags, depth - 1, alpha, beta)
            
            # Update variables
            # Store the newly explored move in tree
            encoded=encode(*move)
            moveTree[encoded] = newTree
            # Take the highest value
            # If gain higher score, update value and list to show we take the step
            if (newValue > value):
                best_move = move
                value = newValue
                moveList = newList
            
            # Update alpha-beta value
            # A max node can update alpha
            if (value > alpha):
                alpha = value
            # pruning all the rest branch if alpha > beta
            if alpha >= beta:   #here >= is necessary
                break
            
    # For the minimizing Player
    else:
        # initialize value
        value = math.inf
        # Iterate over all moves that are legal from the current board position
        for move in generateMoves(side,board,flags):
            # Evaluate the result of taking current step
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            # Expand from current position
            newValue, newList, newTree = alphabeta(newside, newboard, newflags, depth - 1, alpha, beta)
            
            # Update variables
            # Store the newly explored move in tree
            encoded=encode(*move)
            moveTree[encoded] = newTree
            # Take the lowest value
            # If gain lower score, update value and list to show we take the step
            if (newValue < value):
                best_move = move
                value = newValue
                moveList = newList
                
            # Update alpha-beta value
            # A max node can update alpha
            if (value < beta):
                beta = value
            # pruning all the rest branch if alpha > beta
            if alpha >= beta:
                break
                
    # Add the best move at the current step to the front of list
    moveList.insert(0, best_move)
    return (value, moveList, moveTree)    

def stochastic(side, board, flags, depth, breadth, chooser):
    '''
    Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
    Return: (value, moveList, moveTree)
      value (float): average board value of the paths for the best-scoring move
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
      breadth: number of different paths 
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    
    #raise NotImplementedError("you need to write this!")    
    # Define Basic return variables
    value = 0   # value of the final board in the minimax-optimal move sequence
    moveList = []  # the minimax-optimal move sequence, as a list of moves
    moveTree = {}  # a tree of moves that were evaluated in the search process
    
    # All next possible moves
    moves = []
    for move in generateMoves(side, board, flags):
        moves.append(move)
    
    # Base: if depth = 0 or node is a terminal node then
    if (depth == 0) or (len(moves) == 0):
        # evaluate the heuristic value of node
        value = evaluate(board)
        # directly return
        return (value, moveList, moveTree)
    
    # Recursive case
    # For the maximizing Player     
    if (side == False):
        # initialize value
        value = -math.inf
        # Iterate over all moves that are legal from the current board position
        for move in generateMoves(side,board,flags):
            # Evaluate the result of taking current step
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            # Expand from current position
            newValue, newList, newTree = track_path(newside, newboard, newflags, depth - 1, breadth, chooser)
            
            # Update variables
            # Store the newly explored move in tree
            encoded=encode(*move)
            moveTree[encoded] = newTree
            # Take the largest value
            # If gain larger score, update value and list to show we take the step
            if (newValue > value):
                value = newValue
                moveList = [move] + newList
    # For the minimizing Player
    else:
        # initialize value
        value = math.inf
        # Iterate over all moves that are legal from the current board position
        for move in generateMoves(side,board,flags):
            # Evaluate the result of taking current step
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            # Expand from current position
            newValue, newList, newTree = track_path(newside, newboard, newflags, depth - 1, breadth, chooser)
            
            # Update variables
            # Store the newly explored move in tree
            encoded=encode(*move)
            moveTree[encoded] = newTree
            # Take the lowest value
            # If gain lower score, update value and list to show we take the step
            if (newValue < value):
                value = newValue
                moveList = [move] + newList
    
    return (value, moveList, moveTree)    

    
def track_path(side, board, flags, depth, breadth, chooser):
    '''
    a helper function to track each path within the breath recursively
    '''
    # Define local variables
    pathValue = 0   # value of the final board in the minimax-optimal move sequence
    pathList = []  # the minimax-optimal move sequence, as a list of moves
    pathTree = {}  # a tree of moves that were evaluated in the search process

    # All next possible moves
    path_moves = []
    for path_move in generateMoves(side, board, flags):
        path_moves.append(path_move)
        
    # Base: if depth = 0 or node is a terminal node then
    if ((depth == 0) or (len(path_moves) == 0)):
        # evaluate the heuristic value of node
        pathValue = evaluate(board)
        # directly return
        return (pathValue, pathList, pathTree)        

    # Recursive step
    tot_path_value = 0
    # For every possible initial move, you should evaluate exactly breadth paths
    # chosen at random using the function chooser
    # Iterate over all path through the breadth
    for path in range(breadth):
        # choose path at random using the function chooser
        path_move = chooser(path_moves)
        # Evaluate the result of taking current step
        newside, newboard, newflags = makeMove(side, board, path_move[0], path_move[1], flags, path_move[2])
        # Expand from current position
        newValue, pathList, newTree = track_depth(newside, newboard, newflags, depth - 1, breadth, chooser)
        
        # Update variables
        # Store the newly explored move in tree
        encoded=encode(*path_move)
        pathTree[encoded] = newTree
        # update the path value
        tot_path_value = tot_path_value + newValue
    # average the path values in order to find the value of the initial move
    pathValue = tot_path_value / breadth
        
    return (pathValue, pathList, pathTree)    

    
def track_depth(side, board, flags, depth, breadth, chooser):
    '''
    This function takes one step/depth of random move in the path recursively
    '''
    # Define local variables
    deepValue = 0   # value of the final board in the minimax-optimal move sequence
    deepList = []  # the minimax-optimal move sequence, as a list of moves
    deepTree = {}  # a tree of moves that were evaluated in the search process

    # All next possible moves
    deep_moves = []
    for deep_move in generateMoves(side, board, flags):
        deep_moves.append(deep_move)
        
    # Base: if depth = 0 or node is a terminal node then
    if ((depth == 0) or (len(deep_moves) == 0)):
        # evaluate the heuristic value of node
        deepValue = evaluate(board)
        # directly return
        return (deepValue, deepList, deepTree)        

    # Recursive step
    # choose a random step
    deep_move = chooser(deep_moves)
    # Evaluate the result of taking current step
    newside, newboard, newflags = makeMove(side, board, deep_move[0], deep_move[1], flags, deep_move[2])
    # Expand from current position
    deepValue, newList, newTree = track_depth(newside, newboard, newflags, depth - 1, breadth, chooser)
    
    # Update variables
    # Store the newly explored move in tree
    encoded=encode(*deep_move)
    deepTree[encoded] = newTree
    deepList = [deep_move] + newList
    
    return (deepValue, deepList, deepTree)
