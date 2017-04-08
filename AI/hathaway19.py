import random
import sys
import math
import numpy as np

sys.path.append("..")  # so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *


##
# AIPlayer
# Description: This AI uses a neural network to approximate the state evaluation
# function.
# Assignment: Homework #5: Artificial Neural Network
#
# Due Date: April 8th, 2017
#
# @names: Justin Hathaway (no partner)
##
class AIPlayer(Player):
    # list of nodes for search tree
    node_list = []

    # maximum depth
    max_depth = 3

    # current index - for recursive function
    cur_array_index = 0

    # highest evaluated move - to be reset every time the generate_states method is called
    highest_evaluated_move = None

    # highest move score - useful for finding highest evaluated move - to be reset
    highest_move_eval = -1

    # this AI's playerID
    me = -1

    # whether or not the playerID has been set up yet
    me_set_up = False

    # __init__
    # Description: Creates a new Player
    #
    # Parameters:
    #   inputPlayerId - The id to give the new player (int)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer, self).__init__(inputPlayerId, "theNeuralNetAI")

        self.learningRate = 0.8
        self.numOfNodes = 9

        print "rate of learning: ", self.learningRate

        # weights array which has already been learned
        # self.weights = [0.81, 0.5, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8,
        #                 0.81, 0.5, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8,
        #                 0.81, 0.5, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8,
        #                 0.81, 0.5, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8,
        #                 0.81, 0.5, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8,
        #                 0.81, 0.5, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8,
        #                 0.81, 0.5, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8,
        #                 0.81, 0.5, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8,
        #                 0.81, 0.5, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8,
        #                 0.81, 0.5, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8,
        #                 0.22]

        self.weights = []
        # Calls a method to assign random values between 0 and 1 for all the weights
        self.assignRandomWeights()

        print self.weights

    # Method to create a node containing the state, evaluation, move, current depth,
    # the parent node, and the index
    def create_node(self, state, evaluation, move, current_depth, parent_index, index):
        node = [state, evaluation, move, current_depth, parent_index, index]
        self.node_list.append(node)

    ##
    # getPlacement
    #
    # Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    # Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    # Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        numToPlace = 0
        # implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:  # stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    # Choose any x location
                    x = random.randint(0, 9)
                    # Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    # Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        # Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:  # stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    # Choose any x location
                    x = random.randint(0, 9)
                    # Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    # Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        # Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]

    ##
    # getMove
    # Description: Gets the next move from the Player.
    #
    # Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    # Return: The Move to be made
    ##
    def getMove(self, currentState):

        if not self.me_set_up:
            self.me = currentState.whoseTurn

        # searches for best move
        selectedMove = self.move_search(currentState, 0, -(float)("inf"), (float)("inf"))

        # if not None, return move, if None, end turn
        if not selectedMove == None:
            return selectedMove
        else:
            return Move(END, None, None)

    ##
    # getAttack
    # Description: Gets the attack to be made from the Player
    #
    # Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        # Attack a random enemy.
        return enemyLocations[0]

    ##
    # move_search - recursive
    #
    # uses Minimax with alpha beta pruning to search for best next move
    #
    # Parameters:
    #   game_state - current state
    #   curr_depth - current search depth
    #   alpha      - the parent node's alpha value
    #   beta       - the previous node's beta value
    #
    # Return
    #   returns a move object
    ##
    def move_search(self, game_state, curr_depth, alpha, beta):

        # if max depth surpassed, return state evaluation
        if curr_depth == self.max_depth + 1:
            return self.evaluate_state(game_state)

        # list all legal moves
        move_list = listAllLegalMoves(game_state)

        # remove end turn move if the list isn't empty
        if not len(move_list) == 1:
            move_list.pop()

        # list of nodes, which contain the state, move, and eval
        node_list = []

        # generate states based on moves, evaluate them and put them into a list in node_list
        for move in move_list:
            state_eval = 0
            state = getNextStateAdversarial(game_state, move)
            state_eval = self.evaluate_state(state)
            # print(state_eval)
            if not state_eval == 0.00001:
                node_list.append([state, move, state_eval])

        self.mergeSort(node_list)

        if not self.me == game_state.whoseTurn:
            move_list.reverse()

        best_nodes = []

        for i in range(0, 5):  # temporary
            if not len(node_list) == 0:
                best_nodes.append(node_list.pop())

        # best_val = -1

        # if not at the max depth, expand all the nodes in node_list and return
        if curr_depth <= self.max_depth:
            for node in best_nodes:
                score = self.move_search(node[0], curr_depth + 1, alpha, beta)
                if game_state.whoseTurn == self.me:
                    if score > alpha:
                        alpha = score
                    if alpha >= beta:
                        # print("Pruned")
                        break
                else:
                    if score < beta:
                        beta = score
                    if alpha >= beta:
                        # print("Pruned")
                        break

        # if not curr_depth == 0:
        if game_state.whoseTurn == self.me and not curr_depth == 0:
            return alpha
        elif not game_state == self.me and not curr_depth == 0:
            return beta
        else:
            best_eval = -1
            best_node = []

            for node in best_nodes:
                if node[2] > best_eval:
                    best_eval = node[2]
                    best_node = node

            # print(len(best_node))
            if not best_node == []:
                return best_node[1]
            else:
                return None

    ##
    # get_closest_enemy_dist - helper function
    #
    # returns distance to closest enemy from an ant
    ##
    def get_closest_enemy_dist(self, my_ant_coords, enemy_ants):
        closest_dist = 100
        for ant in enemy_ants:
            if not ant.type == WORKER:
                dist = approxDist(my_ant_coords, ant.coords)
                if dist < closest_dist:
                    closest_dist = dist
        return closest_dist

    ##
    # get_closest_enemy_worker_dist - helper function
    #
    # returns distance to closest enemy worker ant
    ##
    def get_closest_enemy_worker_dist(self, my_ant_coords, enemy_ants):
        closest_dist = 100
        for ant in enemy_ants:
            if ant.type == WORKER:
                dist = approxDist(my_ant_coords, ant.coords)
                if dist < closest_dist:
                    closest_dist = dist
        return closest_dist

    ##
    # get_closest_enemy_food_dist - helper function
    #
    # returns distance to closest enemy food
    ##
    def get_closest_enemy_food_dist(self, my_ant_coords, enemy_food_coords):

        enemy_food1_dist = approxDist(my_ant_coords, enemy_food_coords[0])
        enemy_food2_dist = approxDist(my_ant_coords, enemy_food_coords[1])

        if enemy_food1_dist < enemy_food2_dist:
            return enemy_food1_dist
        else:
            return enemy_food2_dist

    ##
    # evaluate_state
    #
    # Evaluates and scores a GameState Object
    #
    # Parameters
    #   state - the GameState object to evaluate
    #
    # Return
    #   a double between 0 and 1 inclusive
    ##
    def evaluate_state(self, state):
        # The AI's player ID
        me = state.whoseTurn
        # The opponent's ID
        enemy = (state.whoseTurn + 1) % 2

        # Get a reference to the player's inventory
        my_inv = state.inventories[me]
        # Get a reference to the enemy player's inventory
        enemy_inv = state.inventories[enemy]

        # Gets both the player's queens
        my_queen = getAntList(state, me, (QUEEN,))
        enemy_queen = getAntList(state, enemy, (QUEEN,))

        # Sees if winning or loosing conditions are already met
        if (my_inv.foodCount == 11) or (enemy_queen is None):
            return 1.0
        if (enemy_inv.foodCount == 11) or (my_queen is None):
            return 0.0

        # the starting value, not winning or losing
        eval = 0.0

        # important number
        worker_count = 0
        drone_count = 0

        food_coords = []
        enemy_food_coords = []

        foods = getConstrList(state, None, (FOOD,))

        # Gets a list of all of the food coords
        for food in foods:
            if food.coords[1] < 5:
                food_coords.append(food.coords)
            else:
                enemy_food_coords.append(food.coords)

        # coordinates of this AI's tunnel
        tunnel = my_inv.getTunnels()
        t_coords = tunnel[0].coords

        # coordinates of this AI's anthill
        ah_coords = my_inv.getAnthill().coords

        # A list that stores the evaluations of each worker
        wEval = []

        # A list that stores the evaluations of each drone, if they exist
        dEval = []

        # queen evaluation
        qEval = 0

        # iterates through ants and scores positioning
        for ant in my_inv.ants:

            # scores queen
            if ant.type == QUEEN:

                qEval = 50.0

                # if queen is on anthill, tunnel, or food it's bad
                if ant.coords == ah_coords or ant.coords == t_coords or ant.coords == food_coords[0] or ant.coords == \
                        food_coords[1]:
                    qEval -= 10

                # if queen is out of rows 0 or 1 it's bad
                if ant.coords[0] > 1:
                    qEval -= 10

                # the father from enemy ants, the better
                qEval -= 2 * self.get_closest_enemy_dist(ant.coords, enemy_inv.ants)

            # scores worker to incentivize food gathering
            elif ant.type == WORKER:

                # tallies up workers
                worker_count += 1

                # if carrying, the closer to the anthill or tunnel, the better
                if ant.carrying:

                    wEval.append(100.0)

                    # distance to anthill
                    ah_dist = approxDist(ant.coords, ah_coords)

                    # distance to tunnel
                    t_dist = approxDist(ant.coords, t_coords)

                    # finds closest and scores
                    # if ant.coords == ah_coords or ant.coords == t_coords:
                    # print("PHill")
                    # eval += 100000000
                    if t_dist < ah_dist:
                        wEval[worker_count - 1] -= 5 * t_dist
                    else:
                        wEval[worker_count - 1] -= 5 * ah_dist

                # if not carrying, the closer to food, the better
                else:

                    wEval.append(80.0)

                    # distance to foods
                    f1_dist = approxDist(ant.coords, food_coords[0])
                    f2_dist = approxDist(ant.coords, food_coords[1])

                    # finds closest and scores
                    # if ant.coords == food_coords[0] or ant.coords == food_coords[1]:
                    # print("PFood")
                    # eval += 500

                    if f1_dist < f2_dist:
                        wEval[worker_count - 1] -= 5 * f1_dist
                    else:
                        wEval[worker_count - 1] -= 5 * f2_dist

                        # the father from enemy ants, the better
                        # eval += -5 + self.get_closest_enemy_dist(ant.coords, enemy_inv.ants)

            # scores soldiers to incentivize the disruption of the enemy economy
            else:

                # tallies up soldiers
                drone_count += 1

                dEval.append(50.0)

                nearest_enemy_worker_dist = self.get_closest_enemy_worker_dist(ant.coords, enemy_inv.ants)

                # if there is an enemy worker
                if not nearest_enemy_worker_dist == 100:
                    dEval[drone_count - 1] -= 5 * nearest_enemy_worker_dist

                # if there isn't an enemy worker, go to the food
                else:
                    dEval[drone_count - 1] -= 5 * self.get_closest_enemy_food_dist(ant.coords, enemy_food_coords)

        # scores other important things

        # state eval
        sEval = 0

        # assesses worker inventory
        if worker_count == 2:
            sEval += 50
        elif worker_count < 2:
            sEval -= 10
        elif worker_count > 2:
            eval_num = 0.00001
            return eval_num

        # assesses drone inventory
        if drone_count == 2:
            sEval += 50
        elif drone_count < 2:
            sEval -= 10
        elif drone_count > 2:
            eval_num = 0.00001
            return eval_num

        # assesses food
        if not sEval == 0:
            sEval += 20 * my_inv.foodCount

        # a temporary variable
        temp = 0

        # scores workers
        for val in wEval:
            temp += val
        if worker_count == 0:
            wEvalAv = 0
        else:
            wEvalAv = temp / worker_count

        temp = 0

        # scores drones
        for val in dEval:
            temp += val

        if not drone_count == 0:
            dEvalAv = temp / drone_count
        else:
            dEvalAv = 0

        # total possible score
        total_possible = 100.0 + 50.0 + 50.0 + 300.0

        # scores total evaluation and returns
        eval = (qEval + wEvalAv + dEvalAv + sEval) / total_possible
        if eval <= 0:
            eval = 0.00002

        return eval

    ##
    # merge_sort
    #
    # useful for sorting the move list from least to greatest in nlog(n) time
    ##
    def mergeSort(self, alist):
        if len(alist) > 1:
            mid = len(alist) // 2
            lefthalf = alist[:mid]
            righthalf = alist[mid:]

            self.mergeSort(lefthalf)
            self.mergeSort(righthalf)

            i = 0
            j = 0
            k = 0
            while i < len(lefthalf) and j < len(righthalf):
                if lefthalf[i][2] < righthalf[j][2]:
                    alist[k] = lefthalf[i]
                    i = i + 1
                else:
                    alist[k] = righthalf[j]
                    j = j + 1
                k = k + 1

            while i < len(lefthalf):
                alist[k] = lefthalf[i]
                i = i + 1
                k = k + 1

            while j < len(righthalf):
                alist[k] = righthalf[j]
                j = j + 1
                k = k + 1

    ##
    # assignRandomWeights
    # Description: All the weights are given random values
    #
    # Parameters: nada
    #
    # Returns: void
    ##
    def assignRandomWeights(self):
        numOfWeights = (self.numOfNodes) * (self.numOfNodes)
        print "num of weights: ", numOfWeights
        for i in range(numOfWeights):
            self.weights.append(random.uniform(0.0, 1.0))

    ##
    # thresholdFunc
    # Description: This method sets the threshold function (or the 'g' function)
    # that is used in the neural network to see if the neuron fires/activates or not.
    #
    # Parameters:
    #   input - the sum of the inputs the neuron receives to apply to the threshold
    #           function
    #
    # Returns: either the output of the threshold function or its derivative
    ##
    def thresholdFunc(self, input, derivative=False):
        # If we are looking for the delta or the slope
        if derivative:
            return input * (1.0 - input)
        # Regular threshold function to find output of node
        else:
            return 1.0 / (1.0 + math.exp(-input))

    ##
    # processNetwork
    # Description: Calculates the output of the current neural network
    #
    # Parameters:
    #   inputs - list of inputs brought into the neural network
    #
    # Return:
    ##
    def processNetwork(self, inputs):
        # Makes an array big enough to hold the outputs of the hidden nodes and output
        # hidden nodes is indexes 0 - (self.numOfNodes - 1)
        # output is the last index
        nodeValues = []
        for i in range(self.numOfNodes):
            nodeValues.append(0)

        weightIndex = 0
        # Get the weights of the hidden nodes and the output
        while weightIndex < self.numOfNodes:
            nodeValues[weightIndex] = self.weights[weightIndex]
            weightIndex += 1

        # Calculate the values of the nodes based on the inputs and their weights
        # Go through all the inputs
        for inputIndex in range(len(inputs)):
            # Go through all the nodes
            for hiddenIndex in range(self.numOfNodes - 1):
                nodeValues[hiddenIndex] += inputs[inputIndex] * self.weights[weightIndex]
                weightIndex += 1

        # Place the resulting node output values into the threshold function
        for j in range(self.numOfNodes - 1):
            nodeValues[j] = self.thresholdFunc(nodeValues[j])

        # Find the sum of the nodes' outputs to help find the output of network
        for hiddenWeightIndex in range(self.numOfNodes - 1):
            nodeValues[self.numOfNodes - 1] += nodeValues[hiddenWeightIndex] * self.weights[weightIndex]

        # Place the sum of the hidden nodes into the threshold function to see the final output
        nodeValues[self.numOfNodes - 1] = self.thresholdFunc(nodeValues[self.numOfNodes - 1])

        # Returns list of the outputs of the hidden and output nodes
        return nodeValues

    ##
    # backPropagate
    # Description: Goes backward through the neural network to find the error from the desired
    # outputs and changes the weights to minimize the outputs from the desired goal
    #
    # Parameters:
    #   inputs - list of inputs brought into the neural network
    #   currentOutput - list of outputs from the nodes including output of network
    #
    # Return:
    ##
    def backPropagate(self, inputs, desiredOutput, currentOutput):
        # Number of inputs for the network
        numOfInputs = len(inputs)
        # Number of weights for the network (inputs, node biases, hidden node outputs, and output bias)
        numOfWeights = len(self.weights)
        # Number of inputs into the hidden nodes
        numOfInputWeights = (self.numOfNodes - 1) * (numOfInputs - 1) + self.numOfNodes
        # Arrays to hold errors and deltas for each of the nodes
        errorOfHiddenNodes = []
        deltaOfHiddenNodes = []

        # error = target - actual
        errorOfOutput = desiredOutput - currentOutput[self.numOfNodes - 1]
        # get the delta or the derivative of the threshold function
        deltaOfOutput = self.thresholdFunc(inputs[self.numOfNodes - 1], derivative=True)
        # deltaValue = currentOutput[self.numOfNodes - 1] *\
        #     (1 - currentOutput[self.numOfNodes - 1]) * errorOfOutput

        # Places enough spots in arrays to hold the errors and deltas for each node
        for i in range(self.numOfNodes - 1):
            errorOfHiddenNodes.append(0)
            deltaOfHiddenNodes.append(0)

        # Calculate the deltas and errors of the hidden nodes and not the output of network
        for j in range(self.numOfNodes - 1):
            # Add the delta for the output last to the array of deltas
            if j == self.numOfNodes - 1:
                deltaOfHiddenNodes.append(deltaOfOutput)
            else:
                # Find the error of the current node
                errorOfHiddenNodes[j] = self.weights[j + numOfInputWeights + 1] * deltaOfOutput
                # Find the delta based on the error
                deltaOfHiddenNodes[j] = self.thresholdFunc(currentOutput[j]) * errorOfHiddenNodes[j]

        # Go through all the weights in the network
        for currentWeightIndex in range(numOfWeights - 1):
            # Check to see which node the weight belongs to
            # (if in the part of weights that belong to inputs into the nodes)
            if currentWeightIndex < self.numOfNodes:
                currentNodeIndex = currentWeightIndex % self.numOfNodes
            elif currentWeightIndex > self.numOfNodes:




# Unit Tests

##
#  Test 1 (2 inputs, 2 hidden nodes, 1 output)
##
print "*** Test Case 1 ***",
player = AIPlayer(0)
player.numOfNodes = 3 #(2 hidden, 1 output)
player.learningRate = 0.2
# 9 weights total (2 input, 2 bias, 2 output of hidden, 1 output)
player.weights = [0.1, 0.2, 0.4, 0.5, 0.8,
                  0.66, 0.3, 0.14, 0.22]
# 2 inputs to test network
inputs = [1, 1]
# output that we should receive
desiredOutput = 0.2
# current output that we are getting
currentOutput = player.processNetwork(inputs)
# error between the desired output and the current output
# error = target - actual
error = desiredOutput - currentOutput[player.numOfNodes - 1]
# Check to see if the error is small enough to stop
if (-0.1 < error) and (error < 0.1):
    print "error is in acceptable limits",
else:
    print "network still needs to learn",
    print "current error: ", error
#edit the weights by back propagating through the network
#player.backPropagate(inputs, desiredOutput, currentOutput[player.numOfNodes - 1])

# ##
# #  Test 2 (8 inputs, 8 hidden nodes, 1 output)
# ##
# print "***Test Case 2 ***",
# player = AIPlayer(0)
# player.numOfNodes = 9 #(8 hidden, 1 output)
# player.learningRate = 0.5
# # 81 weights total (64 input, 8 bias, 8 output of hidden, 1 output)
# player.weights = [0.1, 0.2, 0.4, 0.5, 0.8, 0.66, 0.3, 0.14, 0.6,
#                   0.1, 0.2, 0.4, 0.5, 0.8, 0.66, 0.3, 0.14, 0.7,
#                   0.1, 0.2, 0.4, 0.5, 0.8, 0.66, 0.3, 0.14, 0.7,
#                   0.1, 0.2, 0.4, 0.5, 0.8, 0.66, 0.3, 0.14, 0.7,
#                   0.1, 0.2, 0.4, 0.5, 0.8, 0.66, 0.3, 0.14, 0.7,
#                   0.1, 0.2, 0.4, 0.5, 0.8, 0.66, 0.3, 0.14, 0.7,
#                   0.1, 0.2, 0.4, 0.5, 0.8, 0.66, 0.3, 0.14, 0.7,
#                   0.22]
# # 8 inputs to test network
# inputs = [1, 1, 0, 0, 1, 0, 1, 1]
# # output that we should receive
# desiredOutput = 0.2
# # current output that we are getting
# currentOutput = player.processNetwork(inputs)
# # error between the desired output and the current output
# # error = target - actual
# error = desiredOutput - currentOutput[player.numOfNodes - 1]
# # Check to see if the error is small enough to stop
# if (-0.1 < error) and (error < 0.1):
#     print "error is in acceptable limits",
# else:
#     print "network still needs to learn",
# # edit the weights by back propagating through the network
# player.backPropagate(inputs, desiredOutput, currentOutput[player.numOfNodes - 1])
