# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        currentFood = currentGameState.getFood()
        #newGhostStates = successorGameState.getGhostStates()
        #newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #print(successorGameState)
        #print(newPos)
        #print(newFood)
        #print(newGhostStates)
        #print(newScaredTimes)
        "*** YOUR CODE HERE ***"
        ghostpos = successorGameState.getGhostPositions()
        foodpos = currentFood.asList()
        fooddistance = []
        for food in foodpos:
            fooddistance.append(util.manhattanDistance(food, newPos))
        if action == 'Stop':
            return -float("inf")
        for ghost in ghostpos:
            if ghost == tuple(newPos):
                return -float("inf")
        return -min(fooddistance)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)

    your minimax tree will have multiple min layers (one for each ghost) for every max layer.
    expand the game tree to an arbitrary depth
    access to self.depth and self.evalutationFunction
    make sure to use the appropriate time to expand GameState.generateSuccessor

    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        """
        if the state is a terminal state: return the state's utility
        if the next state is MAX: return maxvalue
        if the next state is MIN: return minvalue
        """
        "*** YOUR CODE HERE ***"

        #agentnum = gameState.getNumAgents()

        def PacmanGhost(gameState, agentIndex, depth):
            if agentIndex >= gameState.getNumAgents():
                agentIndex = 0
                depth = depth +1
            if gameState.isWin() or gameState.isLose() or depth ==self.depth:
                return [self.evaluationFunction(gameState), ""]
            return PGAction(gameState, agentIndex, depth)

        def PGAction(gameState, agentindex, depth):
            if agentindex == 0:
                value = float('-inf')
            else:
                value = float('inf')
            goodaction = ""
            for action in gameState.getLegalActions(agentindex):
                successor = gameState.generateSuccessor(agentindex, action)
                childvalueaction = PacmanGhost(successor, agentindex + 1, depth)
                successorvalue = childvalueaction[0]
                if agentindex == 0:
                    value0 = max(value, successorvalue)
                    if value0 != value:
                        goodaction = action
                    value = value0
                else:
                    value0 = min(value, successorvalue)
                    if value0 != value:
                        goodaction = action
                    value = value0
            return value, goodaction
        return PacmanGhost(gameState, 0, 0)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def PacmanGhost(gameState, agentIndex, depth, alpha, beta):
            if agentIndex >= gameState.getNumAgents():
                agentIndex = 0
                depth = depth +1
            if gameState.isWin() or gameState.isLose() or depth ==self.depth:
                return [self.evaluationFunction(gameState), "", float('-inf'), float('inf')]
            return PGAction(gameState, agentIndex, depth, alpha, beta)

        def PGAction(gameState, agentindex, depth, alpha, beta):
            if agentindex == 0:
                value = float('-inf')
            else:
                value = float('inf')
            goodaction = ""
            for action in gameState.getLegalActions(agentindex):
                successor = gameState.generateSuccessor(agentindex, action)
                childvalueaction = PacmanGhost(successor, agentindex + 1, depth, alpha, beta)
                successorvalue = childvalueaction[0]
                if agentindex == 0:
                    value0 = max(value, successorvalue)
                    if value0 > beta:
                        #value = value0
                        #goodaction = action
                        return value0, action, alpha, beta
                    if value0 != value:
                        goodaction = action
                    value = value0
                    alpha = max(alpha, value)
                else:
                    value0 = min(value, successorvalue)
                    if value0 < alpha:
                        #value = value0
                        #goodaction = action
                        return value0, action, alpha, beta
                    if value != value0:
                        goodaction = action
                    value = value0
                    beta = min(beta, value)
            return value, goodaction, alpha, beta
        return PacmanGhost(gameState, 0, 0, float('-inf'), float('inf'))[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def PacmanGhost(gameState, agentIndex, depth):
            if agentIndex >= gameState.getNumAgents():
                agentIndex = 0
                depth = depth +1
            if gameState.isWin() or gameState.isLose() or depth ==self.depth:
                return [self.evaluationFunction(gameState), ""]
            return PGAction(gameState, agentIndex, depth)

        def PGAction(gameState, agentindex, depth):
            if agentindex == 0:
                value = float('-inf')
            else:
                value = 0
            goodaction = ""
            probability = float( 1 / len( gameState.getLegalActions(agentindex) ) )
            for action in gameState.getLegalActions(agentindex):
                successor = gameState.generateSuccessor(agentindex, action)
                childvalueaction = PacmanGhost(successor, agentindex + 1, depth)
                successorvalue = childvalueaction[0]
                if agentindex == 0:
                    value0 = max(value, successorvalue)
                    if value0 != value:
                        goodaction = action
                    value = value0
                else:
                    #value0 = min(value, successorvalue)
                    #if value0 != value:
                    #    goodaction = action
                    #value = value0
                    value = value + probability * successorvalue
            return value, goodaction
        return PacmanGhost(gameState, 0, 0)[1]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """

    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')

    foodpos = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    score = currentGameState.getScore()
    pacmanpos = currentGameState.getPacmanPosition()
    ghoststates = currentGameState.getGhostStates()

    #the least distance to one food
    disfood = []
    for food in foodpos:
       disfood.append(util.manhattanDistance(food, pacmanpos))

    disghost = []
    numghost = len(ghoststates)
    for ghost in ghoststates:
        if ghost.scaredTimer <= 0:
            numghost = numghost - 1
            #disghost.append(0)
        else:
            disghost.append(util.manhattanDistance(ghost.getPosition(), pacmanpos))

    if len(disfood)==0 and len(disghost)==0:
        return currentGameState.getScore()
    foodfeature = -min(disfood)-20*len(disfood)
    if len(disghost)==0:
        return foodfeature + score
    ghostfeature = -20*numghost-min(disghost)

    return score + foodfeature + ghostfeature + 100*len(capsules)

# Abbreviation
better = betterEvaluationFunction
