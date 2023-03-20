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


import random
import util
import math

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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #Set up stating score from previous state
        score = successorGameState.getScore()


        foodDistances = []
        for food in newFood.asList():
            #Get manhattanDistance between new position and every food
            foodDistances.append(util.manhattanDistance(newPos, food))
        if len(foodDistances):
            # if foodDistances not empty
            # food score is the mean of food distance from new position
            foodScore = sum(foodDistances)/len(foodDistances)
        else: 
            foodScore = 1

        ghostDistances = []
        for ghost in newGhostStates:
            #Get manhattanDistance between new position and every ghost
            ghostDistances.append(util.manhattanDistance(newPos, ghost.getPosition()))

        # The closer is the ghost, the more its dangerous
        ghostScore = min(ghostDistances)
        #If too close, penalized by reducing the score
        if ghostScore <= 3:
            ghostScore = -1

        # The distance from the nearest ghost is inversely proportional 
        #   to the mean of food distance
        # For instance if a ghost is near but there are a lot of food around
        #   It is worth staying around
        score += ghostScore/foodScore
        return score

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

        "*** YOUR CODE HERE ***"     
        def baseCase(gameState, depth):
                # Base Case - Game is Over or reached max depth
                # Stop recursion if any is true
                return gameState.isWin() or gameState.isLose() or (depth == self.depth)  

        def minimax(gameState, agentIndex, depth):
            if baseCase(gameState, depth):
                return self.evaluationFunction(gameState), ''
                
            elif agentIndex == 0:
                # Want to maxize Pacman (evalScore the hight the better)
                node = maxValue(gameState, agentIndex, depth)
            else:
                #Minimize the ghosts actions
                node = minValue(gameState, agentIndex, depth)
            return node

        def minValue(gameState, agentIndex, depth):
            val = math.inf
            act = ''

            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1:
                    # Recursed in All agent. Back to pacman in new depth
                    node = minimax(gameState.generateSuccessor(agentIndex, action), 0, depth + 1)
                else:
                    # Recurse in next ghost 
                    node = minimax(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)

                # Find action that minimize val
                if val > node[0]:
                    val = node[0]
                    act = action
            return val, act

        def maxValue(gameState, agentIndex, depth):
            val = -math.inf
            act = ''

            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1:
                    # Recursed in All agent. Back to pacman in new depth
                    node = minimax(gameState.generateSuccessor(agentIndex, action), 0, depth + 1)
                else:
                    # Recurse in next ghost 
                    node = minimax(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)

                # Find action that maximize val
                if val < node[0]:
                    val = node[0]
                    act = action
            return val, act
            
        return minimax(gameState, agentIndex=0, depth=0)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def baseCase(gameState, depth):
            # Base Case - Game is Over or reached max depth
            # Stop recursion if any is true
            return gameState.isWin() or gameState.isLose() or (depth == self.depth)  

        def alphaBeta(gameState, agentIndex, depth, alpha, beta):
            if baseCase(gameState, depth):
                return self.evaluationFunction(gameState), ''
                
            elif agentIndex == 0:
                # Want to maxize Pacman (evalScore the hight the better)
                node = maxValue(gameState, agentIndex, depth, alpha, beta)
            else:
                #Minimize the ghosts actions
                node = minValue(gameState, agentIndex, depth, alpha, beta)
            return node
       
        def minValue(gameState, agentIndex, depth, alpha, beta):
            val = math.inf
            act = ''

            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1:
                    # Recursed in All agent. Back to pacman in new depth
                    node = alphaBeta(gameState.generateSuccessor(agentIndex, action), 0, depth + 1, alpha, beta)
                else:
                    # Recurse in next ghost 
                    node = alphaBeta(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth, alpha, beta)

                # Find action that minimize val
                if val > node[0]:
                    val = node[0]
                    act = action
                
                if node[0] < alpha:
                    # Prunning
                    return val, act
                beta = min(node[0], beta)

            return val, act

        def maxValue(gameState, agentIndex, depth, alpha, beta):
            val = -math.inf
            act = ''

            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1:
                    # Recursed in All agent. Back to pacman in new depth
                    node = alphaBeta(gameState.generateSuccessor(agentIndex, action), 0, depth + 1, alpha, beta)
                else:
                    # Recurse in next ghost 
                    node = alphaBeta(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth, alpha, beta)

                # Find action that maximize val
                if val < node[0]:
                    val = node[0]
                    act = action
                
                if node[0] > beta:
                    # Prunning
                    return val, act 
                alpha = max(node[0], alpha)
            return val, act          
        
        return alphaBeta(gameState, agentIndex=0, depth=0,alpha=-math.inf, beta=math.inf)[1]



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
        def baseCase(gameState, depth):
            # Base Case - Game is Over or reached max depth
            # Stop recursion if any is true
            return gameState.isWin() or gameState.isLose() or (depth == self.depth)  

        def expectiMax(gameState, agentIndex, depth):
            if baseCase(gameState, depth):
                return self.evaluationFunction(gameState), ''
                
            elif agentIndex == 0:
                # Want to maxize Pacman (evalScore the hight the better)
                node = maxValue(gameState, agentIndex, depth)
            else:
                # specificity of expectiminimax
                node = helper(gameState, agentIndex, depth)
            return node

        def helper(gameState, agentIndex, depth):
            prob = 0
            fraction = 1/len(gameState.getLegalActions(agentIndex))
            act = ''

            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1:
                    # Recursed in All agent. Back to pacman in new depth
                    prob += fraction * expectiMax(gameState.generateSuccessor(agentIndex, action), 0, depth + 1)[0]
                else:
                    # Recurse in next ghost 
                    prob += fraction * expectiMax(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)[0]

            return prob, act

        def maxValue(gameState, agentIndex, depth):
            val = -math.inf
            act = ''

            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1:
                    # Recursed in All agent. Back to pacman in new depth
                    node = expectiMax(gameState.generateSuccessor(agentIndex, action), 0, depth + 1)
                else:
                    # Recurse in next ghost 
                    node = expectiMax(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)

                # Find action that maximize val
                if val < node[0]:
                    val = node[0]
                    act = action
            return val, act
            
        return expectiMax(gameState, agentIndex=0, depth=0)[1]  


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Same base algo as the first eval function
                    added capules into account to calculate eval score
                        Since eating shorst is +300
                 See comments below for explaination
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newCapsule = currentGameState.getCapsules()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    #Set up stating score from previous state
    score = currentGameState.getScore()
    # Initiate Capsule score to 0
    capsuleScore = 0

    foodDistances = []
    for food in newFood.asList():
        #Get manhattanDistance between new position and every food
        foodDistances.append(util.manhattanDistance(newPos, food))
    if len(foodDistances):
        # if foodDistances not empty
        # food score is the mean of food distance from new position
        foodScore = sum(foodDistances)/len(foodDistances)
        # Add the closest food 
        #   If ghort is where mean food is, Pacman wont move
        #   This helps solving pacman getting stuck
        foodScore += min(foodDistances)
    else: 
        # So we dont divide by 0
        foodScore = 1

    ghostDistances = []
    for ghost in newGhostStates:
        #Get manhattanDistance between new position and every ghost
        ghostDistances.append(util.manhattanDistance(newPos, ghost.getPosition()))

    # The closer is the ghost, the more its dangerous
    ghostScore = min(ghostDistances)

    # If ghost eatable
    if newScaredTimes[0] != 0:
        # If ghost not to far
        if ghostScore <= 9:
            # Reward by going toward ghost
            # Eating  ghost = +300 in game score
            ghostScore += 1
    # If not ghost eatable
    else:
        if ghostScore <= 3:
            #If too close, penalized by reducing the score
            ghostScore += -1
        # encourage going toward closest capsule
        if newCapsule:
            capsuleScore += sum(min(newCapsule))

    # The distance from the nearest ghost is inversely proportional 
    #   to the mean of food distance
    # For instance if a ghost is near but there are a lot of food around
    #   It is worth staying around

    # Did a bit of tuning with formula to obtimaze game score
    score += ((ghostScore*0.5)/(foodScore*4 + capsuleScore**2)**2)
    return score


# Abbreviation
better = betterEvaluationFunction
