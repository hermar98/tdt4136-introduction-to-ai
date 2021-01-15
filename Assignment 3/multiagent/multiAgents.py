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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        return successorGameState.getScore()


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

        # Defining current depth and legal moves
        currentDepth = 0
        legalMoves = gameState.getLegalActions(0)

        # Calculating scores for all legal moves using the minValue-function
        scores = [self.minValue(gameState.generateSuccessor(0, action), 1, currentDepth) for action in legalMoves]

        # Selecting the best score
        bestScore = max(scores)

        # Finding index value(s) for the best score(s)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]

        # If there are multiple actions with best score, we choose a random one
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

    # Function for MAX agent. Takes a state, agentIndex and the current depth and returns a utility value (score)
    def maxValue(self, gameState, agentIndex, currentDepth):
        # If we are in a terminal state (win, loose or reached specified depth) we use the evaluation function to
        # calculate the utility value (score).
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(gameState)

        # Defining initial utility value (score) and legal moves
        score = float("-inf")
        legalMoves = gameState.getLegalActions(0)

        # Calling minValue-function for all legal moves to find the maximum score
        for action in legalMoves:
            score = max(score, self.minValue(gameState.generateSuccessor(agentIndex, action), 1, currentDepth))

        return score

    # Function for MIN agent. Takes a state, agentIndex and the current depth and returns a utility value (score)
    def minValue(self, gameState, agentIndex, currentDepth):
        # If we are in a terminal state (win or loose) we use the evaluation function to calculate the
        # utility value (score).
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # Defining initial utility value (score) and legal moves
        score = float("inf")
        legalMoves = gameState.getLegalActions(agentIndex)

        # Different function call depending on whether this is the last ghost or not
        if agentIndex < gameState.getNumAgents() - 1:
            # Call minValue-function for all legal moves to find the minimum score (increase agentIndex)
            for action in legalMoves:
                score = min(score, self.minValue(gameState.generateSuccessor(agentIndex, action), agentIndex + 1,
                                                 currentDepth))
        else:
            # Call maxValue-function for all legal moves to find the minimum score (increase depth)
            for action in legalMoves:
                score = min(score, self.maxValue(gameState.generateSuccessor(agentIndex, action), 0, currentDepth + 1))

        return score


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        # Defining current depth and legal moves
        currentDepth = 0
        legalMoves = gameState.getLegalActions(0)

        # Setting initial alpha, beta and score values
        alpha = float("-inf")
        beta = float("inf")
        score = float("-inf")

        bestAction = None

        # Calculating scores for all legal moves using the minValue-function
        for action in legalMoves:
            newScore = self.minValue(gameState.generateSuccessor(0, action), 1, currentDepth, alpha, beta)

            # Updating score and best action if new max score is found
            if newScore > score:
                score = newScore
                bestAction = action

            # Returning current action if score is bigger than beta
            if score > beta:
                return action

            # Updating value of alpha
            alpha = max(alpha, score)

        return bestAction

    # Function for MAX agent. Takes a state, agentIndex, current depth, alpha value and beta value, and returns a
    # utility value (score).
    def maxValue(self, gameState, agentIndex, currentDepth, alpha, beta):
        # If we are in a terminal state (win, loose or reached specified depth) we use the evaluation function to
        # calculate the utility value (score).
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(gameState)

        # Defining initial utility value (score) and legal moves
        score = float("-inf")
        legalMoves = gameState.getLegalActions(agentIndex)

        # Calling minValue-function for all legal moves to find the maximum score
        for action in legalMoves:
            score = max(score, self.minValue(gameState.generateSuccessor(0, action), 1, currentDepth, alpha, beta))

            # Return score if bigger than beta
            if score > beta:
                return score

            # Updating value of alpha
            alpha = max(alpha, score)

        return score

    # Function for MIN agent. Takes a state, agentIndex, current depth, alpha value and beta value, and returns a
    # utility value (score).
    def minValue(self, gameState, agentIndex, currentDepth, alpha, beta):
        # If we are in a terminal state (win or loose) we use the evaluation function to calculate the
        # utility value (score).
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # Defining initial utility value (score) and legal moves
        score = float("inf")
        legalMoves = gameState.getLegalActions(agentIndex)

        # Different function call depending on whether we are the last ghost or not
        if agentIndex < gameState.getNumAgents() - 1:
            # Call minValue-function for all legal moves to find the minimum score (increase agentIndex)
            for action in legalMoves:
                score = min(score,
                            self.minValue(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, currentDepth,
                                          alpha, beta))

                # Return score if smaller than alpha
                if score < alpha:
                    return score

                # Update value of beta
                beta = min(beta, score)
        else:
            # Call maxValue-function for all legal moves to find the minimum score (increase depth)
            for action in legalMoves:
                score = min(score,
                            self.maxValue(gameState.generateSuccessor(agentIndex, action), 0, currentDepth + 1,
                                          alpha, beta))

                # Return score if smaller than alpha
                if score < alpha:
                    return score

                # Update value of beta
                beta = min(beta, score)

        return score


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
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
