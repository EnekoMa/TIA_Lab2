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
import sys # Importado para usar un valor de infinito

import util
from game import Agent
from util import manhattanDistance


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


        return successorGameState.getScore()#Tendr�is que comentar esta linea y devolver el valor que calculeis


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
        super().__init__()
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        
        # Llama a la función principal de Minimax (maxValue, ya que Pacman es el agente 0)
        # Retorna el par (score, action). Solo necesitamos la acción.
        _, action = self.maxValue(game_state, 0, 0)
        return action
    
    # Función de utilidad para alternar entre MAX y MIN.
    def value(self, gameState, agentIndex, currentDepth):
        # Si Pacman (agente 0), maximiza.
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, currentDepth)
        # Si un fantasma (agente > 0), minimiza.
        else:
            return self.minValue(gameState, agentIndex, currentDepth)

    # Nodos MAX: Pacman (agente 0)
    def maxValue(self, gameState, agentIndex, currentDepth):
        # Caso base: estado terminal (win/lose) o límite de profundidad
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            # Retorna (score, action=None)
            return (self.evaluationFunction(gameState), None)

        v = -sys.maxsize  # Inicializa con -infinito
        bestAction = None
        
        # Iterar sobre las acciones legales
        for action in gameState.getLegalActions(agentIndex):
            # Generar el estado sucesor
            successorState = gameState.generateSuccessor(agentIndex, action)
            
            # Llamar a la siguiente capa del árbol: el primer fantasma (agentIndex + 1)
            # La profundidad se mantiene en el mismo "ciclo" de agentes
            score, _ = self.value(successorState, agentIndex + 1, currentDepth)

            # Actualizar si se encuentra un valor mayor
            if score > v:
                v = score
                bestAction = action
        
        return (v, bestAction)

    # Nodos MIN: Fantasmas (agentes > 0)
    def minValue(self, gameState, agentIndex, currentDepth):
        # Caso base: estado terminal (win/lose) o límite de profundidad
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            # Retorna (score, action=None)
            return (self.evaluationFunction(gameState), None)
        
        v = sys.maxsize  # Inicializa con +infinito
        bestAction = None
        
        # Iterar sobre las acciones legales
        for action in gameState.getLegalActions(agentIndex):
            # Generar el estado sucesor
            successorState = gameState.generateSuccessor(agentIndex, action)
            
            # Determinar el siguiente agente y profundidad
            nextAgentIndex = agentIndex + 1
            nextDepth = currentDepth
            
            # Si el agente actual es el ÚLTIMO fantasma, pasamos a Pacman y AUMENTAMOS la profundidad
            if nextAgentIndex == gameState.getNumAgents():
                nextAgentIndex = 0 # Pacman
                nextDepth += 1     # Siguiente nivel (incremento de profundidad)

            # Llamar a la siguiente capa del árbol
            score, _ = self.value(successorState, nextAgentIndex, nextDepth)

            # Actualizar si se encuentra un valor menor
            if score < v:
                v = score
                bestAction = action
        
        return (v, bestAction)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()



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
        # Expectimax sigue la misma estructura, comenzando con el nodo MAX (Pacman)
        # Retorna el par (score, action). Solo necesitamos la acción.
        _, action = self.maxValue(gameState, 0, 0)
        return action
    
    # Función de utilidad (despachador) para alternar entre MAX y EXPECTED
    def value(self, gameState, agentIndex, currentDepth):
        # Si Pacman (agente 0), maximiza.
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, currentDepth)
        # Si un fantasma (agente > 0), calcula el valor esperado.
        else:
            return self.expectedValue(gameState, agentIndex, currentDepth)

    # Nodos MAX: Pacman (agente 0)
    def maxValue(self, gameState, agentIndex, currentDepth):
        # Caso base: estado terminal (win/lose) o límite de profundidad
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            # Retorna (score, action=None)
            return (self.evaluationFunction(gameState), None)

        v = -sys.maxsize  # Inicializa con -infinito
        bestAction = None
        
        # Iterar sobre las acciones legales
        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            
            # Llama a la siguiente capa (expectedValue para el primer fantasma)
            score, _ = self.value(successorState, agentIndex + 1, currentDepth)

            # Actualizar si se encuentra un valor mayor
            if score > v:
                v = score
                bestAction = action
        
        return (v, bestAction)

    # Nodos CHANCE/EXPECTED: Fantasmas (agentes > 0)
    def expectedValue(self, gameState, agentIndex, currentDepth):
        # Caso base: estado terminal (win/lose) o límite de profundidad
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            return (self.evaluationFunction(gameState), None)
        
        legalMoves = gameState.getLegalActions(agentIndex)
        numMoves = len(legalMoves)
        # Si no hay movimientos legales (debería ser raro en Pacman), retorna la evaluación
        if numMoves == 0:
            return (self.evaluationFunction(gameState), None)
            
        probability = 1.0 / numMoves # Probabilidad uniforme
        expectedScore = 0.0
        
        # Iterar sobre las acciones legales para calcular el promedio ponderado
        for action in legalMoves:
            successorState = gameState.generateSuccessor(agentIndex, action)
            
            # Determinar el siguiente agente y profundidad
            nextAgentIndex = agentIndex + 1
            nextDepth = currentDepth
            
            # Si es el ÚLTIMO fantasma, pasamos a Pacman y AUMENTAMOS la profundidad
            if nextAgentIndex == gameState.getNumAgents():
                nextAgentIndex = 0 # Pacman
                nextDepth += 1     # Siguiente nivel (incremento de profundidad)

            # Llamar a la siguiente capa del árbol
            score, _ = self.value(successorState, nextAgentIndex, nextDepth)

            # Acumular el valor esperado: (Probabilidad * Score del sucesor)
            expectedScore += probability * score
        
        # En el nodo Chance, la acción devuelta es irrelevante (None), ya que no se elige una acción, se promedian.
        return (expectedScore, None)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    pacman_pos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
