"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


class SearchNode:
    def __init__(self, action, result, value):
        self.action = action
        self.result = result
        self.value = value


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    count_x = 0
    count_o = 0
    for list in board:
        count_x += list.count(X)
        count_o += list.count(O)
    if count_x > count_o:
        return O
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()
    if terminal(board):
        return set((0, 0))
    for i, list in enumerate(board):
        for j, element in enumerate(list):
            if element == EMPTY:
                actions.add((i, j))
    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i, j = action
    if i < 0 or i >= 3 or j < 0 or j >= 3 or board[i][j] != EMPTY:
        raise TypeError("Invalid move")

    new_board = copy.deepcopy(board)
    new_board[i][j] = player(board)
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    if board[0][0] != EMPTY and all(x == board[0][0] for x in board[0]):
        return board[0][0]
    if board[1][0] != EMPTY and all(x == board[1][0] for x in board[1]):
        return board[1][0]
    if board[2][0] != EMPTY and all(x == board[2][0] for x in board[2]):
        return board[2][0]

    if board[0][0] != EMPTY and all(x == board[0][0] for x in [board[0][0], board[1][0], board[2][0]]):
        return board[0][0]
    if board[0][1] != EMPTY and all(x == board[0][1] for x in [board[0][1], board[1][1], board[2][1]]):
        return board[0][1]
    if board[0][2] != EMPTY and all(x == board[0][2] for x in [board[0][2], board[1][2], board[2][2]]):
        return board[0][2]

    if board[0][0] != EMPTY and all(x == board[0][0] for x in [board[0][0], board[1][1], board[2][2]]):
        return board[0][0]
    if board[0][2] != EMPTY and all(x == board[0][2] for x in [board[0][2], board[1][1], board[2][0]]):
        return board[0][2]
    
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True
    
    for list in board:
        if EMPTY in list:
            return False
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    win = winner(board)
    if win == X:
        return 1
    elif win == O:
        return -1
    else:
        return 0
    

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    if board == initial_state():
        return (0, 0)

    if player(board) == X:
        return max_node(board).action
    else:
        return min_node(board).action


def max_node(board):
    if terminal(board):
        return SearchNode(None, board, utility(board))
    node = SearchNode((0, 0), initial_state(), -math.inf)
    for action in actions(board):
        result_node = min_node(result(board, action))
        if result_node.value > node.value:
            node = SearchNode(action, result_node.result, result_node.value)
    return node


def min_node(board):
    if terminal(board):
        return SearchNode(None, board, utility(board))
    node = SearchNode((0, 0), initial_state(), math.inf)
    for action in actions(board):
        result_node = max_node(result(board, action))
        if result_node.value < node.value:
            node = SearchNode(action, result_node.result, result_node.value)
    return node