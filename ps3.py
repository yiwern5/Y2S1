import copy
from typing import Union, Tuple, Iterable

import utils

Score = Union[int, float]
Move = Tuple[Tuple[int, int], Tuple[int, int]]

def evaluate(board) -> Score:
    bcount = 0
    wcount = 0
    for r, row in enumerate(board):
        for tile in row:
            if tile == 'B':
                if r == 5:
                    return utils.WIN
                bcount += 1
            elif tile == 'W':
                if r == 0:
                    return -utils.WIN
                wcount += 1
    if wcount == 0:
        return utils.WIN
    if bcount == 0:
        return -utils.WIN
    return bcount - wcount

def generate_valid_moves(board) -> Iterable[Move]:
    '''
    Generates a list (or iterable, if you want to) containing 
    all possible moves in a particular position for black.
    '''
    # TODO: Replace this with your own implementation
    moves = []
    for row in range(utils.ROW):
        for col in range(utils.COL):
            if board[row][col] == 'B':
                src = (row, col)
                for dr in range(row+1, min(utils.ROW, row+2)):
                    for dc in range(max(0, col-1), min(utils.COL, col+2)):
                        dst = (dr, dc)
                        if utils.is_valid_move(board, src, dst):
                            moves.append((src, dst))
    return moves

def test_11():
    board1 = [
        ['_', '_', 'B', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_']
    ]
    assert sorted(generate_valid_moves(board1)) == [((0, 2), (1, 1)), ((0, 2), (1, 2)), ((0, 2), (1, 3))]

    board2 = [
        ['_', '_', '_', '_', 'B', '_'],
        ['_', '_', '_', '_', '_', 'B'],
        ['_', 'W', '_', '_', '_', '_'],
        ['_', '_', 'W', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', 'B', '_', '_', '_']
    ]
    assert sorted(generate_valid_moves(board2)) == [((0, 4), (1, 3)), ((0, 4), (1, 4)), ((1, 5), (2, 4)), ((1, 5), (2, 5))]

    board3 = [
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', 'B', '_', '_', '_'],
        ['_', 'W', 'W', 'W', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_']
    ]
    assert sorted(generate_valid_moves(board3)) == [((1, 2), (2, 1)), ((1, 2), (2, 3))]

# test_11()

def minimax(board, depth, max_depth, is_black: bool) -> Tuple[Score, Move]:
    '''
    Finds the best move for the input board state.
    Note that you are black.
    Parameters
    ----------
    board: 2D list of lists. Contains characters 'B', 'W' and '_' 
    representing black pawn, white pawn and empty cell, respectively.
    
    depth: int, the depth to search for the best move. When this is equal
    to `max_depth`, you should get the evaluation of the position using 
    the provided heuristic function.
    max_depth: int, the maximum depth for cutoff.
    
    is_black: bool. True when finding the best move for black, False 
    otherwise.
    Returns
    -------
    A tuple (evalutation, ((src_row, src_col), (dst_row, dst_col))):
    evaluation: the best score that black can achieve after this move.
    src_row, src_col: position of the pawn to move.
    dst_row, dst_col: position to move the pawn to.
    '''
    # TODO: relace with your own implementation
    def max_value(board, depth):
        if depth == max_depth or utils.is_game_over(board):
            return (evaluate(board), None)
            
        v = -utils.INF
        best_move = None
        for move in generate_valid_moves(board):
            src, dst = move
            next_state = utils.invert_board(utils.state_change(board, src, dst, False), False)
            score, _ = minimax(next_state, depth + 1, max_depth, False)
            if score > v:
                v = score
                best_move = move
        return (v, best_move)
    
    def min_value(board, depth):
        if depth == max_depth or utils.is_game_over(board):
            return (evaluate(utils.invert_board(board)), None)
        
        v = utils.INF
        best_move = None
        
        for move in generate_valid_moves(board):
            src, dst = move
            next_state = utils.invert_board(utils.state_change(board, src, dst, False), False)
            score, _ = minimax(next_state, depth+1, max_depth, True)
            if score < v:
                v = score
                best_move = move
        return (v, best_move)
    
    if is_black:
        return max_value(board, depth)
    else:
        return min_value(board, depth)
def test_21():
    board1 = [
        list("______"),
        list("___B__"),
        list("____BB"),
        list("___WB_"),
        list("_B__WW"),
        list("_WW___"),
    ]
    score1, (src1, dst1) = minimax(copy.deepcopy(board1), 0, 1, True)
    assert score1 == utils.WIN, "black should win in 1"
    assert utils.is_valid_move(board1, src1, dst1), "move should be valid"

    board2 = [
        list("______"),
        list("___B__"),
        list("____BB"),
        list("_BW_B_"),
        list("____WW"),
        list("_WW___"),
    ]
    score2, (src2, dst2) = minimax(copy.deepcopy(board2), 0, 3, True)
    assert score2 == utils.WIN, "black should win in 3"
    assert utils.is_valid_move(board2, src2, dst2), "move should be valid"

    board3 = [
        list("______"),
        list("__B___"),
        list("_WWW__"),
        list("______"),
        list("______"),
        list("______"),
    ]
    score3, (src3, dst3) = minimax(copy.deepcopy(board3), 0, 4, True)
    assert score3 == -utils.WIN, "white should win in 4"
    assert utils.is_valid_move(board3, src3, dst3), "move should be valid"

# test_21()

def negamax(board, depth, max_depth) -> Tuple[Score, Move]:
    '''
    Finds the best move for the input board state.
    Note that you are black.
    Parameters
    ----------
    board: 2D list of lists. Contains characters 'B', 'W' and '_' 
    representing black pawn, white pawn and empty cell, respectively.
    
    depth: int, the depth to search for the best move. When this is equal
    to `max_depth`, you should get the evaluation of the position using 
    the provided heuristic function.
    max_depth: int, the maximum depth for cutoff.
    Notice that you no longer need the parameter `is_black`.
    Returns
    -------
    A tuple (evalutation, ((src_row, src_col), (dst_row, dst_col))):
    evaluation: the best score that black can achieve after this move.
    src_row, src_col: position of the pawn to move.
    dst_row, dst_col: position to move the pawn to.
    '''
    # TODO: replace with your own implementation
    if depth == max_depth or utils.is_game_over(board):
        return evaluate(board), None
    best_score = -utils.INF
    best_move = None
    for move in generate_valid_moves(board):
        src, dst = move
        next_state = utils.invert_board(utils.state_change(board, src, dst, False), False)
        score, _ = negamax(next_state, depth + 1, max_depth)
        score = -score
        if score > best_score:
            best_score = score
            best_move = move
    return best_score, best_move

def test_22():
    board1 = [
        list("______"),
        list("___B__"),
        list("____BB"),
        list("___WB_"),
        list("_B__WW"),
        list("_WW___"),
    ]
    score1, (src1, dst1) = negamax(copy.deepcopy(board1), 0, 1)
    assert score1 == utils.WIN, "black should win in 1"
    assert utils.is_valid_move(board1, src1, dst1), "move should be valid"

    board2 = [
        list("______"),
        list("___B__"),
        list("____BB"),
        list("_BW_B_"),
        list("____WW"),
        list("_WW___"),
    ]
    score2, (src2, dst2) = negamax(copy.deepcopy(board2), 0, 3)
    assert score2 == utils.WIN, "black should win in 3"
    assert utils.is_valid_move(board2, src2, dst2), "move should be valid"

    board3 = [
        list("______"),
        list("__B___"),
        list("_WWW__"),
        list("______"),
        list("______"),
        list("______"),
    ]
    score3, (src3, dst3) = negamax(copy.deepcopy(board3), 0, 4)
    assert score3 == -utils.WIN, "white should win in 4"
    assert utils.is_valid_move(board3, src3, dst3), "move should be valid"

# test_22()

def minimax_alpha_beta(board, depth, max_depth, alpha, beta, is_black: bool) -> Tuple[Score, Move]:
    # TODO: Replace this with your own implementation
    def max_value(board, depth, alpha, beta):
        if depth == max_depth or utils.is_game_over(board):
            return (evaluate(board), None)
            
        v = -utils.INF
        best_move = None
        for move in generate_valid_moves(board):
            src, dst = move
            next_state = utils.invert_board(utils.state_change(board, src, dst, False), False)
            score, _ = minimax_alpha_beta(next_state, depth + 1, max_depth, alpha, beta, False)
            if score > v:
                v = score
                best_move = move
            if v >= beta:
                return (v, best_move)
            
            alpha = max(alpha, v)
        return (v, best_move)
    
    def min_value(board, depth, alpha, beta):
        if depth == max_depth or utils.is_game_over(board):
            return (evaluate(utils.invert_board(board)), None)
        
        v = float('inf')
        best_move = None
        
        for move in generate_valid_moves(board):
            src, dst = move
            next_state = utils.invert_board(utils.state_change(board, src, dst, False), False)
            score, _ = minimax_alpha_beta(next_state, depth + 1, max_depth, alpha, beta, True)
            if score < v:
                v = score
                best_move = move
            if v <= alpha:
                return (v, best_move)
            
            beta = min(beta, v)
        return (v, best_move)
    
    if is_black:
        return max_value(board, depth, alpha, beta)
    else:
        return min_value(board, depth, alpha, beta)

def test_31():
    board1 = [
        list("______"),
        list("__BB__"),
        list("____BB"),
        list("WBW_B_"),
        list("____WW"),
        list("_WW___"),
    ]
    score1, (src1, dst1) = minimax_alpha_beta(copy.deepcopy(board1), 0, 3, -utils.INF, utils.INF, True)
    assert score1 == utils.WIN, "black should win in 3"
    assert utils.is_valid_move(board1, src1, dst1), "move should be valid"

    board2 = [
        list("____B_"),
        list("___B__"),
        list("__B___"),
        list("_WWW__"),
        list("____W_"),
        list("______"),
    ]
    score2, (src2, dst2) = minimax_alpha_beta(copy.deepcopy(board2), 0, 5, -utils.INF, utils.INF, True)
    assert score2 == utils.WIN, "black should win in 5"
    assert utils.is_valid_move(board2, src2, dst2), "move should be valid"

    board3 = [
        list("____B_"),
        list("__BB__"),
        list("______"),
        list("_WWW__"),
        list("____W_"),
        list("______"),
    ]
    score3, (src3, dst3) = minimax_alpha_beta(copy.deepcopy(board3), 0, 6, -utils.INF, utils.INF, True)
    assert score3 == -utils.WIN, "white should win in 6"
    assert utils.is_valid_move(board3, src3, dst3), "move should be valid"

# test_31()

def negamax_alpha_beta(board, depth, max_depth, alpha, beta) -> Tuple[Score, Move]:
    # TODO: Replace this with your own implementation
    if depth == max_depth or utils.is_game_over(board):
        return evaluate(board), None
    best_score = -utils.INF
    best_move = None
    for move in generate_valid_moves(board):
        src, dst = move
        next_state = utils.invert_board(utils.state_change(board, src, dst, False), False)
        score, _ = negamax_alpha_beta(next_state, depth + 1, max_depth, -beta, -alpha)
        score = -score
        if score > best_score:
            best_score = score
            best_move = move
            if best_score >= beta:
                return (best_score, best_move)
            
            alpha = max(alpha, best_score)
    return best_score, best_move

def test_32():
    board1 = [
        list("______"),
        list("__BB__"),
        list("____BB"),
        list("WBW_B_"),
        list("____WW"),
        list("_WW___"),
    ]
    score1, (src1, dst1) = negamax_alpha_beta(copy.deepcopy(board1), 0, 3, -utils.INF, utils.INF)
    assert score1 == utils.WIN, "black should win in 3"
    assert utils.is_valid_move(board1, src1, dst1), "move should be valid"

    board2 = [
        list("____B_"),
        list("___B__"),
        list("__B___"),
        list("_WWW__"),
        list("____W_"),
        list("______"),
    ]
    score2, (src2, dst2) = negamax_alpha_beta(copy.deepcopy(board2), 0, 5, -utils.INF, utils.INF)
    assert score2 == utils.WIN, "black should win in 5"
    assert utils.is_valid_move(board2, src2, dst2), "move should be valid"

    board3 = [
        list("____B_"),
        list("__BB__"),
        list("______"),
        list("_WWW__"),
        list("____W_"),
        list("______"),
    ]
    score3, (src3, dst3) = negamax_alpha_beta(copy.deepcopy(board3), 0, 6, -utils.INF, utils.INF)
    assert score3 == -utils.WIN, "white should win in 6"
    assert utils.is_valid_move(board3, src3, dst3), "move should be valid"

# test_32()

# Uncomment and implement the function.
# Note: this will override the provided `evaluate` function.

def evaluate(board) -> Score:
    # TODO: replace this with your own implementation
    bcount = 0
    wcount = 0
    bmarks = 0
    wmarks = 0
    for r, row in enumerate(board):
        for tile in row:
            if tile == 'B':
                if r == 5:
                    return utils.WIN
                if r == 0:
                    bmarks += 10
                if r == 1:
                    bmarks += 10
                if r == 2:
                    bmarks += 20
                if r == 3:
                    bmarks += 40
                if r == 4:
                    bmarks += 50
                bcount += 1
            elif tile == 'W':
                if r == 0:
                    return -utils.WIN
                if r == 1:
                    wmarks += 50
                if r == 2:
                    wmarks += 40
                if r == 3:
                    wmarks += 20
                if r == 4:
                    wmarks += 10
                if r == 5:
                    wmarks += 10
                wcount += 1
    if wcount == 0:
        return utils.WIN
    if bcount == 0:
        return -utils.WIN
    
    return bmarks - wmarks

def test_41():
    board1 = [
        ['_', '_', '_', 'B', '_', '_'],
        ['_', '_', '_', 'W', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', 'B', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_']
    ]
    assert evaluate(board1) == 0

    board2 = [
        ['_', '_', '_', 'B', 'W', '_'],
        ['_', '_', '_', 'W', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_']
    ]
    assert evaluate(board2) == -utils.WIN

    board3 = [
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', 'B', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_']
    ]
    assert evaluate(board3) == utils.WIN

# test_41()

class PlayerAI:

    def make_move(self, board) -> Move:
        '''
        This is the function that will be called from main.py
        You should combine the functions in the previous tasks
        to implement this function.

        Parameters
        ----------
        self: object instance itself, passed in automatically by Python.
        
        board: 2D list-of-lists. Contains characters 'B', 'W', and '_' 
        representing black pawn, white pawn and empty cell respectively.
        
        Returns
        -------
        Two tuples of coordinates (row_index, col_index).
        The first tuple contains the source position of the black pawn
        to be moved, the second tuple contains the destination position.
        '''
        # TODO: Replace starter code with your AI
        ################
        # Starter code #
        ################
        for r in range(len(board)):
            for c in range(len(board[r])):
                # check if B can move forward directly
                if board[r][c] == 'B' and board[r + 1][c] == '_':
                    src = r, c
                    dst = r + 1, c
                    return src, dst # valid move
        return (0, 0), (0, 0) # invalid move

class PlayerNaive:
    '''
    A naive agent that will always return the first available valid move.
    '''
    def make_move(self, board):
        return utils.generate_rand_move(board)

##########################
# Game playing framework #
##########################
if __name__ == "__main__":
    assert utils.test_move([
        ['B', 'B', 'B', 'B', 'B', 'B'], 
        ['_', 'B', 'B', 'B', 'B', 'B'], 
        ['_', '_', '_', '_', '_', '_'], 
        ['_', 'B', '_', '_', '_', '_'], 
        ['_', 'W', 'W', 'W', 'W', 'W'], 
        ['W', 'W', 'W', 'W', 'W', 'W']
    ], PlayerAI())

    assert utils.test_move([
        ['_', 'B', 'B', 'B', 'B', 'B'], 
        ['_', 'B', 'B', 'B', 'B', 'B'], 
        ['_', '_', '_', '_', '_', '_'], 
        ['_', 'B', '_', '_', '_', '_'], 
        ['W', 'W', 'W', 'W', 'W', 'W'], 
        ['_', '_', 'W', 'W', 'W', 'W']
    ], PlayerAI())

    assert utils.test_move([
        ['_', '_', 'B', 'B', 'B', 'B'], 
        ['_', 'B', 'B', 'B', 'B', 'B'], 
        ['_', '_', '_', '_', '_', '_'], 
        ['_', 'B', 'W', '_', '_', '_'], 
        ['_', 'W', 'W', 'W', 'W', 'W'], 
        ['_', '_', '_', 'W', 'W', 'W']
    ], PlayerAI())

    # generates initial board
    board = utils.generate_init_state()
    res = utils.play(PlayerAI(), PlayerNaive(), board)
    # Black wins means your agent wins.
    print(res)