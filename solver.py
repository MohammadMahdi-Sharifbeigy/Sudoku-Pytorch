import math

class SudokuSolver:
    """Sudoku puzzle solver using backtracking algorithm."""
    def __init__(self, board):
        self.board = board

    def find_next_empty(self, empty_val=0):
        num_rows = len(self.board) 
        num_cols = len(self.board[0])
        for row in range(num_rows):
            for col in range(num_cols):
                if self.board[row][col] == empty_val:
                    return (row, col) 
        return None

    def is_valid_number(self, board, number, position):
        num_rows = len(board)
        square_size = int(math.sqrt(num_rows)) 
        row_idx, col_idx = position

        if number in board[row_idx]: return False
        
        current_column_values = [board[row][col_idx] for row in range(num_rows)]
        if number in current_column_values: return False 

        square_x_idx = col_idx // square_size
        square_y_idx = row_idx // square_size
        for row in range(square_y_idx * square_size, (square_y_idx * square_size) + square_size):
            for col in range(square_x_idx * square_size, (square_x_idx * square_size) + square_size):
                if board[row][col] == number and (row, col) != position: 
                    return False
        return True

    def solve(self):
        next_empty_pos = self.find_next_empty() 
        if not next_empty_pos: return True
        else: row, col = next_empty_pos

        for i in range(1, 10):
            if self.is_valid_number(board=self.board, number=i, position=(row, col)):
                self.board[row][col] = i
                if self.solve(): return True
        self.board[row][col] = 0
        return False