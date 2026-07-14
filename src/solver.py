import math

class SudokuSolver:
    """Sudoku puzzle solver using backtracking with bitmask constraints + MRV heuristic."""

    def __init__(self, board):
        self.board = board
        self.n = len(board)
        self.box_size = int(math.sqrt(self.n))
        self.rows = [0] * self.n
        self.cols = [0] * self.n
        self.boxes = [0] * self.n
        self.valid = True  # becomes False if the givens themselves conflict

        for r in range(self.n):
            for c in range(self.n):
                val = self.board[r][c]
                if val != 0:
                    bit = 1 << val
                    b = self._box_idx(r, c)
                    # Detect duplicate givens immediately (e.g. bad OCR/vision reads)
                    if (self.rows[r] & bit) or (self.cols[c] & bit) or (self.boxes[b] & bit):
                        self.valid = False
                    self.rows[r] |= bit
                    self.cols[c] |= bit
                    self.boxes[b] |= bit

    def _box_idx(self, row, col):
        return (row // self.box_size) * self.box_size + (col // self.box_size)

    def _candidates(self, row, col):
        used = self.rows[row] | self.cols[col] | self.boxes[self._box_idx(row, col)]
        return [i for i in range(1, self.n + 1) if not (used & (1 << i))]

    def _find_best_empty(self):
        best = None
        best_candidates = None
        best_count = self.n + 1

        for row in range(self.n):
            for col in range(self.n):
                if self.board[row][col] == 0:
                    cands = self._candidates(row, col)
                    if len(cands) < best_count:
                        best = (row, col)
                        best_candidates = cands
                        best_count = len(cands)
                        if best_count == 0:
                            return best, best_candidates
                        if best_count == 1:
                            return best, best_candidates
        return best, best_candidates

    def _place(self, row, col, number):
        bit = 1 << number
        self.board[row][col] = number
        self.rows[row] |= bit
        self.cols[col] |= bit
        self.boxes[self._box_idx(row, col)] |= bit

    def _remove(self, row, col, number):
        bit = ~(1 << number)
        self.board[row][col] = 0
        self.rows[row] &= bit
        self.cols[col] &= bit
        self.boxes[self._box_idx(row, col)] &= bit

    def solve(self):
        if not self.valid:
            return False  # givens already conflict — fail fast, no search needed

        pos, candidates = self._find_best_empty()
        if pos is None:
            return True
        if not candidates:
            return False

        row, col = pos
        for number in candidates:
            self._place(row, col, number)
            if self.solve():
                return True
            self._remove(row, col, number)
        return False

    def is_valid_number(self, board, number, position):
        row, col = position
        bit = 1 << number
        used = self.rows[row] | self.cols[col] | self.boxes[self._box_idx(row, col)]
        return not (used & bit)