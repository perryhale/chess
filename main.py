import chess
import numpy as np

from lib.data import *

board = chess.Board()
arr = board_to_vec837(board)

print(board)
print(arr)
