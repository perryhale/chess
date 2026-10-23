import chess
import numpy as np

from lib.data import (
	board_to_vec837,
	move_to_idx4672,
	idx4672_to_move,
	pgn_to_supervised
)

# test state encoding
board = chess.Board()
state = board_to_vec837(board)
print(f"{board}")
print(f"{''.join(['|' if bool(s) else '.' for s in state])}")

# test action encoding
action_uci = "b2b4"
action = move_to_idx4672(chess.Move.from_uci(action_uci))
print(f"\"{action_uci}\":{action}")

# test data loader
data_x, data_y = pgn_to_supervised("data/lichess_db_standard_rated_2017-03.pgn", max_games=500, max_memory=0.99)
#data_yh = np.array([idx4672_to_move(x,b)]) ###! need inverse state function
print(f"{data_x.shape=}")
print(f"{data_y.shape=}")

import pdb; pdb.set_trace() ###! numpy array take up more memory than the lists !?
