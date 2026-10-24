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

# test implicit promotion
state_fen_impl = "7k/P7/8/8/8/8/8/7K w - - 0 1"
action_uci_impl = "a7a8"
board_impl = chess.Board(state_fen_impl)
move_impl = idx4672_to_move(move_to_idx4672(chess.Move.from_uci(action_uci_impl)), board_to_vec837(board_impl))
print(f"{state_fen_impl}->{action_uci_impl}: {move_impl.uci()=}, {move_impl in board_impl.legal_moves=}")

# test data loader
data_x, data_y = pgn_to_supervised("data/lichess_db_standard_rated_2017-03.pgn", max_games=10)
data_yh = np.array([idx4672_to_move(i,s) for i,s in zip(np.argmax(data_y, axis=1), data_x)])
print(f"{data_x.shape=} {data_x.dtype=}")
print(f"{data_y.shape=} {data_y.dtype=}")
print(f"{data_yh.shape=} {data_yh.dtype=}")

#import pdb; pdb.set_trace() ###! numpy array take up more memory than the lists !?
