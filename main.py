import chess
import numpy as np

from lib.data import (
	board_to_vec837,
	move_to_idx4672,
	idx4672_to_move,
	stream_pgn_to_supervised
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
print(f"({state_fen_impl}, {action_uci_impl}):{move_impl.uci()=}, {move_impl in board_impl.legal_moves=}")

# test data loader
lichess_pgn = open("data/lichess_db_standard_rated_2017-03.pgn", "r", encoding="utf-8")

data_x, data_y = stream_pgn_to_supervised(lichess_pgn, max_moves=50)#, debug=True)
data_yh = np.array([idx4672_to_move(i,s) for i,s in zip(np.argmax(data_y, axis=-1), data_x)])

data_x1, data_y1 = stream_pgn_to_supervised(lichess_pgn, max_moves=50)#, debug=True)
data_yh1 = np.array([idx4672_to_move(i,s) for i,s in zip(np.argmax(data_y1, axis=-1), data_x1)])

data_x_differ = bool(np.any(data_x != data_x1))
data_y_differ = bool(np.any(data_y != data_y1))

print(f"{data_x.shape=}, {data_x.dtype=}")
print(f"{data_y.shape=}, {data_y.dtype=}")
print(f"{data_yh.shape=}, {data_yh.dtype=}")
print(f"{data_x1.shape=}, {data_x1.dtype=}")
print(f"{data_y1.shape=}, {data_y1.dtype=}")
print(f"{data_yh1.shape=}, {data_yh1.dtype=}")
print(f"{data_x_differ=}")
print(f"{data_y_differ=}")

try:
	_ = stream_pgn_to_supervised(lichess_pgn, max_memory=0.4, debug=10_000)
except RuntimeError as e:
	print(e)

lichess_pgn.close()

#import pdb; pdb.set_trace() ###! numpy array take up more memory than the lists !?
