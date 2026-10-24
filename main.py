import time
import jax
import jax.numpy as jnp
import optax
import numpy as np
import chess

from lib.data import (
	board_to_vec837,
	move_to_idx4672,
	idx4672_to_move,
	stream_pgn_to_supervised
)
from lib.model import (
	init_dense_neural_network,
	dense_neural_network,
	fit
)


# start timer
start_time = time.time()
print(f"[Elapsed time: {time.time() - start_time:.2f}s]")


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

data_x, data_y = stream_pgn_to_supervised(lichess_pgn, max_moves=50) # ; import pdb; pdb.set_trace() ###! numpy array takes up more memory than lists
data_yh = np.array([idx4672_to_move(int(i),s) for i,s in zip(data_y, data_x)])

data_x1, data_y1 = stream_pgn_to_supervised(lichess_pgn, max_moves=50)
data_yh1 = np.array([idx4672_to_move(int(i),s) for i,s in zip(data_y1, data_x1)])

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
print(f"[Elapsed time: {time.time() - start_time:.2f}s]")

try:
	_ = stream_pgn_to_supervised(lichess_pgn, max_memory=0.33, debug=10_000)
except RuntimeError as e:
	print(e)
	print(f"[Elapsed time: {time.time() - start_time:.2f}s]")

lichess_pgn.close()


# test model end-to-end
lichess_pgn = open("data/lichess_db_standard_rated_2017-03.pgn", "r", encoding="utf-8")

k0 = jax.random.key(9)
params = init_dense_neural_network(k0, [837, 512, 512, 4672])
model = jax.vmap(lambda p, x, m: dense_neural_network(p, x) + m, in_axes=(None, 0, 0))
loss = lambda p, xs, ys, ms: jnp.mean(optax.softmax_cross_entropy_with_integer_labels(model(p, xs, ms), ys))

train_block_count = 8
train_block_size = 256
batch_size = 32
train_steps = train_block_size // batch_size

start_time = time.time()
for _ in range(train_block_count):
	train_x, train_y = stream_pgn_to_supervised(lichess_pgn, max_moves=train_block_size)
	train_x = jnp.array(train_x)
	train_y = jnp.array(train_y)
	train_mask = jnp.array([vec837_to_mask4672(x) for x in train_x]) ###! requires env calls
	k0, k1 = jax.random.split(k0)
	params, history = fit(
		k1,
		params,
		loss,
		(train_x, train_y, train_steps), ###! must pre-compute mask, cannot be jitted, rework fit(), model(), loss() for optional mask
		batch_size=batch_size,
		epochs=1,
		opt=optax.adamw(1e-5),
		start_time=start_time
	)

lichess_pgn.close()
