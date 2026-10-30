import time

import jax
import jax.numpy as jnp
import optax

from lib.data import stream_pgn_to_supervised
from lib.model import init_dense_neural_network, dense_neural_network, fit


###! parameters


# start timer
start_time = time.time()
print(f"[Elapsed time: {time.time() - start_time:.2f}s]")


###! try cache


# supervised training
lichess_pgn = open("data/lichess_db_standard_rated_2017-03.pgn", "r", encoding="utf-8")

k0 = jax.random.key(9)
params = init_dense_neural_network(k0, [837, 512, 512, 512, 4672])
model = jax.vmap(lambda p, x: jnp.where(x[837:], -jnp.inf, dense_neural_network(p, x[:837].astype(jnp.float32))), in_axes=(None, 0))
loss = lambda p, xs, ys: jnp.mean(optax.softmax_cross_entropy_with_integer_labels(model(p, xs), ys))

batch_size = 256
train_block_count = 10
train_block_size = 200_000
val_block_size = 8192

val_x, val_y = stream_pgn_to_supervised(lichess_pgn, max_moves=val_block_size, debug=val_block_size)
val_steps = int(jnp.ceil(val_block_size / batch_size))
train_steps = int(jnp.ceil(train_block_size / batch_size))

for _ in range(train_block_count):
	train_x, train_y = stream_pgn_to_supervised(lichess_pgn, max_moves=train_block_size, debug=train_block_size//5)
	
	k0, k1 = jax.random.split(k0)
	params, history = fit(
		k1,
		params,
		loss,
		(train_x, train_y, train_steps),
		val_data=(val_x, val_y, val_steps),
		batch_size=batch_size,
		epochs=1,
		opt=optax.adamw(1e-5),
		start_time=start_time
	)
	
	del train_x
	del train_y

lichess_pgn.close()


###! write cache
