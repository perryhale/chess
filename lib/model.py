import time

import jax
import jax.numpy as jnp
import optax

from .data import batch_generator




### model

# type: (int, list[int], int, int) -> jax.Array
def init_glorot_uniform(key, shape, fan_in, fan_out):
	"""
	# shape inZ+ (*)
	# fan_in, fan_out inZ+
	# w inR shape
	"""
	limit = jnp.sqrt(6) / jnp.sqrt(fan_in+fan_out)
	w = jax.random.uniform(key, shape, minval=-limit, maxval=limit)
	return w


# type: (int, list[int]) -> list[tuple[jax.Array, jax.Array]]
def init_dense_neural_network(key, layers):
	
	# split key
	k0, key = jax.random.split(key)
	
	# initialise input layer
	w_list = [init_glorot_uniform(k0, (layers[1], layers[0]), layers[0], layers[1])]
	b_list = [jnp.zeros((layers[1], 1))]
	
	# initialise hidden layers
	for i in range(1, len(layers)-1):
		k1, key = jax.random.split(key)
		w_list.append(init_glorot_uniform(k1, (layers[i+1], layers[i]), layers[i], layers[i+1]))
		b_list.append(jnp.zeros((layers[i+1], 1)))
	
	return list(map(tuple, zip(w_list, b_list)))


# type: (list[tuple[jax.Array, jax.Array]], jax.Array, Callable[[jax.Array]]) -> jax.Array
def dense_neural_network(params, x, ha=jax.nn.sigmoid):
	"""
	# x inR (in, 1)
	# z inR (out, 1)
	"""
	
	# shape-safe input activation
	h = x.reshape(-1, 1)
	
	# hidden activations
	for w,b in params[:-1]:
		h = ha(jnp.dot(w, h) + b)
	
	# output activation
	wn, bn = params[-1]
	z = jnp.dot(wn, h) + bn
	
	return z.reshape(-1)




### optimiser


def fit(
		key,
		params,
		loss_fn,
		train_data,
		val_data=None,
		batch_size=64,
		epochs=1,
		opt=optax.sgd(1e-3),
		start_time=None
	):
	"""
	iteratively update params using gradient descent and scalar loss function on (data_x, data_y, train_steps=data_len/batch_size) training data.
	validation data can be provided in the same format for evaluation _after_ every epoch.
	batch_size:int, epochs:int and opt:base.GradientTransformationExtraArgs are provided as defaulting call arguments,
	start_time:float optional argument for stdout time reporting.
	
	type: (
		jax.Array,
		PyTree[jax.Array],
		Callable[[PyTree[jax.Array], jax.Array, jax.Array], float],
		tuple[jax.Array, jax.Array, int],
		tuple[jax.Array, jax.Array, int] | None,
		int,
		int,
		optax.GradientTransformation,
		float | None
	) -> tuple[PyTree[jax.Array], Dict[str:list[float]]]
	"""
	
	# initialise
	train_x, train_y, train_steps = train_data
	opt_state = opt.init(params)
	history = {'batch_loss':[], 'train_loss':[]}
	if val_data is not None:
		val_x, val_y, val_steps = val_data
		val_generator = batch_generator(val_x, val_y, batch_size)
		history.update({'val_loss':[]})
	if start_time is None:
		start_time = time.time()
	
	# jit optimiser core
	@jax.jit
	def opt_step(opt_state_, params_, x, y):
		loss, grad = jax.value_and_grad(loss_fn)(params_, x, y)
		updates, opt_state_ = opt.update(grad, opt_state_, params_)
		params_ = optax.apply_updates(params_, updates)
		return loss, opt_state_, params_
	
	# optimise
	for i in range(epochs):
		
		# shuffle train batches
		key, subkey = jax.random.split(key, 2)
		train_generator = batch_generator(train_x, train_y, batch_size, shuffle_key=subkey)
		
		# iterate optimiser
		train_loss = 0.
		for j in range(train_steps):
			batch_loss, opt_state, params = opt_step(opt_state, params, *next(train_generator))
			train_loss += batch_loss / train_steps
			history['batch_loss'].append(batch_loss)
		history['train_loss'].append(train_loss)
		
		# determine validation loss
		if val_data is not None:
			val_loss = 0.
			for _ in range(val_steps):
				val_loss += loss_fn(params, *next(val_generator)) / val_steps
			history['val_loss'].append(val_loss)
		
		# trace to stdout
		print(
			f"epoch={i+1}, train_loss={train_loss:.4f},"
			+ (f" val_loss={val_loss:.4f}" if val_data is not None else "")
			+ f" [Elapsed time: {time.time() - start_time:.2f}s]"
		)
	
	return params, history




### metrics

# type: () ->
def accuracy_score(yh, y):
	"""	
	# yh, y inR (*, n_classes)
	# accuracy inR
	"""
	yhc = jnp.argmax(yh, axis=-1)
	yc = jnp.argmax(y, axis=-1)
	accuracy = jnp.mean(jnp.array(yhc==yc, dtype='int32'))
	return accuracy

# type: (list[tuple[jax.Array]]) -> int
def count_params(params):
	if isinstance(params, jnp.ndarray):
		return jnp.prod(jnp.array(params.shape))
	elif isinstance(params, (list, tuple)):
		return jnp.sum(jnp.array([count_params(item) for item in params]))
	return 0
