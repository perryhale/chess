# core
import psutil

# numerics
import jax
import jax.numpy as jnp
import numpy as np

# environment
import chess # python-chess
import chess.pgn



### utility

def one_hot(class_idx, n_classes):
	x = np.zeros((n_classes), dtype=np.float32)
	x[class_idx] = 1.0
	return x




### state encoding (positions OHE + special states, concatenated)

# type: (chess.Board) -> np.ndarray
def board_to_vec837(board):
	
	# init array
	vec837 = np.zeros((837), dtype=np.float32)
	
	# init constants
	board_offset = len(board.piece_map())
	
	# set position state OHE vectors: (class) 0-5:white, 6-11:black
	for pos_idx, piece in board.piece_map().items():
		class_idx = piece.piece_type - 1 # classes: [pawn, knight, bishop, rook, queen, king]
		if not piece.color:
			class_idx += 6
		vec837[pos_idx * 12 + class_idx] = 1.0
	
	# set turn state: (value) 0:black, 1:white
	vec837[board_offset + 0] = board.turn
	
	# set castling states: (value) 0: unavailable, 1: available
	vec837[board_offset + 1] = board.has_kingside_castling_rights(chess.WHITE)
	vec837[board_offset + 2] = board.has_queenside_castling_rights(chess.WHITE)
	vec837[board_offset + 3] = board.has_kingside_castling_rights(chess.BLACK)
	vec837[board_offset + 4] = board.has_queenside_castling_rights(chess.BLACK)
	
	# set en-passant states: (value) 0: unavailable, 1: available
	if board.ep_square is not None:
		vec837[board_offset + 5 + board.ep_square] = 1.0
	
	return vec837




### action encoding (AlphaZero encoding) (python-chess dependency)

# define inter/+cardinal directions, (8*7) 0-55
E4672_DIR = [
	(0, 1),   # N
	(1, 1),   # NE
	(1, 0),   # E
	(1, -1),  # SE
	(0, -1),  # S
	(-1, -1), # SW
	(-1, 0),  # W
	(-1, 1),  # NW
]

# define knight translations, (8) 56-63
E4672_KNIGHT = [
	(1, 2),   # up right
	(2, 1),   # right up
	(2, -1),  # right down
	(1, -2),  # down right
	(-1, -2), # down left
	(-2, -1), # 
	(-2, 1),
	(-1, 2),
]

# define underpromotions, directional: (3*3) 64-72
E4672_UNDER_PROM = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


# type: (chess.Move) -> int
def move_to_idx4672(move):
	
	# determine to/from file/rank (col/row) indices
	r0, f0 = divmod(move.from_square, 8)
	r1, f1 = divmod(move.to_square, 8)
	df = f1 - f0
	dr = r1 - r0
	
	# handle promotion move offset
	if move.promotion and move.promotion in E4672_UNDER_PROM:
		df_signed = df if dr > 0 else -df # if black pawn flip file sign
		class_offset = 64 + (df_signed + 1) * 3 + (move.promotion - chess.KNIGHT)
	
	# handle knight move offset
	elif (df, dr) in E4672_KNIGHT:
		class_offset = 56 + E4672_KNIGHT.index((df, dr))
	
	# handle standard move offset
	else:
		distance = max(abs(df), abs(dr))
		direction = (np.sign(df), np.sign(dr))
		class_offset = E4672_DIR.index(direction) * 7 + distance - 1
	
	# determine class index and return
	return (r0 * 8 + f0) * 73 + class_offset


# type: (int, chess.Board) -> str
def idx4672_to_move(idx4672, board):
	
	# derive from-rank/from-file/action
	from_square, action = divmod(idx4672, 73)
	r0, f0 = divmod(from_square, 8)
	
	# initialise promotion class (int)
	promotion_class = None # no promotion
	
	# decode regular inter/+cardinal translation
	if action < 56:
		dir_idx, distance = divmod(action, 7)
		df, dr = E4672_DIR[dir_idx]
		df *= distance + 1
		dr *= distance + 1
	
	# decode knight translation
	elif channel < 64:
		df, dr = E4672_KNIGHT[action - 56]
	
	# decode under-promotion translation and character
	else:
		dir_class, piece_idx = divmod(channel - 64, 3)
		dr = 1 if r0 == 6 else -1 # up if white, down if black
		df = (dir_class - 1) * dr # right if white, left if black
		promotion_class = E4672_UNDER_PROM[piece_idx]
	
	# derive to-rank/to-file
	r1 = r0 + dr
	f1 = f0 + df
	to_square = r1 * 8 + f1
	
	# handle implicit promotion
	if board.piece_at(r0 * 8 + f0) == chess.PAWN and r1 in (0, 7) and promotion_str == None:
		promotion_class = chess.QUEEN
	
	# construct chess.Move and return
	return chess.Move(from_square, to_square, promotion=promotion_class)


###! E4672 with python-chess dependency

# E4672_DIR = [
    # (0, 1), (1, 1), (1, 0), (1, -1),
    # (0, -1), (-1, -1), (-1, 0), (-1, 1),
# ]

# KNIGHT_OFFSETS = [
    # (1, 2), (2, 1), (2, -1), (1, -2),
    # (-1, -2), (-2, -1), (-2, 1), (-1, 2),
# ]

# UNDERPROMOTION_PIECES = {
    # chess.KNIGHT: 0,
    # chess.BISHOP: 1,
    # chess.ROOK: 2,
# }


# def uci_to_vec4672(uci: str) -> np.ndarray:
    # """Encode a UCI move as a 4,672-element one-hot vector."""
    # move = chess.Move.from_uci(uci)

    # from_file = chess.square_file(move.from_square)
    # from_rank = chess.square_rank(move.from_square)
    # to_file = chess.square_file(move.to_square)
    # to_rank = chess.square_rank(move.to_square)

    # df = to_file - from_file
    # dr = to_rank - from_rank

    # if move.promotion in UNDERPROMOTION_PIECES:
        # # Infer forward direction from the promotion move's rank change.
        # # Relative left/right is from the moving player's perspective.
        # relative_df = df if dr > 0 else -df
        # direction_idx = {-1: 0, 0: 1, 1: 2}[relative_df]
        # plane = 64 + direction_idx * 3 + UNDERPROMOTION_PIECES[move.promotion]

    # elif (df, dr) in KNIGHT_OFFSETS:
        # plane = 56 + KNIGHT_OFFSETS.index((df, dr))

    # else:
        # distance = max(abs(df), abs(dr))
        # direction = (df // distance, dr // distance)
        # direction_idx = E4672_DIR.index(direction)
        # plane = direction_idx * 7 + distance - 1

    # index = move.from_square * 73 + plane
    # vec = np.zeros(64 * 73, dtype=np.float32)
    # vec[index] = 1.0
    # return vec


# def vec4672_to_uci(vec: np.ndarray, board: chess.Board) -> str:
    # """Decode a 4,672-element policy vector, checking legality on the board."""
    # if vec.size != 64 * 73:
        # raise ValueError("vec must contain 4672 values")

    # index = int(np.argmax(vec))
    # from_square, plane = divmod(index, 73)

    # from_file = chess.square_file(from_square)
    # from_rank = chess.square_rank(from_square)
    # promotion = None

    # if plane < 56:
        # direction_idx, distance_idx = divmod(plane, 7)
        # df, dr = E4672_DIR[direction_idx]
        # distance = distance_idx + 1
        # to_file = from_file + df * distance
        # to_rank = from_rank + dr * distance

    # elif plane < 64:
        # df, dr = KNIGHT_OFFSETS[plane - 56]
        # to_file = from_file + df
        # to_rank = from_rank + dr

    # else:
        # direction_idx, piece_idx = divmod(plane - 64, 3)
        # relative_df = (-1, 0, 1)[direction_idx]
        # df = relative_df if board.turn == chess.WHITE else -relative_df
        # dr = 1 if board.turn == chess.WHITE else -1
        # to_file = from_file + df
        # to_rank = from_rank + dr
        # promotion = (chess.KNIGHT, chess.BISHOP, chess.ROOK)[piece_idx]

    # if not (0 <= to_file < 8 and 0 <= to_rank < 8):
        # raise ValueError("Vector selects a move off the board")

    # to_square = chess.square(to_file, to_rank)

    # # Queen promotions use the ordinary direction/distance planes.
    # piece = board.piece_at(from_square)
    # if (
        # promotion is None
        # and piece is not None
        # and piece.piece_type == chess.PAWN
        # and to_rank in (0, 7)
    # ):
        # promotion = chess.QUEEN

    # move = chess.Move(from_square, to_square, promotion=promotion)
    # if not board.is_legal(move):
        # raise ValueError(f"Vector selects an illegal move: {move.uci()}")

    # return move.uci()


###! naive complete categorical with promotions

# PROMO_TO_CLASS = {
	# "": 0,   # no promotion
	# "q": 1,
	# "r": 2,
	# "b": 3,
	# "n": 4,
# }

# def uci_to_idx20480(uci: str) -> int:
	# from_file = ord(uci[0]) - ord("a")
	# from_rank = ord(uci[1]) - ord("1")
	# to_file = ord(uci[2]) - ord("a")
	# to_rank = ord(uci[3]) - ord("1")
	
	# from_square = from_rank * 8 + from_file
	# to_square = to_rank * 8 + to_file
	# promo_class = PROMO_TO_CLASS[uci[4:].lower()]
	
	# return (from_square * 64 + to_square) * 5 + promo_class


# CLASS_TO_PROMO = ["", "q", "r", "b", "n"]

# def idx20480_to_uci(index: int) -> str:
    # if not 0 <= index < 64 * 64 * 5:
        # raise ValueError("index must be in the range [0, 20480)")

    # pair_index, promo_class = divmod(index, 5)
    # from_square, to_square = divmod(pair_index, 64)

    # from_file, from_rank = from_square % 8, from_square // 8
    # to_file, to_rank = to_square % 8, to_square // 8

    # return (
        # chr(ord("a") + from_file)
        # + chr(ord("1") + from_rank)
        # + chr(ord("a") + to_file)
        # + chr(ord("1") + to_rank)
        # + CLASS_TO_PROMO[promo_class]
    # )


###! naive complete categorical with promotions and python-chess dependency

# ACTION_SIZE = 64 * 64 * 5

# PROMOTION_TO_CLASS = {
	# None: 0,
	# chess.QUEEN: 1,
	# chess.ROOK: 2,
	# chess.BISHOP: 3,
	# chess.KNIGHT: 4,
# }
# CLASS_TO_PROMOTION = {v: k for k, v in PROMOTION_TO_CLASS.items()}


# def uci_to_vec20480(uci: str) -> np.ndarray:
	# move = chess.Move.from_uci(uci)

	# pair_idx = move.from_square * 64 + move.to_square
	# promotion_class = PROMOTION_TO_CLASS[move.promotion]
	# idx = pair_idx * 5 + promotion_class

	# vec = np.zeros(ACTION_SIZE, dtype=np.float32)
	# vec[idx] = 1.0
	# return vec


# def vec20480_to_uci(vec: np.ndarray, board: chess.Board) -> str:
	# idx = int(np.argmax(vec))
	# pair_idx, promotion_class = divmod(idx, 5)
	# from_sq, to_sq = divmod(pair_idx, 64)

	# promotion = CLASS_TO_PROMOTION[promotion_class]
	# move = chess.Move(from_sq, to_sq, promotion=promotion)

	# return move.uci()


###! E4672 3D with python chess dependency

# E4672_DIR = [
	# (0, 1),    # N
	# (1, 1),    # NE
	# (1, 0),    # E
	# (1, -1),   # SE
	# (0, -1),   # S
	# (-1, -1),  # SW
	# (-1, 0),   # W
	# (-1, 1),   # NW
# ]

# KNIGHT_MOVES = [
	# (1, 2), (2, 1), (2, -1), (1, -2),
	# (-1, -2), (-2, -1), (-2, 1), (-1, 2),
# ]

# UNDERPROMOTIONS = [
	# chess.KNIGHT, chess.BISHOP, chess.ROOK
# ]


# def uci_to_vec4672(uci: str) -> np.ndarray:
	# move = chess.Move.from_uci(uci)

	# f0 = chess.square_file(move.from_square)
	# r0 = chess.square_rank(move.from_square)
	# f1 = chess.square_file(move.to_square)
	# r1 = chess.square_rank(move.to_square)

	# df, dr = f1 - f0, r1 - r0

	# if move.promotion in UNDERPROMOTIONS:
		# # -1, 0, +1 files relative to origin
		# if abs(df) > 1 or abs(dr) != 1:
			# raise ValueError("Invalid underpromotion geometry")

		# direction_idx = df + 1
		# piece_idx = UNDERPROMOTIONS.index(move.promotion)
		# channel = 64 + direction_idx * 3 + piece_idx

	# elif (df, dr) in KNIGHT_MOVES:
		# channel = 56 + KNIGHT_MOVES.index((df, dr))

	# else:
		# distance = max(abs(df), abs(dr))

		# if distance == 0 or not (
			# df == 0 or dr == 0 or abs(df) == abs(dr)
		# ):
			# raise ValueError("Invalid move geometry")

		# direction = (
			# int(np.sign(df)),
			# int(np.sign(dr))
		# )

		# direction_idx = E4672_DIR.index(direction)
		# channel = direction_idx * 7 + distance - 1

	# vec = np.zeros((8, 8, 73), dtype=np.float32)
	# vec[r0, f0, channel] = 1.0

	# return vec


# def vec4672_to_uci(vec: np.ndarray, board: chess.Board) -> str:
	# """
	# Decode a policy tensor, selecting the highest-scoring
	# legal move. Accepts one-hot vectors, logits or probabilities.
	# """
	# scores = np.asarray(vec).reshape(8, 8, 73)

	# legal_moves = list(board.legal_moves)

	# if not legal_moves:
		# raise ValueError("No legal moves available")

	# best_move = max(
		# legal_moves,
		# key=lambda move: scores[
			# chess.square_rank(move.from_square),
			# chess.square_file(move.from_square),
			# np.argmax(uci_to_vec4672(move.

# type: (np.ndarray, int, bool) ~> tuple[np.ndarray, np.ndarray]
def batch_generator_mono(data_x, batch_size, shuffle_key=None):
	
	# yield infinite batches optionally shuffled
	while True:
		n_samples = len(data_x)
		data_idx = jax.random.permutation(shuffle_key, n_samples) if (shuffle_key is not None) else jnp.array(range(n_samples))
		for batch_data_idx in range(0, n_samples, batch_size):
			batch_idx = data_idx[batch_data_idx:batch_data_idx+batch_size]
			batch_x = data_x[batch_idx]
			yield batch_x


###! no testing docs

# # type: (list, (list)->list, int) -> list
# import multiprocessing
# def multiprocess_list(list_data, list_fn, max_threads=999):
	
	# # determine thread count
	# n_threads = min(multiprocessing.cpu_count(), max_threads)
	# print(f"Using {n_threads} CPUs..")
	
	# # prepare jobs
	# job_size = len(list_data) // n_threads
	# job_data = [list_data[i:i+job_size] for i in range(0, len(list_data), job_size)]
	
	# # pool worker results
	# with multiprocessing.Pool(n_threads) as pool:
		# result = pool.map(list_fn, job_data)
		# result = [item for sublist in result for item in sublist]
	
	# return result


###! does not generalise

# processing
#from sklearn.preprocessing import MinMaxScaler

# # type: (jax.PRNGKey, List[Tuple[float*4]], float, float, float, float) -> Tuple(Tuple(jnp.array*2, int)*3, MinMaxScaler)
# def train_val_test_split(key, data_points, batch_size, part_buffer=0.0, part_train=0.7, part_val=0.05, part_test=0.25):
	
	# data_points = jnp.array(data_points)
	
	# # time-series partition with buffer, shuffle val+test together
	# n_data = data_points.shape[0]
	# n_buffer = int(part_buffer * n_data)
	# n_train = int(part_train * n_data)
	# n_val = int(part_val * n_data)
	# n_test = int(part_test * n_data)
	
	# shuffle_idx = n_buffer + n_train + jax.random.permutation(key, n_val + n_test)
	# data_train = data_points[n_buffer:n_buffer+n_train]
	# data_val = data_points[shuffle_idx[:n_val]]
	# data_test = data_points[shuffle_idx[n_val:]]
	
	# # unit scale
	# data_scaler = MinMaxScaler(feature_range=(0, 1))
	# data_scaler.fit(data_train)
	# data_train = data_scaler.transform(data_train)
	# data_val = data_scaler.transform(data_val)
	# data_test = data_scaler.transform(data_test)
	
	# # supervised split
	# train_x, train_y = data_train[:,:-1], data_train[:,-1] # xyt, z
	# val_x, val_y = data_val[:,:-1], data_val[:,-1]
	# test_x, test_y = data_test[:,:-1], data_test[:,-1]
	
	# # determine batch counts
	# train_steps = int(jnp.ceil(train_x.shape[0] / batch_size))
	# val_steps = int(jnp.ceil(val_x.shape[0] / batch_size))
	# test_steps = int(jnp.ceil(test_x.shape[0] / batch_size))
	
	# return (train_x, train_y, train_steps), (val_x, val_y, val_steps), (test_x, test_y, test_steps), data_scaleruci())) % 73
		# ]
	# )

	# return best_move.uci()


###! naive incomplete categorical with python-chess dependency

# def uci_to_vec4096(uci: str) -> np.ndarray:
	# move = chess.Move.from_uci(uci)

	# idx = move.from_square * 64 + move.to_square

	# vec = np.zeros(ACTION_SIZE, dtype=np.float32)
	# vec[idx] = 1.0
	# return vec


# def vec4096_to_uci(vec: np.ndarray, board: chess.Board) -> str:
	# idx = int(np.argmax(vec))

	# from_sq, to_sq = divmod(idx, 64)

	# move = chess.Move(from_sq, to_sq)

	# # Automatically promote to queen when appropriate
	# piece = board.piece_at(from_sq)

	# if (
		# piece is not None
		# and piece.piece_type == chess.PAWN
		# and chess.square_rank(to_sq) in (0, 7)
	# ):
		# move.promotion = chess.QUEEN

	# return move.uci()




### pipeline


###! do: stream to batch from pgn / stream chunks to memory (+async)

# type: (str) -> tuple[np.ndarray, np.ndarray]
def pgn_to_supervised(pgn_path, max_games=-1, max_memory=0.99):
	states = []
	actions = []
	game_counter = 0
	max_memory_percent = 100 * max_memory
	
	# stream up to #max_games games from lichess PGN file
	with open(pgn_path, "r", encoding="utf-8") as pgn:
		
		# read games
		while max_games == -1 or game_counter < max_games:
			
			# mem safety
			memory_percent = psutil.virtual_memory().percent
			if memory_percent > max_memory_percent:
				raise RuntimeError(f"memory_percent > max_memory_percent, {memory_percent} > {max_memory_percent}")
			
			# init game if available
			game = chess.pgn.read_game(pgn)
			if game is None:
				break
			game_counter += 1
			board = game.board()
			
			print(f"{game_counter} {memory_percent}%")
			
			# unpack moves from game and update
			for move in game.mainline_moves():
				states.append(board_to_vec837(game.board()))
				actions.append(one_hot(move_to_idx4672(move), 4672))
				board.push(move)
	
	# construct numpy arrays
	data_x = np.asarray(states, dtype=np.float32)
	data_y = np.asarray(actions, dtype=np.float32)
	
	return data_x, data_y


# type: (np.ndarray, np.ndarray, int, bool) ~> tuple[np.ndarray, np.ndarray]
def batch_generator(data_x, data_y, batch_size, shuffle_key=None):
	
	# assertions
	assert (len(data_x)==len(data_y))
	
	# yield infinite batches optionally shuffled
	while True:
		n_samples = len(data_x)
		data_idx = jax.random.permutation(shuffle_key, n_samples) if (shuffle_key is not None) else jnp.array(range(n_samples))
		for batch_data_idx in range(0, n_samples, batch_size):
			batch_idx = data_idx[batch_data_idx:batch_data_idx+batch_size]
			batch_x = data_x[batch_idx]
			batch_y = data_y[batch_idx]
			yield batch_x, batch_y


# type: (np.ndarray, int, bool) ~> tuple[np.ndarray, np.ndarray]
def batch_generator_mono(data_x, batch_size, shuffle_key=None):
	
	# yield infinite batches optionally shuffled
	while True:
		n_samples = len(data_x)
		data_idx = jax.random.permutation(shuffle_key, n_samples) if (shuffle_key is not None) else jnp.array(range(n_samples))
		for batch_data_idx in range(0, n_samples, batch_size):
			batch_idx = data_idx[batch_data_idx:batch_data_idx+batch_size]
			batch_x = data_x[batch_idx]
			yield batch_x


###! no testing docs

# # type: (list, (list)->list, int) -> list
# import multiprocessing
# def multiprocess_list(list_data, list_fn, max_threads=999):
	
	# # determine thread count
	# n_threads = min(multiprocessing.cpu_count(), max_threads)
	# print(f"Using {n_threads} CPUs..")
	
	# # prepare jobs
	# job_size = len(list_data) // n_threads
	# job_data = [list_data[i:i+job_size] for i in range(0, len(list_data), job_size)]
	
	# # pool worker results
	# with multiprocessing.Pool(n_threads) as pool:
		# result = pool.map(list_fn, job_data)
		# result = [item for sublist in result for item in sublist]
	
	# return result


###! does not generalise

# processing
#from sklearn.preprocessing import MinMaxScaler

# # type: (jax.PRNGKey, List[Tuple[float*4]], float, float, float, float) -> Tuple(Tuple(jnp.array*2, int)*3, MinMaxScaler)
# def train_val_test_split(key, data_points, batch_size, part_buffer=0.0, part_train=0.7, part_val=0.05, part_test=0.25):
	
	# data_points = jnp.array(data_points)
	
	# # time-series partition with buffer, shuffle val+test together
	# n_data = data_points.shape[0]
	# n_buffer = int(part_buffer * n_data)
	# n_train = int(part_train * n_data)
	# n_val = int(part_val * n_data)
	# n_test = int(part_test * n_data)
	
	# shuffle_idx = n_buffer + n_train + jax.random.permutation(key, n_val + n_test)
	# data_train = data_points[n_buffer:n_buffer+n_train]
	# data_val = data_points[shuffle_idx[:n_val]]
	# data_test = data_points[shuffle_idx[n_val:]]
	
	# # unit scale
	# data_scaler = MinMaxScaler(feature_range=(0, 1))
	# data_scaler.fit(data_train)
	# data_train = data_scaler.transform(data_train)
	# data_val = data_scaler.transform(data_val)
	# data_test = data_scaler.transform(data_test)
	
	# # supervised split
	# train_x, train_y = data_train[:,:-1], data_train[:,-1] # xyt, z
	# val_x, val_y = data_val[:,:-1], data_val[:,-1]
	# test_x, test_y = data_test[:,:-1], data_test[:,-1]
	
	# # determine batch counts
	# train_steps = int(jnp.ceil(train_x.shape[0] / batch_size))
	# val_steps = int(jnp.ceil(val_x.shape[0] / batch_size))
	# test_steps = int(jnp.ceil(test_x.shape[0] / batch_size))
	
	# return (train_x, train_y, train_steps), (val_x, val_y, val_steps), (test_x, test_y, test_steps), data_scaler
