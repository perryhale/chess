# numerics
import jax
import jax.numpy as jnp
import numpy as np

# environment
import chess # python-chess


### action encoding














DIRECTIONS = [
    (0, 1),    # N
    (1, 1),    # NE
    (1, 0),    # E
    (1, -1),   # SE
    (0, -1),   # S
    (-1, -1),  # SW
    (-1, 0),   # W
    (-1, 1),   # NW
]

KNIGHT_MOVES = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
]

UNDERPROMOTIONS = [
    chess.KNIGHT, chess.BISHOP, chess.ROOK
]


def uci_to_vec4672(uci: str) -> np.ndarray:
    move = chess.Move.from_uci(uci)

    f0 = chess.square_file(move.from_square)
    r0 = chess.square_rank(move.from_square)
    f1 = chess.square_file(move.to_square)
    r1 = chess.square_rank(move.to_square)

    df, dr = f1 - f0, r1 - r0

    if move.promotion in UNDERPROMOTIONS:
        # -1, 0, +1 files relative to origin
        if abs(df) > 1 or abs(dr) != 1:
            raise ValueError("Invalid underpromotion geometry")

        direction_idx = df + 1
        piece_idx = UNDERPROMOTIONS.index(move.promotion)
        channel = 64 + direction_idx * 3 + piece_idx

    elif (df, dr) in KNIGHT_MOVES:
        channel = 56 + KNIGHT_MOVES.index((df, dr))

    else:
        distance = max(abs(df), abs(dr))

        if distance == 0 or not (
            df == 0 or dr == 0 or abs(df) == abs(dr)
        ):
            raise ValueError("Invalid move geometry")

        direction = (
            int(np.sign(df)),
            int(np.sign(dr))
        )

        direction_idx = DIRECTIONS.index(direction)
        channel = direction_idx * 7 + distance - 1

    vec = np.zeros((8, 8, 73), dtype=np.float32)
    vec[r0, f0, channel] = 1.0

    return vec


def vec4672_to_uci(vec: np.ndarray, board: chess.Board) -> str:
    """
    Decode a policy tensor, selecting the highest-scoring
    legal move. Accepts one-hot vectors, logits or probabilities.
    """
    scores = np.asarray(vec).reshape(8, 8, 73)

    legal_moves = list(board.legal_moves)

    if not legal_moves:
        raise ValueError("No legal moves available")

    best_move = max(
        legal_moves,
        key=lambda move: scores[
            chess.square_rank(move.from_square),
            chess.square_file(move.from_square),
            np.argmax(uci_to_vec4672(move.uci())) % 73
        ]
    )

    return best_move.uci()













def uci_to_vec4096(uci: str) -> np.ndarray:
    move = chess.Move.from_uci(uci)

    idx = move.from_square * 64 + move.to_square

    vec = np.zeros(ACTION_SIZE, dtype=np.float32)
    vec[idx] = 1.0
    return vec


def vec4096_to_uci(vec: np.ndarray, board: chess.Board) -> str:
    idx = int(np.argmax(vec))

    from_sq, to_sq = divmod(idx, 64)

    move = chess.Move(from_sq, to_sq)

    # Automatically promote to queen when appropriate
    piece = board.piece_at(from_sq)

    if (
        piece is not None
        and piece.piece_type == chess.PAWN
        and chess.square_rank(to_sq) in (0, 7)
    ):
        move.promotion = chess.QUEEN

    return move.uci()























### state encoding


# type: (chess.Board) -> np.ndarray
def board_to_vec837(board):
	
	# init array
	vec837 = np.zeros((837), dtype=np.float32)
	
	# init constants
	board_offset = len(board.piece_map())
	
	# set position state OHE vectors: (class) 0-5:white, 6-11:black
	for pos_idx, piece in board.piece_map().items():
		cat_idx = piece.piece_type - 1
		if not piece.color:
			cat_idx += 6
		vec837[pos_idx * 12 + cat_idx] = 1.0
	
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




### pipeline


# type: (str) -> tuple[np.ndarray, list[str]]
def pgn_to_supervised(pgn_path, max_games=None):
	states = []
	actions = []
	game_counter = 0
	
	# stream up to #max_games games from lichess PGN file
	with open(pgn_path, "r", encoding="utf-8") as pgn:
		while True:
			
			# init game if available
			game = chess.pgn.read_game(pgn)
			if game is None or game_counter >= max_games:
				break
			game_counter += 1
			
			# unpack moves from game
			for move in game.mainline_moves():
				states.append(board_to_vec837(game.board()))
				actions.append(move.uci())
				board.push(move)
	
	return np.asarray(states, dtype=np.float32), actions


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
