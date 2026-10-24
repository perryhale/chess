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




### state encoder / decoder / masker (positions OHE + turn + castling + enpassant)

# type: (chess.Board) -> np.ndarray
def board_to_vec837(board):
	
	# init array
	vec837 = np.zeros((837), dtype=np.float32)
	
	# init constants
	board_offset = 64 * 12
	
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


###! + state decoder

# # type: (np.ndarray) -> chess.Board
# def vec837_to_board(vec837):
	# vec837 = np.asarray(vec837)
	# if vec837.size != 837:
		# raise ValueError(f"Expected 837 values, got {vec837.size}")

	# vec837 = vec837.reshape(837)
	# board = chess.Board(None)  # start with an empty board

	# # Decode piece positions: 64 squares × 12 classes
	# pieces = vec837[:768].reshape(64, 12)
	# for square, row in enumerate(pieces):
		# class_idx = int(np.argmax(row))
		# if row[class_idx] > 0.5:
			# piece_type = (class_idx % 6) + 1
			# color = class_idx < 6  # classes 0–5 white, 6–11 black
			# board.set_piece_at(square, chess.Piece(piece_type, color))

	# # Decode turn
	# state = vec837[768:]
	# board.turn = bool(state[0])

	# # Decode castling rights. python-chess stores these as rook-square bits.
	# rights = chess.BB_EMPTY
	# for color, kingside_idx, queenside_idx in (
		# (chess.WHITE, 1, 2),
		# (chess.BLACK, 3, 4),
	# ):
		# rank = 0 if color == chess.WHITE else 7
		# king = board.king(color)

		# if king is not None:
			# if state[kingside_idx] > 0.5:
				# rooks = [
					# sq for sq in chess.SQUARES
					# if chess.square_rank(sq) == rank
					# and chess.square_file(sq) > chess.square_file(king)
					# and board.piece_at(sq) == chess.Piece(chess.ROOK, color)
				# ]
				# if rooks:
					# rights |= chess.BB_SQUARES[max(rooks)]

			# if state[queenside_idx] > 0.5:
				# rooks = [
					# sq for sq in chess.SQUARES
					# if chess.square_rank(sq) == rank
					# and chess.square_file(sq) < chess.square_file(king)
					# and board.piece_at(sq) == chess.Piece(chess.ROOK, color)
				# ]
				# if rooks:
					# rights |= chess.BB_SQUARES[min(rooks)]

	# board.castling_rights = rights

	# # Decode en-passant square (state indices 5–68)
	# ep = state[5:69]
	# board.ep_square = int(np.argmax(ep)) if ep.max() > 0.5 else None

	# return board


###! + E4672 action legality mask

# type: (np.ndarray) -> np.ndarray
def vec837_to_mask4672(vec837):
	mask = np.full(4672, -np.inf, dtype=np.float32)
	for move in vec837_to_board(vec837).legal_moves:
		mask[move_to_idx4672(move)] = 0.0
	return mask




### action encoder / decoder (AlphaZero format)

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


# type: (int, np.ndarray) -> chess.Move
def idx4672_to_move(idx4672, vec837):
	
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
	elif action < 64:
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
	
	# handle implicit promotion (directly from state encoding)
	is_pawn = vec837[(r0 * 8 + f0) * 12] in (1, 6)
	is_prom = r1 in (0, 7)
	is_impl = promotion_class == None
	if is_pawn and is_prom and is_impl:
		promotion_class = chess.QUEEN
	
	# construct chess.Move and return
	return chess.Move(from_square, to_square, promotion=promotion_class)




### pipeline

###! impl scope: stream to batch from pgn / stream blocks to memory (+async)

# type: (TextIO, int, float, int) -> tuple[np.ndarray, np.ndarray]
def stream_pgn_to_supervised(pgn_textio, max_moves=-1, max_memory=0.95, debug=-1):
	states = []
	actions = []
	move_counter = 0
	max_memory_percent = 100 * max_memory
	
	# stream up to #max_games games from lichess PGN file
	while max_moves < 0 or move_counter < max_moves:
		
		# mem safety
		memory_percent = psutil.virtual_memory().percent
		if memory_percent > max_memory_percent:
			raise RuntimeError(f"memory_percent > max_memory_percent, {memory_percent} > {max_memory_percent}")
		
		# initialise game (check EOF)
		game = chess.pgn.read_game(pgn_textio)
		if game is None:
			break
		board = game.board()
		
		# encode all states and moves from game (check max_moves)
		for move in game.mainline_moves():
			if max_moves < 0 or move_counter < max_moves:
				states.append(board_to_vec837(board))
				actions.append(move_to_idx4672(move))
				board.push(move)
				move_counter += 1
				if debug > 0 and move_counter % debug == 0:
					print(f"{move_counter} {memory_percent}%")
	
	# construct numpy arrays
	data_x = np.asarray(states, dtype=np.float32)
	data_y = np.asarray(actions, dtype=np.int32)
	
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
