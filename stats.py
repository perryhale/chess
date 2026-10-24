import time
import chess.pgn

# start timer
T0 = time.time()

def count_games(pgn_path):
	count = 0
	print(f"{count} [Elapsed time: {time.time()-T0:.2f}s]")
	with open(pgn_path, encoding="utf-8") as pgn:
		while True:
			game = chess.pgn.read_game(pgn)
			if game is None:
				break
			for move in game.mainline_moves():
				count += 1
			if count % 1000 == 0:
				print(f"{count} [Elapsed time: {time.time()-T0:.2f}s]")

count_games("data/lichess_db_standard_rated_2017-03.pgn")
