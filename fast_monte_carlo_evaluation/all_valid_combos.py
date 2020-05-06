from itertools import combinations

card_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]

holecard_combos = list(combinations(card_indices[0:4], 2))
board_combos = list(combinations(card_indices[4:], 3))
results = []
for holecard_combo in holecard_combos:
	for board_combo in board_combos:
		results.append(list(holecard_combo) + list(board_combo))
for result in results:
	print("{", end="")
	for x in range(0, len(result)):
		out = str(result[x])
		if x != len(result) - 1:
			out += ", "
		print(out, end="")
	print("},")
print(len(results))