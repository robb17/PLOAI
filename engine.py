from itertools import combinations
from copy import deepcopy
from multiprocessing import Process, Manager
from ctypes import *
fast_mc_evaluation = CDLL("fast_monte_carlo_evaluation/podds.so")

import multiprocessing as mp
import random
import time
import pickle
import sys
import kmeans
import numpy as np
#from scipy.cluster.vq import kmeans
from scipy.stats import wasserstein_distance
from sklearn.cluster import SpectralClustering


N_CORES = 16

P1_STACK = 0
P2_STACK = P1_STACK + 1
P1_BET = P2_STACK + 1
P2_BET = P1_BET + 1
P1_HOLECARDS = P2_BET + 1
P2_HOLECARDS = P1_HOLECARDS + 1
POT = P2_HOLECARDS + 1
TO_ACT = POT + 1
CURRENT_STREET = TO_ACT + 1
BOARD = CURRENT_STREET + 1
ACTIONS = BOARD + 1

LOWEST_BUCKET = 5

P1 = "P1"
P2 = "P2"

STARTING_STACK = 200

# Example action histories
# ["B1", "R1", "C1", "X", -1, "X", "B2", "C2", -1, "X", "X", -1, "B2", "R10", "R10", "C10"]
# ["B1", "R1", "F"]

DECK = ['As', 'Ks', 'Qs', 'Js', 'Ts', '9s', '8s', '7s', '6s', '5s', '4s', '3s', '2s', \
		'Ac', 'Kc', 'Qc', 'Jc', 'Tc', '9c', '8c', '7c', '6c', '5c', '4c', '3c', '2c', \
		'Ah', 'Kh', 'Qh', 'Jh', 'Th', '9h', '8h', '7h', '6h', '5h', '4h', '3h', '2h', \
		'Ad', 'Kd', 'Qd', 'Jd', 'Td', '9d', '8d', '7d', '6d', '5d', '4d', '3d', '2d']

CARDS_TO_DEAL_BY_STREET = [4, 3, 1, 1]

RANK_DICT = {"2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6, "9": 7, "T": 8, "J": 9, "Q": 10, "K": 11, "A": 12}
SUIT_DICT = {"s": 0, "h": 1, "d": 2, "c": 3}

#class MKMeans:
#	def __init__(self, mat, iterations = 1000, n_clusters = 10):
#		self.mat = mat
#		self.iterations = iterations
#		self.n_clusters = 10
#		self.means = self.initialize_means()
#
#	def initialize_means(self):
#		means = []
#		for z in range(0, self.n_clusters):
#			means.append(self.select_means())
#		return means
#
#	def select_means(self):
#		x = 0
#		y = 0
#		while True:
#			x = random.randint(0, len(self.mat) - 1)
#			y = random.randint(0, len(self.mat) - 1)
#			if y > x:
#				break
#		return (x, y)
#
#	def fit(self):
#		for i in range(0, self.iterations):
#			means = self.select_means()


class memoize:
	def __init__(self, f):
		self.f = f
		self.memo = {}

	def __call__(self, *args):
		key_str = str(self.f.__name__)
		for arg in args:
			try:
				key_str += "".join([str(item) for item in arg])
			except TypeError as e:
				key_str += str(arg)		#it's not a list, it's an index
		if key_str not in self.memo:
			if len(args) == 1:
				self.memo[key_str] = self.f(args[0])
			elif len(args) == 2:
				self.memo[key_str] = self.f(args[0], args[1])
			elif len(args) == 3:
				self.memo[key_str] = self.f(args[0], args[1], args[2])
			elif len(args) == 4:
				self.memo[key_str] = self.f(args[0], args[1], args[2], args[3])
		return self.memo[key_str]

	def clear(self):
		self.memo = {}

	def getLength(self):
		print(len(self.memo))

class DSBD():
	''' Distinct-set bidirectional dictionary: keys are guaranteed to be distinct from values
	'''
	def __init__(self, d = None, inverse = None):
		self.d = d
		if not d:
			self.d = {}
		self.inverse = inverse
		if not self.inverse:
			self.inverse = {}

	def __setitem__(self, key, value):
		if self.d.get(key):
			self.d[key].append(value)
		else:
			self.d[key] = [value]
		self.inverse[value] = key

	def __delitem__(self, key):
		all_vals = self.d[key]
		self.d.remove(key)
		for val in all_vals:
			self.inverse.remove(val)

	def __getitem__(self, key):
		return self.d[key]

	def keys(self):
		return self.d.keys()

	def values(self):
		return self.d.values()

	def __len__(self):
		return len(self.d)

	def get(self, key):
		return self.d.get(key)


class Game:
	@staticmethod
	def generate_quantized_actions(gamestate):
		''' Generate either 5 (if checked to player) or 6 actions with
			quantized betting sizes, ranging from 1/4 to full pot
		'''
		if gamestate[ACTIONS][-1] == -1:
			return []
		most_recent_action = gamestate[ACTIONS][-1][0]
		if gamestate[P1_STACK] == 0 or gamestate[P2_STACK] == 0:
			return ['X']	# force check-down because no more chips to bet
		if most_recent_action == "F":
			return []
		actions = []
		if most_recent_action == 'B' or most_recent_action == 'R':
			opponent_last_bet = P1_BET if gamestate[TO_ACT] == P2 else P2_BET
			current_player_last_bet = P1_BET if opponent_last_bet == P2_BET else P2_BET
			call_size = gamestate[opponent_last_bet] - gamestate[current_player_last_bet]
			actions.append("F")
			actions.append("C" + str(call_size))
			for x in range(1, 5):
				raise_size = int(((gamestate[POT] + call_size) * x) / 4)	# calling the prior bet is implicit in a raise size
				if Game._is_legal_raise(raise_size, gamestate, gamestate[current_player_last_bet], gamestate[opponent_last_bet]):
					actions.append("R" + str(raise_size))
		elif most_recent_action == 'X' or (most_recent_action == 'C' and \
				len(gamestate[ACTIONS]) == 3):	# "C" to give BB the option
			actions.append("X")
			for x in range(1, 5):
				size = int((gamestate[POT] * x) / 4)
				if size >= 2:	# must bet at least the BB
					actions.append("B" + str(size))
		return actions

	@staticmethod
	def perform_action(gamestate, action):
		''' Apply the action to the gamestate and return the resulting gamestate
		'''
		gamestate[ACTIONS].append(action)
		winner = None
		if action[0] == "F":
			winner = P1 if gamestate[TO_ACT] == P2 else P2
		amount = 0 if action[0] == "X" or action[0] == "F" else int(action[1:])
		player_stack = P1_STACK if gamestate[TO_ACT] == P1 else P2_STACK
		gamestate[player_stack] -= amount
		gamestate[POT] += amount
		player_bet = P1_BET if gamestate[TO_ACT] == P1 else P2_BET
		gamestate[player_bet] += amount
		gamestate[TO_ACT] = P1 if gamestate[TO_ACT] == P2 else P2
		action_is_alive = Game._is_action_alive(gamestate)
		if not action_is_alive:
			gamestate[P1_BET] = 0   		# reset all bets (push them to the pot)
			gamestate[P2_BET] = 0
			gamestate[TO_ACT] = P2 			# P2 always acts first after the flop
			gamestate[CURRENT_STREET] += 1
			gamestate[ACTIONS].append(-1)				# dummy value to indicate break in streets
		if winner:
			gamestate[TO_ACT] = winner		# set to_act to winner, for awarding pot later
		return gamestate, not action_is_alive

	@staticmethod
	def _is_action_alive(gamestate):
		'''	Returns True if the newly created gamestate represents one in which further
			action is possible on the current street
		'''
		if gamestate[ACTIONS][-1] == -1:
			return False
		most_recent_action = gamestate[ACTIONS][-1][0]
		return most_recent_action == "B" or most_recent_action == "R" or \
				(most_recent_action == "X" and gamestate[ACTIONS][-2] != -1 and gamestate[ACTIONS][-2][0] == "X") \
				or (most_recent_action == "C" and len(gamestate[ACTIONS]) == 3)	# give BB the option

	@staticmethod
	def _is_legal_raise(size, gamestate, current_player_last_bet, opponent_last_bet):
		''' Raise must be of size at least 2 and be >= the previous bet/raise
		'''
		return (size >= 2 and size >= (opponent_last_bet - current_player_last_bet))

	@staticmethod
	def generate_all_dealt_card_combos(gamestate):
		'''	Generate all possible ways to deal the number of cards appropriate
			for the given street.

			Perhaps memoization would help here.
		'''
		deck = deepcopy(DECK)
		if gamestate[P1_HOLECARDS] != None:
			for card in gamestate[P1_HOLECARDS]:
				deck.remove(card)
			for card in gamestate[P2_HOLECARDS]:
				deck.remove(card)
		for card in gamestate[BOARD]:
			deck.remove(card)
		combos = combinations(deck, CARDS_TO_DEAL_BY_STREET[gamestate[CURRENT_STREET]])
		new_combos = [] #[0 for x in range(0, 270725 * 194580)]
		if gamestate[CURRENT_STREET] == 0:
			for combo in combos:
				new_deck = deepcopy(deck)
				for card in combo:
					new_deck.remove(card)
				additional_card_combos = combinations(new_deck, CARDS_TO_DEAL_BY_STREET[gamestate[CURRENT_STREET]])
				for additional_cardset in additional_card_combos:
					new_combos.append("".join(combo) + "".join(additional_cardset))
		return new_combos

# Memoization on sorted hand would help with determining hand strength/comparing hands
class PotManager:
	@staticmethod
	def award_pot(gamestate):
		if gamestate[ACTIONS][-2] == "F":
			PotManager._add_to_winning_stack(gamestate, gamestate[TO_ACT])
		else:
			hand_1 = Hand(gamestate[P1_HOLECARDS], gamestate[BOARD], 2, True)
			hand_2 = Hand(gamestate[P2_HOLECARDS], gamestate[BOARD], 2, True)
			if hand_1.compare_to(hand_2) == 1:
				PotManager._add_to_winning_stack(P1)
			elif hand_1.compare_to(hand_2) == -1:
				PotManager._add_to_winning_stack(P2)
			else:	# chop
				gamestate[P1_STACK] = STARTING_STACK
				gamestate[P2_STACK] = STARTING_STACK
		return gamestate

	@staticmethod
	def _add_to_winning_stack(gamestate, winner):
		winning_stack = P1_STACK if winner == P1 else P2_STACK
		gamestate[winning_stack] += gamestate[POT]
		gamestate[POT] = 0
		return gamestate

class BucketManager:
	def __init__(self, existing_pickle = None, pickle_to_write = None, existing_superbucket_pickle = None, superbucket_pickle_to_write = None, size_of_postflop_dictionary = 0.01):
		self.all_mappings = [None, None, None, None]
		self.existing_pickle = existing_pickle
		self.existing_superbucket_pickle = existing_superbucket_pickle
		self.superbucket_pickle_to_write = superbucket_pickle_to_write
		self.pickle_to_write = pickle_to_write
		self.size_of_postflop_dictionary = size_of_postflop_dictionary
		self.additional_cards = {7: 3, 8: 1, 9: 1}
		self.map()
		self.generate_superbuckets_wrap()
		#print(len(self.holecard_bucket_map))
		#test_holecards = ["As", "2h", "3d", "4c"]
		#test_holecards.sort(key=lambda x: (RANK_DICT[x[0]], SUIT_DICT[x[1]]), reverse=True)
		#test_bucket = self.bucket_holecards(test_holecards)
		#print(test_bucket)
		#test_holecards = ["As", "2h", "3d", "4c"]
		#test_holecards.sort(key=lambda x: (RANK_DICT[x[0]], SUIT_DICT[x[1]]), reverse=True)
		#test_bucket = self.bucket_holecards(test_holecards)
		#print(test_bucket)
		#print(self.holecard_bucket_map[self.bucket_holecards(test_holecards)])

	def map(self):
		bucket_recalculation_needed = True
		if self.existing_pickle:
			for x in range(0, len(self.all_mappings)):
				try:
					self.all_mappings[x] = pickle.load(open(self.existing_pickle + str(x), "rb"))
				except Exception as e:
					break
			bucket_recalculation_needed = False
		if bucket_recalculation_needed:
			self.all_mappings[0] = self.build_preflop_dict()
			self.all_mappings[1] = self.build_postflop_dict(7)
			self.all_mappings[2] = self.build_postflop_dict(8)
			self.all_mappings[3] = self.build_postflop_dict(9)
		if self.pickle_to_write:
			for x in range(0, len(self.all_mappings)):
				pickle.dump(self.all_mappings[x], open(self.pickle_to_write + str(x), "wb"))

	def generate_superbuckets_wrap(self):
		superbucket_generation_needed = False
		if self.existing_superbucket_pickle:
			try:
				self.all_groupings = pickle.load(open(self.existing_superbucket_pickle, "rb"))
			except Exception as e:
				superbucket_generation_needed = True
		if not self.existing_superbucket_pickle or superbucket_generation_needed:
			self.generate_superbuckets()
		if self.superbucket_pickle_to_write:
			pickle.dump(self.all_groupings, open(self.superbucket_pickle_to_write, "wb"))

	def generate_superbuckets(self):
		n_superbuckets = 10
		self.all_groupings = {}
		manager = Manager()
		opponent_hands_to_explore = manager.list()
		for street in range(0, 4):
			reference_superbucket_grouping = DSBD()
			superbucket_grouping = DSBD()
			print(reference_superbucket_grouping.d)
			print(superbucket_grouping.d)
			all_strengths = {}
			# First create reference superbuckets using the odds solver written in C
			
			raw_strength_values = []
			for bucket_key in self.all_mappings[street].keys():
				bucket = self.all_mappings[street][bucket_key]
				bucket_strength = []
				for x in range(0, max(1, min(len(bucket) // 2, 3))):
					sampled_hand = random.choice(bucket)
					remaining_deck = deepcopy(DECK)
					for y in range(0, len(sampled_hand) - 1, 2):
						remaining_deck.remove(sampled_hand[y:y + 2])
					current_player_hand = sampled_hand[:8]
					board = sampled_hand[8:]
					argc = len(current_player_hand) // 2 + len(board) // 2 + 2
					current_player_hand_txt = " ".join(current_player_hand[i:i+2] for i in range(0, len(current_player_hand), 2))
					board_txt = " ".join(board[i:i+2] for i in range(0, len(board), 2))
					args = "driver 2 " + current_player_hand_txt + " " + board_txt
					arguments = ((str(argc) + " " + args).strip()).encode()
					strength = fast_mc_evaluation.driver(arguments) / 10000.0
					#strength = manager.list()
					#jobs = []
					#n_comparison_hands = 75
					#for x in range(0, N_CORES):
					#	jobs.append(Process(target=self.monte_carlo_hand_strength_solver, args=(n_comparison_hands // N_CORES, remaining_deck, board, current_player_hand, strength,)))
					#for job in jobs:
					#	job.start()
					#for job in jobs:
					#	job.join()
					print("hand " + current_player_hand + " on a board of " + board + " wins with p = " + str(strength))
					bucket_strength.append(strength)
				all_strengths[bucket_key] = sum(bucket_strength) / len(bucket_strength)
				raw_strength_values.append(sum(bucket_strength) / len(bucket_strength))
			data = [((int(raw_strength_values[x] * 255), 0, 0), 1) for x in range(0, len(raw_strength_values))]
			centers = kmeans.kmeans(data, n_superbuckets)
			centers_lst = [element[0] / 255 for element in centers]
			centers_lst.sort()
			for bucket_key in self.all_mappings[street].keys():
				if all_strengths.get(bucket_key):
					reference_superbucket_grouping[self.group(all_strengths[bucket_key], centers_lst)] = bucket_key
					#print(bucket_key, self.group(all_strengths[bucket_key], centers_lst))

			# Now, compute new superbuckets using the histograms obtained when comparing buckets against the old superbuckets
			bucket_strength_against_all_superbuckets = {}
			for bucket_key in self.all_mappings[street].keys():
				bucket = self.all_mappings[street][bucket_key]
				for x in range(0, max(1, min(len(bucket) // 2, 3))):
					current_hand_strength_against_all_superbuckets = []
					sampled_hand = random.choice(bucket)
					remaining_deck = deepcopy(DECK)
					for y in range(0, len(sampled_hand) - 1, 2):
						remaining_deck.remove(sampled_hand[y:y + 2])
					current_player_hand = sampled_hand[:8]
					board = sampled_hand[8:]
					current_player_hand_txt = " ".join(current_player_hand[i:i+2] for i in range(0, len(current_player_hand), 2))
					board_txt = " ".join(board[i:i+2] for i in range(0, len(board), 2))
					if street != 0:
						jobs = []
						for x in range(0, N_CORES):
							jobs.append(Process(target=self.find_hands_to_satisfy_board, args=(opponent_hands_to_explore, board, remaining_deck,)))
						for job in jobs:
							job.start()
						for job in jobs:
							job.join()
						#print(opponent_hands_to_explore)
						opponent_hands_to_explore_dict = DSBD()
						for pair in opponent_hands_to_explore:
							#print(pair[1])
							opponent_hands_to_explore_dict[pair[0]] = pair[1]
							#print(opponent_hands_to_explore_dict.d)
						#print(opponent_hands_to_explore_dict.d)
					for z in range(0, n_superbuckets):
						if reference_superbucket_grouping.get(z):
							bucket_strength_against_superbucket = []
							for trial in range(0, 5):
								# only select hands with cards not in other bucket!
								player_2_bucket_keys = reference_superbucket_grouping[z]
								player_2_hand = None
								if street == 0:
									player_2_hand = self.iterate_until_distinct_sets(current_player_hand, None, street, player_2_bucket_keys)
								else: # find hand that, in combination with the given board, would yield the same bucket key
									for player_2_bucket_key in player_2_bucket_keys:
										if player_2_bucket_key in opponent_hands_to_explore_dict.keys():
											print(player_2_bucket_key)
											player_2_hand = random.choice(opponent_hands_to_explore_dict[player_2_bucket_key])
											break
									#player_2_hand = self.find_hand_to_satisfy_board_and_superbucket(board, player_2_bucket_keys, remaining_deck, street)
								if player_2_hand == None:	# no non-overlapping hands to pick from OR couldn't find hand to satisfy board and superbucket
									continue
								player_2_hand_txt = " ".join(player_2_hand[i:i+2] for i in range(0, len(player_2_hand), 2))
								argc = len(current_player_hand) // 2 + len(board) // 2 + len(player_2_hand) // 2 + 2
								args = "driver 2! " + current_player_hand_txt + " " + board_txt + " " + player_2_hand_txt
								arguments = ((str(argc) + " " + args).strip()).encode()
								strength = fast_mc_evaluation.driver(arguments) / 10000.0
								print("hand " + current_player_hand + " on a board of " + board + " wins against " + player_2_hand + " with p = " + str(strength))
								bucket_strength_against_superbucket.append(strength)
							if len(bucket_strength_against_superbucket) > 0:
								current_hand_strength_against_all_superbuckets.append(sum(bucket_strength_against_superbucket) / len(bucket_strength_against_superbucket))
							else:
								current_hand_strength_against_all_superbuckets.append(0.5)	# nothing to compete with: call it a draw
						else:
							current_hand_strength_against_all_superbuckets.append(0)
					opponent_hands_to_explore[:] = []
					bucket_strength_against_all_superbuckets = self.update_global_bucket_stats(bucket_strength_against_all_superbuckets, bucket_key, current_hand_strength_against_all_superbuckets)
				bucket_strength_against_all_superbuckets = self.get_average_bucket_performance(bucket_strength_against_all_superbuckets, bucket_key, n_superbuckets)

			bucket_em_matrix = np.zeros((len(self.all_mappings[street].keys()), len(self.all_mappings[street].keys())))
			all_keys = list(bucket_strength_against_all_superbuckets.keys())
			for x in range(0, len(all_keys)):
				bucket_key = all_keys[x]
				histogram = bucket_strength_against_all_superbuckets[bucket_key]
				for y in range(0, len(all_keys)):
					second_bucket_key = all_keys[y]
					em_distance = 0
					if bucket_key != second_bucket_key:
						second_histogram = bucket_strength_against_all_superbuckets[second_bucket_key]
						em_distance = wasserstein_distance(histogram, second_histogram)
						#print(em_distance, histogram, second_histogram)
					bucket_em_matrix[x][y] = em_distance
			new_clustering = SpectralClustering(n_superbuckets).fit_predict(bucket_em_matrix)

			for x in range(0, len(self.all_mappings[street].keys())):
				bucket_key = list(self.all_mappings[street].keys())[x]
				superbucket_grouping[new_clustering[x]] = bucket_key

			print(superbucket_grouping.inverse)
			self.all_groupings[street] = superbucket_grouping

	def iterate_until_distinct_sets(self, hand1, hand2, street, player_2_bucket_keys):
		count = 0
		while hand2 == None and count < 25:
			bucket = random.choice(player_2_bucket_keys)
			hand2 = random.choice(self.all_mappings[street][bucket])
			if self.check_for_duplicates(hand1, hand2):
				hand2 = None
			count += 1
		return hand2

	def check_for_duplicates(self, hand1, hand2):
		for x in range(0, min(len(hand1), len(hand2)), 2):
			substring = hand1[x] + hand1[x + 1]
			if substring in hand2:
				return True
		return False

	def find_hands_to_satisfy_board(self, opponent_hands_to_explore, board, remaining_deck):
		deck = deepcopy(remaining_deck)
		for x in range(0, 300 // N_CORES):
			cards = []
			deck_length = len(deck)
			for y in range(0, 4):
				index = random.randint(0, deck_length - 1)
				cards.append(deck[index])
				temp = deck[deck_length - 1]
				deck[deck_length - 1] = deck[index]
				deck[index] = temp
				deck_length -= 1
			card_str = "".join(cards)
			hand = Hand(card_str, board)
			label = hand.collect_categorical_data()
			opponent_hands_to_explore.append((label, card_str))


	def find_hand_to_satisfy_board_and_superbucket(self, board, bucket_labels, remaining_deck, street):
		#for bucket_label in bucket_labels:
		#	if self.all_mappings[street].get(bucket_label):
		#		all_cards = self.all_mappings[street][bucket_label]
		#		for cards in all_cards:
		#			if self.test_string_set_equality(cards[4:], board):
		#				print("success!")
		#				return cards[:4]
		deck = deepcopy(remaining_deck)
		for x in range(0, 500):
			cards = []
			deck_length = len(deck)
			for y in range(0, 4):
				index = random.randint(0, deck_length - 1)
				cards.append(deck[index])
				temp = deck[deck_length - 1]
				deck[deck_length - 1] = deck[index]
				deck[index] = temp
				deck_length -= 1
			card_str = "".join(cards)
			hand = Hand(card_str, board)
			label = hand.collect_categorical_data()
			if label in bucket_labels:
				return card_str
		return None

	def test_string_set_equality(self, str1, str2):
		if len(str1) != len(str2):
			return False
		for x in range(0, len(str1), 2):
			if str1[x: x + 2] not in str2:
				return False
		return True


	def update_global_bucket_stats(self, global_dict, bucket_key, new_values):
		if not global_dict.get(bucket_key):
			global_dict[bucket_key] = [new_values]
		else:
			global_dict[bucket_key].append(new_values)
		return global_dict

	def get_average_bucket_performance(self, d, key, n_superbuckets):
		n_lsts = len(d[key])
		new_lst = [0 for x in range(0, n_superbuckets)]
		for lst in d[key]:
			for x in range(0, n_superbuckets):
				new_lst[x] += lst[x]
		for x in range(0, len(new_lst)):
			new_lst[x] = new_lst[x] / n_lsts
		d[key] = new_lst
		return d

	@staticmethod
	def group(val, centers):
		for x in range(0, len(centers)):
			if val < centers[x]:
				if x == 0:
					return 0
				cmp1 = centers[x - 1]
				cmp2 = centers[x]
				if val - cmp1 < cmp2 - val:
					return x - 1
				else:
					return x
		return len(centers) - 1

	# too slow! replaced with c implementation
	def monte_carlo_hand_strength_solver(self, iterations, remaining_deck, board, current_player_hand, strength):
		for opponent_hand_index in range(0, iterations):
			opponent_hand = ""
			new_deck = deepcopy(remaining_deck)
			for z in range(0, 4):
				card = random.choice(new_deck)
				opponent_hand += card
				new_deck.remove(card)
			n_required_cards = (10 - len(board)) // 2
			wins = 0
			draws = 0
			n_iterations = 500
			for runout_iteration in range(0, n_iterations):
				new_cards = random.sample(new_deck, n_required_cards)
				board += "".join(new_cards)
				p1_hand = Hand(current_player_hand, board)
				p2_hand = Hand(opponent_hand, board)
				comparison_val = p1_hand.compare_to(p2_hand)
				if comparison_val == 1:
					wins += 1
				elif comparison_val == 0:
					draws += 1
				#print(board, len(new_deck))
				board = board[:len(board) - (n_required_cards * 2)]
			strength.append((wins + 0.5 * draws) / n_iterations)

	def build_preflop_dict(self):
		preflop_buckets = DSBD()
		all_holecards = combinations(DECK, 4)
		for holecards in all_holecards:
			bucket = self.bucket_holecards(list(holecards))
			key = list(holecards)
			key.sort(key=lambda x: (RANK_DICT[x[0]], SUIT_DICT[x[1]]), reverse=True)
			preflop_buckets[bucket] = "".join([str(item) for item in key])
		print("completed preflop bucket")
		return preflop_buckets

	def bucket_holecards(self, holecards):
		''' Doesn't capture A's power as a low card, but that's a pretty insignificant oversight
		'''
		encoding = ""
		holecards.sort(key=lambda x: (RANK_DICT[x[0]], SUIT_DICT[x[1]]), reverse=True)

		suit_counts = [0, 0, 0, 0]
		suit_highcards = [-1, -1, -1, -1]
		for card in holecards:
			card_suit = SUIT_DICT[card[1]]
			suit_counts[card_suit] += 1
			if RANK_DICT[card[0]] > suit_highcards[card_suit]:
				suit_highcards[card_suit] = RANK_DICT[card[0]]
		for x in range(0, 4):
			if suit_highcards[x] < 11 and suit_counts[x] == 1:
				suit_counts[x] = 0

		current_high_card = 12
		count = 0
		local_lowest_bucket = LOWEST_BUCKET
		for card in holecards:
			card_suit = SUIT_DICT[card[1]]
			card_rank = RANK_DICT[card[0]]
			encoding += str(min(current_high_card - card_rank, local_lowest_bucket))
			current_high_card = RANK_DICT[card[0]]
			local_lowest_bucket = max(2, local_lowest_bucket - 1)
			count += 1
			if count == 1 and current_high_card <= 9:
				local_lowest_bucket = 2
			if card_rank == suit_highcards[card_suit]:
				encoding += str(suit_counts[card_suit])
			else:
				encoding += str(0)
		if encoding[2] == "2" and encoding[4] == "1":
			''' Bucket hands like KQT9 and KJT9 together
			'''
			new_encoding = ""
			for x in range(0, len(encoding)):
				if x == 2:
					new_encoding += "1"
				elif x == 4:
					new_encoding += "2"
				else:
					new_encoding += encoding[x]
			encoding = new_encoding
		return encoding

	def build_postflop_dict(self, n_cards):
		manager = Manager()
		hand_to_bucket = manager.dict()
		previous_map = self.all_mappings[n_cards - 7]
		all_keys = list(previous_map.keys())
		jobs = []
		for x in range(0, N_CORES):
			#self.build_postflop_dict_help(n_cards, x, bucket_to_hand, hand_to_bucket, all_keys)
			jobs.append(Process(target=self.build_postflop_dict_help, args=(n_cards, x, hand_to_bucket, all_keys,)))
		for job in jobs:
			job.start()
		for job in jobs:
			job.join()
		print("built reverse dict for n_cards = " + str(n_cards))
		bucket_to_hand = {}
		hand_to_bucket = dict(hand_to_bucket)
		for hand in hand_to_bucket.keys():
			categorical_data = hand_to_bucket[hand]
			if bucket_to_hand.get(categorical_data):
				bucket_to_hand[categorical_data].append(hand)
			else:
				bucket_to_hand[categorical_data] = [hand]
		return_dict = DSBD(bucket_to_hand, hand_to_bucket)
		print("built forward dict for n_cards = " + str(n_cards))
		return return_dict

	def build_postflop_dict_help(self, n_cards, index, hand_to_bucket, all_keys):
		len_keys = len(all_keys)
		previous_map = self.all_mappings[n_cards - 7]
		old_len_bucket_to_hand = 0
		for bucket_key in all_keys[(index * len_keys) // N_CORES:((index + 1) * len_keys) // N_CORES]:
			#sampled_holecards_lst = random.choice(previous_map[bucket_key])
			n_iterations = 1 if n_cards == 7 else max(1, int(self.size_of_postflop_dictionary * 500)) # scale size of turn/river buckets with size of flop buckets
			for it in range(0, n_iterations):
				sampled_holecards = random.choice(previous_map[bucket_key])
				remaining_deck = deepcopy(DECK)
				for x in range(0, len(sampled_holecards) - 1, 2):
					remaining_deck.remove(sampled_holecards[x:x + 2])
				combos = combinations(remaining_deck, self.additional_cards[n_cards])
				for combo in combos:
					if n_cards == 7 and random.random() > self.size_of_postflop_dictionary:
						continue
					#print(sampled_holecards[:8], sampled_holecards[8:] + "".join([str(item) for item in combo]))
					hand = Hand(sampled_holecards[:8], sampled_holecards[8:] + "".join([str(item) for item in combo]))
					categorical_data = hand.collect_categorical_data()
					#if bucket_to_hand.get(categorical_data):
					#	bucket_to_hand[categorical_data].append(hand.holecard_str + hand.board_str)
					#else:
					#	bucket_to_hand[categorical_data].append(hand.holecard_str + hand.board_str)
					hand_to_bucket[hand.holecard_str + hand.board_str] = categorical_data


class Suit:
	Lookup = {0: "S", 1: "H", 2: "D", 3: "C", 4: "?"}

	def __init__(self, suit_int):
		self.suit_int = suit_int
		self.val = Suit.Lookup[suit_int]

	def __repr__(self):
		return self.val

	def as_int(self):
		return self.suit_int

class Rank:
	Lookup = {0: "2", 1: "3", 2: "4", 3: "5", 4: "6", 5: "7", 6: "8", 7: "9", 8: "T", 9: "J", 10: "Q", 11: "K", 12: "A", 13: "?"}

	def __init__(self, rank_int):
		self.rank_int = rank_int
		self.val = Rank.Lookup[rank_int]

	def __repr__(self):
		return self.val

	def as_int(self):
		return self.rank_int

class Card:
	def __init__(self, rank_int, suit_int):
		self.rank = Rank(rank_int)
		self.suit = Suit(suit_int)

	def __repr__(self):
		return repr(self.rank) + repr(self.suit)


class Hand:
	HandValue = {
		"straight flush": 9,
		"four of a kind": 8,
		"full house": 7,
		"flush": 6,
		"straight": 5,
		"three of a kind": 4,
		"two pair": 3,
		"pair": 2,
		"high card": 1
	}

	PairingStrengthRefactor = {
		1: 0,
		2: 1,
		3: 2,
		4: 3,
		7: 4,
		8: 4,
		9: 4
	}

	NKindStrength = {
		1: 1,
		2: 2,
		3: 4,
		4: 8
	}

	NKindIndex = {
		1: 1,
		2: 2,
		4: 3,
		8: 4
	}

	TwoSetsStrength = {
		3: 7,
		2: 3
	}

	def __init__(self, holecard_str, board, n_holecards_play = 2, is_exact = True):
		self.holecard_str = Hand._to_string(holecard_str)
		self.board_str = Hand._to_string(board)
		self.holecards = Hand._to_cards(holecard_str)
		self.board = Hand._to_cards(board)
		#self.remaining_deck_dict = self._to_cards(DECK)
		#for card in self.holecards + self.board:
		#	self.remaining_deck_dict.pop(repr(card))
		self.n_holecards_play = n_holecards_play
		self.is_exact = is_exact
		self.all_cards = []
		self.used_holecards = []
		self.used_holecard_ranks = []
		self.board_ranks = []
		self.board_suits = []
		self.ranks = []
		self.suits = []
		self.holecard_ranks = []
		self.holecard_suits = []
		self.max_strength = 1
		self.best_hand_index = 0 # useful for PLO
		self.straights_flushes = 0
		self.pairing_strength = 0
		self.generate_all_possible_hands()
		self.count_ranks_and_suits()
		self.count_holecard_ranks_and_suits()
		self.determine_canonical_hand_strength()

	def generate_all_possible_hands(self):
		for subset in combinations(self.holecards, self.n_holecards_play):
			possible_hand = []
			for card in subset:
				possible_hand.append(card)
			if self.is_exact:
				for board_subset in combinations(self.board, max(0, 5 - self.n_holecards_play)):
					possible_hand_with_board = deepcopy(possible_hand)
					for card in board_subset:
						possible_hand_with_board.append(card)
					possible_hand_with_board.sort(key=lambda c: c.rank.as_int())
					self.all_cards.append(possible_hand_with_board)
					self.used_holecards.append(subset)
			else:
				for card in self.board:
					possible_hand.append(card)
				possible_hand.sort(key=lambda c: c.rank.as_int())
				self.all_cards.append(possible_hand)

	def count_ranks_and_suits(self):
		for hand_no in range(0, len(self.all_cards)):
			self.ranks.append([0 for y in range(0, 13)])
			self.suits.append([0 for y in range(0, 4)])
			self.used_holecard_ranks.append([0 for y in range(0, 13)])
			for card in self.all_cards[hand_no]:
				self.ranks[hand_no][card.rank.as_int()] += 1
				self.suits[hand_no][card.suit.as_int()] += 1
			for card in self.used_holecards[hand_no]:
				self.used_holecard_ranks[hand_no][card.rank.as_int()] += 1

	def count_holecard_ranks_and_suits(self):
		self.holecard_ranks = [0 for y in range(0, 13)]
		self.holecard_suits = [0 for y in range(0, 4)]
		self.board_ranks = [0 for y in range(0, 13)]
		self.board_suits = [0 for y in range(0, 4)]
		for card in self.holecards:
			self.holecard_ranks[card.rank.as_int()] += 1
			self.holecard_suits[card.suit.as_int()] += 1
		for card in self.board:
			self.board_ranks[card.rank.as_int()] += 1
			self.board_suits[card.suit.as_int()] += 1

	def determine_canonical_hand_strength(self):
		for index in range(0, len(self.all_cards)):
			for i in range(4, 0, -1):
				if (Hand.NKindStrength[i] >= self.max_strength):
					if (Hand.is_n_kind(i, self.ranks[index])):
						if (self.max_strength == Hand.NKindStrength[i]):
							cmpr = self.is_better_n_kind(self.ranks[index], self.ranks[self.best_hand_index], i)
							if cmpr != -1:	# if better, this is the actual best hand. Don't worry about tie (only care when comparing others' hands)
								self.best_hand_index = index
						else:
							self.best_hand_index = index
						self.max_strength = Hand.NKindStrength[i]
						self.pairing_strength = Hand.PairingStrengthRefactor[self.max_strength]
						break
			for i in range(3, 1, -1):
				if (Hand.TwoSetsStrength[i] >= self.max_strength):
					if (Hand.is_two_sets(i, self.ranks[index])):
						if (self.max_strength == Hand.TwoSetsStrength[i]):
							cmpr = self.is_better_double_set(self.ranks[index], self.ranks[self.best_hand_index], i)
							if cmpr != -1:	# if better, this is the actual best hand. Don't worry about tie (only care when comparing others' hands)
								self.best_hand_index = index
						else:
							self.best_hand_index = index
						self.max_strength = Hand.TwoSetsStrength[i]
						self.pairing_strength = Hand.PairingStrengthRefactor[self.max_strength]
						break
			if (Hand.is_flush(self.suits[index])):
				if (self.max_strength == Hand.HandValue["flush"]):
					cmpr = self.is_better_flush(self.all_cards[index], self.suits[index], self.all_cards[self.best_hand_index], self.suits[self.best_hand_index])
					if cmpr != -1:	# if better, this is the actual best hand. Don't worry about tie (only care when comparing others' hands)
						self.best_hand_index = index
				else:
					self.best_hand_index = index
				self.max_strength = Hand.HandValue["flush"] if (Hand.HandValue["flush"] >= self.max_strength) else self.max_strength
				self.straights_flushes = 2
			if (Hand.is_straight(self.ranks[index])):
				if (self.max_strength == Hand.HandValue["straight"]):
					cmpr = self.is_better_straight(self.ranks[index], self.ranks[self.best_hand_index])
					if cmpr != -1:	# if better, this is the actual best hand. Don't worry about tie (only care when comparing others' hands)
						self.best_hand_index = index
				else:
					self.best_hand_index = index
				self.max_strength = Hand.HandValue["straight"] if (Hand.HandValue["straight"] >= self.max_strength) else self.max_strength
				self.straights_flushes = max(self.straights_flushes, 1)
			if (self.is_straight_flush(index)):
				if (self.max_strength == Hand.HandValue["straight flush"]):
					cmpr = self.is_better_straight_flush(self.all_cards[index], self.all_cards[self.best_hand_index])
					if cmpr != -1:	# if better, this is the actual best hand. Don't worry about tie (only care when comparing others' hands)
						self.best_hand_index = index
				else:
					self.best_hand_index = index
				self.max_strength = Hand.HandValue["straight flush"]
				self.pairing_strength = Hand.PairingStrengthRefactor[self.max_strength]

	def collect_categorical_data(self):
		# check for flushes, straights
		high_holecard_suit_indices = []
		for card in self.holecards:
			if card.rank.as_int() == 12 or card.rank.as_int() == 11:
				high_holecard_suit_indices.append(card.suit.as_int())
		#n_sd_backdoor_turns = 0
		n_sds = 0
		is_flush = False
		is_straight = False
		visited_straight_outs = []
		visited_sd_backdoors = []
		f_possible = Hand.flush_possible(self.board_suits)
		n_straights_p = Hand.n_straights_possible(self.board_ranks)
		for index in range(0, len(self.all_cards)):
			is_flush = True if Hand.is_flush(self.suits[index]) else is_flush
			is_straight = True if Hand.is_straight(self.ranks[index]) else is_straight
			incremental_sds, visited_straight_outs = Hand.n_straight_outs(self.ranks[index], self.used_holecard_ranks[index], visited_straight_outs)
			n_sds += incremental_sds
			#incremental_sd_backdoors, visited_sd_backdoors = Hand.n_sd_equity_increasing_turns(self.ranks[index], self.used_holecard_ranks[index], visited_sd_backdoors)
			#n_sd_backdoor_turns += incremental_sd_backdoors
		f_blockers = False if is_flush else Hand.is_blocking_flush(self.board_suits, self.holecard_suits, high_holecard_suit_indices)
		s_blockers = 0 if is_straight else Hand.n_straight_blockers(self.board_ranks, self.holecard_ranks)
		n_fd_backdoor_turns = 0 if is_flush or len(self.board) > 3 or self.pairing_strength == 4 else Hand.n_fd_equity_increasing_turns(self.board_suits, self.holecard_suits)
		#n_sd_backdoor_turns = 0 if is_straight or len(self.board) > 3 or self.pairing_strength == 4 else n_sd_backdoor_turns
		n_sds = 0 if is_straight or self.pairing_strength == 4 else n_sds
		n_fds = 0 if is_flush or self.pairing_strength == 4 else Hand.n_flush_draws(self.board_suits, self.holecard_suits)
		#overcards = Hand.has_overcards(self.board_ranks, self.holecard_ranks) if (self.pairing_strength <= 3 and self.straights_flushes == 0) else False

		''' Condense representation as follows
		'''
		f_blockers = 1 if f_blockers else 0
		n_sds = min(int(n_sds // 2), 2)
		s_blockers = 1 if s_blockers > 2 else 0
		n_straights_p = 1 if n_straights_p >= 2 else 0
		f_possible = 1 if f_possible else 0
		n_fds = 1 if n_fds == 0 and n_fd_backdoor_turns > 16 else n_fds
		n_fds = 0 if len(self.board) == 5 else n_fds	# draws are worthless on the river, doesn't matter if you have a draw or not--you can play all your bluffs at this point the same
		n_sds = 0 if len(self.board) == 5 else n_sds


		return n_fds, f_blockers, n_sds, s_blockers, self.pairing_strength, self.straights_flushes, n_straights_p, f_possible, Hand.n_board_pairs(self.board_ranks)
		

	@staticmethod
	@memoize
	def _to_cards(input_rep, dict = False):
		''' Accepts list and string representations of cards
		'''
		cards = []
		if isinstance(input_rep, list) or isinstance(input_rep, tuple):
			for card in input_rep:
				cards.append(Card(RANK_DICT[card[0]], SUIT_DICT[card[1]]))
		else:
			for x in range(0, len(input_rep) // 2):
				cards.append(Card(RANK_DICT[input_rep[2 * x]], SUIT_DICT[input_rep[(2 * x) + 1]]))
		return cards

	@staticmethod
	@memoize
	def _to_string(cards):
		''' Accepts list and string representations of cards
		'''
		if isinstance(cards, str):
			return cards
		else:
			return "".join(cards)

	def get_hand_strength(self):
		return self.max_strength

	def compare_to(self, hand2):
		val = -1
		if self.max_strength > hand2.get_hand_strength():
			return 1
		if self.max_strength == hand2.get_hand_strength():
			if self.max_strength in self.NKindStrength.values():
				n_kindedness = self.NKindIndex[self.max_strength]
				return self.is_better_n_kind(self.ranks[self.best_hand_index], hand2.ranks[hand2.best_hand_index], n_kindedness)
			elif self.max_strength in self.TwoSetsStrength.values():
				setedness = 2 if self.max_strength == 3 else 3
				return self.is_better_double_set(self.ranks[self.best_hand_index], hand2.ranks[hand2.best_hand_index], setedness)
			elif self.max_strength == 5:
				return self.is_better_straight(self.ranks[self.best_hand_index], hand2.ranks[hand2.best_hand_index])
			elif self.max_strength == 6:
				return self.is_better_flush(self.all_cards[self.best_hand_index], self.suits[self.best_hand_index], hand2.all_cards[hand2.best_hand_index], hand2.suits[hand2.best_hand_index])
			elif self.max_strength == 9:
				return self.is_better_straight_flush(self.all_cards[self.best_hand_index], hand2.all_cards[hand2.best_hand_index])
		return -1

	@staticmethod
	@memoize
	def flush_possible(board_suits):
		'''	Returns whether or not the board contains at least three of the same suit
		'''
		for x in range(0, len(board_suits)):
			if (board_suits[x] >= 3):
				return True
		return False

	@staticmethod
	@memoize
	def n_straights_possible(board_ranks):
		'''	Counts the number of straights possible on the given board texture
		'''
		count = 0
		for x in range(0, len(board_ranks)):
			if board_ranks[x] >= 1:
				continue
			board_ranks[x] += 1
			for y in range(x, len(board_ranks)):
				if x == y or board_ranks[y] >= 1:
					continue
				board_ranks[y] += 1
				if Hand.is_straight(board_ranks):
					count += 1
				board_ranks[y] -= 1
			board_ranks[x] -= 1
		return count

	@staticmethod
	@memoize
	def n_board_pairs(board_ranks):
		''' Returns 0: no pair
					1: one pair
					2: two pairs
					3: 3K on board or better
		'''
		pair = 0
		for x in range(0, len(board_ranks)):
			if board_ranks[x] == 2:
				pair += 1
			elif board_ranks[x] >= 3:
				pair = 3
				break
		return pair

	@staticmethod
	@memoize
	def n_fd_equity_increasing_turns(board_suits, holecard_suits):
		''' Quantized because we really only care if one of the desirable
			suits hits or not. Not interested in the exact number
		'''
		count = 0
		remaining_suits = [13 - i for i in board_suits]
		for x in range(0, len(remaining_suits)):
			remaining_suits[x] -= holecard_suits[x]
		for index in range(0, 4):
			if board_suits[index] == 1 and holecard_suits[index] >= 2:
				count += remaining_suits[index]
		return count

	@staticmethod
	@memoize
	def n_sd_equity_increasing_turns(ranks, used_holecard_ranks, visisted_backdoors, visited_straight_outs):
		''' Quantized because having two good turns as opposed to 3 isn't all that different
		'''
		count = 0
		non_populated_indices = [i for i, a in enumerate(ranks) if a == 0]
		decrementable_indices = [i for i, a in enumerate(ranks) if a >= 1 and ranks[i] > used_holecard_ranks[i]]

		for d_i in decrementable_indices:
			new_ranks = deepcopy(ranks)
			new_ranks[d_i] -= 1
			for index in non_populated_indices:
				if index in visisted_backdoors:	# dont double-count the same out
					continue
				new_ranks[index] += 1
				possible_outs, visited_straight_outs = Hand.n_straight_outs(new_ranks, used_holecard_ranks, visited_straight_outs)
				if possible_outs > current_outs:
					visisted_backdoors.append(index)
					count += 1
				new_ranks[index] -= 1
		return count, visisted_backdoors, visited_straight_outs

	@staticmethod
	@memoize
	def n_straight_outs(ranks, used_holecard_ranks, visited_straight_outs):
		'''	Returns many values because a hand could have anywhere from 0 to 6 outs
		'''
		count = 0
		non_populated_indices = [i for i, a in enumerate(ranks) if a == 0]
		decrementable_indices = [i for i, a in enumerate(ranks) if a >= 1 and ranks[i] > used_holecard_ranks[i]]

		for d_i in decrementable_indices:
			new_ranks = deepcopy(ranks)
			new_ranks[d_i] -= 1
			for index in non_populated_indices:
				if index in visited_straight_outs:	# dont double-count the same out
					continue
				new_ranks[index] += 1
				if Hand.is_straight(new_ranks):
					visited_straight_outs.append(index)
					count += 1
				new_ranks[index] -= 1
		return count, visited_straight_outs

	@staticmethod
	@memoize
	def n_flush_draws(board_suits, holecard_suits):
		'''	Can be 2-hot because you either have two of the required suit or not
		'''
		count = 0
		for x in range(0, len(board_suits)):
			if (board_suits[x] == 2) and holecard_suits[x] >= 2:
				count += 1
		return count


	@staticmethod
	@memoize
	def fd_suits(suits, holecard_suits):
		''' Returns each of the suits in which the player has a flush draw
		'''
		fd_suits = []
		if Hand.n_flush_draws(suits, holecard_suits) == 1:
			for x in range(0, len(suits)):
				if suits[x] == 4 and holecard_suits[x] == 2:
					fd_suits.append(x)
		return fd_suits

	@staticmethod
	@memoize
	def has_overcards(board_ranks, holecard_ranks):
		''' Returns if at least two holecards are non-paired overcards
		'''
		count = 0
		for x in range(len(holecard_ranks) - 1, -1, -1):
			if board_ranks[x] >= holecard_ranks[x]:
				break
			if holecard_ranks[x] == 1:	# only count non-paired cards
				count += 1
		return count >= 2

	@staticmethod
	@memoize
	def is_blocking_flush(board_suits, holecard_suits, high_holecard_suit_indices):
		'''	Determine if holecards contain an A or K of a suit that appears
			on the board three, four, or five times
		'''
		for index in high_holecard_suit_indices:
			if board_suits[index] >= 3 and holecard_suits[index] == 1:
				return True
		return False

	@staticmethod
	@memoize
	def n_straight_blockers(board_ranks, holecard_ranks):
		'''	Count the number of straight blockers contained within the holecards

			Abuses the is_straight method, which is only supposed to take a
			5-card representation of a hand, but it works nonetheless and is more
			efficient than creating 5-card representations for each holecard
			added into board_ranks

			Also abuses the n_straight_outs method for the same reason
		'''
		count = 0
		populated_holecard_indices = [i for i, a in enumerate(holecard_ranks) if a > 0]
		populated_board_indices = [i for i, a in enumerate(board_ranks) if a > 0]

		board_ranks[populated_board_indices[0]] += 1	# increment this so it can be decremented by n_straight_outs

		for holecard_index in populated_holecard_indices:
			used_holecard_ranks = [0 for x in range(0, 13)]
			used_holecard_ranks[holecard_index] = 1
			board_ranks[holecard_index] += 1
			if Hand.is_straight(board_ranks) or Hand.n_straight_outs(board_ranks, used_holecard_ranks, [])[0] > 0:
				count += 1
			board_ranks[holecard_index] -= 1

		board_ranks[populated_board_indices[0]] -= 1	# cleanup
		return count

	@staticmethod
	@memoize
	def is_n_kind(n, ranks):
		for n_counts in ranks:
			if n_counts == n:
				return True
		return False

	@staticmethod
	def is_better_n_kind(rank_lst1, rank_lst2, n):
		''' Returns 1 if the first hand represents a better n_kind,
			0 if tie,
			-1 if the first hand represents a worse n_kind
		'''
		index_1 = 0
		index_2 = 0
		for x in range(0, len(rank_lst1)):	# will get index for the highest_n_kind
			if rank_lst1[x] == n:
				index_1 = x
			if rank_lst2[x] == n:
				index_2 = x

		count = 0
		tie_breaker = 0
		if (index_1 == index_2):
			for x in range(len(rank_lst1) - 1, -1, -1):	# look for divergence in remaining cards
				if not x == index_1 and (rank_lst1[x] >= 1 or rank_lst2[x] >= 1):
					count += min(rank_lst1[x], rank_lst2[x])
				if count >= 5 - n:
					break
				if rank_lst1[x] > rank_lst2[x]:
					tie_breaker = 1
					break
				elif rank_lst2[x] > rank_lst1[x]:
					tie_breaker = 2
					break

		val = -1
		if index_1 > index_2 or (index_1 == index_2 and tie_breaker == 1):
			val = 1
		elif index_1 == index_2 and tie_breaker == 0:
			val = 0
		return val

	@staticmethod
	@memoize
	def is_two_sets(n, ranks):
		'''	Returns if the collected ranks contain a pair and either another,
			unique pair (if n == 2) or three of a kind (if n == 3)
		'''
		pair = False
		second_set = False
		for n_counts in ranks:
			if ((pair and second_set) or n_counts > 3):
				break
			elif (n_counts == 2 and not pair):
				pair = True
			elif (n_counts == n):
				if not second_set:
					second_set = True
				elif second_set and n == 3:
					pair = True		# Super boats must register as boats
		return pair and second_set

	@staticmethod
	def is_better_double_set(rank_lst1, rank_lst2, n):
		''' Returns 1 if the first hand represents a better double set,
			0 if tie,
			-1 if the first hand represents a worse double set
		'''
		index_1_1 = -1
		index_1_2 = -1
		index_2_1 = -1
		index_2_2 = -1
		high_card_1 = -1
		high_card_2 = -1
		if (n == 3):
			for x in range(0, len(rank_lst1)):
				if rank_lst1[x] == 3:
					index_1_1 = x
				if rank_lst2[x] == 3:
					index_2_1 = x
				if rank_lst1[x] == 2:
					index_1_2 = x
				if rank_lst2[x] == 2:
					index_2_2 = x
			if index_1_2 == -1:		# if superboat, wouldnt register as a boat, just 3K without the following
				for x in range(len(rank_lst1) - 1, -1, -1):
					if rank_lst1[x] == 3:
						index_1_2 = x
			if index_2_1 == -1:
				for x in range(len(rank_lst2) - 1, -1, -1):
					if rank_lst2[x] == 3:
						index_2_2 = x
		else:
			for x in range(len(rank_lst1) - 1, -1, -1):	# will get index of highest 2-pair
				if index_1_2 == -1 and index_1_1 != -1 and rank_lst1[x] == 2:
					index_1_2 = x
				if index_2_2 == -1 and index_2_1 != -1 and rank_lst2[x] == 2:
					index_2_2 = x
				if index_1_1 == -1 and rank_lst1[x] == 2:
					index_1_1 = x
				if index_2_1 == -1 and rank_lst2[x] == 2:
					index_2_1 = x
				if high_card_1 == -1 and not index_1_1 == x and not index_1_2 == x and rank_lst1[x] >= 1:
					high_card_1 = x
				if high_card_2 == -1 and not index_2_1 == x and not index_2_2 == x and rank_lst2[x] >= 1:
					high_card_2 = x
		val = -1
		if index_1_1 > index_2_1 or (index_1_1 == index_2_1 and index_1_2 > index_2_2) or (index_1_1 == index_2_1 and index_1_2 == index_2_2 and high_card_1 > high_card_2):
			val = 1
		elif index_1_1 == index_2_1 and index_1_2 == index_2_2 and high_card_1 == high_card_2:
			val = 0
		return val

	@staticmethod
	@memoize
	def is_flush(suits):
		for n_counts in suits:
			if (n_counts >= 5):
				return True
		return False

	@staticmethod
	def is_better_flush(lst_1, suits_1, lst_2, suits_2):
		suit_1 = -1
		suit_2 = -1
		high_cards_1 = []
		high_cards_2 = []
		for x in range(0, len(suits_1)):
			if (suits_1[x] >= 5):
				suit_1 = x
			if (suits_2[x] >= 5):
				suit_2 = x
		for card in lst_1:
			if card.suit.as_int() == suit_1:
				high_cards_1.append(card.rank.as_int())
		for card in lst_2:
			if card.suit.as_int() == suit_2:
				high_cards_2.append(card.rank.as_int())
		val = 0
		high_cards_1.reverse()
		high_cards_2.reverse()
		min_length = min(len(high_cards_1), len(high_cards_2))
		for x in range(0, min_length):
			if high_cards_1[x] > high_cards_2[x]:
				val = 1
				break
			elif high_cards_1[x] < high_cards_2[x]:
				val = -1
				break
		return val

	@staticmethod
	@memoize
	def is_straight(ranks):
		''' Returns if the given five-card representation
			of a hand is a straight
		'''
		streakiness = 0
		if (ranks[12] >= 1):
			streakiness += 1
		for n_counts in ranks:
			if n_counts >= 1:
				streakiness += 1
				if (streakiness >= 5):
					break
			else:
				streakiness = 0
		return streakiness >= 5

	@staticmethod
	def is_better_straight(rank_lst1, rank_lst2):
		streak_1 = 0
		streak_2 = 0
		best_rank_1 = 0
		best_rank_2 = 0
		if (rank_lst1[12] >= 1):
			streak_1 += 1
		if (rank_lst2[12] >= 1):
			streak_2 += 1
		for x in range(0, len(rank_lst1)):
			if rank_lst1[x] >= 1:
				streak_1 += 1
				if (streak_1 >= 5):
					best_rank_1 = x
			elif rank_lst1[x] == 0:
				streak_1 = 0
			if rank_lst2[x] >= 1:
				streak_2 += 1
				if (streak_2 >= 5):
					best_rank_2 = x
			elif rank_lst2[x] == 0:
				streak_2 = 0
		val = -1
		if best_rank_1 > best_rank_2:
			val = 1
		elif best_rank_1 == best_rank_2:
			val = 0
		return val

	def is_straight_flush(self, index):
		'''	Returns if the hand given at index contains a continuous sequence
			of five or more cards of the same suit
			REQUIRES and assumes all hands to be sorted
		'''
		#print(self.max_strength)
		if self.max_strength < 5:	# if no straight or flush, can't be straight flush
			return False
		return Hand.sf_highcard(self.all_cards[index])

	@staticmethod
	def is_better_straight_flush(all_cards_1, all_cards_2):
		'''	Returns 1 if the sf represented by all_cards_1 is better than that
			represented by all_cards_2,
			0 if they are of equal strength, and
			-1 if the latter is better than the former
		'''
		best_highcard_1 = Hand.sf_highcard(all_cards_1)
		best_highcard_2 = Hand.sf_highcard(all_cards_2)
		val = -1
		if best_highcard_1 > best_highcard_2:
			val = 1
		elif best_highcard_1 == best_highcard_2:
			val = 0
		return val

	@staticmethod
	@memoize
	def sf_highcard(all_cards):
		previous_rank_for_each_suit = [-10 for x in range(0, 4)]
		streakiness_for_each_suit = [0 for x in range(0, 4)]
		highcard_for_each_suit = [False for x in range(0, 4)]

		# catch small aces
		for x in range(0, len(all_cards)):
			card = all_cards[x]
			if (card.rank.as_int() == 12):
				previous_rank_for_each_suit[card.suit.as_int()] = -1

		for x in range(0, len(all_cards)):
			card = all_cards[x]
			#print(previous_rank_for_each_suit[card.suit.as_int()], card.suit.as_int(), streakiness_for_each_suit[card.suit.as_int()])
			if (card.rank.as_int() == previous_rank_for_each_suit[card.suit.as_int()] + 1):
				streakiness_for_each_suit[card.suit.as_int()] += 1
				if (streakiness_for_each_suit[card.suit.as_int()]) >= 4:
					highcard_for_each_suit[card.suit.as_int()] = card.rank.as_int()
			else:
				streakiness_for_each_suit[card.suit.as_int()] = 0
			previous_rank_for_each_suit[card.suit.as_int()] = card.rank.as_int()

		best_highcard = -1
		for highcard in highcard_for_each_suit:
			if highcard:
				best_highcard = max(best_highcard, highcard)
		return best_highcard if best_highcard > -1 else False	# must return at least 3 if True

	def __repr__(self):
		return str(self.all_cards)

def test_all_card_combos():
	gamestate = {P1_STACK: STARTING_STACK - 1, P2_STACK: STARTING_STACK - 2, P1_BET: 1, P2_BET: 2, P1_HOLECARDS: None, P2_HOLECARDS: None, POT: 3, TO_ACT: P1, CURRENT_STREET: 0, ACTIONS: ["B1", "R1"], BOARD: ""}
	Game.generate_all_dealt_card_combos(gamestate)

def test_actions():
	gamestate = {P1_STACK: STARTING_STACK - 1, P2_STACK: STARTING_STACK - 2, P1_BET: 1, P2_BET: 2, P1_HOLECARDS: None, P2_HOLECARDS: None, POT: 3, TO_ACT: P1, CURRENT_STREET: 0, ACTIONS: ["B1", "R1"], BOARD: ""}
	actions = Game.generate_quantized_actions(gamestate)
	print(actions)
	gamestate, new_street = Game.perform_action(gamestate, actions[1])
	print(gamestate, new_street)
	actions = Game.generate_quantized_actions(gamestate)
	print(actions)
	gamestate, new_street = Game.perform_action(gamestate, actions[0])
	print(gamestate, new_street)
	actions = Game.generate_quantized_actions(gamestate)
	print(actions)

def test_hands():
	'''returned: n_fd_backdoor_turns, n_fds, f_blockers, n_sd_backdoor_turns, n_sds, s_blockers, overcards, pairing strength, straights flushes, n_straights_possible, flush_possible
	'''
	hand = Hand("Qs3d3s2s", "KsJd9d")
	print(hand.collect_categorical_data())
	hand = Hand("AsKdQsJs", "KhJd9s")
	print(hand.collect_categorical_data())
	hand = Hand("AsAdQsJd", "KsJd9s")
	print(hand.collect_categorical_data())
	hand = Hand("AsKdQsJd", "KsJdQd9d")
	print(hand.collect_categorical_data())
	hand = Hand("AsKdQsJd", "KsJdQd9d8d")
	print(hand.collect_categorical_data())
	hand = Hand("QsTd3s4c", "Ks5h6dJd")
	print(hand.collect_categorical_data())
	hand = Hand("QsJsTc9c", "KsTs8c2c")
	print(hand.collect_categorical_data())
	hand = Hand("QsJsTc9c", "KsTd8c")
	print(hand.collect_categorical_data())
	start = time.time()
	for x in range(0, 100):
		holecard_str = ""
		board_str = ""
		for y in range(0, 4):
			holecard_str += str(random.choice(DECK))
		for z in range(0, random.randint(3, 5)):
			board_str += str(random.choice(DECK))
		hand = Hand(holecard_str, board_str)
	print(time.time() - start)

if __name__ == '__main__':
	pickle_to_write = None
	existing_pickle = None
	existing_superbucket_pickle = None
	superbucket_pickle_to_write = None
	size_of_postflop_dictionary = 0.01
	for x in range(1, len(sys.argv)):
		if sys.argv[x] == "-write_buckets" and len(sys.argv) > x + 1:
			pickle_to_write = sys.argv[x + 1]
		elif sys.argv[x] == "-read_buckets":
			existing_pickle = sys.argv[x + 1]
		elif sys.argv[x] == "-read_superbuckets":
			existing_superbucket_pickle = sys.argv[x + 1]
		elif sys.argv[x] == "-write_superbuckets":
			superbucket_pickle_to_write = sys.argv[x + 1]
		elif sys.argv[x] == "-handspace_explored":
			size_of_postflop_dictionary = float(sys.argv[x + 1])
	bucketman = BucketManager(existing_pickle, pickle_to_write, existing_superbucket_pickle, superbucket_pickle_to_write, size_of_postflop_dictionary)


