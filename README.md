PLOAI - Towards a Pot-Limit Omaha Agent

Please see https://docs.google.com/presentation/d/1xx-S2HE10ZZ7cnxWcXCnddAgX2nU1-tm62EJN41ATQs/edit?usp=sharing for an in-depth walkthrough of the ideas implemented in this project.

This work aims to produce a game-playing agent that is at least as good as a "decent" human player through two rounds of game abstraction and then learning via CFR or Monte Carlo CFR.

Contents:

engine.py
- The file responsible for most game logic and abstraction.

fast_monte_carlo_evaluation
- Subdirectory responsible for quickly evaluating the expectation value of a given hand. Adapted from https://github.com/lostella/podds

Usage:

Run make in fast_monte_carlo_evaluation
python3 engine.py
	[-write_buckets buckets_pickle_file]: pickle and write buckets to the specified filename
	[-read_buckets buckets_pickle_file]: read buckets from the specified filename
	[-write_superbuckets buckets_pickle_file]: pickle and write superbuckets to the specified filename
	[-read_superbuckets buckets_pickle_file]: read superbuckets from the specified filename
	[-handspace_explored (0-1]]: allows specification of the total handspace he or she wishes the program to explore when bucketing, default = 0.01

You may wish to adjust the number of threads specified in both engine.py and fast_monte_carlo_evaluation/podds.c