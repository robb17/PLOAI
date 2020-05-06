PLOAI - Towards a Pot-Limit Omaha Agent

Please see https://docs.google.com/presentation/d/1xx-S2HE10ZZ7cnxWcXCnddAgX2nU1-tm62EJN41ATQs/edit?usp=sharing for an in-depth walkthrough of the ideas implemented by this project.

This project aims to produce a game-playing agent that is at least as good as a "decent" human player through two rounds of game abstraction and then learning via CFR or Monte Carlo CFR.

Contents:

engine.py
	- The file responsible for most game logic and abstraction.

fast_monte_carlo_evaluation
	- Subdirectory responsible for quickly evaluating the expectation value of a given hand. Adapted from https://github.com/lostella/podds

Usage:

Run make in fast_monte_carlo_evaluation
python3 engine.py [-write_buckets buckets_pickle_file] [-read_buckets buckets_pickle_file] [-read_superbuckets buckets_pickle_file] [-write_superbuckets buckets_pickle_file]

You may wish to adjust the number of threads specified in both engine.py and fast_monte_carlo_evaluation/podds.c