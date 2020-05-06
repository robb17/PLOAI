/*
This file is part of podds.
*/

#include <stdint.h>

#include "xorshift.h"

uint32_t xorshift32_rand(uint32_t * seed) {
	seed[0] ^= seed[0] << 13;
	seed[0] ^= seed[0] >> 17;
	seed[0] ^= seed[0] << 5;
	return seed[0];
}

uint32_t xorshift32_randint(uint32_t * seed, uint32_t hi) {
  return (uint32_t)((double)xorshift32_rand(seed) / ((double)UINT32_MAX + 1) * hi);
}
