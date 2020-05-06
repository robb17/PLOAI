/*
This file is part of podds.

*/

#ifndef __XORSHIFT_H__
#define __XORSHIFT_H__

#include <stdint.h>

uint32_t xorshift32_rand(uint32_t *);
uint32_t xorshift32_randint(uint32_t *, uint32_t);

#endif // __XORSHIFT_H__
