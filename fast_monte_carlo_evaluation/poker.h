/*
This file is part of podds.
*/

#ifndef __POKER_H__
#define __POKER_H__

#include <stdint.h>

/* shifts to build up scores properly */
#define SFLUSH_SHIFT    42
#define FOAK_SHIFT      38
#define FULL_SHIFT      37
#define FLUSH_SHIFT     36
#define STRAIGHT_SHIFT  32
#define TOAK_SHIFT      28
#define PAIR2_SHIFT     24
#define PAIR1_SHIFT     20
#define HC_SHIFT        0

/* some constants */
#define LOSS            ((int64_t)0)
#define DRAW            ((int64_t)1)
#define WIN             ((int64_t)2)
#define HC              ((int64_t)3)
#define PAIR            ((int64_t)4)
#define TWOPAIRS        ((int64_t)5)
#define TOAK            ((int64_t)6)
#define STRAIGHT        ((int64_t)7)
#define FLUSH           ((int64_t)8)
#define FULLHOUSE       ((int64_t)9)
#define FOAK            ((int64_t)10)
#define STRFLUSH        ((int64_t)11)

typedef struct {
  uint32_t c[52];
  uint32_t n;
  uint32_t s;
} deck;

uint32_t randint(uint32_t hi);
deck * newdeck();
void initdeck(deck *, uint32_t);
uint32_t draw(deck *);
void pick(deck *, uint32_t);
void sort(uint32_t *);
uint32_t hand(int64_t s);
int64_t eval5(uint32_t *);
int64_t eval7(uint32_t *);
int64_t comp7(uint32_t *, int64_t s);
int64_t eval9(uint32_t *);
int64_t comp9(uint32_t *, int64_t s);

#endif // __POKER_H__
