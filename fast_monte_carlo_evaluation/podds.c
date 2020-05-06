/*
podds - Texas Hold 'Em Poker odds calculator
Copyright (C) 2011-2017  Lorenzo Stella

Heavily modified by Rob Brunstad for
    1) PLO
    2) use as a program driven by engine.py

*/

#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>

#include "poker.h"

/* total number of games for the simulation */
#define MAXGAMES        5000

/*~~ Argument parsing ~~~~~~~~~~~~~~~~~*/

#define SYMBOL_TEN        84 // 'T'
#define SYMBOL_JACK       74 // 'J'
#define SYMBOL_QUEEN      81 // 'Q'
#define SYMBOL_KING       75 // 'K'
#define SYMBOL_ACE        65 // 'A'

#define SYMBOL_HEARTS     104 // 'h'
#define SYMBOL_DIAMONDS   100 // 'd'
#define SYMBOL_CLUBS      99  // 'c'
#define SYMBOL_SPADES     115 // 's'

uint32_t char2rank(char c) {
  // 50 = '2', 57 = '9'
  if (c >= 50 && c <= 57) return c - 50;
  else if (c == SYMBOL_TEN) return 8;
  else if (c == SYMBOL_JACK) return 9;
  else if (c == SYMBOL_QUEEN) return 10;
  else if (c == SYMBOL_KING) return 11;
  else if (c == SYMBOL_ACE) return 12;
  return 13;
}

uint32_t char2suit(char c) {
  if (c == SYMBOL_HEARTS) return 0;
  else if (c == SYMBOL_DIAMONDS) return 1;
  else if (c == SYMBOL_CLUBS) return 2;
  else if (c == SYMBOL_SPADES) return 3;
  return 4;
}

uint32_t string2index(char * str) {
  uint32_t r, s;
  r = char2rank(str[0]);
  s = char2suit(str[1]);
  if (r >= 13 || s >= 4) return 52;
  return s*13 + r;
}

/*~~ Global (shared) data ~~~~~~~~~~~~~*/

int64_t counters[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
uint32_t np, kc, as[9], draw_for_p2, p2_cards[4];

/*~~ Threading ~~~~~~~~~~~~~~~~~~~~~~~~*/

#include <pthread.h>
pthread_t * tpool;
pthread_mutex_t tlock;

void * simulator(void * v) {
  uint32_t ngames = ((uint32_t *)v)[0];
  uint32_t * ohs = (uint32_t *)malloc(4*(np-1)*sizeof(uint32_t));
  uint32_t cs[9], myas[9], cs0, cs1, result, result1, i, j, k;
  uint32_t mycounters[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  deck * d = newdeck();
  for (i=0; i<kc; i++) {
    pick(d, as[i]);
    myas[i] = as[i];
  }
  if (draw_for_p2 == 0) {
    for (i = 0; i < 4; i++) {
        pick(d, p2_cards[i]);
        //printf("selected %d", p2_cards[i]);
        ohs[i] = p2_cards[i];
    }
  }
  for (i=0; i<ngames; i++) {
    int64_t score;
    int64_t deck_length = (draw_for_p2 == 1 ? 52 - kc : 52 - 4 - kc);
    initdeck(d, deck_length);
    for (j=0; j<4*(np-1); j++) {
      if (draw_for_p2 == 1) {
        ohs[j] = draw(d);
      }
    }
    for (j=kc; j<9; j++) {
      myas[j] = draw(d);
      //printf("%d\n", myas[j]);
    }
    for (j=0; j<9; j++) cs[j] = myas[j];
    //sort(cs);
    score = eval9(cs);
    result = WIN;
    for (j=0; j<np-1; j++) {
      cs[0] = ohs[4*j];
      cs[1] = ohs[4*j+1];
      cs[2] = ohs[4*j+2];
      cs[3] = ohs[4*j+3];
      //printf("%d%d%d%d\n", cs[j], cs[j + 1], cs[j + 2], cs[j + 3]);
      for (k=4; k<9; k++) cs[k] = myas[k];
      //sort(cs);
      result1 = comp9(cs, score);
      if (result1 < result) result = result1;
      if (result == LOSS) break;
    }
    mycounters[result]++;
    mycounters[hand(score)]++;
  }
  pthread_mutex_lock(&tlock);
  for (i=0; i<12; i++) {
    counters[i] += mycounters[i];
  }
  pthread_mutex_unlock(&tlock);
  free(ohs);
  free(d);
  return NULL;
}

/*~~ Main program ~~~~~~~~~~~~~~~~~~~~~*/
int driver(char *args);

float run(int argc, char **argv) {
  //driver("3 4 2 Ac Td");
  uint32_t i, cs0, cs1, checksum;
  uint32_t nthreads, ngames, totgames;
  if (argc < 6) {
    fprintf(stderr, "incorrect number of arguments\n");
    fprintf(stderr, "required: <#players> <card1> <card2> <card3> <card4>\n");
    return 1;
  }
  nthreads = 20; //sysconf(_SC_NPROCESSORS_ONLN);
  ngames = MAXGAMES/nthreads;
  totgames = ngames*nthreads;
  //printf("cores:%d\n", nthreads);
  //printf("games:%d\n", totgames);
  // read the arguments and create the known cards
  if (strcmp(argv[1], "2!") != 0) {
    np = atoi(argv[1]);
    draw_for_p2 = 1;
    kc = argc-2;
  } else {
    np = 2;
    draw_for_p2 = 0;
    kc = argc-6;
  }
  for (i=0; i<kc; i++) {
    as[i] = string2index(argv[i+2]);
    //printf("%d ", as[i]);
    if (as[i] >= 52) {
      fprintf(stderr, "wrong card identifier: %s\n", argv[i+2]);
      return 1;
    }
  }
  if (draw_for_p2 == 0) {
    for (i = kc; i < kc + 4; i++) {
      p2_cards[i - kc] = string2index(argv[i + 2]);
      //printf("%d ", p2_cards[i - kc]);
    }
    //printf("\n");
    ngames = ngames / 10;
    totgames = totgames / 10;
  }

  // initialize the rng seed and the mutex
  srand(time(NULL));
  pthread_mutex_init(&tlock, NULL);
  pthread_mutex_unlock(&tlock);
  // run the simulation threads
  tpool = (pthread_t *)malloc(nthreads*sizeof(pthread_t));
  for (i=0; i<nthreads; i++) {
    pthread_create(&tpool[i], NULL, simulator, (void *)(&ngames));
  }
  // wait for the threads to finish
  for (i=0; i<nthreads; i++) {
    pthread_join(tpool[i], NULL);
  }
  // check correctness (sum counters)
  checksum = 0;
  for (i=0; i<3; i++) checksum += counters[i];
  if (checksum != totgames) {
    fprintf(stderr, "counters do not sum up, checksum = %d, totgames = %d\n", checksum, totgames);
    return 1;
  }
  checksum = 0;
  for (i=3; i<12; i++) checksum += counters[i];
  if (checksum != totgames) {
    fprintf(stderr, "counters do not sum up\n");
    return 1;
  }
  // show the results
  //printf("win:%.3f\n", ((float)counters[WIN])/totgames);
  //printf("draw:%.3f\n", ((float)counters[DRAW])/totgames);
  //printf("pair:%.3f\n", ((float)counters[PAIR])/totgames);
  //printf("two-pairs:%.3f\n", ((float)counters[TWOPAIRS])/totgames);
  //printf("three-of-a-kind:%.3f\n", ((float)counters[TOAK])/totgames);
  //printf("straight:%.3f\n", ((float)counters[STRAIGHT])/totgames);
  //printf("flush:%.3f\n", ((float)counters[FLUSH])/totgames);
  //printf("full-house:%.3f\n", ((float)counters[FULLHOUSE])/totgames);
  //printf("four-of-a-kind:%.3f\n", ((float)counters[FOAK])/totgames);
  //printf("straight-flush:%.3f\n", ((float)counters[STRFLUSH])/totgames);
  // clear all
  float winning = ((float)counters[WIN])/totgames;
  pthread_mutex_destroy(&tlock);
  // reset counters
  for (i=0; i<12; i++) {
    counters[i] = 0;
  }
  return winning;
}

int driver(char *input) {
  //int argc = ((int) args[0]) - 48;
  char *args = malloc(sizeof(int) * strlen(input));
  strcpy(args, input);
  printf("%s\n", args);

  char *tok = strtok(args, " ");
  int argc = ((int) args[0]) - 48;
  if (args[0] == '1') {
    argc = argc * 10 + ((int) args[1]) - 48;
  }
  char **new_args = malloc(sizeof(int *) * (argc + 1));
  int count = 0;
  while (tok != NULL) {
    //printf("%s\n", tok);
    tok = strtok(NULL, " ");
    if (tok == NULL) {
      break;
    }
    new_args[count] = malloc(sizeof(int) * strlen(tok));
    strcpy(new_args[count], tok);
    count++;
  }

//  printf("%d\n", argc + 1);
//
//  int *sizes_of_args = malloc(sizeof(int) * argc);
//
//  for (int x = 1; x < argc; x++) {
//    int count = 1;
//    while (args[x] != ' ') {
//      x += 1;
//      count += 1;
//    }
//    sizes_of_args[x] = count;
//    printf("%d\n", count);
//  }
//
//  for (int x = 1; x < argc; x++) {
//    char *arg = malloc(sizeof(char) * sizes_of_args[x]);
//    int count = 0;
//    while (args[x] != ' ') {
//      arg[count] = args[x];
//      x += 1;
//      count += 1;
//    }
//    new_args[x] = arg;
//    printf("%s\n", arg);
//  }
  //for (int i = 0; i < count; i++) {
  //  printf("%s", new_args[i]);
  //}

  float res = run(argc, new_args);
  for (int i = 0; i < count; i++) {
    free(new_args[i]);
  }
  free(args);
  free(new_args);
  return (int) (res * 10000);
}

int main(int argc, char **argv) {
  printf("%f\n", run(argc, argv));
  return 1;
}
