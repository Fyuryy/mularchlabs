/******************************************************************************
* FILE: pthreads_hello.c
* DESCRIPTION:
*   Pthreads Example - Hello World - C/C++ Version
*   In this simple example, the master thread forks a parallel region.
*   All threads in the team obtain their unique thread number and
*   a random number and print them. Only the master thread prints the
*   total number of threads. set_clock and elapsed_time functions
*   are used to measure the time. The three *_rand functions are
*   used to manage the random number generator.
******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "utility.h"

#define NTHREADS 8

void* hello_world(void *thread_num){
  rand_gen gen = init_rand_pthreads(*(int *)thread_num);
  double rand_number = next_rand(gen);

  printf("Hello World from thread = %d, this is my random number: %f\n", *(int *)thread_num, rand_number);

  free_rand(gen);
  return NULL;
}

int main (int argc, char *argv[]) {
  int nthreads = NTHREADS;
  pthread_t tid[NTHREADS];
  int thread_num[NTHREADS];

  set_clock();

  for(int i = 0; i < nthreads; i++){
    thread_num[i] = i;
    pthread_create(&tid[i], NULL, hello_world, (void *)(thread_num + i));
  }

  /* Only master thread does this */
  printf("Master thread: Number of threads = %d\n", nthreads);

  /* All threads join master thread and disband */
  for(int i = 0; i < nthreads; i++){
    pthread_join(tid[i], NULL);
  }

  printf("- Using %d threads: Hello World took %.4gs.\n", nthreads, elapsed_time());

  return 0;
}
