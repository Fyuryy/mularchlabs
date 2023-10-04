/*
============================================================================
Filename    : pi.c
Author      : Your names goes here
SCIPER		: Your SCIPER numbers
============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include "utility.h"

double calculate_pi (int num_threads, int samples);

int compute(int *arg);

int main (int argc, const char *argv[]) {

    int num_threads, num_samples;
    double pi;

    if (argc != 3) {
		printf("Invalid input! Usage: ./pi <num_threads> <num_samples> \n");
		return 1;
	} else {
        num_threads = atoi(argv[1]);
        num_samples = atoi(argv[2]);
	}

    set_clock();
    pi = calculate_pi (num_threads, num_samples);

    printf("- Using %d threads: pi = %.15g computed in %.4gs.\n", num_threads, pi, elapsed_time());

    return 0;
}


double calculate_pi (int num_threads, int samples) {
    double pi;

    /* Your code goes here */
    int nthreads = num_threads;
    pthread_t tid[num_threads];
    int thread_num[num_threads];

    //*Number of samples that will be taken care by each thread
    int local_samples = (int) (samples/nthreads);
    int insides[nthreads];

    for(int i = 0; i < nthreads; i++){
    thread_num[i] = i;
    pthread_create(&tid[i], NULL, compute, (int *)&local_samples);
    }


    /* All threads join master thread and disband */
    for(int i = 0; i < nthreads; i++){
    pthread_join(tid[i], insides + i);
  }
    for(int i = 0; i < nthreads; i++){
    pi += (double)insides[i] / local_samples;
    }

    pi = 4.0 * pi / num_threads;
    return pi;
}

int compute(int *arg){

    //*Compute x and y coords
    int thread_insides = 0;
    rand_gen r = init_rand(); 

    for(int i = 0; i < *arg; ++i){
            double x = next_rand(r);
            double y = next_rand(r);

            double distance = x * x + y * y;

            if (distance <= 1.0)
            {
                thread_insides += 1;
            }

    }
    return thread_insides;
}
