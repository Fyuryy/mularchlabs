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


int perform_bucket_computation(int, int, int);

int main(int argc, const char *argv[])
{
    int num_threads, num_samples, num_buckets;

    if (argc != 4)
    {
        printf("Invalid input! Usage: ./sharing <num_threads> <num_samples> <num_buckets> \n");
        return 1;
    }
    else
    {
        num_threads = atoi(argv[1]);
        num_samples = atoi(argv[2]);
        num_buckets = atoi(argv[3]);
    }

    set_clock();
    if (perform_buckets_computation(num_threads, num_samples, num_buckets) == -1)
    {
        printf("Error in allocating memory for histogram\n");
        return 1;
    }

    printf("Using %d threads: %d operations completed in %.4gs.\n", num_threads, num_samples, elapsed_time());
    return 0;
}

int perform_buckets_computation(int num_threads, int num_samples, int num_buckets)
{
    volatile int *histogram = (int *)calloc(num_buckets, sizeof(int));

    int **tmp_histogram = (int **)calloc(num_threads, sizeof(int *));

    omp_set_num_threads(num_threads);

    for (int i = 0; i < num_threads; i++)
    {
        tmp_histogram[i] = (int *)calloc(num_buckets, sizeof(int));
    }

    rand_gen generator = init_rand();

#pragma omp parallel for shared(tmp_histogram) 
    for (int i = 0; i < num_samples; i++)
    {
        int tid = omp_get_thread_num();
        int bucket = next_rand(generator) * num_buckets;
        tmp_histogram[tid][bucket]++;
    }


#pragma omp parallel for shared(tmp_histogram, histogram) 
for (int i = 0; i < num_buckets; i++)
{   
    int tid;
    for (tid = 0; tid < num_threads; tid++)
        histogram[i] += tmp_histogram[tid][i];
    
}


free_rand(generator);
free(tmp_histogram);

return 0;
}
