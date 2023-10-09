/*
============================================================================
Filename    : pi.c
Author      : Pedro Gouveia & Gustave Charles
SCIPER		: 345768 &
============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include "utility.h"

double calculate_pi(int num_threads, int samples);

int main(int argc, const char *argv[])
{

    int num_threads, num_samples;
    double pi;

    if (argc != 3)
    {
        printf("Invalid input! Usage: ./pi <num_threads> <num_samples> \n");
        return 1;
    }
    else
    {
        num_threads = atoi(argv[1]);
        num_samples = atoi(argv[2]);
    }

    set_clock();
    pi = calculate_pi(num_threads, num_samples);

    printf("- Using %d threads: pi = %.15g computed in %.4gs.\n", num_threads, pi, elapsed_time());

    return 0;
}

double calculate_pi(int num_threads, int samples)
{
    double pi;
      
#pragma omp parallel num_threads(num_threads)
    {

        rand_gen r = init_rand();
        //*Number of samples that will be taken care by each thread
        double local_samples = samples / num_threads; 

        //*seed is the variable where we'll stock the random number, who we'll divide by RAND_MAX to have smth between 0 and 1
        
        //*inside's gonna keep count of the points inside the cercle
        int inside = 0;

        for (int i = 0; i < local_samples; i++)
        {
            //*Compute x and y coords
            double x = next_rand(r);
            double y = next_rand(r);

            double distance = x * x + y * y;

            if (distance <= 1.0)
            {
                inside++;
            }
        }

//*This critical section allows us to update the pi variable ONE THREAD at the time
#pragma omp critical
        {
            pi += (double)inside / local_samples;
        }
    }


    pi = 4.0 * pi / num_threads;
    return pi;
}
