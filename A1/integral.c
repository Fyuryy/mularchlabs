/*
============================================================================
Filename    : integral.c
Author      : Your names goes here
SCIPER		: Your SCIPER numbers
============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include "utility.h"
#include "function.c"

double integrate(int num_threads, int samples, int a, int b, double (*f)(double));

int main(int argc, const char *argv[])
{

    int num_threads, num_samples, a, b;
    double integral;

    if (argc != 5)
    {
        printf("Invalid input! Usage: ./integral <num_threads> <num_samples> <a> <b>\n");
        return 1;
    }
    else
    {
        num_threads = atoi(argv[1]);
        num_samples = atoi(argv[2]);
        a = atoi(argv[3]);
        b = atoi(argv[4]);
    }

    set_clock();

    /* You can use your self-defined funtions by replacing identity_f. */
    integral = integrate(num_threads, num_samples, a, b, identity_f);

    printf("- Using %d threads: integral on [%d,%d] = %.15g computed in %.4gs.\n", num_threads, a, b, integral, elapsed_time());

    return 0;
}

double integrate(int num_threads, int samples, int a, int b, double (*f)(double))
{
    double integral = 0.0;

#pragma omp parallel num_threads(num_threads) 
    {
        double inside = 0.0;
        double s = 0.0;
        rand_gen r = init_rand();

        double local_samples = samples / num_threads;

#pragma omp parallel for
    for (int i = 0; i < (int)local_samples; i++)
    {
        double x = next_rand(r) * (b - a) + a;
        s += f(x) * (b - a);
    }

#pragma omp critical
    {
        integral += s;
    }
    }

integral = integral / samples;
return integral;
}
