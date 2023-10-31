/*
============================================================================
Filename    : algorithm.c
Author      : Your names go here
SCIPER      : Your SCIPER numbers

============================================================================
*/
#include <math.h>

#define INPUT(I,J) input[(I)*length+(J)]
#define OUTPUT(I,J) output[(I)*length+(J)]

#define TILE_SIZE 8

void simulate(double *input, double *output, int threads, int length, int iterations)
{
    double *temp;
    int i,j,i1,j1;
    
    // Parallelize this!!
    omp_set_num_threads(threads);  // Set the number of threads

   
    for(int n=0; n < iterations; n++)
    {
        #pragma omp parallel for private(i, j, i1, j1) collapse(2)
        for(i=1; i<length-1; i += TILE_SIZE)
        { 
            for(j=1; j<length-1; j += TILE_SIZE)
            {

                        int i_end = i + TILE_SIZE < length ? i + TILE_SIZE : length-1;
                        int j_end = j + TILE_SIZE < length ? j + TILE_SIZE : length-1;

                        for (i1 = i; i1 < i_end; ++i1){
                            for (j1 = j; j1 < j_end; ++j1){
                                if ( ((i1 == length/2-1) || (i1== length/2))
                                && ((j1 == length/2-1) || (j1 == length/2)) )
                                continue;

                                OUTPUT(i1,j1) = (INPUT(i1-1,j1-1) + INPUT(i1-1,j1) + INPUT(i1-1,j1+1) +
                                       INPUT(i1,j1-1)   + INPUT(i1,j1)   + INPUT(i1,j1+1)   +
                                       INPUT(i1+1,j1-1) + INPUT(i1+1,j1) + INPUT(i1+1,j1+1) )/9;
                            }
                        }

            }
        }

        temp = input;
        input = output;
        output = temp;
    }
}
