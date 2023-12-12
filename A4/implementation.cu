/*
============================================================================
Filename    : algorithm.c
Author      : Your name goes here
SCIPER      : Your SCIPER number
============================================================================
*/

#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <cuda_runtime.h>
using namespace std;

// CPU Baseline
__global__ void array_process(double *input, double *output, int length, int iterations)
{
    double *temp;

    for(int n=0; n<(int) iterations; n++)
    {
        for(int i=1; i<length-1; i++)
        {
            for(int j=1; j<length-1; j++)
            {
                output[(i)*(length)+(j)] = (input[(i-1)*(length)+(j-1)] +
                                            input[(i-1)*(length)+(j)]   +
                                            input[(i-1)*(length)+(j+1)] +
                                            input[(i)*(length)+(j-1)]   +
                                            input[(i)*(length)+(j)]     +
                                            input[(i)*(length)+(j+1)]   +
                                            input[(i+1)*(length)+(j-1)] +
                                            input[(i+1)*(length)+(j)]   +
                                            input[(i+1)*(length)+(j+1)] ) / 9;

            }
        }
        output[(length/2-1)*length+(length/2-1)] = 1000;
        output[(length/2)*length+(length/2-1)]   = 1000;
        output[(length/2-1)*length+(length/2)]   = 1000;
        output[(length/2)*length+(length/2)]     = 1000;

        temp = input;
        input = output;
        output = temp;
    }
}


// GPU Optimized function
void GPU_array_process(double *input, double *output, int length, int iterations)
{
    //Cuda events for calculating elapsed time
    cudaEvent_t cpy_H2D_start, cpy_H2D_end, comp_start, comp_end, cpy_D2H_start, cpy_D2H_end;
    cudaEventCreate(&cpy_H2D_start);
    cudaEventCreate(&cpy_H2D_end);
    cudaEventCreate(&cpy_D2H_start);
    cudaEventCreate(&cpy_D2H_end);
    cudaEventCreate(&comp_start);
    cudaEventCreate(&comp_end);

    /* Preprocessing goes here */
    double *d_input, *d_output;
    cudaMalloc((void**)&d_input, length*length*sizeof(double));
    cudaMalloc((void**)&d_output, length*length*sizeof(double));

    cudaEventRecord(cpy_H2D_start);
    /* Copying array from host to device goes here */
    cudaMemcpy(d_input, input, length, cudaMemcpyHostToDevice);

    cudaEventRecord(cpy_H2D_end);
    cudaEventSynchronize(cpy_H2D_end);

    //Copy array from host to device
    cudaEventRecord(comp_start);
    /* GPU calculation goes here */

    dim3 dimBlock(16, 16);
    dim3 dimGrid((length + dimBlock.x -1) / dimBlock.x, (length + dimBlock.y -1) / dimBlock.y);

    for(int n = 0; n < iterations; n++){
        array_process<<<dimGrid, dimBlock>>>(d_input, d_output, length, iterations);

        double *temp = d_input;
        d_input = d_output;
        d_output = temp;
    }


    cudaEventRecord(comp_end);
    cudaEventSynchronize(comp_end);

    cudaEventRecord(cpy_D2H_start);
    /* Copying array from device to host goes here */

    cudaMemcpy(output, d_output, length, cudaMemcpyDeviceToHost);

    cudaEventRecord(cpy_D2H_end);
    cudaEventSynchronize(cpy_D2H_end);

    /* Postprocessing goes here */
    cudaFree(d_input);
    cudaFree(d_output);
    float time;
    cudaEventElapsedTime(&time, cpy_H2D_start, cpy_H2D_end);
    cout<<"Host to Device MemCpy takes "<<setprecision(4)<<time/1000<<"s"<<endl;

    cudaEventElapsedTime(&time, comp_start, comp_end);
    cout<<"Computation takes "<<setprecision(4)<<time/1000<<"s"<<endl;

    cudaEventElapsedTime(&time, cpy_D2H_start, cpy_D2H_end);
    cout<<"Device to Host MemCpy takes "<<setprecision(4)<<time/1000<<"s"<<endl;
}