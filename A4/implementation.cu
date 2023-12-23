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

#define MAX_HEAT 1000

using namespace std;

// CPU Baseline
void array_process(double *input, double *output, int length, int iterations)
{
    double *temp;

    for (int n = 0; n < (int)iterations; n++)
    {
        for (int i = 1; i < length - 1; i++)
        {
            for (int j = 1; j < length - 1; j++)
            {
                output[(i) * (length) + (j)] = (input[(i - 1) * (length) + (j - 1)] +
                                                input[(i - 1) * (length) + (j)] +
                                                input[(i - 1) * (length) + (j + 1)] +
                                                input[(i) * (length) + (j - 1)] +
                                                input[(i) * (length) + (j)] +
                                                input[(i) * (length) + (j + 1)] +
                                                input[(i + 1) * (length) + (j - 1)] +
                                                input[(i + 1) * (length) + (j)] +
                                                input[(i + 1) * (length) + (j + 1)]) /
                                               9;
            }
        }
        output[(length / 2 - 1) * length + (length / 2 - 1)] = 1000;
        output[(length / 2) * length + (length / 2 - 1)] = 1000;
        output[(length / 2 - 1) * length + (length / 2)] = 1000;
        output[(length / 2) * length + (length / 2)] = 1000;

        temp = input;
        input = output;
        output = temp;
    }
}

__global__ void kernel(double *input, double *output, size_t length)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int index = (i * length) + j;

    //*border threads can be ignored as borders are always 0.


    if(i < length - 1 && j < length -1 && i > 0 && j > 0){
    
        output[index] = (input[(i - 1) * (length) + (j - 1)] +
                     input[(i - 1) * (length) + (j)] +
                     input[(i - 1) * (length) + (j + 1)] +
                     input[(i) * (length) + (j - 1)] +
                     input[(i) * (length) + (j)] +
                     input[(i) * (length) + (j + 1)] +
                     input[(i + 1) * (length) + (j - 1)] +
                     input[(i + 1) * (length) + (j)] +
                     input[(i + 1) * (length) + (j + 1)]) /
                    9;

        output[(length / 2 - 1) * length + (length / 2 - 1)] = MAX_HEAT;
        output[(length / 2) * length + (length / 2 - 1)] = MAX_HEAT;
        output[(length / 2 - 1) * length + (length / 2)] = MAX_HEAT;
        output[(length / 2) * length + (length / 2)] = MAX_HEAT;
    }
}

// GPU Optimized function
void GPU_array_process(double *input, double *output, int length, int iterations)
{
    // Cuda events for calculating elapsed time
    cudaEvent_t cpy_H2D_start, cpy_H2D_end, comp_start, comp_end, cpy_D2H_start, cpy_D2H_end;
    cudaEventCreate(&cpy_H2D_start);
    cudaEventCreate(&cpy_H2D_end);
    cudaEventCreate(&cpy_D2H_start);
    cudaEventCreate(&cpy_D2H_end);
    cudaEventCreate(&comp_start);
    cudaEventCreate(&comp_end);

    /* Preprocessing goes here */

    double *d_input, *d_output;
    size_t size = length * length * sizeof(double);

    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);

    /* Copying array from host to device goes here */
    cudaEventRecord(cpy_H2D_start);

    cudaMemcpyAsync((double*) d_input, (double*)input, size, cudaMemcpyHostToDevice);

    cudaEventRecord(cpy_H2D_end);
    cudaEventSynchronize(cpy_H2D_end);
    // Copy array from host to device

    /* GPU calculation goes here */

    // dim3 nbBlocks(ceil(length / 32), ceil(length / 32), 1); // 32x32 blocks
    // dim3 nbThreads(32, 32, 1);

    // We organize the thread blocks into 2D arrays of threads.

    dim3 blockD(32, 32);
    dim3 nbThreads( (length-2) / blockD.x + 1, (length-2)  / blockD.y + 1);

    double *temp;
    cudaEventRecord(comp_start);

    for (int i = 0; i < iterations; i++)
    {
        kernel<<<nbThreads, blockD>>>(d_input, d_output, length);

        temp = d_input;
        d_input = d_output;
        d_output = temp;
    }

    cudaEventRecord(comp_end);
    cudaEventSynchronize(comp_end);
    cudaEventRecord(cpy_D2H_start);
    /* Copying array from device to host goes here */
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(cpy_D2H_end);
    cudaEventSynchronize(cpy_D2H_end);
    /* Postprocessing goes here */
    cudaFree(d_input);
    cudaFree(d_output);
    float time;
    cudaEventElapsedTime(&time, cpy_H2D_start, cpy_H2D_end);
    cout << "Host to Device MemCpy takes " << setprecision(4) << time / 1000 << "s" << endl;
    cudaEventElapsedTime(&time, comp_start, comp_end);
    cout << "Computation takes " << setprecision(4) << time / 1000 << "s" << endl;
    cudaEventElapsedTime(&time, cpy_D2H_start, cpy_D2H_end);
    cout << "Device to Host MemCpy takes " << setprecision(4) << time / 1000 << "s" << endl;
}
