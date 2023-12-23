/*
============================================================================
Filename    : algorithm.c
Author      : Thösam, Khalil
SCIPER      : 330163, 310698
============================================================================
*/

#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <cuda_runtime.h>

#define MAX_HEAT 1000

using namespace std;

// CPU Baseline
void array_process(double *input, double *output, int length, int iterations) {
    double *temp;

    for (int n = 0; n < (int) iterations; n++) {
        for (int i = 1; i < length - 1; i++) {
            for (int j = 1; j < length - 1; j++) {
                output[(i) * (length) + (j)] = (input[(i - 1) * (length) + (j - 1)] +
                                                input[(i - 1) * (length) + (j)] +
                                                input[(i - 1) * (length) + (j + 1)] +
                                                input[(i) * (length) + (j - 1)] +
                                                input[(i) * (length) + (j)] +
                                                input[(i) * (length) + (j + 1)] +
                                                input[(i + 1) * (length) + (j - 1)] +
                                                input[(i + 1) * (length) + (j)] +
                                                input[(i + 1) * (length) + (j + 1)]) / 9;

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

__global__ void siuuu(double *in, double *out, int l) {
    int i = blockIdx.x + 1;
    int j = threadIdx.x + 1;

    int top_line = (i - 1) * (l);
    int middle_line = (i) * (l);
    int bottom_line = (i + 1) * (l);

    int left = (j - 1);
    int mid = j;
    int right = (j + 1);

    out[i * l + j] = (
                             in[top_line + left]  + in[middle_line + left]  + in[bottom_line + left] +
                             in[top_line + mid]   + in[middle_line + mid]   + in[bottom_line + mid]  +
                             in[top_line + right] + in[middle_line + right] + in[bottom_line + right]
                     ) / 9;


    out[(l / 2 - 1) * l + (l / 2 - 1)] = MAX_HEAT;
    out[(l / 2) * l + (l / 2 - 1)]     = MAX_HEAT;
    out[(l / 2 - 1) * l + (l / 2)]     = MAX_HEAT;
    out[(l / 2) * l + (l / 2)]         = MAX_HEAT;
}

// GPU Optimized function
void GPU_array_process(double *input, double *output, int length, int iterations) {
    //Cuda events for calculating elapsed time
    cudaEvent_t cpy_H2D_start, cpy_H2D_end, comp_start, comp_end, cpy_D2H_start, cpy_D2H_end;
    cudaEventCreate(&cpy_H2D_start);
    cudaEventCreate(&cpy_H2D_end);
    cudaEventCreate(&cpy_D2H_start);
    cudaEventCreate(&cpy_D2H_end);
    cudaEventCreate(&comp_start);
    cudaEventCreate(&comp_end);

    /* Preprocessing goes here */

    double *input_gpu;
    double *output_gpu;
    size_t size = length * length * sizeof(double);

    if (cudaMalloc((void **) &input_gpu, size) != cudaSuccess) {
        return;
    }

    if (cudaMalloc((void **) &output_gpu, size) != cudaSuccess) {
        cudaFree(input_gpu);
        return;
    }

    cudaEventRecord(cpy_H2D_start);
    /* Copying array from host to device goes here */
    if (cudaMemcpy(input_gpu, input, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(input_gpu);
        cudaFree(output_gpu);
        return;
    };

    cudaEventRecord(cpy_H2D_end);
    cudaEventSynchronize(cpy_H2D_end);

    //Copy array from host to device
    cudaEventRecord(comp_start);
    /* GPU calculation goes here */

    int l = length - 2;

    double *temp;
    int thrdsPerBlk = 0;
    if (l <= 512) { thrdsPerBlk = l; }
    else { thrdsPerBlk = 512; }

    int casePerBlk = (l * l + 512 - 1) / 512;

    for (int i = 0; i < iterations; i++) {
        siuuu<<<l, l>>>(input_gpu, output_gpu, length);

        // eventual implementation if we have
        // more than 1024 per block
        //siuuu2<<<l, thrdsPerBlk>>>(input_gpu, output_gpu, length, casePerBlk);

        // cuda synchronize not needed, as we are instantiating a new Cuda Kernel every loop
        // this would be unnecessary overhead

        temp = input_gpu;
        input_gpu = output_gpu;
        output_gpu = temp;
    }

    cudaEventRecord(comp_end);
    cudaEventSynchronize(comp_end);

    cudaEventRecord(cpy_D2H_start);
    /* Copying array from device to host goes here */
    if (iterations % 2 == 0) {
        cudaMemcpy(output, input_gpu, size, cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(output, output_gpu, size, cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(cpy_D2H_end);
    cudaEventSynchronize(cpy_D2H_end);

    /* Postprocessing goes here */
    cudaFree(input_gpu);
    cudaFree(output_gpu);

    float time;
    cudaEventElapsedTime(&time, cpy_H2D_start, cpy_H2D_end);
    cout << "Host to Device MemCpy takes " << setprecision(4) << time / 1000 << "s" << endl;

    cudaEventElapsedTime(&time, comp_start, comp_end);
    cout << "Computation takes " << setprecision(4) << time / 1000 << "s" << endl;

    cudaEventElapsedTime(&time, cpy_D2H_start, cpy_D2H_end);
    cout << "Device to Host MemCpy takes " << setprecision(4) << time / 1000 << "s" << endl;
}