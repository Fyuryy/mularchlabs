/*
============================================================================
Filename    : implementation.cu 
Author      : Jacopo Ferro, Luna Godier
SCIPER      : 299301, 296817
============================================================================
*/

#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <cuda_runtime.h>
using namespace std;

// // CUDA kernel
// #define BLOCK_SIZE 32
// #define TILE_SIZE 32

// #define HEAT_CENTER 1000
// define constants
#define HEAT_CENTER 1000
#define MAX_THREADS_PER_BLOCK 512
#define DOUBLE_SIZE sizeof(double)

//define functions to be used later
__global__ void compute(double *input, double *output, int length);


// CPU Baseline
void array_process(double *input, double *output, int length, int iterations)
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


__global__ 
void compute(double *input, double *output, int length, int iterations)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // set the heat source
    if (i == length/2-1 && j == length/2-1) {
        output[i*length+j] = HEAT_CENTER;
        output[(i+1)*length+j] = HEAT_CENTER;
        output[i*length+j+1] = HEAT_CENTER;
        output[(i+1)*length+j+1] = HEAT_CENTER;
    }

    if (i < length && j < length)
    {
        for(int n=0; n<iterations; n++)
        {
            if (i > 0 && i < length-1 && j > 0 && j < length-1)
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
            else
            {
                output[(i)*(length)+(j)] = input[(i)*(length)+(j)];
            }
        }
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

    /* Copy input to device */
    double *d_input, *d_output;
    cudaMalloc((void**)&d_input, length*length*sizeof(double));
    cudaMalloc((void**)&d_output, length*length*sizeof(double));


    cudaEventRecord(cpy_H2D_start);
    /* Copying array from host to device goes here */
    cudaEventRecord(cpy_H2D_end);
    cudaEventSynchronize(cpy_H2D_end);

    //Copy array from host to device
    cudaEventRecord(comp_start);
    /* GPU calculation goes here */
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((length + dimBlock.x - 1) / dimBlock.x, (length + dimBlock.y - 1) / dimBlock.y);
    compute<<<dimGrid, dimBlock>>>(d_input, d_output, length, iterations);

    cudaEventRecord(comp_end);
    cudaEventSynchronize(comp_end);

    cudaEventRecord(cpy_D2H_start);
    /* Copying array from device to host goes here */
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


// // GPU Optimized function
// void GPU_array_process(double *input, double *output, int length, int iterations)
// {
//     //Cuda events for calculating elapsed time
//     cudaEvent_t cpy_H2D_start, cpy_H2D_end, comp_start, comp_end, cpy_D2H_start, cpy_D2H_end;
//     cudaEventCreate(&cpy_H2D_start);
//     cudaEventCreate(&cpy_H2D_end);
//     cudaEventCreate(&cpy_D2H_start);
//     cudaEventCreate(&cpy_D2H_end);
//     cudaEventCreate(&comp_start);
//     cudaEventCreate(&comp_end);

//     /* Preprocessing goes here */

//     //first we allocate memory for the gpu arrays

//     const size_t SIZE = DOUBLE_SIZE*length*length;
//     double* input_gpu;
//     cudaMalloc(&input_gpu, SIZE);
//     double* output_gpu;
//     cudaMalloc(&output_gpu, SIZE);
    
//     cudaEventRecord(cpy_H2D_start);
//     /* Copying array from host to device goes here */

//     cudaMemcpy(input_gpu, input, SIZE, cudaMemcpyHostToDevice);  
//     cudaMemcpy(output_gpu, output, SIZE, cudaMemcpyHostToDevice);

//     cudaEventRecord(cpy_H2D_end);
//     cudaEventSynchronize(cpy_H2D_end);

//     //Copy array from host to device
//     cudaEventRecord(comp_start);
//     /* GPU calculation goes here */
//     const int CEIL_THREADS = ceil((double)length/MAX_THREADS_PER_BLOCK);
//     double *temp_array;

//     dim3 blocks(length,CEIL_THREADS); 
//     int threads = min(length, MAX_THREADS_PER_BLOCK);
//     size_t sharedMemSize = 3*threads*DOUBLE_SIZE;
//     for (size_t k = 0; k < iterations; k++) {
//         compute <<< blocks, threads, sharedMemSize >>> (input_gpu, output_gpu, length);
//         //need to wait after each interation for all to be done
//         cudaDeviceSynchronize(); 
        
//         //SWAP JUST LIKE IN ASSIGNEMENT 2

//         temp_array = input_gpu;
//         input_gpu = output_gpu; 
//         output_gpu = temp_array;
//     }   

//     cudaEventRecord(comp_end);
//     cudaEventSynchronize(comp_end);

//     cudaEventRecord(cpy_D2H_start);
//     /* Copying array from device to host goes here */

//     cudaMemcpy(output, input_gpu, SIZE, cudaMemcpyDeviceToHost);

//     cudaEventRecord(cpy_D2H_end);
//     cudaEventSynchronize(cpy_D2H_end);

//     /* Postprocessing goes here */

//     //free memory allocated previously
//     cudaFree(input_gpu);
//     cudaFree(output_gpu);

//     float time;
//     cudaEventElapsedTime(&time, cpy_H2D_start, cpy_H2D_end);
//     cout<<"Host to Device MemCpy takes "<<setprecision(4)<<time/1000<<"s"<<endl;

//     cudaEventElapsedTime(&time, comp_start, comp_end);
//     cout<<"Computation takes "<<setprecision(4)<<time/1000<<"s"<<endl;

//     cudaEventElapsedTime(&time, cpy_D2H_start, cpy_D2H_end);
//     cout<<"Device to Host MemCpy takes "<<setprecision(4)<<time/1000<<"s"<<endl;
// }

// __global__
// void compute(double *input, double *output, int length)
// {
//   int i = blockIdx.x + 1; //shifting to not consider
//   int j = threadIdx.x + 1;
//   output[i*length + j] = (input[(i-1)*(length)+(j-1)] +
//                       input[(i-1)*(length)+(j)]   +
//                       input[(i-1)*(length)+(j+1)] +
//                       input[(i)*(length)+(j-1)]   +
//                       input[(i)*(length)+(j)]     +
//                       input[(i)*(length)+(j+1)]   +
//                       input[(i+1)*(length)+(j-1)] +
//                       input[(i+1)*(length)+(j)]   +
//                       input[(i+1)*(length)+(j+1)] ) / 9;

//   output[(length/2-1)*length+(length/2-1)] = HEAT_CENTER;
//   output[(length/2)*length+(length/2-1)]   = HEAT_CENTER;
//   output[(length/2-1)*length+(length/2)]   = HEAT_CENTER;
//   output[(length/2)*length+(length/2)]     = HEAT_CENTER;
// }