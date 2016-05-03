/*
 * María Fernanda Mora Alba, 103596
 * Arquitectura de computadoras - Maestría en Ciencias en Computación
 * Práctica de Introducción a los conceptos de CUDA
 * Multiplicación de matrices
 */


#include <stdio.h>
#include <stdlib.h>

void checkCUDAError(const char*);

/*definimos función kernel matrix_mult a ejecutarse en devide*/
__global__ void matrix_mult(int *d_a, int *d_b, int *d_c, int DIM)
{
    int idx = blockIdx.x;
    int jdx = threadIdx.x;
    int sum = 0;
    for(int k=0; k < DIM; k++){
        sum += d_a[idx*DIM + k]*d_b[DIM*k + jdx];
    }
    d_c[threadIdx.x + (blockIdx.x * blockDim.x)] = sum;
}

/*definimos dimensiones*/

#define N 10
#define NUM_BLOCKS N
#define THREADS_PER_BLOCK N
#define ARR_SIZE N*N

int main(int argc, char *argv[])
{

/*eventos para contar tiempo*/
    cudaEvent_t start, stop;
    float time;

/*declaración de arreglos en host y device*/
    int *h_a, *h_b, *h_c; /* Arreglos del host */
    int *d_a, *d_b, *d_c;/* Arreglos del device */

    int i;

/*alocación memoria en host*/
    size_t sz = N * N * sizeof(int);
    h_a = (int *) malloc(sz);
    h_b = (int *) malloc(sz);
    h_c = (int *) malloc(sz);

/*inicializamos eventos*/
    // timer for timing CUDA calculation
    //PPunsigned int timer = 0;
    //PPcutCreateTimer( &timer );
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

/*alocación de memoria en device*/
    cudaMalloc((void**) &d_a, sz);
    cudaMalloc((void**) &d_b, sz);
    cudaMalloc((void**) &d_c, sz);

/*inicializamos arreglos*/
    for (i = 0; i < ARR_SIZE; i++) {
        h_a[i] = rand()%255;
        h_b[i] = rand()%255;
        h_c[i] = 0;
    }

/*copiamos bloques de memoria de host a device*/
    cudaMemcpy(d_a, h_a, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, sz, cudaMemcpyHostToDevice);

/*definimos dimensiones*/
    dim3 dimBlock(32,32);
    dim3 dimGrid((N+31)/32,(N+31)/32);
    cudaEventRecord(start,0);
    matrix_mult<<<dimGrid,dimBlock>>>(d_a,d_b,d_c,N);
/*sincronizamos hilos*/
    cudaThreadSynchronize();
    checkCUDAError("kernel invocation");

/*copiamos resultado de device a host*/
    cudaMemcpy(h_a,d_a,sz,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b,d_b,sz,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c,d_c,sz,cudaMemcpyDeviceToHost);


    checkCUDAError("memcpy");

/*al ejecutarse todo, detenemos los eventos*/
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime( &time, start, stop );

/*imprimimos resultados*/
    printf("\nTIEMPO DE EJECUCIÓN: %f mSeg\n\n", time);

/*liberamos memorias en host y device*/
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}

/* Utility function to check for and report CUDA errors */
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}
