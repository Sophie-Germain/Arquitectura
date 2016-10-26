/*
 * María Fernanda Mora Alba, 103596
 * Arquitectura de computadoras - Maestría en Ciencias en Computación
 * Programa de Introducción a los conceptos de CUDA
 * Multiplicación de matrices usando memoria compartida
 */

#include <stdlib.h>
#include <math.h>
#include <stdio.h>

/* Utilidad para checar errores de CUDA */
void checkCUDAError(const char*);


typedef struct{
	int width;
	int height;
	int stride;
	int* elements;
}Matrix;

#define BLOCK_SIZE 16
#define N 1500
#define NUM_BLOCKS N
#define THREADS_PER_BLOCK N
#define ARR_SIZE N*N

__device__ int GetElement(const Matrix A, int row, int col)
{
	return A.elements[row * A.stride + col];
}

__device__ void SetElement(Matrix A, int row, int col, int value)
{
	A.elements[row * A.stride + col] = value;
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
	Matrix Asub;
	Asub.width = BLOCK_SIZE;
	Asub.height = BLOCK_SIZE;
	Asub.stride = A.stride;
	Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];

	return Asub;
}

__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

int main(int argc, char *argv[])
{
/*eventos para timing*/
    cudaEvent_t start, stop;
    float time;

/*declaración de matrices y dimensiones*/
	int *a, *b, *c;
    Matrix h_A, h_B, h_C;
    h_A.width = h_A.height = h_A.stride = N;
    h_B.width = h_B.height = h_B.stride = N;
    h_C.width = h_C.height = h_C.stride = N;
    Matrix d_A, d_B, d_C;

    int i;
/*alocación memoria en host*/
    size_t sz = N * N * sizeof(int);
    a = (int *) malloc(sz);
    b = (int *) malloc(sz);
    c = (int *) malloc(sz);

/*inicialización de vectores*/
    for (i = 0; i < ARR_SIZE; i++) {
        a[i] = rand()%255;
        b[i] = rand()%255;
        c[i] = 0;
    }

/*eventos para timing*/
    // Create timer for timing CUDA calculation
    //PPunsigned int timer = 0;
    //PPcutCreateTimer( &timer );
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /*for(int i=0;i<ARR_SIZE; i++){
      printf("%d",a[i]);
      printf("%c",((i%N)<N-1) ? '\t':'\n');
    }
    printf("\n\n");*/

    /*for(int i=0;i<ARR_SIZE; i++){
      printf("%d",b[i]);
      printf("%c",((i%N)<N-1) ? '\t':'\n');
    }
    printf("\n\n");*/

/*alocación de memoria en devide*/
    d_A.width = d_A.stride = h_A.width; d_A.height = h_A.height;
    h_A.elements = a;
    cudaMalloc((void**) &d_A.elements, sz);

    d_B.width = d_B.stride = h_B.width; d_B.height = h_B.height;
    h_B.elements = b;
    cudaMalloc((void**) &d_B.elements, sz);

    d_C.width = d_C.stride = h_C.width; d_C.height = h_C.height;
    h_C.elements = c;
    cudaMalloc((void**) &d_C.elements, sz);

/*Copiar bloques de memoria de host al device*/
    cudaMemcpy(d_A.elements, h_A.elements, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B.elements, h_B.elements, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C.elements, h_C.elements, sz, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(N/dimBlock.x,N/dimBlock.y);
    cudaEventRecord(start,0);
    MatMulKernel<<<dimGrid,dimBlock>>>(d_A, d_B, d_C);

    //vect_add<<<1,ARRAY_SIZE>>>(d_a,d_b,d_c);
    //matrix_mult<<<N,N>>>(d_a,d_b,d_c,N);

    /* Esperar a que todos los threads sincronizados acaben y esperar errores */
    cudaThreadSynchronize();
    checkCUDAError("kernel invocation");

    /* Copiar del resultado del GPU al CPU */
    cudaMemcpy(h_C.elements,d_C.elements,sz,cudaMemcpyDeviceToHost);


    checkCUDAError("memcpy");

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( &time, start, stop );

    /*for(int i=0;i<ARR_SIZE; i++){
      printf("%d",h_C.elements[i]);
      printf("%c",((i%N)<N-1) ? '\t':'\n');
    }
    printf("\n\n");*/

    printf("\nTIEMPO DE EJECUCIÓN: %f mSeg\n\n", time);

/* Liberar memoria */
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);

    free(a);
    free(b);
    free(c);

    return 0;
}

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	Matrix Csub = GetSubMatrix(C,blockRow,blockCol);

	float Cvalue = 0;

	int row = threadIdx.y;
	int col = threadIdx.x;

	for(int m = 0; m < (A.width / BLOCK_SIZE); ++m){
		Matrix Asub = GetSubMatrix(A, blockRow, m);
		Matrix Bsub = GetSubMatrix(B,m,blockCol);
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		As[row][col] = GetElement(Asub,row,col);
		Bs[row][col] = GetElement(Bsub,row,col);

		__syncthreads();

		for(int e=0;e<BLOCK_SIZE;++e)
			Cvalue += As[row][e] * Bs[e][col];

		__syncthreads();
	}
	SetElement(Csub,row,col,Cvalue);
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
