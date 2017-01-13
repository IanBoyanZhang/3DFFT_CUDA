// 1D fft vector test
// Complex to Complex
// Normalized
// Forward
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

//#include "types.h"
//#include "utils.h"
#include <sys/time.h>
#include <stdio.h>

#define NX	128
#define NY	128
//#define BATCH	10
#define BATCH	128
//#define NRANK	3
#define NRANK	1

void genMatrix( float *a, unsigned int m, unsigned int n);
void verify( float *C, unsigned int m, unsigned int n, float eps, char *mesg);
void verify( float *c_d, float *c_h, unsigned int m, unsigned int n, float eps, char *mesg);
void printMatrix( float *a, unsigned int m, unsigned int n);
double getTime();
//double gflops(int n, int niter, double time);

// declaration forward
void runTest1d(cufftComplex *, int);
static __device__ __host__ inline cufftComplex ComplexScale(cufftComplex, float);
static __global__ void ComplexPointwiseScale(cufftComplex*, int, float);
// Program main

int main(int argc, char *argv[]){

	//int n[NRANK] = {NX, NY, NZ};
	/* Host memory allocation */	
	cufftComplex* h_data = (cufftComplex*)malloc(sizeof(cufftComplex)*NX*NY*BATCH);
	const int size = NX * NY* BATCH;
	/* source data creation */
	for (unsigned int i=0; i<NX*NY*BATCH; i++){
		h_data[i].x = 1.0f;
		h_data[i].y = 1.0f;
	}
	
	runTest1d(h_data, size);
	free(h_data);	

	return 0;
}

// ! Run a simple test with Transform size --
void runTest1d(cufftComplex* dataPtr, int size){
	
	const int SCALE = 10;

	cufftHandle plan;
	cufftComplex *devPtr;

	cufftComplex *dummy_devPtr;
	
	/* GPU memory allocation */
	cudaMalloc((void**)&devPtr, sizeof(cufftComplex)*NX*NY*BATCH);
	cudaMalloc((void**)&dummy_devPtr, sizeof(cufftComplex)*NX*NY*BATCH);
	/* Error Checker */
	if (cudaGetLastError()!= cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		// return 1;
	}

	/* transfer to GPU memory */
	cudaMemcpy(devPtr, dataPtr, sizeof(cufftComplex)*NX*NY*BATCH, cudaMemcpyHostToDevice);
	/* cpy data to dummy ptr memory */	
	cudaMemcpy(dummy_devPtr, dataPtr, sizeof(cufftComplex)*NX*NY*BATCH, cudaMemcpyHostToDevice);
	if (cudaGetLastError()!= cudaSuccess){
		fprintf(stderr, "cudaMemcpy failed!\n");
		// return 1;
	}
	/* One device memory */
//	cudaMemset(data, 1, sizeof())
	// Start the timer
#ifdef CUDA_TIMER
	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);
#endif


	/* Warm up */
	/* execute cufft forward transform */
	/* Create a 1D FFT plan */
	if (cufftPlan1d(&plan, NX, CUFFT_C2C, NY*BATCH) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed");
		//return 1;
	}

	/* executes Forward  FFT */
	/* Identical pointers to input and output arrays implies in-place transformation*/
	if (cufftExecC2C(plan, devPtr, devPtr, CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stderr, "Cuda error: ExecC2C Forward failed\n");
		// return 1;
	}
	if (cudaThreadSynchronize() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
		// return 1;
	}
#ifdef CUDA_TIMER
	cudaEventRecord(start_event, 0);
	float t_device;
#else
	cudaThreadSynchronize();
	double t_device = -getTime();
#endif
	/* multi-run for timing */
	for(int r=0; r<SCALE; r++)
	{
		/* Create a 1D FFT plan */
		if (cufftPlan1d(&plan, NX, CUFFT_C2C, NY*BATCH) != CUFFT_SUCCESS){
			fprintf(stderr, "CUFFT error: Plan creation failed");
			//return 1;
		}

		/* executes Forward  FFT */
		/* Identical pointers to input and output arrays implies in-place transformation*/
		if (cufftExecC2C(plan, dummy_devPtr, dummy_devPtr, CUFFT_FORWARD) != CUFFT_SUCCESS){
			fprintf(stderr, "Cuda error: ExecC2C Forward failed\n");
			// return 1;
		}	
		if (cudaThreadSynchronize() != cudaSuccess){
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
			// return 1;
		}
	}

#ifdef CUDA_TIMER
	cudaEventRecord(stop_event, 0);
	cudaEventSynchronize(stop_event);
	/* cudaEventElapsedTime returns value in milliseconds. Resolution ~0.5ms */
	cudaEventElapsedTime(&t_device, start_event, stop_event);
	t_device /= SCALE;
	//t_device /= 1000.0;

#else
	/* block until the device has finished */
	cudaThreadSynchronize();
	/* stop the timer */
	t_device += getTime();
#endif

	/* Normalizing Scale down */
	/* 1D Thread Structure */
	dim3 blocks(size / 128);
	/* host calling */
//	ComplexPointwiseScale<<<blocks, 128>>>(devPtr, size, 1.0f/128);
	/* executes Inverse FFT */
//	cufftExecC2C(plan, devPtr, devPtr, CUFFT_INVERSE);

	/* transfer results from GPU memory */
	cudaMemcpy(dataPtr, devPtr, sizeof(cufftComplex)*NX*NY*BATCH, cudaMemcpyDeviceToHost);

	/* deletes CUFFT plan */
	cufftDestroy(plan);

	/* frees GPU memory */
	cudaFree(devPtr);
	cudaFree(dummy_devPtr);
	for (int i = 0; i < NX; i++ )
	{
		printf("dataPtr[%d] = %f %f\n", i, dataPtr[i].x, dataPtr[i].y);
	}
	// free(data);

	printf("Time: %f\n", t_device);	

	//double gflops_d = gflops(size, SCALE, t_device);
	//printf("Device computation time: %f sec. [%f gflops]\n", t_device, gflops_d);
	
	cudaThreadExit();

	// return 0;
}

static __device__ __host__ inline cufftComplex ComplexScale(cufftComplex a, float s)
{
	cufftComplex c;
	c.x = s * a.x;
	c.y = s * a.y;
	return c;
}

static __global__ void ComplexPointwiseScale(cufftComplex* a, int size, float scale)
{
	/* 128 to 128 one to one mapping */
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = threadID; i < size; i += numThreads)
		a[i] = ComplexScale(a[i], scale);
}

double getTime()
{
	const double kMicro = 1.0e-6;
	struct timeval TV;

	const int RC = gettimeofday(&TV, NULL);
	if(RC == -1)
	{
		printf("ERROR: Bad call to gettimeofday\n");
		return(-1);
	}

	return( ((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec) );

}
/*
double gflops(int n, int reps, double time){

    // Total number of entries
    long long int n2 = n;
    n2 *= n;
    // Updates
    const long long int updates =  n2 * (long long) reps;
    // Number of flops
    const long long int flops =  (long long ) n * 2L * updates;
    double flop_rate = (double) flops / time;
    return ( flop_rate/1.0e9);
}
*/
