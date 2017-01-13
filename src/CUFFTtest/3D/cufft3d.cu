// Multi-Dimensional cufft test
// Complex to Complex
// Un-normalized
#include <math.h>
#include <cuda.h>
//#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
//#include <cutil_inline.h>

/* Define a type mult-dim data input to be a struct with integer members x,y,z */
typedef struct {
	int x;
	int y;
	int z;
} data_size;
// declaration forward
void runTest(cufftComplex *, data_size, int, int, int *);
static __device__ __host__ inline cufftComplex ComplexScale(cufftComplex, float);
// Program main
int main(int argc, char *argv[]){
	
	data_size data_sz;
		
	/* from user cmd line input */
	int nx = 128;
	int ny = 128;
	int nz = 128;
	int batch = 1;
	// Transform Dimension
	const int nrank = 3;
	/* Check dimension match or not */
	if ((nz>1 && nrank<3) || (ny>1 && nrank<2)){
		fprintf(stderr, "Input data and transform dimension do not match, quit\n");
		return 1;
	}		

	data_sz.x = nx;
	data_sz.y = ny;
	data_sz.z = nz;	

	//int n[nrank] = {nx, ny};
	int n[nrank] = {nx, ny, nz};
	/* Host memory allocation */	
	cufftComplex* data = (cufftComplex*)malloc(sizeof(cufftComplex)*nx*ny*nz*batch);
	/* source data creation */
	for (unsigned int i=0; i<nx*ny*nz; i++){
		data[i].x = 1.0f;
		data[i].y = 1.0f;
	}
	
	runTest(data, data_sz, nrank, batch, n);
	free(data);	
	// printf("Seg\n");

	/* for (int i=0; i<NX*NY*BATCH; i++){
		printf("data[%d] %f %f\n", i, data[i].x, data[i].y);
	}
	*/
	return 0;
}

// ! Run a simple test with Transform size --
//void runTest2d(cufftComplex* dataPtr, int NX, int NY, int NRANK, int BATCH, int * n){
void runTest(cufftComplex* dataPtr, data_size data_sz, int NRANK, int BATCH, int * n){
	
	cufftHandle plan;
	cufftComplex *devPtr;
	int NX = data_sz.x;
	int NY = data_sz.y;
	int NZ = data_sz.z;
	/* GPU memory allocation */
	//cudaMalloc((void**)&devPtr, sizeof(cufftComplex)*NX*NY*NZ*BATCH);
	cudaMalloc((void**)&devPtr, sizeof(cufftComplex)*NX*NY*NZ*BATCH);

	/* Error Checker */
	if (cudaGetLastError()!= cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		// return 1;
	}

	/* transfer to GPU memory */
	cudaMemcpy(devPtr, dataPtr, sizeof(cufftComplex)*NX*NY*NZ*BATCH, cudaMemcpyHostToDevice);
	if (cudaGetLastError()!= cudaSuccess){
		fprintf(stderr, "cudaMemcpy failed!\n");
		// return 1;
	}
	
	/* One device memory */
//	cudaMemset(data, 1, sizeof());

#ifdef CUDA_TIMER
	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);
#endif

#ifdef CUDA_TIMER
	cudaEventRecord(start_event, 0);
	float t_device;
#else
	cudaThreadSynchronize();
	double t_device = -getTime();
#endif

	/* Create a mult-D FFT plan */
	if (cufftPlanMany(&plan, NRANK, n, NULL, 1, NX*NY*NZ, // *inembed, istride, idist
					NULL, 1, NX*NY*NZ, // *onembed, ostride, odist
					CUFFT_C2C, BATCH) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed\n");
		//return 1;
	}

	/* Use the CUFFT plan to transform the signal in place */
	/* executes Forward  FFT */
	if (cufftExecC2C(plan, devPtr, devPtr, CUFFT_FORWARD) != CUFFT_SUCCESS)		{
		fprintf(stderr, "Cuda error: ExecC2C Forward failed\n");
		// return 1;
	}

	if (cudaThreadSynchronize() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
		// return 1;
	}
#ifdef CUDA_TIMER
	cudaEventRecord(stop_event, 0);
	cudaEventSynchronize(stop_event);
	cudaEventElapsedTime(&t_device, start_event, stop_event);

#else
	/* block until the device has finished */
	cudaThreadSynchronize();
	/* stop the timer */
	t_device += getTime();
#endif
	fprintf(stdout, "Time: %f\n", t_device);
	
	/* executes Inverse FFT */
	cufftExecC2C(plan, devPtr, devPtr, CUFFT_INVERSE);
	/* transfer results from GPU memory */
	cudaMemcpy(dataPtr, devPtr, sizeof(cufftComplex)*NX*NY*NZ*BATCH, cudaMemcpyDeviceToHost);
	/* deletes CUFFT plan */
	cufftDestroy(plan);
	/* frees GPU memory */
	cudaFree(devPtr);
	// free(data);
	cudaThreadExit();
	// return 0;
}
