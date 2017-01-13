#define TIMING
#define FFT
#define TRANSPOSE
#define FFT3D
#define OUTPUT_DATA

#define TEST_ITER 1
#include <iostream>
#include "fft.cu"

void init2DHostData(float **real, float **img, int length);
void init2DDeviceData(float **p_devReal, float **p_devImg, float *hostReal,
  float *hostImg, int length);
void init3DHostData(float **real, float **img, int length);
void init3DDeviceData(float **p_devReal, float **p_devImg, float *hostReal,
  float *hostImg, int length);
void freeDeviceData(float *devReal, float *devImg);
void freeHostData(float *real, float *img);
void get2DTransformData(float **hostReal, float **hostImg, float *devReal,
  float *devImg, int length);
void get3DTransformData(float **hostReal, float **hostImg, float *devReal,
  float *devImg, int length);
void malloc3DDeviceData(float **p_devReal, float **p_devImg, int length);
int main(int argc, char *argv[])
{
  float *hReal, *hImg;
  float *dReal, *dImg;
  float *d2Real, *d2Img;

  init3DHostData(&hReal, &hImg, TLEN);
  init3DDeviceData(&dReal, &dImg, hReal, hImg, TLEN);
  malloc3DDeviceData(&d2Real, &d2Img, TLEN);

#ifdef FFT
  int blockD = TLEN;
  int gridD = TLEN*TLEN/HEIGHT;
#endif

  dim3 transposeXYBlock(TILE_DIM, TILE_DIM, 1);
  dim3 transposeXZBlock(TILE_DIM, 1, TILE_DIM);
  dim3 gridXYBlock(TLEN/TILE_DIM, TLEN/TILE_DIM, TLEN);
  dim3 gridXZBlock(TLEN/TILE_DIM, TLEN, TLEN/TILE_DIM);

  #ifdef TIMING
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  #endif

  #pragma unroll
  for(int i = 0; i<TEST_ITER; i++)
  {

#ifdef FFT
    FFT1D<<<gridD, blockD>>>(dReal, dImg);
#endif

#ifdef FFT3D

#ifdef TRANSPOSE
    transposeDiagonal_xy<<<gridXYBlock, transposeXYBlock>>>(d2Real, dReal, TLEN);
    transposeDiagonal_xy<<<gridXYBlock, transposeXYBlock>>>(d2Img, dImg, TLEN);
#endif

#ifdef FFT
    FFT1D<<<gridD, blockD>>>(d2Real, d2Img);
#endif

#ifdef TRANSPOSE
    transposeDiagonal_xz<<<gridXZBlock, transposeXZBlock>>>(dReal, d2Real, TLEN);
    transposeDiagonal_xz<<<gridXZBlock, transposeXZBlock>>>(dImg, d2Img, TLEN);
#endif

#ifdef FFT
    FFT1D<<<gridD, blockD>>>(dReal, dImg);
#endif

#ifdef TRANSPOSE
    transposeDiagonal_xy<<<gridXYBlock, transposeXYBlock>>>(d2Real, dReal, TLEN);
    transposeDiagonal_xy<<<gridXYBlock, transposeXYBlock>>>(d2Img, dImg, TLEN);
    transposeDiagonal_xz<<<gridXZBlock, transposeXZBlock>>>(dReal, d2Real, TLEN);
    transposeDiagonal_xz<<<gridXZBlock, transposeXZBlock>>>(dImg, d2Img, TLEN);
#endif

#endif
  }

  #ifdef TIMING
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  #endif

#ifdef OUTPUT_DATA
  get3DTransformData(&hReal, &hImg, dReal, dImg, TLEN);
  for(int i = 0; i<TLEN; i++)
    std::cout<<"\n"<<i<<" | "<<hReal[i]<<" "<<hImg[i];
  std::cout<<"\n";
#endif
  freeDeviceData(dReal, dImg);
  freeHostData(hReal, hImg);

  #ifdef TIMING
  std::cout<<"\nTiming (ms) : "<<elapsedTime/TEST_ITER<<std::endl;
  #endif
  return 0;
}
