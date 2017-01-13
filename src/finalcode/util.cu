#include <iostream>

void init3DHostData(float **real, float **img, int length)
{
  float *funcReal = new float[length*length*length];
  float *funcImg = new float[length*length*length];

  for (int i = 0; i < length; i++)
  {
    for (int j = 0; j < length; j++)
    {
      for (int k = 0; k < length; k++)
      {
        funcReal[i*length*length + j*length + k] = 1;
        funcImg[i*length*length + j*length + k] = 0;
      }
    }
  }

  *real = funcReal;
  *img = funcImg;
}

void init3DDeviceData(float **p_devReal, float **p_devImg, float *hostReal,
  float *hostImg, int length)
{
  float *devReal, *devImg;
  size_t transferSize = length*length*length*sizeof(float);
  cudaMalloc((void **)&devReal, transferSize);
  cudaMalloc((void **)&devImg, transferSize);

  //copy to device
  cudaMemcpy(devReal, hostReal, transferSize, cudaMemcpyHostToDevice);
  cudaMemcpy(devImg, hostImg, transferSize, cudaMemcpyHostToDevice);

  *p_devReal = devReal;
  *p_devImg = devImg;
}

void malloc3DDeviceData(float **p_devReal, float **p_devImg, int length)
{
  float *devReal, *devImg;
  size_t transferSize = length*length*length*sizeof(float);
  cudaMalloc((void **)&devReal, transferSize);
  cudaMalloc((void **)&devImg, transferSize);

  *p_devReal = devReal;
  *p_devImg = devImg;
}
//transform length
void init2DHostData(float **real, float **img, int length)
{
  float *funcReal = new float[length*length];
  float *funcImg = new float[length*length];

  for (int i = 0; i < length; i++)
  {
    for (int j = 0; j < length; j++)
    {
      funcReal[i*length + j] = j;
      funcImg[i*length + j] = j;
    }
  }

  *real = funcReal;
  *img = funcImg;
}

void init2DDeviceData(float **p_devReal, float **p_devImg, float *hostReal,
  float *hostImg, int length)
{
  float *devReal, *devImg;
  size_t transferSize = length*length*sizeof(float);
  cudaMalloc((void **)&devReal, transferSize);
  cudaMalloc((void **)&devImg, transferSize);

  //copy to device
  cudaMemcpy(devReal, hostReal, transferSize, cudaMemcpyHostToDevice);
  cudaMemcpy(devImg, hostImg, transferSize, cudaMemcpyHostToDevice);

  *p_devReal = devReal;
  *p_devImg = devImg;
}

void get3DTransformData(float **hostReal, float **hostImg, float *devReal,
  float *devImg, int length)
{
  size_t transferSize = length*length*sizeof(float);
  cudaMemcpy(*hostReal, devReal, transferSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(*hostImg, devImg, transferSize, cudaMemcpyDeviceToHost);
}

void get2DTransformData(float **hostReal, float **hostImg, float *devReal,
  float *devImg, int length)
{
  size_t transferSize = length*length*length*sizeof(float);
  cudaMemcpy(*hostReal, devReal, transferSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(*hostImg, devImg, transferSize, cudaMemcpyDeviceToHost);
}

void freeDeviceData(float *devReal, float *devImg)
{
  cudaFree(devReal);
  cudaFree(devImg);
}

void freeHostData(float *real, float *img)
{
  delete[] real;
  delete[] img;
}
