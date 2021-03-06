//transform length
#define TLEN 128
#define TILE_DIM 16
#define HEIGHT 8 //1, 2, 4, 8
#define NEG_2PI_BY_TLEN -0.04908738521f //-2*PI/128
#define STRIDE_STAGE_1 0x00000040 //64
#define STRIDE_STAGE_2 0x00000020 //32
#define STRIDE_STAGE_3 0x00000010 //16
#define STRIDE_STAGE_4 0x00000008 //08
#define STRIDE_STAGE_5 0x00000004 //04
#define STRIDE_STAGE_6 0x00000002 //02
#define STRIDE_STAGE_7 0x00000001 //01

#define M12 0x60
#define M34 0x18
#define M35 0x14
#define M46 0x0a
#define M246 0x2a
#define M135 0x54
#define M357 0x15
#define M56 0x06
#define M5 0x04

__device__ int bitreverse(int tid)
{
  int revtid = tid;
  revtid = ((0xf0f0f0f0 & revtid) >> 4) | ((0x0f0f0f0f & revtid) << 4);
  revtid = ((0xcccccccc & revtid) >> 2) | ((0x33333333 & revtid) << 2);
  revtid = ((0xaaaaaaaa & revtid) >> 2) | ((0x55555555 & revtid));
  return revtid;
}

__global__ void FFT1D(float *real, float *img)
{
  float rVal[HEIGHT], iVal[HEIGHT];
  //assuming that thread block size is 128
  __shared__ float shReal[HEIGHT][TLEN];
  __shared__ float shImg[HEIGHT][TLEN];

  #pragma unroll
  for(int I = 0; I < HEIGHT; I++)
  {
    rVal[I] = real[((blockIdx.x*HEIGHT) + I)*blockDim.x + threadIdx.x];
    iVal[I] = img[((blockIdx.x*HEIGHT) + I)*blockDim.x + threadIdx.x];
  }

  #pragma unroll
  for(int I = 0; I < HEIGHT; I++)
  {
  shReal[I][threadIdx.x] = rVal[I];
  shImg[I][threadIdx.x] = iVal[I];
  }
  __syncthreads();

  if((threadIdx.x & STRIDE_STAGE_1) == STRIDE_STAGE_1)
  {
      rVal[0] = shReal[0][threadIdx.x - STRIDE_STAGE_1] - rVal[0];
      iVal[0] = shImg[0][threadIdx.x - STRIDE_STAGE_1] - iVal[0];

#if HEIGHT > 1
      rVal[1] = shReal[1][threadIdx.x - STRIDE_STAGE_1] - rVal[1];
      iVal[1] = shImg[1][threadIdx.x - STRIDE_STAGE_1] - iVal[1];
#endif

#if HEIGHT > 2
      rVal[2] = shReal[2][threadIdx.x - STRIDE_STAGE_1] - rVal[2];
      iVal[2] = shImg[2][threadIdx.x - STRIDE_STAGE_1] - iVal[2];

      rVal[3] = shReal[3][threadIdx.x - STRIDE_STAGE_1] - rVal[3];
      iVal[3] = shImg[3][threadIdx.x - STRIDE_STAGE_1] - iVal[3];
#endif

#if HEIGHT > 4
      rVal[4] = shReal[4][threadIdx.x - STRIDE_STAGE_1] - rVal[4];
      iVal[4] = shImg[4][threadIdx.x - STRIDE_STAGE_1] - iVal[4];

      rVal[5] = shReal[5][threadIdx.x - STRIDE_STAGE_1] - rVal[5];
      iVal[5] = shImg[5][threadIdx.x - STRIDE_STAGE_1] - iVal[5];

      rVal[6] = shReal[6][threadIdx.x - STRIDE_STAGE_1] - rVal[6];
      iVal[6] = shImg[6][threadIdx.x - STRIDE_STAGE_1] - iVal[6];

      rVal[7] = shReal[7][threadIdx.x - STRIDE_STAGE_1] - rVal[7];
      iVal[7] = shImg[7][threadIdx.x - STRIDE_STAGE_1] - iVal[7];
#endif
  }
else
  {
      rVal[0] = shReal[0][threadIdx.x + STRIDE_STAGE_1] + rVal[0];
      iVal[0] = shImg[0][threadIdx.x + STRIDE_STAGE_1] + iVal[0];

#if HEIGHT > 1
      rVal[1] = shReal[1][threadIdx.x + STRIDE_STAGE_1] + rVal[1];
      iVal[1] = shImg[1][threadIdx.x + STRIDE_STAGE_1] + iVal[1];
#endif

#if HEIGHT > 2
      rVal[2] = shReal[2][threadIdx.x + STRIDE_STAGE_1] + rVal[2];
      iVal[2] = shImg[2][threadIdx.x + STRIDE_STAGE_1] + iVal[2];

      rVal[3] = shReal[3][threadIdx.x + STRIDE_STAGE_1] + rVal[3];
      iVal[3] = shImg[3][threadIdx.x + STRIDE_STAGE_1] + iVal[3];
#endif

#if HEIGHT > 4
      rVal[4] = shReal[4][threadIdx.x + STRIDE_STAGE_1] + rVal[4];
      iVal[4] = shImg[4][threadIdx.x + STRIDE_STAGE_1] + iVal[4];

      rVal[5] = shReal[5][threadIdx.x + STRIDE_STAGE_1] + rVal[5];
      iVal[5] = shImg[5][threadIdx.x + STRIDE_STAGE_1] + iVal[5];

      rVal[6] = shReal[6][threadIdx.x + STRIDE_STAGE_1] + rVal[6];
      iVal[6] = shImg[6][threadIdx.x + STRIDE_STAGE_1] + iVal[6];

      rVal[7] = shReal[7][threadIdx.x + STRIDE_STAGE_1] + rVal[7];
      iVal[7] = shImg[7][threadIdx.x + STRIDE_STAGE_1] + iVal[7];
#endif

  }

  if((threadIdx.x & STRIDE_STAGE_2) == STRIDE_STAGE_2)
  {
    //calculate Q twiddle factor
    float Q = (float)((threadIdx.x & STRIDE_STAGE_1)>>1);
    Q = Q*NEG_2PI_BY_TLEN;

    float c, s, T;
    __sincosf(Q, &s, &c);

    #pragma unroll
    for(int I = 0; I < HEIGHT; I++)
    {
      T = rVal[I];
      rVal[I] = rVal[I]*c - iVal[I]*s;
      iVal[I] = T*s + iVal[I]*c;
    }
  }

  #pragma unroll
  for(int I = 0; I < HEIGHT; I++)
  {
  shReal[I][threadIdx.x] = rVal[I];
  shImg[I][threadIdx.x] = iVal[I];
  }
  __syncthreads();

  //stage 2
  if((threadIdx.x & STRIDE_STAGE_2) == STRIDE_STAGE_2)
  {
      rVal[0] = shReal[0][threadIdx.x - STRIDE_STAGE_2] - rVal[0];
      iVal[0] = shImg[0][threadIdx.x - STRIDE_STAGE_2] - iVal[0];

#if HEIGHT > 1
      rVal[1] = shReal[1][threadIdx.x - STRIDE_STAGE_2] - rVal[1];
      iVal[1] = shImg[1][threadIdx.x - STRIDE_STAGE_2] - iVal[1];
#endif

#if HEIGHT > 2
      rVal[2] = shReal[2][threadIdx.x - STRIDE_STAGE_2] - rVal[2];
      iVal[2] = shImg[2][threadIdx.x - STRIDE_STAGE_2] - iVal[2];

      rVal[3] = shReal[3][threadIdx.x - STRIDE_STAGE_2] - rVal[3];
      iVal[3] = shImg[3][threadIdx.x - STRIDE_STAGE_2] - iVal[3];
#endif

#if HEIGHT > 4
      rVal[4] = shReal[4][threadIdx.x - STRIDE_STAGE_2] - rVal[4];
      iVal[4] = shImg[4][threadIdx.x - STRIDE_STAGE_2] - iVal[4];

      rVal[5] = shReal[5][threadIdx.x - STRIDE_STAGE_2] - rVal[5];
      iVal[5] = shImg[5][threadIdx.x - STRIDE_STAGE_2] - iVal[5];

      rVal[6] = shReal[6][threadIdx.x - STRIDE_STAGE_2] - rVal[6];
      iVal[6] = shImg[6][threadIdx.x - STRIDE_STAGE_2] - iVal[6];

      rVal[7] = shReal[7][threadIdx.x - STRIDE_STAGE_2] - rVal[7];
      iVal[7] = shImg[7][threadIdx.x - STRIDE_STAGE_2] - iVal[7];
#endif

  }
else
  {
      rVal[0] = shReal[0][threadIdx.x + STRIDE_STAGE_2] + rVal[0];
      iVal[0] = shImg[0][threadIdx.x + STRIDE_STAGE_2] + iVal[0];

#if HEIGHT > 1
      rVal[1] = shReal[1][threadIdx.x + STRIDE_STAGE_2] + rVal[1];
      iVal[1] = shImg[1][threadIdx.x + STRIDE_STAGE_2] + iVal[1];
#endif

#if HEIGHT > 2
      rVal[2] = shReal[2][threadIdx.x + STRIDE_STAGE_2] + rVal[2];
      iVal[2] = shImg[2][threadIdx.x + STRIDE_STAGE_2] + iVal[2];

      rVal[3] = shReal[3][threadIdx.x + STRIDE_STAGE_2] + rVal[3];
      iVal[3] = shImg[3][threadIdx.x + STRIDE_STAGE_2] + iVal[3];
#endif

#if HEIGHT > 4
      rVal[4] = shReal[4][threadIdx.x + STRIDE_STAGE_2] + rVal[4];
      iVal[4] = shImg[4][threadIdx.x + STRIDE_STAGE_2] + iVal[4];

      rVal[5] = shReal[5][threadIdx.x + STRIDE_STAGE_2] + rVal[5];
      iVal[5] = shImg[5][threadIdx.x + STRIDE_STAGE_2] + iVal[5];

      rVal[6] = shReal[6][threadIdx.x + STRIDE_STAGE_2] + rVal[6];
      iVal[6] = shImg[6][threadIdx.x + STRIDE_STAGE_2] + iVal[6];

      rVal[7] = shReal[7][threadIdx.x + STRIDE_STAGE_2] + rVal[7];
      iVal[7] = shImg[7][threadIdx.x + STRIDE_STAGE_2] + iVal[7];
#endif

  }

  //end of stage 2
  //multiply with twiddle for beginning of stage 3
  //STRIDE_STAGE_3 16
  if((threadIdx.x & STRIDE_STAGE_3) == STRIDE_STAGE_3)
  {
    //calculate Q twiddle factor
    float Q = (float)((threadIdx.x & STRIDE_STAGE_2) | ((threadIdx.x & STRIDE_STAGE_1)>>2));
    Q = Q*NEG_2PI_BY_TLEN;

    float c, s, T;
    __sincosf(Q, &s, &c);

    #pragma unroll
    for(int I = 0; I < HEIGHT; I++)
    {
      T = rVal[I];
      rVal[I] = rVal[I]*c - iVal[I]*s;
      iVal[I] = T*s + iVal[I]*c;
    }
  }

  #pragma unroll
  for(int I = 0; I < HEIGHT; I++)
  {
  shReal[I][threadIdx.x] = rVal[I];
  shImg[I][threadIdx.x] = iVal[I];
  }
  __syncthreads();

  //stage3
  if((threadIdx.x & STRIDE_STAGE_3) == STRIDE_STAGE_3)
  {
      rVal[0] = shReal[0][threadIdx.x - STRIDE_STAGE_3] - rVal[0];
      iVal[0] = shImg[0][threadIdx.x - STRIDE_STAGE_3] - iVal[0];

#if HEIGHT > 1
      rVal[1] = shReal[1][threadIdx.x - STRIDE_STAGE_3] - rVal[1];
      iVal[1] = shImg[1][threadIdx.x - STRIDE_STAGE_3] - iVal[1];
#endif

#if HEIGHT > 2
      rVal[2] = shReal[2][threadIdx.x - STRIDE_STAGE_3] - rVal[2];
      iVal[2] = shImg[2][threadIdx.x - STRIDE_STAGE_3] - iVal[2];

      rVal[3] = shReal[3][threadIdx.x - STRIDE_STAGE_3] - rVal[3];
      iVal[3] = shImg[3][threadIdx.x - STRIDE_STAGE_3] - iVal[3];
#endif

#if HEIGHT > 4
      rVal[4] = shReal[4][threadIdx.x - STRIDE_STAGE_3] - rVal[4];
      iVal[4] = shImg[4][threadIdx.x - STRIDE_STAGE_3] - iVal[4];

      rVal[5] = shReal[5][threadIdx.x - STRIDE_STAGE_3] - rVal[5];
      iVal[5] = shImg[5][threadIdx.x - STRIDE_STAGE_3] - iVal[5];

      rVal[6] = shReal[6][threadIdx.x - STRIDE_STAGE_3] - rVal[6];
      iVal[6] = shImg[6][threadIdx.x - STRIDE_STAGE_3] - iVal[6];

      rVal[7] = shReal[7][threadIdx.x - STRIDE_STAGE_3] - rVal[7];
      iVal[7] = shImg[7][threadIdx.x - STRIDE_STAGE_3] - iVal[7];
#endif

  }
else
  {
      rVal[0] = shReal[0][threadIdx.x + STRIDE_STAGE_3] + rVal[0];
      iVal[0] = shImg[0][threadIdx.x + STRIDE_STAGE_3] + iVal[0];

#if HEIGHT > 1
      rVal[1] = shReal[1][threadIdx.x + STRIDE_STAGE_3] + rVal[1];
      iVal[1] = shImg[1][threadIdx.x + STRIDE_STAGE_3] + iVal[1];
#endif

#if HEIGHT > 2
      rVal[2] = shReal[2][threadIdx.x + STRIDE_STAGE_3] + rVal[2];
      iVal[2] = shImg[2][threadIdx.x + STRIDE_STAGE_3] + iVal[2];

      rVal[3] = shReal[3][threadIdx.x + STRIDE_STAGE_3] + rVal[3];
      iVal[3] = shImg[3][threadIdx.x + STRIDE_STAGE_3] + iVal[3];
#endif

#if HEIGHT > 4
      rVal[4] = shReal[4][threadIdx.x + STRIDE_STAGE_3] + rVal[4];
      iVal[4] = shImg[4][threadIdx.x + STRIDE_STAGE_3] + iVal[4];

      rVal[5] = shReal[5][threadIdx.x + STRIDE_STAGE_3] + rVal[5];
      iVal[5] = shImg[5][threadIdx.x + STRIDE_STAGE_3] + iVal[5];

      rVal[6] = shReal[6][threadIdx.x + STRIDE_STAGE_3] + rVal[6];
      iVal[6] = shImg[6][threadIdx.x + STRIDE_STAGE_3] + iVal[6];

      rVal[7] = shReal[7][threadIdx.x + STRIDE_STAGE_3] + rVal[7];
      iVal[7] = shImg[7][threadIdx.x + STRIDE_STAGE_3] + iVal[7];
#endif

  }
  //end of stage 3
  //multiply with twiddle for beginning of stage 4
  //STRIDE_STAGE_4 8
  if((threadIdx.x & STRIDE_STAGE_4) == STRIDE_STAGE_4)
  {
    //calculate Q twiddle factor
    float Q = (float)(((threadIdx.x & STRIDE_STAGE_1)>>3) |
      ((threadIdx.x & STRIDE_STAGE_2)>>1) | ((threadIdx.x & STRIDE_STAGE_3)<<1));
    Q = Q*NEG_2PI_BY_TLEN;

    float c, s, T;
    __sincosf(Q, &s, &c);

    #pragma unroll
    for(int I = 0; I < HEIGHT; I++)
    {
      T = rVal[I];
      rVal[I] = rVal[I]*c - iVal[I]*s;
      iVal[I] = T*s + iVal[I]*c;
    }
  }

  #pragma unroll
  for(int I = 0; I < HEIGHT; I++)
  {
  shReal[I][threadIdx.x] = rVal[I];
  shImg[I][threadIdx.x] = iVal[I];
  }
  __syncthreads();

  //stage 4
  if((threadIdx.x & STRIDE_STAGE_4) == STRIDE_STAGE_4)
  {
      rVal[0] = shReal[0][threadIdx.x - STRIDE_STAGE_4] - rVal[0];
      iVal[0] = shImg[0][threadIdx.x - STRIDE_STAGE_4] - iVal[0];

#if HEIGHT > 1
      rVal[1] = shReal[1][threadIdx.x - STRIDE_STAGE_4] - rVal[1];
      iVal[1] = shImg[1][threadIdx.x - STRIDE_STAGE_4] - iVal[1];
#endif

#if HEIGHT > 2
      rVal[2] = shReal[2][threadIdx.x - STRIDE_STAGE_4] - rVal[2];
      iVal[2] = shImg[2][threadIdx.x - STRIDE_STAGE_4] - iVal[2];

      rVal[3] = shReal[3][threadIdx.x - STRIDE_STAGE_4] - rVal[3];
      iVal[3] = shImg[3][threadIdx.x - STRIDE_STAGE_4] - iVal[3];
#endif

#if HEIGHT > 4
      rVal[4] = shReal[4][threadIdx.x - STRIDE_STAGE_4] - rVal[4];
      iVal[4] = shImg[4][threadIdx.x - STRIDE_STAGE_4] - iVal[4];

      rVal[5] = shReal[5][threadIdx.x - STRIDE_STAGE_4] - rVal[5];
      iVal[5] = shImg[5][threadIdx.x - STRIDE_STAGE_4] - iVal[5];

      rVal[6] = shReal[6][threadIdx.x - STRIDE_STAGE_4] - rVal[6];
      iVal[6] = shImg[6][threadIdx.x - STRIDE_STAGE_4] - iVal[6];

      rVal[7] = shReal[7][threadIdx.x - STRIDE_STAGE_4] - rVal[7];
      iVal[7] = shImg[7][threadIdx.x - STRIDE_STAGE_4] - iVal[7];
#endif

  }
else
  {
      rVal[0] = shReal[0][threadIdx.x + STRIDE_STAGE_4] + rVal[0];
      iVal[0] = shImg[0][threadIdx.x + STRIDE_STAGE_4] + iVal[0];

#if HEIGHT > 1
      rVal[1] = shReal[1][threadIdx.x + STRIDE_STAGE_4] + rVal[1];
      iVal[1] = shImg[1][threadIdx.x + STRIDE_STAGE_4] + iVal[1];
#endif

#if HEIGHT > 2
      rVal[2] = shReal[2][threadIdx.x + STRIDE_STAGE_4] + rVal[2];
      iVal[2] = shImg[2][threadIdx.x + STRIDE_STAGE_4] + iVal[2];

      rVal[3] = shReal[3][threadIdx.x + STRIDE_STAGE_4] + rVal[3];
      iVal[3] = shImg[3][threadIdx.x + STRIDE_STAGE_4] + iVal[3];
#endif

#if HEIGHT > 4
      rVal[4] = shReal[4][threadIdx.x + STRIDE_STAGE_4] + rVal[4];
      iVal[4] = shImg[4][threadIdx.x + STRIDE_STAGE_4] + iVal[4];

      rVal[5] = shReal[5][threadIdx.x + STRIDE_STAGE_4] + rVal[5];
      iVal[5] = shImg[5][threadIdx.x + STRIDE_STAGE_4] + iVal[5];

      rVal[6] = shReal[6][threadIdx.x + STRIDE_STAGE_4] + rVal[6];
      iVal[6] = shImg[6][threadIdx.x + STRIDE_STAGE_4] + iVal[6];

      rVal[7] = shReal[7][threadIdx.x + STRIDE_STAGE_4] + rVal[7];
      iVal[7] = shImg[7][threadIdx.x + STRIDE_STAGE_4] + iVal[7];
#endif

  }
  //end of stage 4
  //multiply with twiddle for beginning of stage 5
  //STRIDE_STAGE_5 4
  if((threadIdx.x & STRIDE_STAGE_5) == STRIDE_STAGE_5)
  {
    //calculate Q twiddle factor
    int q = ((threadIdx.x & M12)>>4) | (threadIdx.x & M34);
    q = (q & M35) | ((q & M46)<<2);
    float Q = q*NEG_2PI_BY_TLEN;

    float c, s, T;
    __sincosf(Q, &s, &c);

    #pragma unroll
    for(int I = 0; I < HEIGHT; I++)
    {
      T = rVal[I];
      rVal[I] = rVal[I]*c - iVal[I]*s;
      iVal[I] = T*s + iVal[I]*c;
    }
  }

  #pragma unroll
  for(int I = 0; I < HEIGHT; I++)
  {
  shReal[I][threadIdx.x] = rVal[I];
  shImg[I][threadIdx.x] = iVal[I];
  }
  __syncthreads();

  //stage 5
  if((threadIdx.x & STRIDE_STAGE_5) == STRIDE_STAGE_5)
  {
      rVal[0] = shReal[0][threadIdx.x - STRIDE_STAGE_5] - rVal[0];
      iVal[0] = shImg[0][threadIdx.x - STRIDE_STAGE_5] - iVal[0];

#if HEIGHT > 1
      rVal[1] = shReal[1][threadIdx.x - STRIDE_STAGE_5] - rVal[1];
      iVal[1] = shImg[1][threadIdx.x - STRIDE_STAGE_5] - iVal[1];
#endif

#if HEIGHT > 2
      rVal[2] = shReal[2][threadIdx.x - STRIDE_STAGE_5] - rVal[2];
      iVal[2] = shImg[2][threadIdx.x - STRIDE_STAGE_5] - iVal[2];

      rVal[3] = shReal[3][threadIdx.x - STRIDE_STAGE_5] - rVal[3];
      iVal[3] = shImg[3][threadIdx.x - STRIDE_STAGE_5] - iVal[3];
#endif

#if HEIGHT > 4
      rVal[4] = shReal[4][threadIdx.x - STRIDE_STAGE_5] - rVal[4];
      iVal[4] = shImg[4][threadIdx.x - STRIDE_STAGE_5] - iVal[4];

      rVal[5] = shReal[5][threadIdx.x - STRIDE_STAGE_5] - rVal[5];
      iVal[5] = shImg[5][threadIdx.x - STRIDE_STAGE_5] - iVal[5];

      rVal[6] = shReal[6][threadIdx.x - STRIDE_STAGE_5] - rVal[6];
      iVal[6] = shImg[6][threadIdx.x - STRIDE_STAGE_5] - iVal[6];

      rVal[7] = shReal[7][threadIdx.x - STRIDE_STAGE_5] - rVal[7];
      iVal[7] = shImg[7][threadIdx.x - STRIDE_STAGE_5] - iVal[7];
#endif

  }
else
  {
      rVal[0] = shReal[0][threadIdx.x + STRIDE_STAGE_5] + rVal[0];
      iVal[0] = shImg[0][threadIdx.x + STRIDE_STAGE_5] + iVal[0];

#if HEIGHT > 1
      rVal[1] = shReal[1][threadIdx.x + STRIDE_STAGE_5] + rVal[1];
      iVal[1] = shImg[1][threadIdx.x + STRIDE_STAGE_5] + iVal[1];
#endif

#if HEIGHT > 2
      rVal[2] = shReal[2][threadIdx.x + STRIDE_STAGE_5] + rVal[2];
      iVal[2] = shImg[2][threadIdx.x + STRIDE_STAGE_5] + iVal[2];

      rVal[3] = shReal[3][threadIdx.x + STRIDE_STAGE_5] + rVal[3];
      iVal[3] = shImg[3][threadIdx.x + STRIDE_STAGE_5] + iVal[3];
#endif

#if HEIGHT > 4
      rVal[4] = shReal[4][threadIdx.x + STRIDE_STAGE_5] + rVal[4];
      iVal[4] = shImg[4][threadIdx.x + STRIDE_STAGE_5] + iVal[4];

      rVal[5] = shReal[5][threadIdx.x + STRIDE_STAGE_5] + rVal[5];
      iVal[5] = shImg[5][threadIdx.x + STRIDE_STAGE_5] + iVal[5];

      rVal[6] = shReal[6][threadIdx.x + STRIDE_STAGE_5] + rVal[6];
      iVal[6] = shImg[6][threadIdx.x + STRIDE_STAGE_5] + iVal[6];

      rVal[7] = shReal[7][threadIdx.x + STRIDE_STAGE_5] + rVal[7];
      iVal[7] = shImg[7][threadIdx.x + STRIDE_STAGE_5] + iVal[7];
#endif

  }
  //end of stage 5
  //multiply with twiddle for beginning of stage 6
  //STRIDE_STAGE_6 2
  if((threadIdx.x & STRIDE_STAGE_6) == STRIDE_STAGE_6)
  {
    //calculate Q twiddle factor
    int q = ((threadIdx.x & M12)>>5) | ((threadIdx.x & M34)>>1) |
      ((threadIdx.x & M5)<<3);
    q = (q & M246) | ((q & M357)<<2);
    float Q = q*NEG_2PI_BY_TLEN;

    float c, s, T;
    __sincosf(Q, &s, &c);

    #pragma unroll
    for(int I = 0; I < HEIGHT; I++)
    {
      T = rVal[I];
      rVal[I] = rVal[I]*c - iVal[I]*s;
      iVal[I] = T*s + iVal[I]*c;
    }
  }

  #pragma unroll
  for(int I = 0; I < HEIGHT; I++)
  {
  shReal[I][threadIdx.x] = rVal[I];
  shImg[I][threadIdx.x] = iVal[I];
  }
  __syncthreads();

  //stage 6
  if((threadIdx.x & STRIDE_STAGE_6) == STRIDE_STAGE_6)
  {
      rVal[0] = shReal[0][threadIdx.x - STRIDE_STAGE_6] - rVal[0];
      iVal[0] = shImg[0][threadIdx.x - STRIDE_STAGE_6] - iVal[0];

#if HEIGHT > 1
      rVal[1] = shReal[1][threadIdx.x - STRIDE_STAGE_6] - rVal[1];
      iVal[1] = shImg[1][threadIdx.x - STRIDE_STAGE_6] - iVal[1];
#endif

#if HEIGHT > 2
      rVal[2] = shReal[2][threadIdx.x - STRIDE_STAGE_6] - rVal[2];
      iVal[2] = shImg[2][threadIdx.x - STRIDE_STAGE_6] - iVal[2];

      rVal[3] = shReal[3][threadIdx.x - STRIDE_STAGE_6] - rVal[3];
      iVal[3] = shImg[3][threadIdx.x - STRIDE_STAGE_6] - iVal[3];
#endif

#if HEIGHT > 4
      rVal[4] = shReal[4][threadIdx.x - STRIDE_STAGE_6] - rVal[4];
      iVal[4] = shImg[4][threadIdx.x - STRIDE_STAGE_6] - iVal[4];

      rVal[5] = shReal[5][threadIdx.x - STRIDE_STAGE_6] - rVal[5];
      iVal[5] = shImg[5][threadIdx.x - STRIDE_STAGE_6] - iVal[5];

      rVal[6] = shReal[6][threadIdx.x - STRIDE_STAGE_6] - rVal[6];
      iVal[6] = shImg[6][threadIdx.x - STRIDE_STAGE_6] - iVal[6];

      rVal[7] = shReal[7][threadIdx.x - STRIDE_STAGE_6] - rVal[7];
      iVal[7] = shImg[7][threadIdx.x - STRIDE_STAGE_6] - iVal[7];
#endif

  }
else
  {
      rVal[0] = shReal[0][threadIdx.x + STRIDE_STAGE_6] + rVal[0];
      iVal[0] = shImg[0][threadIdx.x + STRIDE_STAGE_6] + iVal[0];

#if HEIGHT > 1
      rVal[1] = shReal[1][threadIdx.x + STRIDE_STAGE_6] + rVal[1];
      iVal[1] = shImg[1][threadIdx.x + STRIDE_STAGE_6] + iVal[1];
#endif

#if HEIGHT > 2
      rVal[2] = shReal[2][threadIdx.x + STRIDE_STAGE_6] + rVal[2];
      iVal[2] = shImg[2][threadIdx.x + STRIDE_STAGE_6] + iVal[2];

      rVal[3] = shReal[3][threadIdx.x + STRIDE_STAGE_6] + rVal[3];
      iVal[3] = shImg[3][threadIdx.x + STRIDE_STAGE_6] + iVal[3];
#endif

#if HEIGHT > 4
      rVal[4] = shReal[4][threadIdx.x + STRIDE_STAGE_6] + rVal[4];
      iVal[4] = shImg[4][threadIdx.x + STRIDE_STAGE_6] + iVal[4];

      rVal[5] = shReal[5][threadIdx.x + STRIDE_STAGE_6] + rVal[5];
      iVal[5] = shImg[5][threadIdx.x + STRIDE_STAGE_6] + iVal[5];

      rVal[6] = shReal[6][threadIdx.x + STRIDE_STAGE_6] + rVal[6];
      iVal[6] = shImg[6][threadIdx.x + STRIDE_STAGE_6] + iVal[6];

      rVal[7] = shReal[7][threadIdx.x + STRIDE_STAGE_6] + rVal[7];
      iVal[7] = shImg[7][threadIdx.x + STRIDE_STAGE_6] + iVal[7];
#endif

  }
  //end of stage 6
  //multiply with twiddle for beginning of stage 7
  //STRIDE_STAGE_7 1
  if((threadIdx.x & STRIDE_STAGE_7) == STRIDE_STAGE_7)
  {
    //calculate Q twiddle factor
    int q = ((threadIdx.x & M12)>>4) | (threadIdx.x & M34) |
      ((threadIdx.x & M56)<<4);
    q = ((q & M135)>>2) | (q & M246);
    float Q = q*NEG_2PI_BY_TLEN;

    float c, s, T;
    __sincosf(Q, &s, &c);

    #pragma unroll
    for(int I = 0; I < HEIGHT; I++)
    {
      T = rVal[I];
      rVal[I] = rVal[I]*c - iVal[I]*s;
      iVal[I] = T*s + iVal[I]*c;
    }
  }

  #pragma unroll
  for(int I = 0; I < HEIGHT; I++)
  {
  shReal[I][threadIdx.x] = rVal[I];
  shImg[I][threadIdx.x] = iVal[I];
  }
  __syncthreads();

  //stage 7
  if((threadIdx.x & STRIDE_STAGE_7) == STRIDE_STAGE_7)
  {
      rVal[0] = shReal[0][threadIdx.x - STRIDE_STAGE_7] - rVal[0];
      iVal[0] = shImg[0][threadIdx.x - STRIDE_STAGE_7] - iVal[0];

#if HEIGHT > 1
      rVal[1] = shReal[1][threadIdx.x - STRIDE_STAGE_7] - rVal[1];
      iVal[1] = shImg[1][threadIdx.x - STRIDE_STAGE_7] - iVal[1];
#endif

#if HEIGHT > 2
      rVal[2] = shReal[2][threadIdx.x - STRIDE_STAGE_7] - rVal[2];
      iVal[2] = shImg[2][threadIdx.x - STRIDE_STAGE_7] - iVal[2];

      rVal[3] = shReal[3][threadIdx.x - STRIDE_STAGE_7] - rVal[3];
      iVal[3] = shImg[3][threadIdx.x - STRIDE_STAGE_7] - iVal[3];
#endif

#if HEIGHT > 4
      rVal[4] = shReal[4][threadIdx.x - STRIDE_STAGE_7] - rVal[4];
      iVal[4] = shImg[4][threadIdx.x - STRIDE_STAGE_7] - iVal[4];

      rVal[5] = shReal[5][threadIdx.x - STRIDE_STAGE_7] - rVal[5];
      iVal[5] = shImg[5][threadIdx.x - STRIDE_STAGE_7] - iVal[5];

      rVal[6] = shReal[6][threadIdx.x - STRIDE_STAGE_7] - rVal[6];
      iVal[6] = shImg[6][threadIdx.x - STRIDE_STAGE_7] - iVal[6];

      rVal[7] = shReal[7][threadIdx.x - STRIDE_STAGE_7] - rVal[7];
      iVal[7] = shImg[7][threadIdx.x - STRIDE_STAGE_7] - iVal[7];
#endif

  }
else
  {
      rVal[0] = shReal[0][threadIdx.x + STRIDE_STAGE_7] + rVal[0];
      iVal[0] = shImg[0][threadIdx.x + STRIDE_STAGE_7] + iVal[0];

#if HEIGHT > 1
      rVal[1] = shReal[1][threadIdx.x + STRIDE_STAGE_7] + rVal[1];
      iVal[1] = shImg[1][threadIdx.x + STRIDE_STAGE_7] + iVal[1];
#endif

#if HEIGHT > 2
      rVal[2] = shReal[2][threadIdx.x + STRIDE_STAGE_7] + rVal[2];
      iVal[2] = shImg[2][threadIdx.x + STRIDE_STAGE_7] + iVal[2];

      rVal[3] = shReal[3][threadIdx.x + STRIDE_STAGE_7] + rVal[3];
      iVal[3] = shImg[3][threadIdx.x + STRIDE_STAGE_7] + iVal[3];
#endif

#if HEIGHT > 4
      rVal[4] = shReal[4][threadIdx.x + STRIDE_STAGE_7] + rVal[4];
      iVal[4] = shImg[4][threadIdx.x + STRIDE_STAGE_7] + iVal[4];

      rVal[5] = shReal[5][threadIdx.x + STRIDE_STAGE_7] + rVal[5];
      iVal[5] = shImg[5][threadIdx.x + STRIDE_STAGE_7] + iVal[5];

      rVal[6] = shReal[6][threadIdx.x + STRIDE_STAGE_7] + rVal[6];
      iVal[6] = shImg[6][threadIdx.x + STRIDE_STAGE_7] + iVal[6];

      rVal[7] = shReal[7][threadIdx.x + STRIDE_STAGE_7] + rVal[7];
      iVal[7] = shImg[7][threadIdx.x + STRIDE_STAGE_7] + iVal[7];
#endif

  }
  int revtid = bitreverse(threadIdx.x);
/*
  #pragma unroll
  for(int I = 0; I < HEIGHT; I++)
  {
  shReal[I][revtid] = rVal[I];
  shImg[I][revtid] = iVal[I];
  }
*/
  shReal[0][revtid] = rVal[0];
  shImg[0][revtid] = iVal[0];
#if HEIGHT > 1
  shReal[1][revtid] = rVal[1];
  shImg[1][revtid] = iVal[1];
#endif

#if HEIGHT > 2
  shReal[2][revtid] = rVal[2];
  shImg[2][revtid] = iVal[2];
  shReal[3][revtid] = rVal[3];
  shImg[3][revtid] = iVal[3];
#endif

#if HEIGHT > 4
  shReal[4][revtid] = rVal[4];
  shImg[4][revtid] = iVal[4];
  shReal[5][revtid] = rVal[5];
  shImg[5][revtid] = iVal[5];
  shReal[6][revtid] = rVal[6];
  shImg[6][revtid] = iVal[6];
  shReal[7][revtid] = rVal[7];
  shImg[7][revtid] = iVal[7];
#endif

  __syncthreads();

  #pragma unroll
  for(int I = 0; I < HEIGHT; I++)
  {
    real[((blockIdx.x*HEIGHT) + I)*blockDim.x + threadIdx.x] = shReal[I][threadIdx.x];
    img[((blockIdx.x*HEIGHT) + I)*blockDim.x + threadIdx.x] = shImg[I][threadIdx.x];
  }
}

__global__ void transposeDiagonal_xy(float *odatar, float *odatai, float *idatar, float *idatai, int width)
{
  __shared__ float tile[TILE_DIM][TILE_DIM+1];

  int blockIdx_x, blockIdx_y;

  // do diagonal reordering
  blockIdx_y = blockIdx.x;
  blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;

  // from here on the code is same as previous kernel except blockIdx_x replaces blockIdx.x
  // and similarly for y

  int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex + blockIdx.z*width)*width;

  xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx_x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex + blockIdx.z*width)*width;

  tile[threadIdx.y][threadIdx.x] = idatar[index_in];
  __syncthreads();
  odatar[index_out] = tile[threadIdx.x][threadIdx.y];
  __syncthreads();
  tile[threadIdx.y][threadIdx.x] = idatai[index_in];
  __syncthreads();
  odatai[index_out] = tile[threadIdx.x][threadIdx.y];
}

__global__ void transposeDiagonal_xz(float *odatar, float *odatai, float *idatar, float *idatai, int width)
{
  __shared__ float tile[TILE_DIM][TILE_DIM+1];

  int blockIdx_x, blockIdx_z;

  // do diagonal reordering
  blockIdx_z = blockIdx.x;
  blockIdx_x = (blockIdx.x+blockIdx.z)%gridDim.x;

  // from here on the code is same as previous kernel except blockIdx_x replaces blockIdx.x
  // and similarly for y

  int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
  int zIndex = blockIdx_z * TILE_DIM + threadIdx.z;
  int index_in = xIndex + (blockIdx.y + zIndex*width)*width;

  xIndex = blockIdx_z * TILE_DIM + threadIdx.x;
  zIndex = blockIdx_x * TILE_DIM + threadIdx.z;
  int index_out = xIndex + (blockIdx.y + zIndex*width)*width;

  tile[threadIdx.z][threadIdx.x] = idatar[index_in];
  __syncthreads();
  odatar[index_out] = tile[threadIdx.x][threadIdx.z];
  __syncthreads();
  tile[threadIdx.z][threadIdx.x] = idatai[index_in];
  __syncthreads();
  odatai[index_out] = tile[threadIdx.x][threadIdx.z];
}

__global__ void transposeDiagonal_xy(float *odata, float *idata, int width)
{
  __shared__ float tile[TILE_DIM][TILE_DIM+1];

  int blockIdx_x, blockIdx_y;

  // do diagonal reordering
  blockIdx_y = blockIdx.x;
  blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;

  // from here on the code is same as previous kernel except blockIdx_x replaces blockIdx.x
  // and similarly for y

  int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex + blockIdx.z*width)*width;

  xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx_x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex + blockIdx.z*width)*width;

  tile[threadIdx.y][threadIdx.x] = idata[index_in];
  __syncthreads();
  odata[index_out] = tile[threadIdx.x][threadIdx.y];
}

__global__ void transposeDiagonal_xz(float *odata, float *idata, int width)
{
  __shared__ float tile[TILE_DIM][TILE_DIM+1];

  int blockIdx_x, blockIdx_z;

  // do diagonal reordering
  blockIdx_z = blockIdx.x;
  blockIdx_x = (blockIdx.x+blockIdx.z)%gridDim.x;

  // from here on the code is same as previous kernel except blockIdx_x replaces blockIdx.x
  // and similarly for y

  int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
  int zIndex = blockIdx_z * TILE_DIM + threadIdx.z;
  int index_in = xIndex + (blockIdx.y + zIndex*width)*width;

  xIndex = blockIdx_z * TILE_DIM + threadIdx.x;
  zIndex = blockIdx_x * TILE_DIM + threadIdx.z;
  int index_out = xIndex + (blockIdx.y + zIndex*width)*width;

  tile[threadIdx.z][threadIdx.x] = idata[index_in];
  __syncthreads();
  odata[index_out] = tile[threadIdx.x][threadIdx.z];
}
