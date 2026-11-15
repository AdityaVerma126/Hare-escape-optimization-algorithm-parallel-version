#include <cstdio>
#include <cuda_runtime.h>
__global__ void k(int* p){ p[threadIdx.x] = threadIdx.x; }
int main(){ int n=0; if(cudaGetDeviceCount(&n)!=cudaSuccess||n==0){puts("no device");return 2;}
 int* d; cudaMalloc(&d,32*sizeof(int)); k<<<1,32>>>(d); cudaError_t e=cudaDeviceSynchronize();
 if(e!=cudaSuccess){printf("launch failed: %s\n",cudaGetErrorString(e));return 3;}
 cudaFree(d); printf("ok %d device(s)\n", n); return 0; }
