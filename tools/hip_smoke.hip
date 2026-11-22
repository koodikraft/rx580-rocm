#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void k(float* a){a[threadIdx.x]+=1.f;}
int main(){int n;hipGetDeviceCount(&n);printf("dev=%d\n",n);hipDeviceProp_t p;hipGetDeviceProperties(&p,0);printf("%s %s\n",p.name,p.gcnArchName);hipSetDevice(0);float*d;hipMalloc(&d,1024);k<<<1,256>>>(d);hipDeviceSynchronize();printf("err=%s\n",hipGetErrorString(hipGetLastError()));return 0;}
