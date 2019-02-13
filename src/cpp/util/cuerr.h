#ifndef LS_CUERR_H
#define LS_CUERR_H

#include <cstdio>
#include <cstdlib>

#ifdef HAVE_CUDA

#define CUERR { \
 cudaError_t e=cudaGetLastError(); \
 if(e!=cudaSuccess) { \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
   exit(0); \
 } \
}

#endif

#endif
