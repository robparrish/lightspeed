#ifndef LS_GPU_ATOMIC_H
#define LS_GPU_ATOMIC_H

/**
 * CUDA recommended hack for double-precision atomic addition prior to Pascal.
 * @param address the shared or global memory address requiring atomic protection.
 * @param val the amount to increment by.
 **/
#if __CUDA_ARCH__ < 600 
__device__ __forceinline__
double atomicAdd(double* address, double val)
{ 
    unsigned long long int* address_as_ull = (unsigned long long int*)address; 
    unsigned long long int old = *address_as_ull, assumed; 
    do { 
        assumed = old; 
        old = atomicCAS(address_as_ull, assumed, 
        __double_as_longlong(val + __longlong_as_double(assumed))); // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) 
    } while (assumed != old); 
    return __longlong_as_double(old); 
} 
#endif

#endif 
