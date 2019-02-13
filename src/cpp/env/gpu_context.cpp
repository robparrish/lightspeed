#include <lightspeed/gpu_context.hpp>
#include "../util/cuerr.h"
#include "../util/string.hpp"
#include <stdexcept>

namespace lightspeed {

GPUContext::GPUContext(
    int id,
    size_t cpu_size,
    size_t gpu_size) :
    id_(id),
    cpu_size_(cpu_size),
    gpu_size_(gpu_size)
{
#ifdef HAVE_CUDA
    cudaSetDevice(id); CUERR;
    //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1); CUERR; // TODO: We need to think about this.
   
    cudaStreamCreate(&stream_); CUERR;

    cudaError_t cuerr;
    cuerr = cudaMalloc(&gpu_buf_, gpu_size_);
    if (cuerr != cudaSuccess) {
        throw std::runtime_error("GPUContext: cannot allocate GPU buffer.");
    }
    cuerr = cudaMallocHost(&cpu_buf_, cpu_size_);
    if (cuerr != cudaSuccess) {
        throw std::runtime_error("GPUContext: cannot allocate CPU buffer.");
    }

    cpu_top_ = cpu_buf_;
    gpu_top_ = gpu_buf_;
#else
    throw std::runtime_error("GPUContext: No CUDA - should not construct");
#endif
}
GPUContext::~GPUContext()
{
#ifdef HAVE_CUDA
    cudaError_t cuerr;
    cuerr = cudaSetDevice(id_);
    if (cuerr != cudaSuccess) {
        throw std::runtime_error("GPUContext~: cannot set device.");
    }
    cuerr = cudaStreamDestroy(stream_);
    if (cuerr != cudaSuccess) {
        throw std::runtime_error("GPUContext~: cannot destroy cudaStream.");
    }
    cuerr = cudaFree(gpu_buf_);
    if (cuerr != cudaSuccess) {
        throw std::runtime_error("GPUContext~: cannot free GPU buffer.");
    }
    cuerr = cudaFreeHost(cpu_buf_);
    if (cuerr != cudaSuccess) {
        throw std::runtime_error("GPUContext~: cannot free CPU buffer.");
    }
#endif
}
    
void* GPUContext::cpu_malloc(size_t size)
{
    void* ptr = cpu_top_;
    cpu_top_ += ((size + alignment() - 1) / alignment()) * alignment();
    if (cpu_top_ - cpu_buf_ > cpu_size_) {
        throw std::runtime_error("GPUContext::cpu_malloc: out of memory.");
    }
    return ptr;
}
void* GPUContext::gpu_malloc(size_t size)
{
    void* ptr = gpu_top_;
    gpu_top_ += ((size + alignment() - 1) / alignment()) * alignment();
    if (gpu_top_ - gpu_buf_ > gpu_size_) {
        throw std::runtime_error("GPUContext::gpu_malloc: out of memory.");
    }
    return ptr;
}
void GPUContext::cpu_free(void* ptr)
{
    if (ptr < cpu_buf_ || ptr >= cpu_top_ || ((((char*)ptr) - cpu_buf_) % alignment())) {
        throw std::runtime_error("GPUContext::cpu_free: bad stack pointer.");
    }
    cpu_top_ = (char*) ptr;
}
void GPUContext::gpu_free(void* ptr)
{
    if (ptr < gpu_buf_ || ptr >= gpu_top_ || ((((char*)ptr) - gpu_buf_) % alignment())) {
        throw std::runtime_error("GPUContext::gpu_free: bad stack pointer.");
    }
    gpu_top_ = (char*) ptr;
}
void GPUContext::clear()
{
    cpu_top_ = cpu_buf_;
    gpu_top_ = gpu_buf_;
}
int GPUContext::device_count()
{   
#ifdef HAVE_CUDA
    int count;
    if (cudaGetDeviceCount(&count) == cudaSuccess) {
        return count;
    } else {
        // there is no CUDA-capable device
        return 0;
    }
#else
    return 0;
#endif
}
std::string GPUContext::string() const
{
#ifdef HAVE_CUDA
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, id_ );
    std::string str = "";
    str += sprintf2("\nGPUContext:\n");
    str += sprintf2("  Name:                               %s\n",  devProp.name);
    str += sprintf2("  Device ID:                          %d\n",  id_);
    str += sprintf2("  Compute capability:                 %d.%d\n",  devProp.major,  devProp.minor);
    str += sprintf2("  Total global memory (B):            %u\n",  devProp.totalGlobalMem);
    str += sprintf2("  Total shared memory per block  (B): %u\n",  devProp.sharedMemPerBlock);
    str += sprintf2("  Total registers per block:          %d\n",  devProp.regsPerBlock);
    str += sprintf2("  Warp size:                          %d\n",  devProp.warpSize);
    str += sprintf2("  Maximum memory pitch (B):           %u\n",  devProp.memPitch);
    str += sprintf2("  Maximum threads per block:          %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        str += sprintf2("  Maximum dimension %d of block:       %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        str += sprintf2("  Maximum dimension %d of grid:        %d\n", i, devProp.maxGridSize[i]);
    str += sprintf2("  Clock rate (KHz):                   %d\n",  devProp.clockRate);
    str += sprintf2("  Total constant memory (B):          %u\n",  devProp.totalConstMem);
    str += sprintf2("  Texture alignment:                  %u\n",  devProp.textureAlignment);
    str += sprintf2("  Concurrent copy and execution:      %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    str += sprintf2("  Number of multiprocessors:          %d\n",  devProp.multiProcessorCount);
    str += sprintf2("  Kernel execution timeout:           %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    str += sprintf2("  Peak memory bandwidth (GB/s):       %4.3E\n",  2.0*devProp.memoryClockRate*(devProp.memoryBusWidth/8)/1.0e6);
    str += sprintf2("  CPU buffer size (B):                %zu\n", cpu_size());
    str += sprintf2("  GPU buffer size (B):                %zu\n", gpu_size());
    
    return str;
#else
    throw std::runtime_error("GPUContext: No CUDA.");
#endif
}
std::string GPUContext::name() const
{
#ifdef HAVE_CUDA
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, id_ );
    return std::string(devProp.name);
#else
    throw std::runtime_error("GPUContext: No CUDA.");
#endif
}
float GPUContext::CC() const
{
#ifdef HAVE_CUDA
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, id_ );
    return (devProp.major + 0.1f * devProp.minor);
#else
    throw std::runtime_error("GPUContext: No CUDA.");
#endif
}

} // namespace lightspeed 
