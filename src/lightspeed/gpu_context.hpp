#ifndef LS_GPU_CONTEXT_HPP
#define LS_GPU_CONTEXT_HPP

#include <cstddef>
#include <vector>
#include <string>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace lightspeed {

/**
 * Class GPUContext represents a single logical CUDA GPU, including the
 * following considerations:
 *
 *  -physical device id  (id field)
 *  -a (non-default) cudaStream_t struct through which to interact with the GPU
 *  (stream field)
 *  -a stack-based memory allocation system for both CPU (pinned memory) and
 *  GPU. The user provides the fields cpu_size and gpu_size (allowed memory in
 *  bytes) to the constructor of the GPUContext object. The constructor then
 *  allocates these arrays once (using expensive pinned-memory allocation on
 *  the CPU), and gives away aligned pointers to pieces of these arrays via the
 *  cpu_malloc and gpu_malloc methods. Memory provided by these methods has
 *  essentially zero allocation latency (as this has all been amortized by the
 *  two mallocs in the GPUContext constructor). Moreover, this memory can be
 *  used with CUDA Async memory commands, as the CPU arrays live in pinned
 *  memory. Calls to cpu_free and gpu_free logically release all memory after
 *  the input pointer. Therefore, a common workflow is:
 *
 *  // Enter some heavy CPU-side subroutine calling GPU kernels
 *
 *  // Set the device, in case thread has changed
 *  cudaSetDevice(gpu->id());
 *
 *  // Keep track of the start of the memory stack
 *  char* cpu_tag = (char*) gpu->cpu_malloc(1);
 *  char* gpu_tag = (char*) gpu->gpu_malloc(1);
 *
 *  // Grab some memory on both CPU and GPU
 *  double* cpu_buf = (double*) gpu->cpu_malloc(100*sizeof(double));
 *  double* gpu_buf = (double*) gpu->gpu_malloc(100*sizeof(double));
 *
 *  // Put stuff in cpu_buf
 *
 *  // Copy array to GPU, asynchronously
 *  cudaMemcpyAsync(gpu_buf,cpu_buf,100*sizeof(double),cudaMemcpyHostToDevice,gpu->stream());
 *
 *  // Launch a kernel on GPU, asynchronously, overwriting gpu_buf
 *  kernel<128,128,0,gpu->stream()>(gpu_buf);
 *
 *  // Copy array to CPU, asynchronously
 *  cudaMemcpyAsync(cpu_buf,gpu_buf,100*sizeof(double),cudaDeviceToHost,gpu->stream());
 *
 *  cudaStreamSynchronize(gpu->stream());
 *
 *  // Use the stuff in cpu_buf
 *
 *  // Free the used stack memory
 *  gpu->cpu_free(cpu_tag);
 *  gpu->gpu_free(gpu_tag);
 *
 *  // Do not use cpu_buf or gpu_buf anymore!
 *
 *  Note that only the calls to free cpu_tag and gpu_tag are necessary, no
 *  matter how many calls to cpu_malloc and gpu_malloc are made in the
 *  subroutine. This is due to the stack model of memory management used.
 **/
class GPUContext {

public:



/**
 * Allocating constructor. Copies fields, initializes GPU stream, allocates CPU
 * pinned memory and GPU memory stack arrays. Throws if any allocation or
 * initialization step fails.
 *
 * @param id the CUDA Runtime device id to work with
 * @cpu_size the total size of CPU pinned memory stack, in bytes
 * @gpu_size the total size of GPU memory stack, in bytes
 *
 * Usually cpu_size == gpu_size, but this is ultimately up to the user.
 **/
GPUContext(
    int id,
    size_t cpu_size,
    size_t gpu_size);

// Destructor, frees CPU/GPU stacks and other resources. Note that a "cannot
// set device" or other error from this destructor likely indicates an error
// upstream in the asynchronous CUDA call stack.
~GPUContext();

// CUDA Runtime device id, e.g., used in cudaSetDevice(context->id());
int id() const { return id_; }
#ifdef HAVE_CUDA
// Non-default stream via which to interact with GPU
const cudaStream_t& stream() const { return stream_; }
#endif
// Amount of pinned CPU memory allocated, in bytes
size_t cpu_size() const { return cpu_size_; }
// Amount of GPU memory allocated, in bytes
size_t gpu_size() const { return gpu_size_; }
// Amount of pinned CPU memory used, in bytes
size_t cpu_used() const { return cpu_top_ - cpu_buf_; }
// Amount of GPU memory used, in bytes
size_t gpu_used() const { return gpu_top_ - gpu_buf_; }
// Amount of pinned CPU memory remaining, in bytes
size_t cpu_avail() const { return cpu_size() - cpu_used(); }
// Amount of GPU memory remaining, in bytes
size_t gpu_avail() const { return gpu_size() - gpu_used(); }
// Simulates malloc'ing size bytes of CPU pinned memory. Returns a pointer to
// the aligned top of the CPU pinned memory stack, and advances the top of the
// stack to the next aligned address after size bytes. Throws if the allocation
// (including alignment) exceeds the amount of CPU pinned memory available in
// the stack.
void* cpu_malloc(size_t size);
// Simulates malloc'ing size bytes of GPU memory. Returns a pointer to the
// aligned top of the GPU memory stack, and advances the top of the stack to
// the next aligned address after size bytes. Throws if the allocation
// (including alignment) exceeds the amount of GPU memory available in the
// stack.
void* gpu_malloc(size_t size);
// Moves the top of the CPU pinned memory stack to ptr, effectively freeing ptr
// and all addresses allocated after it. Throws if ptr is not a valid, aligned
// address within the CPU pinned memory stack.
// (roughly, only pointers acquired by cpu_malloc can be freed by cpu_free).
void cpu_free(void* ptr);
// Moves the top of the GPU memory stack to ptr, effectively freeing ptr and
// and all addresses allocated after it. Throws if ptr is not a valid, aligned
// address within the GPU memory stack 
// (roughly, only pointers acquired by gpu_malloc can be freed by gpu_free).
void gpu_free(void* ptr);
// Free all memory on CPU and GPU [calls cpu_free(cpu_buf_); and gpu_free(gpu_buf_);]
void clear();
// The alignment of malloc calls in cpu_malloc and gpu_malloc above (improves
// performance of GPU memory operations)
static int alignment() { return 256; }
// The total number of GPUs visible to the CUDA runtime environment (wraps
// cudaGetDeviceCount)
static int device_count();

// A handy string representation gathering of information about this GPU as the
// following: Name, architecture, total global and shred memory, # of registers
// per block, wrap size, maximum memory pitch, threads, blocks and grids per
// block, GPUs clock rate, total constant memory, # of multiprocessors and peak
// memory bandwidth.
std::string string() const;
// The device name for this GPUContext, e.g., "GeForce GTX 970"
std::string name() const;
// The compute capability (CC) for this GPUContext, e.g., 5.2
float CC() const;

private:

// CUDA RT device id
int id_;
#ifdef HAVE_CUDA
// CUDA stream
cudaStream_t stream_;
#endif
// Size of CPU buffer in bytes
size_t cpu_size_;
// Size of GPU buffer in bytes
size_t gpu_size_;
// CPU buffer (pinned memory)
char* cpu_buf_;
// GPU buffer 
char* gpu_buf_;
// Top of the CPU stack
char* cpu_top_;
// Top of the GPU stack
char* gpu_top_;

};

} // namespace lightspeed

#endif
