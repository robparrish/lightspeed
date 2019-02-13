#ifndef LS_RESOURCE_LIST_HPP
#define LS_RESOURCE_LIST_HPP

#include <cstddef>
#include <memory>
#include <vector>

namespace lightspeed {

class GPUContext;

/**
 * Class ResourceList provides a list of computational resources to use for CPU
 * and/or GPU-based tasks.
 *
 * For CPU-based tasks, the nthread field specifies the number of computational
 * threads (OpenMP, lightspeed, or pthreads) to use. It is assumed that the thread
 * model and/or operating system is responsible for allocating and managing
 * these threads.
 *
 * For GPU-based tasks, the gpus field provides a list of logical GPUContext
 * objects, including device id, streams, and stack-based memory system. These
 * pre-allocated objects amortize expensive pinned memory allocation.
 **/
class ResourceList {

public:

/**
 * Verbatim Constructor: fills fields below (see accessor fields for details)
 **/
ResourceList(
    int nthread,
    const std::vector<std::shared_ptr<GPUContext> >& gpus) :
    nthread_(nthread),
    gpus_(gpus) {}

// => Accessors <= //

// Number of OpenMP, boost, or pthreads to use for CPU-based tasks
int nthread() const { return nthread_; }
// Number of logical GPUs to use for GPU-based tasks
int ngpu() const { return gpus_.size(); }
// Set of GPUContext objects representing logical GPUs for GPU-based tasks
const std::vector<std::shared_ptr<GPUContext> >& gpus() const { return gpus_; } 

// => Static Setup Routines <= //

// Return the maximum number of threads currently visible to the OpenMP environment
static int thread_count();
// Return the maximum number GPUs currently visible to the CUDA runtime environment
static int gpu_count();

// Return a default ResourceList with all available resources utilized
static std::shared_ptr<ResourceList> build(
    size_t cpu_memory=1024,  // Pinned CPU memory in bytes for each GPUContext
    size_t gpu_memory=1024); // GPU memory in bytes for each GPUContext

// Return a default ResourceList with all CPU-only resources utilized
static std::shared_ptr<ResourceList> build_cpu();
// Return information regarding computational resources as:
// Number of computing cores available within this instance of resources list
// List of logical GPUContext available  within this instance ofresources list.
std::string string() const;

private:

// => Fields <= //

int nthread_;
std::vector<std::shared_ptr<GPUContext> > gpus_;

};

} // namespace lightspeed

#endif
