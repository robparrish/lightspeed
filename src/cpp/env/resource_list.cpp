#include <lightspeed/resource_list.hpp>
#include <lightspeed/gpu_context.hpp>
#include "../util/string.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace lightspeed {

int ResourceList::thread_count()
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
int ResourceList::gpu_count()
{
    return GPUContext::device_count();
}
std::shared_ptr<ResourceList> ResourceList::build(
    size_t cpu_memory,
    size_t gpu_memory)
{
    int nthread = ResourceList::thread_count();
    int ngpu = ResourceList::gpu_count();

    std::vector<std::shared_ptr<GPUContext> > gpus;
    for (int id = 0; id < ngpu; id++) {
        gpus.push_back(std::shared_ptr<GPUContext>(
            new GPUContext(id, cpu_memory, gpu_memory)));
    }

    return std::shared_ptr<ResourceList>(new ResourceList(
        nthread, gpus));
}
std::shared_ptr<ResourceList> ResourceList::build_cpu()
{
    int nthread = ResourceList::thread_count();
    std::vector<std::shared_ptr<GPUContext> > gpus;
    return std::shared_ptr<ResourceList>(new ResourceList(
        nthread, gpus));
}
std::string ResourceList::string() const
{
    std::string str = "";
    str += sprintf2("ResourceList:\n");
    str += sprintf2("  CPU threads:  %2d\n", ResourceList::nthread());
    str += sprintf2("  GPU contexts: %2d\n", ResourceList::ngpu());

    if (gpus_.size()) {
        str += "  GPU context details:\n";
        str += sprintf2("  %-2s %2s %20s %3s %11s %11s\n",
            "N",
            "ID",
            "Name",
            "CC",
            "CPU Buffer",
            "GPU Buffer");
        for (size_t ind = 0; ind < gpus_.size(); ind++) {
            str += sprintf2("  %-2d %2d %20s %3.1f %11zu %11zu\n",
                ind,
                gpus_[ind]->id(),
                gpus_[ind]->name().c_str(),
                gpus_[ind]->CC(),
                gpus_[ind]->cpu_size(),          
                gpus_[ind]->gpu_size());
        }
    }
    return str;
}

} // namespace lightspeed 
