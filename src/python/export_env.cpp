#include <boost/python.hpp>
#include <boost/python/overloads.hpp>
#include <memory>
#include <lightspeed/config.hpp>
#include <lightspeed/gpu_context.hpp>
#include <lightspeed/resource_list.hpp>

using namespace lightspeed;
using namespace boost::python;

BOOST_PYTHON_FUNCTION_OVERLOADS(resource_list_build_ov,ResourceList::build,0,2);

namespace lightspeed {
void exit_hooks();
}

void export_env()
{
    def("exit_hooks", &exit_hooks); // Exit hooks to manually destroy global static data

    class_<Config>("Config", no_init)
        .def("git_sha", &Config::git_sha).staticmethod("git_sha")
        .def("git_dirty", &Config::git_dirty).staticmethod("git_dirty")
        .def("has_cuda", &Config::has_cuda).staticmethod("has_cuda")
        .def("has_terachem", &Config::has_terachem).staticmethod("has_terachem")
        .def("has_libxc", &Config::has_libxc).staticmethod("has_libxc")
        .def("has_openmp", &Config::has_openmp).staticmethod("has_openmp")
        ;

    class_<GPUContext, std::shared_ptr<GPUContext> >("GPUContext", init<
        int,
        size_t,
        size_t
        >())
        .add_property("id", &GPUContext::id)
        .add_property("cpu_size", &GPUContext::cpu_size)
        .add_property("gpu_size", &GPUContext::gpu_size)
        .add_property("cpu_used", &GPUContext::cpu_used)
        .add_property("gpu_used", &GPUContext::gpu_used)
        .add_property("cpu_avail", &GPUContext::cpu_avail)
        .add_property("gpu_avail", &GPUContext::gpu_avail)
        .def("alignment", &GPUContext::alignment)
        .staticmethod("alignment")
        .def("device_count", &GPUContext::device_count)
        .staticmethod("device_count")
        .def("__str__", &GPUContext::string)
        .add_property("name", &GPUContext::name)
        .add_property("CC", &GPUContext::CC)
        ;

    class_<ResourceList, std::shared_ptr<ResourceList> >("ResourceList", init<
        int,
        const std::vector<std::shared_ptr<GPUContext> >&
        >())
        .add_property("nthread", &ResourceList::nthread)
        .add_property("ngpu", &ResourceList::ngpu)
        .add_property("gpus", make_function(&ResourceList::gpus, return_internal_reference<>()))
        .def("thread_count", &ResourceList::thread_count)
        .staticmethod("thread_count")
        .def("gpu_count", &ResourceList::gpu_count)
        .staticmethod("gpu_count")
        .def("build", &ResourceList::build, resource_list_build_ov())
        .staticmethod("build")
        .def("build_cpu", &ResourceList::build_cpu)
        .staticmethod("build_cpu")
        .def("__str__", &ResourceList::string)
        ; 

}
