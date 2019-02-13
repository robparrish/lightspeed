#include <boost/python.hpp>
#include <boost/python/overloads.hpp>
#include <lightspeed/solver.hpp>
#include <lightspeed/tensor.hpp>

using namespace lightspeed;
using namespace boost::python;

void export_solver()
{
    class_<Storage, std::shared_ptr<Storage> >("Storage", init<
        size_t,
        bool>())
        .add_property("size", &Storage::size)
        .add_property("is_disk", &Storage::is_disk)
        .def("__str__", &Storage::string)
        .add_property("core_data", make_function(&Storage::core_data, return_internal_reference<>()))
        .add_property("disk_data", &Storage::disk_data)
        .def("to_disk_data", &Storage::to_disk_data) 
        .def("zeros_like", &Storage::zeros_like)
        .staticmethod("zeros_like")
        .def("scale", &Storage::scale)
        .staticmethod("scale")
        .def("dot", &Storage::dot)
        .staticmethod("dot")
        .def("axpby", &Storage::axpby)
        .staticmethod("axpby")
        .def("set_scratch_path", &Storage::set_scratch_path)
        .staticmethod("set_scratch_path")
        .def("scratch_path", &Storage::scratch_path)
        .staticmethod("scratch_path")
        .def("from_tensor", &Storage::from_tensor)
        .staticmethod("from_tensor")
        .def("to_tensor", &Storage::to_tensor)
        .staticmethod("to_tensor")
        .def("from_tensor_vec", &Storage::from_tensor_vec)
        .staticmethod("from_tensor_vec")
        .def("to_tensor_vec", &Storage::to_tensor_vec)
        .staticmethod("to_tensor_vec")
        ; 
    
    class_<DIIS, std::shared_ptr<DIIS> >("DIIS", init<
        size_t>())
        .add_property("max_vectors", &DIIS::max_vectors)
        .add_property("current_vectors", &DIIS::current_vectors)
        .def("__str__", &DIIS::string)
        .def("py_iterate", &DIIS::iterate)
        ;

    class_<Davidson, std::shared_ptr<Davidson>, boost::noncopyable>("Davidson", init<
        size_t,
        size_t,
        double,
        optional<double,
        const std::vector<std::shared_ptr<Storage> >&>
        >())
        .add_property("nstate", &Davidson::nstate) 
        .add_property("nmax", &Davidson::nmax) 
        .add_property("convergence", &Davidson::convergence) 
        .def("__str__", &Davidson::string)
        .add_property("evecs", make_function(&Davidson::evecs, return_internal_reference<>()))
        .add_property("evals", make_function(&Davidson::evals, return_internal_reference<>()))
        .def("py_add_vectors", &Davidson::add_vectors)
        .def("py_add_preconditioned", &Davidson::add_preconditioned)
        .add_property("rs", make_function(&Davidson::rs, return_internal_reference<>()))
        .add_property("rnorms", make_function(&Davidson::rnorms, return_internal_reference<>()))
        .add_property("gs", make_function(&Davidson::gs, return_internal_reference<>()))
        .add_property("hs", make_function(&Davidson::hs, return_internal_reference<>()))
        .add_property("cs", make_function(&Davidson::cs, return_internal_reference<>()))
        .add_property("is_converged", &Davidson::is_converged) 
        .add_property("max_rnorm", &Davidson::max_rnorm) 
        ;


}
