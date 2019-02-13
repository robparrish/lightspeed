#include <boost/python.hpp>
#include <boost/python/overloads.hpp>
#include <lightspeed/tensor.hpp>

using namespace lightspeed;
using namespace boost::python;

dict tensor_array_interface(std::shared_ptr<Tensor> ten){
    dict rv;

    rv["shape"] = boost::python::tuple(ten->shape());
    rv["data"] = boost::python::make_tuple((size_t)ten->data().data(), false);
    
    // Type
    //std::string typestr = is_big_endian() ? ">" : "<"; # TODO
    std::string typestr = "<";
    std::stringstream sstr;
    sstr << (int)sizeof(double);
    typestr += "f" + sstr.str();
    rv["typestr"] = typestr;

    return rv;
}

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(tensor_swap_in_ov, Tensor::swap_in, 0, 1)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(tensor_swap_out_ov, Tensor::swap_out, 0, 1)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(tensor_string_ov, Tensor::string, 0, 4)
BOOST_PYTHON_FUNCTION_OVERLOADS(tensor_potrf_ov, Tensor::potrf, 1, 2)
BOOST_PYTHON_FUNCTION_OVERLOADS(tensor_trtri_ov, Tensor::trtri, 1, 2)
//BOOST_PYTHON_FUNCTION_OVERLOADS(tensor_gesv_ov, Tensor::gesv, 2, 3)
BOOST_PYTHON_FUNCTION_OVERLOADS(tensor_syev_ov, Tensor::syev, 3, 5)
BOOST_PYTHON_FUNCTION_OVERLOADS(tensor_generalized_syev_ov, Tensor::generalized_syev, 4, 6)
BOOST_PYTHON_FUNCTION_OVERLOADS(tensor_gesvd_ov, Tensor::gesvd, 4, 6)
BOOST_PYTHON_FUNCTION_OVERLOADS(tensor_power_ov, Tensor::power, 1, 4)
BOOST_PYTHON_FUNCTION_OVERLOADS(tensor_canonical_orthogonalize_ov, Tensor::canonical_orthogonalize, 1, 2)
BOOST_PYTHON_FUNCTION_OVERLOADS(tensor_chain_ov, Tensor::chain, 2, 5)
BOOST_PYTHON_FUNCTION_OVERLOADS(tensor_permute_ov, Tensor::permute, 3, 6)
BOOST_PYTHON_FUNCTION_OVERLOADS(tensor_einsum_ov, Tensor::einsum, 5, 8)

std::string Tensor__str__(const Tensor& self) {
    return self.string();
}

void export_tensor()
{
    class_<Tensor, std::shared_ptr<Tensor> >("Tensor", init<
        const std::vector<size_t>&,
        optional< 
        const std::string&
        > >())
        .add_property("name", &Tensor::name, &Tensor::set_name)
        .add_property("ndim", &Tensor::ndim)
        .add_property("size", &Tensor::size)
        .add_property("shape", make_function(&Tensor::shape, return_internal_reference<>()))
        .add_property("strides", make_function(&Tensor::strides, return_internal_reference<>()))
        .add_property("data", make_function(&Tensor::data, return_internal_reference<>()))
        .add_property("__array_interface__", tensor_array_interface)
        .def("clone", &Tensor::clone)
        .def("zero", &Tensor::zero)
        .def("identity", &Tensor::identity)
        .def("symmetrize", &Tensor::symmetrize)
        .def("antisymmetrize", &Tensor::antisymmetrize)
        .def("scale", &Tensor::scale)
        .def("copy", &Tensor::copy)
        .def("axpby", &Tensor::axpby)
        .def("vector_dot", &Tensor::vector_dot)
        .def("transpose", &Tensor::transpose)
        .def("string", &Tensor::string, tensor_string_ov())
        .def("__str__", &Tensor__str__)
        .def("ndim_error", &Tensor::ndim_error)
        .def("shape_error", &Tensor::shape_error)
        .def("square_error", &Tensor::square_error)
        .def("syev", &Tensor::syev, tensor_syev_ov())
        .staticmethod("syev")
        .def("generalized_syev", &Tensor::generalized_syev, tensor_generalized_syev_ov())
        .staticmethod("generalized_syev")
        .def("gesvd", &Tensor::gesvd, tensor_gesvd_ov())
        .staticmethod("gesvd")
        .def("power", &Tensor::power, tensor_power_ov())
        .staticmethod("power")
        .def("potrf", &Tensor::potrf, tensor_potrf_ov())
        .staticmethod("potrf")
        .def("trtri", &Tensor::trtri, tensor_trtri_ov())
        .staticmethod("trtri")
        .def("gesv", &Tensor::gesv)
        .staticmethod("gesv")
        .def("py_chain", &Tensor::chain, tensor_chain_ov())
        .staticmethod("py_chain")
        .def("py_permute", &Tensor::permute, tensor_permute_ov())
        .staticmethod("py_permute")
        .def("py_einsum", &Tensor::einsum, tensor_einsum_ov())
        .staticmethod("py_einsum")
        .def("lowdin_orthogonalize", &Tensor::lowdin_orthogonalize)
        .staticmethod("lowdin_orthogonalize")
        .def("cholesky_orthogonalize", &Tensor::cholesky_orthogonalize)
        .staticmethod("cholesky_orthogonalize")
        .def("canonical_orthogonalize", &Tensor::canonical_orthogonalize, tensor_canonical_orthogonalize_ov())
        .staticmethod("canonical_orthogonalize")
        .def("invert_lu", &Tensor::invert_lu)
        .def("total_memory", &Tensor::total_memory)
        .staticmethod("total_memory")
        ;

}
