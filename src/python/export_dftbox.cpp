#include <boost/python.hpp>
#include <boost/python/overloads.hpp>
#include <memory>
#include <lightspeed/dftbox.hpp>
#include <lightspeed/tensor.hpp>

using namespace lightspeed;
using namespace boost::python;

BOOST_PYTHON_FUNCTION_OVERLOADS(dftbox_rks_potential_ov,DFTBox::rksPotential,7,8)
BOOST_PYTHON_FUNCTION_OVERLOADS(dftbox_rks_grad_ov,DFTBox::rksGrad,7,8)

void export_dftbox()
{
    class_<Functional, std::shared_ptr<Functional> >("Functional", no_init)
        .def("build", &Functional::build)
        .staticmethod("build")
        .add_property("name", &Functional::name) 
        .add_property("citation", &Functional::citation) 
        .def("__str__", &Functional::string)
        .add_property("type", &Functional::type) 
        .add_property("has_lsda", &Functional::has_lsda) 
        .add_property("has_gga", &Functional::has_gga) 
        .add_property("deriv", &Functional::deriv) 
        .def("get_param", &Functional::get_param)
        .def("set_param", &Functional::set_param)
        .add_property("alpha", &Functional::alpha)
        .add_property("beta", &Functional::beta)
        .add_property("omega", &Functional::omega)
        .def("set_alpha", &Functional::set_alpha)
        .def("set_beta", &Functional::set_beta)
        .def("set_omega", &Functional::set_omega)
        .add_property("is_alpha_fixed", &Functional::is_alpha_fixed)
        .add_property("is_beta_fixed", &Functional::is_beta_fixed)
        .add_property("is_omega_fixed", &Functional::is_omega_fixed)
        .def("compute", &Functional::compute)
        ;

    class_<DFTBox>("DFTBox", no_init)
        .def("rksPotential", &DFTBox::rksPotential, dftbox_rks_potential_ov())
        .staticmethod("rksPotential")
        .def("rksGrad", &DFTBox::rksGrad, dftbox_rks_grad_ov())
        .staticmethod("rksGrad")
        ;
    
}
