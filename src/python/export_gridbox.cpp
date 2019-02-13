#include <boost/python.hpp>
#include <boost/python/overloads.hpp>
#include <memory>
#include <lightspeed/gridbox.hpp>
#include <lightspeed/pair_list.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/resource_list.hpp>

using namespace lightspeed;
using namespace boost::python;

BOOST_PYTHON_FUNCTION_OVERLOADS(gridbox_lda_density_ov,GridBox::ldaDensity,5,6)
BOOST_PYTHON_FUNCTION_OVERLOADS(gridbox_lda_potential_ov,GridBox::ldaPotential,5,6)
BOOST_PYTHON_FUNCTION_OVERLOADS(gridbox_lda_grad_ov,GridBox::ldaGrad,6,7)
BOOST_PYTHON_FUNCTION_OVERLOADS(gridbox_gga_density_ov,GridBox::ggaDensity,5,6)
BOOST_PYTHON_FUNCTION_OVERLOADS(gridbox_gga_potential_ov,GridBox::ggaPotential,5,6)
BOOST_PYTHON_FUNCTION_OVERLOADS(gridbox_gga_grad_ov,GridBox::ggaGrad,6,7)
BOOST_PYTHON_FUNCTION_OVERLOADS(gridbox_meta_density_ov,GridBox::metaDensity,5,6)
BOOST_PYTHON_FUNCTION_OVERLOADS(gridbox_orbitals_ov,GridBox::orbitals,5,6)
BOOST_PYTHON_FUNCTION_OVERLOADS(gridbox_orbitals_grad_ov,GridBox::orbitalsGrad,6,7)

void export_gridbox()
{
    class_<HashedGrid, std::shared_ptr<HashedGrid> >("HashedGrid", init<
        const std::shared_ptr<Tensor>&,
        double
        >())
        .add_property("xyz", &HashedGrid::xyz)
        .add_property("R", &HashedGrid::R)
        .add_property("nbox", &HashedGrid::nbox)
        .def("__str__", &HashedGrid::string)
        ;

    class_<GridBox>("GridBox", no_init)
        .def("ldaDensity", &GridBox::ldaDensity, gridbox_lda_density_ov())
        .staticmethod("ldaDensity")
        .def("ldaPotential", &GridBox::ldaPotential, gridbox_lda_potential_ov())
        .staticmethod("ldaPotential")
        .def("ldaGrad", &GridBox::ldaGrad, gridbox_lda_grad_ov())
        .staticmethod("ldaGrad")
        .def("ggaDensity", &GridBox::ggaDensity, gridbox_gga_density_ov())
        .staticmethod("ggaDensity")
        .def("ggaPotential", &GridBox::ggaPotential, gridbox_gga_potential_ov())
        .staticmethod("ggaPotential")
        .def("ggaGrad", &GridBox::ggaGrad, gridbox_gga_grad_ov())
        .staticmethod("ggaGrad")
        .def("metaDensity", &GridBox::metaDensity, gridbox_meta_density_ov())
        .staticmethod("metaDensity")
        .def("orbitals", &GridBox::orbitals, gridbox_orbitals_ov())
        .staticmethod("orbitals")
        .def("orbitalsGrad", &GridBox::orbitalsGrad, gridbox_orbitals_grad_ov())
        .staticmethod("orbitalsGrad")
        ;
}
