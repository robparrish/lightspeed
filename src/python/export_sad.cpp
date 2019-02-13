#include <boost/python.hpp>
#include <boost/python/overloads.hpp>
#include <memory>
#include <lightspeed/sad.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/molecule.hpp>

using namespace lightspeed;
using namespace boost::python;

void export_sad()
{
    class_<SAD>("SAD", no_init)
        .def("sad_nocc", &SAD::sad_nocc)
        .staticmethod("sad_nocc")
        .def("sad_nocc_neutral", &SAD::sad_nocc_neutral)
        .staticmethod("sad_nocc_neutral")
        .def("sad_nocc_atoms", &SAD::sad_nocc_atoms)
        .staticmethod("sad_nocc_atoms")
        .def("sad_orbitals", &SAD::sad_orbitals)
        .staticmethod("sad_orbitals")
        .def("project_orbitals", &SAD::project_orbitals)
        .staticmethod("project_orbitals")
        ;
}
