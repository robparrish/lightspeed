#include <boost/python.hpp>
#include <boost/python/overloads.hpp>
#include <memory>
#include <lightspeed/cubic.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/molecule.hpp>

using namespace lightspeed;
using namespace boost::python;

void export_cubic()
{
    class_<CubicGrid, std::shared_ptr<CubicGrid> >("CubicGrid", init<
        const std::vector<double>&,
        const std::vector<double>&,
        const std::vector<size_t>&
        >())
        .def("build_fourier", &CubicGrid::build_fourier)
        .staticmethod("build_fourier") 
        .def("build_next_fourier", &CubicGrid::build_next_fourier)
        .staticmethod("build_next_fourier") 
        .def("build_next_cube", &CubicGrid::build_next_cube)
        .staticmethod("build_next_cube") 
        .def("save_cube_file", &CubicGrid::save_cube_file)
        .add_property("size", &CubicGrid::size)
        .add_property("origin", make_function(&CubicGrid::origin, return_internal_reference<>()))
        .add_property("spacing", make_function(&CubicGrid::spacing, return_internal_reference<>()))
        .add_property("sizing", make_function(&CubicGrid::sizing, return_internal_reference<>()))
        .add_property("xyz", &CubicGrid::xyz)
        .add_property("w", &CubicGrid::w)
        .def("__str__", &CubicGrid::string)
        ;
}
