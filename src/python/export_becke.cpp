#include <boost/python.hpp>
#include <boost/python/overloads.hpp>
#include <memory>
#include <lightspeed/becke.hpp>
#include <lightspeed/tensor.hpp>

using namespace lightspeed;
using namespace boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(becke_grad_ov, BeckeGrid::grad, 2, 3)

void export_becke()
{
    class_<LebedevGrid, std::shared_ptr<LebedevGrid> >("LebedevGrid", no_init)
        .def("build", &LebedevGrid::build)
        .staticmethod("build")
        .def("sizes", &LebedevGrid::sizes)
        .staticmethod("sizes")
        .add_property("size", &LebedevGrid::size)
        .add_property("L", &LebedevGrid::L)
        .add_property("is_positive", &LebedevGrid::size)
        .add_property("xyzw", &LebedevGrid::xyzw)
        .add_property("rtpw", &LebedevGrid::rtpw)
        .def("__str__", &LebedevGrid::string)
        ;

    class_<RadialGrid, std::shared_ptr<RadialGrid> >("RadialGrid", no_init)
        .def("build", &RadialGrid::build)
        .staticmethod("build")
        .def("build_by_N", &RadialGrid::build_by_N)
        .staticmethod("build_by_N")
        .def("build_explicit", &RadialGrid::build_explicit)
        .staticmethod("build_explicit")
        .add_property("name", &RadialGrid::name)
        .add_property("R", &RadialGrid::R)
        .add_property("size", &RadialGrid::size)
        .add_property("rw", &RadialGrid::rw)
        .def("__str__", &RadialGrid::string)
        ;

    class_<AtomGrid, std::shared_ptr<AtomGrid> >("AtomGrid", init<
        size_t,
        double,
        double,
        double,
        const std::shared_ptr<Tensor>&,
        const std::shared_ptr<RadialGrid>&,
        const std::vector<std::shared_ptr<LebedevGrid> >&
        >())
        .add_property("N", &AtomGrid::N)
        .add_property("x", &AtomGrid::x)
        .add_property("y", &AtomGrid::y)
        .add_property("z", &AtomGrid::z)
        .add_property("orientation", &AtomGrid::orientation)
        .add_property("radial", &AtomGrid::radial)
        .add_property("spherical", make_function(&AtomGrid::spherical, return_internal_reference<>()))
        .add_property("size", &AtomGrid::size)
        .add_property("radial_size", &AtomGrid::radial_size)
        .add_property("max_spherical_size", &AtomGrid::max_spherical_size)
        .add_property("is_pruned", &AtomGrid::is_pruned)
        .add_property("spherical_sizes", make_function(&AtomGrid::spherical_sizes, return_internal_reference<>()))
        .add_property("spherical_starts", make_function(&AtomGrid::spherical_starts, return_internal_reference<>()))
        .def("atomic_index", &AtomGrid::atomic_index)
        .def("radial_index", &AtomGrid::radial_index)
        .def("spherical_index", &AtomGrid::spherical_index)
        .def("__str__", &AtomGrid::string)
        ;

    class_<BeckeGrid, std::shared_ptr<BeckeGrid> >("BeckeGrid", init<
        const std::shared_ptr<ResourceList>&,
        const std::string&,
        const std::string&,
        const std::vector<std::shared_ptr<AtomGrid> >&
        >())
        .add_property("name", &BeckeGrid::name)
        .add_property("atomic_scheme", &BeckeGrid::atomic_scheme)
        .add_property("atomic", make_function(&BeckeGrid::atomic, return_internal_reference<>()))
        .add_property("size", &BeckeGrid::size)
        .add_property("natom", &BeckeGrid::natom)
        .add_property("atomic_sizes", make_function(&BeckeGrid::atomic_sizes, return_internal_reference<>()))
        .add_property("atomic_starts", make_function(&BeckeGrid::atomic_starts, return_internal_reference<>()))
        .add_property("radial_sizes", make_function(&BeckeGrid::radial_sizes, return_internal_reference<>()))
        .add_property("spherical_sizes", make_function(&BeckeGrid::spherical_sizes, return_internal_reference<>()))
        .def("total_index", &BeckeGrid::total_index)
        .def("atomic_index", &BeckeGrid::atomic_index)
        .def("radial_index", &BeckeGrid::radial_index)
        .def("spherical_index", &BeckeGrid::spherical_index)
        .add_property("atomic_inds", make_function(&BeckeGrid::atomic_inds, return_internal_reference<>()))
        .add_property("max_spherical_size", &BeckeGrid::max_spherical_size)
        .add_property("max_radial_size", &BeckeGrid::max_radial_size)
        .add_property("max_atomic_size", &BeckeGrid::max_atomic_size)
        .add_property("is_pruned", &BeckeGrid::is_pruned)
        .add_property("xyzw", &BeckeGrid::xyzw)
        .add_property("xyz", &BeckeGrid::xyz)
        .add_property("xyzw_raw", &BeckeGrid::xyzw_raw)
        .def("grad", &BeckeGrid::grad, becke_grad_ov())
        .def("__str__", &BeckeGrid::string)
        ;


}
