#include <boost/python.hpp>
#include <boost/python/overloads.hpp>
#include <memory>
#include <lightspeed/molecule.hpp>
#include <lightspeed/am.hpp>
#include <lightspeed/basis.hpp>
#include <lightspeed/ecp.hpp>
#include <lightspeed/pair_list.hpp>
#include <lightspeed/ewald.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/pure_transform.hpp>
#include <lightspeed/rys.hpp>
#include <lightspeed/boys.hpp>
#include <lightspeed/gh.hpp>
#include <lightspeed/molden.hpp>
#include <lightspeed/local.hpp>

using namespace lightspeed;
using namespace boost::python;

BOOST_PYTHON_FUNCTION_OVERLOADS(pureTransform_allocCart2_ov,PureTransform::allocCart2,2,3)
BOOST_PYTHON_FUNCTION_OVERLOADS(pureTransform_cartToPure2_ov,PureTransform::cartToPure2,3,4)

std::string Molecule__str__(
    const Molecule& self) {
    return self.string();
}

void export_core()
{
    class_<Atom, std::shared_ptr<Atom> >("Atom", init<
        const std::string&,
        const std::string&,
        int,
        double,
        double,
        double,
        double,
        size_t
        >())
        .add_property("label", &Atom::label)
        .add_property("symbol", &Atom::symbol)
        .add_property("N", &Atom::N)
        .add_property("x", &Atom::x)
        .add_property("y", &Atom::y)
        .add_property("z", &Atom::z)
        .add_property("Z", &Atom::Z)
        .add_property("atomIdx", &Atom::atomIdx)
        .def("distance", &Atom::distance)
        ;

    class_<Molecule, std::shared_ptr<Molecule> >("Molecule", init<
        const std::string&,
        const std::vector<Atom>&,
        double,
        double
        >())
        .add_property("name", &Molecule::name, &Molecule::set_name)
        .add_property("atoms", make_function(&Molecule::atoms, return_internal_reference<>()))
        .add_property("natom", &Molecule::natom)
        .add_property("charge", &Molecule::charge, &Molecule::set_charge)
        .add_property("multiplicity", &Molecule::multiplicity, &Molecule::set_multiplicity)
        .def("string", &Molecule::string)
        .def("__str__", Molecule__str__)
        .add_property("nuclear_charge", &Molecule::nuclear_charge)
        .add_property("nuclear_COM", &Molecule::nuclear_COM)
        .add_property("nuclear_I", &Molecule::nuclear_I)
        .def("nuclear_repulsion_energy", &Molecule::nuclear_repulsion_energy)
        .def("nuclear_repulsion_energy_other", &Molecule::nuclear_repulsion_energy_other)
        .def("nuclear_repulsion_grad", &Molecule::nuclear_repulsion_grad)
        .def("subset", &Molecule::subset)
        .def("concatenate", &Molecule::concatenate)
        .staticmethod("concatenate")
        .add_property("xyz", &Molecule::xyz)
        .add_property("Z", &Molecule::Z)
        .add_property("xyzZ", &Molecule::xyzZ)
        .def("update_xyz", &Molecule::update_xyz)
        .def("update_Z", &Molecule::update_Z)
        .def("equivalent", &Molecule::equivalent)
        .staticmethod("equivalent")
        ;

    class_<AngularMomentum, std::shared_ptr<AngularMomentum> >("AngularMomentum", init<int>())
        .def("build", &AngularMomentum::build)
        .staticmethod("build")
        .add_property("L", &AngularMomentum::L)
        .add_property("ncart", &AngularMomentum::ncart)
        .add_property("npure", &AngularMomentum::npure)
        .add_property("ls", make_function(&AngularMomentum::ls, return_internal_reference<>()))
        .add_property("ms", make_function(&AngularMomentum::ms, return_internal_reference<>()))
        .add_property("ns", make_function(&AngularMomentum::ns, return_internal_reference<>()))
        .add_property("ncoef", &AngularMomentum::ncoef)
        .add_property("cart_inds",  make_function(&AngularMomentum::cart_inds, return_internal_reference<>()))
        .add_property("pure_inds",  make_function(&AngularMomentum::pure_inds, return_internal_reference<>()))
        .add_property("cart_coefs", make_function(&AngularMomentum::cart_coefs,return_internal_reference<>()))
        ;

    class_<Primitive>("Primitive", init<
        double,
        double,
        double,
        double,
        double,
        double,
        int,
        bool,
        int,
        int,
        int,
        int,
        int>())
        .add_property("c", &Primitive::c)
        .add_property("e", &Primitive::e)
        .add_property("x", &Primitive::x)
        .add_property("y", &Primitive::y)
        .add_property("z", &Primitive::z)
        .add_property("c0", &Primitive::c0)
        .add_property("L", &Primitive::L)
        .add_property("is_pure", &Primitive::is_pure)
        .add_property("aoIdx", &Primitive::aoIdx)
        .add_property("cartIdx", &Primitive::cartIdx)
        .add_property("primIdx", &Primitive::primIdx)
        .add_property("shellIdx", &Primitive::shellIdx)
        .add_property("atomIdx", &Primitive::atomIdx)
        .add_property("nao", &Primitive::nao)
        .add_property("npure", &Primitive::npure)
        .add_property("ncart", &Primitive::ncart)
        .def("__str__", &Primitive::string)
        ;

    class_<Shell>("Shell", init<
        const std::vector<double>&,
        const std::vector<double>&,
        double,
        double,
        double,
        const std::vector<double>&,
        int,
        bool,
        int,
        int,
        int,
        int,
        int>())
        .add_property("cs", make_function(&Shell::cs, return_internal_reference<>()))
        .add_property("es", make_function(&Shell::es, return_internal_reference<>()))
        .add_property("x", &Shell::x)
        .add_property("y", &Shell::y)
        .add_property("z", &Shell::z)
        .add_property("c0s", make_function(&Shell::c0s, return_internal_reference<>()))
        .add_property("L", &Shell::L)
        .add_property("is_pure", &Shell::is_pure)
        .add_property("aoIdx", &Shell::aoIdx)
        .add_property("cartIdx", &Shell::cartIdx)
        .add_property("primIdx", &Shell::primIdx)
        .add_property("shellIdx", &Shell::shellIdx)
        .add_property("atomIdx", &Shell::atomIdx)
        .add_property("nao", &Shell::nao)
        .add_property("npure", &Shell::npure)
        .add_property("ncart", &Shell::ncart)
        .def("__str__", &Shell::string)
        ;

    class_<Basis, std::shared_ptr<Basis> >("Basis", init<
        const std::string&,
        const std::vector<Primitive>&
        >())
        .add_property("name", &Basis::name)
        .add_property("primitives", make_function(&Basis::primitives, return_internal_reference<>()))
        .add_property("nao", &Basis::nao)
        .add_property("ncart", &Basis::ncart)
        .add_property("nprim", &Basis::nprim)
        .add_property("nshell", &Basis::nshell)
        .add_property("natom", &Basis::natom)
        .add_property("max_L", &Basis::max_L)
        .add_property("max_nao", &Basis::max_nao)
        .add_property("max_npure", &Basis::max_npure)
        .add_property("max_ncart", &Basis::max_ncart)
        .add_property("has_pure", &Basis::has_pure)
        .def("__str__", &Basis::string)
        .def("subset", &Basis::subset)
        .def("concatenate", &Basis::concatenate)
        .staticmethod("concatenate")
        .add_property("xyz", &Basis::xyz)
        .def("update_xyz", &Basis::update_xyz)
        .def("equivalent", &Basis::equivalent)
        .staticmethod("equivalent")
        .add_property("shells", make_function(&Basis::shells, return_internal_reference<>()))
        ;
         
    class_<ECPShell, std::shared_ptr<ECPShell>>("ECPShell", init<
        double,
        double,
        double,
        int,
        bool,
        const std::vector<int>&,
        const std::vector<double>&,
        const std::vector<double>&,
        size_t,
        size_t
        >()) 
        .add_property("x", &ECPShell::x)
        .add_property("y", &ECPShell::y)
        .add_property("z", &ECPShell::z)
        .add_property("L", &ECPShell::L)
        .add_property("is_max_L", &ECPShell::is_max_L)
        .add_property("nprim", &ECPShell::nprim)
        .add_property("ns", make_function(&ECPShell::ns, return_internal_reference<>()))
        .add_property("cs", make_function(&ECPShell::cs, return_internal_reference<>()))
        .add_property("es", make_function(&ECPShell::es, return_internal_reference<>()))
        .add_property("atomIdx", &ECPShell::atomIdx)
        .add_property("shellIdx", &ECPShell::shellIdx)
        ;

    class_<ECPBasis, std::shared_ptr<ECPBasis>>("ECPBasis", init<
        const std::string&,
        const std::vector<ECPShell>&,
        const std::vector<int>& 
        >())
        .add_property("name", &ECPBasis::name)
        .add_property("shells", make_function(&ECPBasis::shells, return_internal_reference<>()))
        .add_property("atoms_to_shell_inds", make_function(&ECPBasis::atoms_to_shell_inds, return_internal_reference<>()))
        .add_property("nelecs", make_function(&ECPBasis::nelecs, return_internal_reference<>()))
        .add_property("natom", &ECPBasis::natom)
        .add_property("nshell", &ECPBasis::nshell)
        .add_property("nprim", &ECPBasis::nprim)
        .add_property("nelec", &ECPBasis::nelec)
        .add_property("max_L", &ECPBasis::max_L)
        .add_property("max_nprim", &ECPBasis::max_nprim)
        .def("__str__", &ECPBasis::string)
        .def("subset", &ECPBasis::subset)
        .def("concatenate", &ECPBasis::concatenate)
        .staticmethod("concatenate")
        .add_property("xyz", &ECPBasis::xyz)
        .def("update_xyz", &ECPBasis::update_xyz)
        .def("equivalent", &ECPBasis::equivalent)
        .staticmethod("equivalent")
        ;
    
    class_<Pair>("Pair", no_init)
        .add_property("prim1", make_function(&Pair::prim1, return_internal_reference<>()))
        .add_property("prim2", make_function(&Pair::prim2, return_internal_reference<>()))
        .add_property("bound", &Pair::bound)
        .def("__str__", &Pair::string)
        ;

    class_<PairListL>("PairListL", no_init)
        .add_property("is_symmetric", &PairListL::is_symmetric)
        .add_property("L1", &PairListL::L1)
        .add_property("L2", &PairListL::L2)
        .add_property("pairs", make_function(&PairListL::pairs, return_internal_reference<>()))
        .def("__str__", &PairListL::string)
        ;

    class_<PairList, std::shared_ptr<PairList> >("PairList", no_init)
        .def("build_schwarz", &PairList::build_schwarz)
        .staticmethod("build_schwarz")
        .add_property("basis1", &PairList::basis1)
        .add_property("basis2", &PairList::basis2)
        .add_property("is_symmetric", &PairList::is_symmetric)
        .add_property("thre", &PairList::thre)
        .add_property("pairlists", make_function(&PairList::pairlists, return_internal_reference<>()))
        .def("__str__", &PairList::string)
        ;

    class_<Ewald, std::shared_ptr<Ewald> >("Ewald", init<
        const std::vector<double>&,
        const std::vector<double>&
        >())
        .def("coulomb", &Ewald::coulomb).staticmethod("coulomb")
        .add_property("scales", make_function(&Ewald::scales, return_internal_reference<>()))
        .add_property("omegas", make_function(&Ewald::omegas, return_internal_reference<>()))
        .add_property("is_coulomb", &Ewald::is_coulomb)
        .add_property("is_sr", &Ewald::is_sr)
        .add_property("sr_scale", &Ewald::sr_scale)
        .add_property("sr_omega", &Ewald::sr_omega)
        .add_property("is_lr", &Ewald::is_lr)
        .add_property("lr_scale", &Ewald::lr_scale)
        .add_property("lr_omega", &Ewald::lr_omega)
        ;

    class_<PureTransform, std::shared_ptr<PureTransform> >("PureTransform", no_init)
        .def("allocCart2", &PureTransform::allocCart2, pureTransform_allocCart2_ov())
        .staticmethod("allocCart2")
        .def("cartToPure2", &PureTransform::cartToPure2, pureTransform_cartToPure2_ov())
        .staticmethod("cartToPure2")
        .def("pureToCart2", &PureTransform::pureToCart2)
        .staticmethod("pureToCart2")
        .def("pureToCart1", &PureTransform::pureToCart1)
        .staticmethod("pureToCart1")
        ;

    class_<Rys>("Rys", no_init)
        .def("compute_t", &Rys::compute_t)
        .staticmethod("compute_t")
        .def("compute_w", &Rys::compute_w)
        .staticmethod("compute_w")
        ;

    class_<Boys>("Boys", no_init)
        .def("compute", &Boys::compute)
        .staticmethod("compute")
        ;

    class_<GH>("GH", no_init)
        .def("compute", &GH::compute)
        .staticmethod("compute")
        ;

    class_<Molden>("Molden", no_init)
        .def("save_molden_file", &Molden::save_molden_file)
        .staticmethod("save_molden_file")
        ;

    class_<Local>("Local", no_init)
        .def("localize", &Local::localize)
        .staticmethod("localize")
        ;
    
}
