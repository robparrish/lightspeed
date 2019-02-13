#include <boost/python.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <memory>
#include <lightspeed/gpu_context.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/solver.hpp>
#include <lightspeed/molecule.hpp>
#include <lightspeed/am.hpp>
#include <lightspeed/basis.hpp>
#include <lightspeed/ecp.hpp>
#include <lightspeed/pair_list.hpp>
#include <lightspeed/becke.hpp>

using namespace lightspeed;
using namespace boost::python;

/** @brief Type that allows for registration of conversions from
 *         Python iterable types.
 */
struct iterable_converter
{
    /** @note Registers converter from a Python iterable type to the
     *  provided type.
     */
    template<typename Container>
    iterable_converter&
    from_python()
    {
        boost::python::converter::registry::push_back(&iterable_converter::convertible,
                                                      &iterable_converter::construct<Container>,
                                                      boost::python::type_id<Container>());

        // support chaining
        return *this;
    }

    /// @brief Check if PyObject is iterable
    static void* convertible(PyObject* object)
    {
        return PyObject_GetIter(object) ? object : NULL;
    }

    /** @brief Convert iterable PyObject to C++ container type.
     *
     * Container concept requirements:
     *
     *   * Container::value_type is CopyConstructable.
     *   * Container can be constructed and populated with two iterators.
     *     i.e. Container(begin, end)
     */
    template<typename Container>
    static void construct(PyObject* object,
                          boost::python::converter::rvalue_from_python_stage1_data* data)
    {
        namespace python = boost::python;

        // Object is borrowed reference, so create a handle indictating it is
        // borrowed for proper reference counting
        python::handle<> handle(python::borrowed(object));

        // Obtain a handle to the memory block that the converter has allocated
        // for the C++ type.
        typedef python::converter::rvalue_from_python_storage<Container> storage_type;

        void* storage = reinterpret_cast<storage_type*>(data)->storage.bytes;

        typedef python::stl_input_iterator<typename Container::value_type> iterator;

        // Allocate the C++ type into the converter's memory block, and assign
        // its handle to the converter's convertible variable. The C++
        // container is populated by passing the begin and end iterators of
        // the python object to the container's constructor.
        new (storage) Container(iterator(python::object(handle)), // begin
                                iterator());                      // end
        data->convertible = storage;
    }
};

void export_collections()
{
    // => Iterable Conversions <= //
    
    iterable_converter()
        .from_python<std::vector<bool> >()
        .from_python<std::vector<int> >()
        .from_python<std::vector<size_t> >()
        .from_python<std::vector<double> >()
        .from_python<std::vector<std::string> >()
        .from_python<std::vector<std::shared_ptr<GPUContext> > >()
        .from_python<std::vector<std::shared_ptr<Tensor> > >()
        .from_python<std::vector<std::vector<std::shared_ptr<Tensor>> > >()
        .from_python<std::vector<std::shared_ptr<Storage> > >()
        .from_python<std::vector<Atom> >()
        .from_python<std::vector<Primitive> >()
        .from_python<std::vector<Shell> >()
        .from_python<std::vector<ECPShell> >()
        .from_python<std::vector<std::shared_ptr<Molecule> > >()
        .from_python<std::vector<std::shared_ptr<Basis> > >()
        .from_python<std::vector<std::shared_ptr<LebedevGrid> > >()
        .from_python<std::vector<std::shared_ptr<AtomGrid> > >()
        ;

    // => Standard Collections <= //

    class_<std::vector<bool> >("BoolVec")
        .def(vector_indexing_suite<std::vector<bool> >())
        ;

    class_<std::vector<int> >("IntVec")
        .def(vector_indexing_suite<std::vector<int> >())
        ;

    class_<std::vector<size_t> >("Size_tVec")
        .def(vector_indexing_suite<std::vector<size_t> >())
        ;

    class_<std::vector<double> >("DoubleVec")
        .def(vector_indexing_suite<std::vector<double> >())
        ;

    // => Env Collections <= //

    class_<std::vector<std::shared_ptr<GPUContext> > >("GPUContextVec")
        .def(vector_indexing_suite<std::vector<std::shared_ptr<GPUContext> >, true>()) // no_proxy flag is needed!
        ;

    // => Tensor Collections <= //

    class_<std::vector<std::shared_ptr<Tensor> > >("TensorVec")
        .def(vector_indexing_suite<std::vector<std::shared_ptr<Tensor> >, true>()) // no_proxy flag is needed!
        ;

    class_<std::vector<std::vector<std::shared_ptr<Tensor>> > >("TensorVecVec")
        .def(vector_indexing_suite<std::vector<std::vector<std::shared_ptr<Tensor>> >, true>()) // no_proxy flag is needed!
        ;
      
    // => Storage Collections <= //

    class_<std::vector<std::shared_ptr<Storage> > >("StorageVec")
        .def(vector_indexing_suite<std::vector<std::shared_ptr<Storage> >, true>()) // no_proxy flag is needed!
        ;

    // => Core Collections <= //

    class_<std::vector<Atom> >("AtomVec")
        .def(vector_indexing_suite<std::vector<Atom> >()) 
        ;

    class_<std::vector<AngularMomentum> >("AngularMomentumVec")
        .def(vector_indexing_suite<std::vector<AngularMomentum> >()) 
        ;

    class_<std::vector<Primitive> >("PrimitiveVec")
        .def(vector_indexing_suite<std::vector<Primitive> >()) 
        ;

    class_<std::vector<Shell> >("ShellVec")
        .def(vector_indexing_suite<std::vector<Shell> >()) 
        ;

    class_<std::vector<ECPShell> >("ECPShellVec")
        .def(vector_indexing_suite<std::vector<ECPShell> >()) 
        ;

    class_<std::vector<Pair> >("PairVec")
        .def(vector_indexing_suite<std::vector<Pair> >()) 
        ;

    class_<std::vector<PairListL> >("PairListLVec")
        .def(vector_indexing_suite<std::vector<PairListL> >()) 
        ;

    // => Becke Collections <= //

    class_<std::vector<std::shared_ptr<LebedevGrid> > >("LebedevGridVec")
        .def(vector_indexing_suite<std::vector<std::shared_ptr<LebedevGrid> >, true>()) // no_proxy flag is needed!
        ;

    class_<std::vector<std::shared_ptr<AtomGrid> > >("AtomGridVec")
        .def(vector_indexing_suite<std::vector<std::shared_ptr<AtomGrid> >, true>()) // no_proxy flag is needed!
        ;


}
