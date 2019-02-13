#include <boost/python.hpp>
#include <boost/python/overloads.hpp>
#include <lightspeed/tensor.hpp>

using namespace lightspeed;
using namespace boost::python;

void print_tensor(lightspeed::shared_ptr<Tensor>& T)
{
    T->print();
}

BOOST_PYTHON_MODULE(pyplugin)
{
    def("print_tensor", &print_tensor);
}

