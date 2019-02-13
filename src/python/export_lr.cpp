#include <boost/python.hpp>
#include <memory>
#include <lightspeed/lr.hpp>
#include <lightspeed/tensor.hpp>

using namespace lightspeed;
using namespace boost::python;

void export_lr()
{
    class_<LaplaceDenom>("LaplaceDenom", no_init)
        .def("o2v2_denom", &LaplaceDenom::o2v2_denom)
        .staticmethod("o2v2_denom")
        .def("o3v3_denom", &LaplaceDenom::o3v3_denom)
        .staticmethod("o3v3_denom")
        .def("onvn_denom", &LaplaceDenom::onvn_denom)
        .staticmethod("onvn_denom")
        ;

    class_<DF>("DF", no_init)
        .def("metric", &DF::metric)
        .staticmethod("metric")
        .def("ao_df", &DF::ao_df)
        .staticmethod("ao_df")
        .def("mo_df", &DF::mo_df)
        .staticmethod("mo_df")
        ;

    class_<THC>("THC", no_init)
        .def("mo_thc_X", &THC::mo_thc_X)
        .staticmethod("mo_thc_X")
        .def("mo_thc_V", &THC::mo_thc_V)
        .staticmethod("mo_thc_V")
        ;
}
