#include <boost/python.hpp>
#include <boost/python/overloads.hpp>
#include <memory>
#include <lightspeed/blurbox.hpp>
#include <lightspeed/gridbox.hpp>
#include <lightspeed/pair_list.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/resource_list.hpp>

using namespace lightspeed;
using namespace boost::python;

BOOST_PYTHON_FUNCTION_OVERLOADS(blurbox_point_density_ov,BlurBox::pointDensity, 6,7)
BOOST_PYTHON_FUNCTION_OVERLOADS(blurbox_point_potential_ov,BlurBox::pointPotential, 8,9)
BOOST_PYTHON_FUNCTION_OVERLOADS(blurbox_point_grad_ov,BlurBox::pointGrad, 8,9)
BOOST_PYTHON_FUNCTION_OVERLOADS(blurbox_lda_density_ov,BlurBox::ldaDensity, 7,8)
BOOST_PYTHON_FUNCTION_OVERLOADS(blurbox_lda_potential_ov,BlurBox::ldaPotential, 8,9)
BOOST_PYTHON_FUNCTION_OVERLOADS(blurbox_lda_grad_ov,BlurBox::ldaGrad, 9, 10)
BOOST_PYTHON_FUNCTION_OVERLOADS(blurbox_lda_density2_ov,BlurBox::ldaDensity2, 7,8)
BOOST_PYTHON_FUNCTION_OVERLOADS(blurbox_lda_potential2_ov,BlurBox::ldaPotential2, 8,9)
BOOST_PYTHON_FUNCTION_OVERLOADS(blurbox_lda_grad2_ov,BlurBox::ldaGrad2, 9, 10)
BOOST_PYTHON_FUNCTION_OVERLOADS(blurbox_potential_sr3_ov,BlurBox::potentialSR3, 5, 6)
BOOST_PYTHON_FUNCTION_OVERLOADS(blurbox_potential_grad_sr3_ov,BlurBox::potentialGradSR3, 6,7)
BOOST_PYTHON_FUNCTION_OVERLOADS(blurbox_esp_sr3_ov,BlurBox::espSR3, 6,7)
BOOST_PYTHON_FUNCTION_OVERLOADS(blurbox_esp_grad_sr3_ov,BlurBox::espGradSR3, 6,7)
BOOST_PYTHON_FUNCTION_OVERLOADS(blurbox_coulomb_sr3_ov,BlurBox::coulombSR3, 6,7)
BOOST_PYTHON_FUNCTION_OVERLOADS(blurbox_coulomb_grad_sr3_ov,BlurBox::coulombGradSR3, 7, 8)

void export_blurbox()
{
    class_<BlurBox>("BlurBox", no_init)
        .def("pointDensity", &BlurBox::pointDensity, blurbox_point_density_ov())
        .staticmethod("pointDensity")
        .def("pointPotential", &BlurBox::pointPotential, blurbox_point_potential_ov())
        .staticmethod("pointPotential")
        .def("pointGrad", &BlurBox::pointGrad, blurbox_point_grad_ov())
        .staticmethod("pointGrad")
        .def("ldaDensity", &BlurBox::ldaDensity, blurbox_lda_density_ov())
        .staticmethod("ldaDensity")
        .def("ldaPotential", &BlurBox::ldaPotential, blurbox_lda_potential_ov())
        .staticmethod("ldaPotential")
        .def("ldaGrad", &BlurBox::ldaGrad, blurbox_lda_grad_ov())
        .staticmethod("ldaGrad")
        .def("ldaDensity2", &BlurBox::ldaDensity2, blurbox_lda_density2_ov())
        .staticmethod("ldaDensity2")
        .def("ldaPotential2", &BlurBox::ldaPotential2, blurbox_lda_potential2_ov())
        .staticmethod("ldaPotential2")
        .def("ldaGrad2", &BlurBox::ldaGrad2, blurbox_lda_grad2_ov())
        .staticmethod("ldaGrad2")
        .def("potentialSR3", &BlurBox::potentialSR3, blurbox_potential_sr3_ov())
        .staticmethod("potentialSR3")
        .def("potentialGradSR3", &BlurBox::potentialGradSR3, blurbox_potential_grad_sr3_ov())
        .staticmethod("potentialGradSR3")
        .def("espSR3", &BlurBox::espSR3, blurbox_esp_sr3_ov())
        .staticmethod("espSR3")
        .def("espGradSR3", &BlurBox::espGradSR3, blurbox_esp_grad_sr3_ov())
        .staticmethod("espGradSR3")
        .def("coulombSR3", &BlurBox::coulombSR3, blurbox_coulomb_sr3_ov())
        .staticmethod("coulombSR3")
        .def("coulombGradSR3", &BlurBox::coulombGradSR3, blurbox_coulomb_grad_sr3_ov())
        .staticmethod("coulombGradSR3")
        ;
}
