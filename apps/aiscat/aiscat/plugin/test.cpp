#include "cpu_aiscat.hpp"
#include "gpu_aiscat.hpp"
#include <lightspeed/tensor.hpp>
#include <lightspeed/resource_list.hpp>
#include <boost/python.hpp>

using namespace lightspeed;
using namespace boost::python;

BOOST_PYTHON_MODULE(pyplugin)
{
    def("isotropic_moments", isotropic_moments);
    def("parallel_moments", parallel_moments);
    def("perpendicular_moments", perpendicular_moments);
    def("aligned_diffraction", aligned_diffraction);
    def("aligned_diffraction2", aligned_diffraction2);
    def("perpendicular_moments_gpu_f_f", perpendicular_moments_gpu_f_f);
    def("perpendicular_moments_gpu_f_d", perpendicular_moments_gpu_f_d);
    def("perpendicular_moments_gpu_d_d", perpendicular_moments_gpu_d_d);
    def("aligned_diffraction_gpu_f_f", aligned_diffraction_gpu_f_f);
    def("aligned_diffraction_gpu_f_d", aligned_diffraction_gpu_f_d);
    def("aligned_diffraction_gpu_d_d", aligned_diffraction_gpu_d_d);
}


    
