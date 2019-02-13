#ifndef LS_GPU_AISCAT_HPP
#define LS_GPU_AISCAT_HPP
    
#include <memory>
    
namespace lightspeed {

class ResourceList;
class Tensor;

// Work in float, accumulate in float
std::shared_ptr<Tensor> perpendicular_moments_gpu_f_f(
    const std::shared_ptr<ResourceList>& resources,
    double L,
    const std::shared_ptr<Tensor>& s,
    const std::shared_ptr<Tensor>& xyzq
    );
    
// Work in float, accumulate in double
std::shared_ptr<Tensor> perpendicular_moments_gpu_f_d(
    const std::shared_ptr<ResourceList>& resources,
    double L,
    const std::shared_ptr<Tensor>& s,
    const std::shared_ptr<Tensor>& xyzq
    );
    
// Work in double, accumulate in double
std::shared_ptr<Tensor> perpendicular_moments_gpu_d_d(
    const std::shared_ptr<ResourceList>& resources,
    double L,
    const std::shared_ptr<Tensor>& s,
    const std::shared_ptr<Tensor>& xyzq
    );

// Work in float, accumulate in float
std::shared_ptr<Tensor> aligned_diffraction_gpu_f_f(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& sxyz,
    const std::shared_ptr<Tensor>& xyzq
    );
    
// Work in float, accumulate in double
std::shared_ptr<Tensor> aligned_diffraction_gpu_f_d(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& sxyz,
    const std::shared_ptr<Tensor>& xyzq
    );
    
// Work in double, accumulate in double
std::shared_ptr<Tensor> aligned_diffraction_gpu_d_d(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& sxyz,
    const std::shared_ptr<Tensor>& xyzq
    );
    
} // namespace lightspeed
    
#endif 
