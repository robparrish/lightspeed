#ifndef LS_CPU_AISCAT_HPP
#define LS_CPU_AISCAT_HPP

#include <memory>

namespace lightspeed {

class Tensor;
class ResourceList;

// => Moment Specialization <= //

std::shared_ptr<Tensor> isotropic_moments(
    const std::shared_ptr<ResourceList>& resources,
    double L,
    const std::shared_ptr<Tensor> s,
    const std::shared_ptr<Tensor> xyzq
    );
    
std::shared_ptr<Tensor> parallel_moments(
    const std::shared_ptr<ResourceList>& resources,
    double L,
    const std::shared_ptr<Tensor> s,
    const std::shared_ptr<Tensor> xyzq
    );

std::shared_ptr<Tensor> perpendicular_moments(
    const std::shared_ptr<ResourceList>& resources,
    double L,
    const std::shared_ptr<Tensor> s,
    const std::shared_ptr<Tensor> xyzq
    );

std::shared_ptr<Tensor> aligned_diffraction(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor> sxyz,
    const std::shared_ptr<Tensor> xyzq
    );

std::shared_ptr<Tensor> aligned_diffraction2(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor> sxyz,
    const std::shared_ptr<Tensor> xyzq
    );


} // namespace lightspeed

#endif
