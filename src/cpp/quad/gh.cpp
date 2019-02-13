#include <lightspeed/gh.hpp>
#include <lightspeed/tensor.hpp>
#include <stdexcept>

namespace lightspeed {
    
std::shared_ptr<Tensor> GH::compute(int n)
{
    if (n < 1 || n > 20) throw std::runtime_error("GH::compute: Invalid number of points.");
    
    std::vector<size_t> dim;
    dim.push_back(n);   
    dim.push_back(2);
    std::shared_ptr<Tensor> ret(new Tensor(dim));
    double* retp = ret->data().data();

    std::shared_ptr<GH> gh = GH::instance();
    const std::vector<double>& tvals = gh->ts()[n];
    const std::vector<double>& wvals = gh->ws()[n];

    for (size_t P = 0; P < n; P++) {
        retp[2*P + 0] = tvals[P];
        retp[2*P + 1] = wvals[P];
    }

    return ret;
}

} // namespace lightspeed 
