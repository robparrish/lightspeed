#include <lightspeed/becke.hpp>
#include <lightspeed/tensor.hpp>
#include "../util/string.hpp"

namespace lightspeed {

std::shared_ptr<Tensor> AtomGrid::xyzw() const 
{
    std::vector<size_t> dim;
    dim.push_back(size());
    dim.push_back(4);
    std::shared_ptr<Tensor> ret(new Tensor(dim));
    double* retp = ret->data().data();
    std::shared_ptr<Tensor> rwr = radial_->rw();
    const double* rwrp = rwr->data().data();
    const double* Up = orientation_->data().data();
    size_t rsize = radial_->size();
    for (size_t rind = 0L, aind = 0L; rind < rsize; rind++) {
        size_t ssize = spherical_[rind]->size();
        std::shared_ptr<Tensor> xyzws = spherical_[rind]->xyzw();
        const double* xyzwsp = xyzws->data().data();
        for (size_t sind = 0L; sind < ssize; sind++, aind++) {
            double x0 = xyzwsp[4*sind + 0] * rwrp[2*rind + 0];
            double y0 = xyzwsp[4*sind + 1] * rwrp[2*rind + 0];
            double z0 = xyzwsp[4*sind + 2] * rwrp[2*rind + 0];
            double w0 = xyzwsp[4*sind + 3] * rwrp[2*rind + 1];
            retp[4*aind + 0] = Up[0*3 + 0] * x0 + Up[1*3 + 0] * y0 + Up[2*3 + 0]* z0 + x_; 
            retp[4*aind + 1] = Up[0*3 + 1] * x0 + Up[1*3 + 1] * y0 + Up[2*3 + 1]* z0 + y_; 
            retp[4*aind + 2] = Up[0*3 + 2] * x0 + Up[1*3 + 2] * y0 + Up[2*3 + 2]* z0 + z_; 
            retp[4*aind + 3] = w0;
        }
    }
    return ret;
}
size_t AtomGrid::max_spherical_size() const 
{
    size_t maxv = 0L;
    for (size_t rind = 0L; rind < spherical_.size(); rind++) {
        maxv = std::max(spherical_[rind]->size(),maxv);
    }
    return maxv;
}
bool AtomGrid::is_pruned() const 
{
    size_t maxv = max_spherical_size();
    bool pruned = false;
    for (size_t rind = 0L; rind < spherical_.size(); rind++) {
        if (spherical_[rind]->size() != maxv) {
            pruned = true;
            break;
        }
    }
    return pruned;
}
size_t AtomGrid::atomic_index(
    size_t radial_index,
    size_t spherical_index) const 
{
    return spherical_starts_[radial_index] + spherical_index;
}      
size_t AtomGrid::radial_index(
    size_t atomic_index) const 
{
    for (size_t ind = 0L; ind < spherical_starts_.size(); ind++) {
        if (spherical_starts_[ind] + spherical_sizes_[ind] > atomic_index)    
            return ind;
    }    
    throw std::runtime_error("AtomGrid: atomic index is too large");
}
size_t AtomGrid::spherical_index(
    size_t atomic_index) const
{
    return atomic_index - spherical_starts_[radial_index(atomic_index)];
}
std::string AtomGrid::string() const
{
    std::string str = "";
    str += sprintf2("AtomGrid:\n");
    str += sprintf2("  N                  = %11d\n", N_);
    str += sprintf2("  xc                 = %11.6f\n", x_);
    str += sprintf2("  yc                 = %11.6f\n", y_);
    str += sprintf2("  zc                 = %11.6f\n", z_);
    str += sprintf2("  Atom Size          = %11zu\n", size());
    str += sprintf2("  Radial Size        = %11zu\n", radial_size());
    str += sprintf2("  Max Spherical Size = %11zu\n", max_spherical_size());
    str += sprintf2("  Pruned             = %11s", (is_pruned() ? "Yes" : "No")); 
    return str;
}

} // namespace lightspeed 
