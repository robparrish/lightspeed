#include <lightspeed/boys.hpp>
#include <lightspeed/tensor.hpp>
#include <stdexcept>

namespace lightspeed {

std::shared_ptr<Boys> Boys::instance_(new Boys());

Boys::Boys()
{
    Tc_ = 30.0; // Hardcoded, crossover is accurate to 10^-14 absolute
    nunit_ = Boys::build_nunit();
    dTinv_.resize(nunit_.size());
    for (size_t k = 0; k < nunit_.size(); k++) {
        dTinv_[k] = nunit_[k] / Tc_;
    }

    boys_Fs_ = build_boys_Fs();
    
    for (int n = 0; n < nunit_.size(); n++) {
        inv_odds_.push_back(1.0 / (2.0 * n + 1.0));
    }
}

std::shared_ptr<Tensor> Boys::compute(
    int nboys,
    const std::shared_ptr<Tensor>& T)
{
    if (nboys > 17) throw std::runtime_error("Boys::compute: nboys is too large.");

    T->ndim_error(1);
    size_t npoint = T->shape()[0];
    const double* Tp = T->data().data();

    std::vector<size_t> dim;
    dim.push_back(npoint);
    dim.push_back(nboys+1);
    std::shared_ptr<Tensor> ret(new Tensor(dim));
    double* retp = ret->data().data(); 

    std::shared_ptr<Boys> boys = Boys::instance();
    const Boys* boysp = boys.get();

    for (size_t P = 0; P < npoint; P++) {
        boysp->interpolate_F(nboys,Tp[P],retp + P*(nboys + 1));
    } 
    
    return ret;
}


} // namespace lightspeed
