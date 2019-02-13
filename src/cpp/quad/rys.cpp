#include <lightspeed/rys.hpp>
#include <lightspeed/tensor.hpp>
#include <stdexcept>
#include <cmath>

namespace lightspeed {

std::shared_ptr<Rys> Rys::instance_(new Rys());

Rys::Rys()
{
    Tc_ = Rys::build_Tc();
    nunit_ = Rys::build_nunit();
    dTinv_.resize(Tc_.size());
    for (size_t k = 1; k < Tc_.size(); k++) {
        dTinv_[k] = nunit_[k] / Tc_[k];
    }

    rys_ts_ = build_rys_ts();
    rys_ws_ = build_rys_ws();
    pegl_ts_ = build_pegl_ts();
    pegl_ws_ = build_pegl_ws();
    pegh_ts_ = build_pegh_ts();
    pegh_ws_ = build_pegh_ws();
}

std::shared_ptr<Tensor> Rys::compute_t(
    int nrys,
    const std::shared_ptr<Tensor>& T)
{
    if (nrys > 9) throw std::runtime_error("Rys::compute: nrys is too large.");

    T->ndim_error(1);
    size_t npoint = T->shape()[0];
    const double* Tp = T->data().data();

    std::vector<size_t> dim;
    dim.push_back(npoint);
    dim.push_back(nrys);
    std::shared_ptr<Tensor> ret(new Tensor(dim));
    double* retp = ret->data().data(); 

    std::shared_ptr<Rys> rys = Rys::instance();
    const Rys* rysp = rys.get();

    for (size_t P = 0; P < npoint; P++) {
        for (int mrys = 0; mrys < nrys; mrys++) {
            retp[P*nrys + mrys] = rysp->interpolate_t(nrys,mrys,Tp[P]);
        } 
    } 
    
    return ret;
}
std::shared_ptr<Tensor> Rys::compute_w(
    int nrys,
    const std::shared_ptr<Tensor>& T)
{
    if (nrys > 9) throw std::runtime_error("Rys::compute: nrys is too large.");

    T->ndim_error(1);
    size_t npoint = T->shape()[0];
    const double* Tp = T->data().data();

    std::vector<size_t> dim;
    dim.push_back(npoint);
    dim.push_back(nrys);
    std::shared_ptr<Tensor> ret(new Tensor(dim));
    double* retp = ret->data().data(); 

    std::shared_ptr<Rys> rys = Rys::instance();
    const Rys* rysp = rys.get();

    for (size_t P = 0; P < npoint; P++) {
        for (int mrys = 0; mrys < nrys; mrys++) {
            retp[P*nrys + mrys] = rysp->interpolate_w(nrys,mrys,Tp[P]);
        } 
    } 
    
    return ret;
}

} // namespace lightspeed
