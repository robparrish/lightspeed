#include "libxc_functional_impl.hpp"
#include <lightspeed/tensor.hpp>
#include "../util/string.hpp"
#include <map>
#include <vector>

namespace lightspeed {
   
LibXCFunctionalImpl::LibXCFunctionalImpl(const std::string& name) :
    initialized_(false)
{
#ifdef HAVE_LIBXC
    std::map<std::string, int> ids;
    // LSDA
    ids["X"] = XC_LDA_X;
    ids["VWN_C"] = XC_LDA_C_VWN;
    // GGA
    ids["B88_X"] = XC_GGA_X_B88;
    ids["PBE_X"] = XC_GGA_X_PBE;
    ids["PBE_C"] = XC_GGA_C_PBE;
    ids["LYP_C"] = XC_GGA_C_LYP;
    // Hybrid GGA
    ids["B3PW91"] = XC_HYB_GGA_XC_B3PW91;
    ids["B3LYP"] = XC_HYB_GGA_XC_B3LYP;
    ids["B3LYP5"] = XC_HYB_GGA_XC_B3LYP5;
    ids["B3P86"] = XC_HYB_GGA_XC_B3P86;
    ids["B97"] = XC_HYB_GGA_XC_B97;
    ids["B97_1"] = XC_HYB_GGA_XC_B97_1;
    ids["B97_2"] = XC_HYB_GGA_XC_B97_2;
    ids["B97_3"] = XC_HYB_GGA_XC_B97_3;
    ids["BHANDH"] = XC_HYB_GGA_XC_BHANDH;
    // LRC Hybrid GGA
    ids["CAM_B3LYP"] = XC_HYB_GGA_XC_CAM_B3LYP;
    ids["WB97"] = XC_HYB_GGA_XC_WB97;
    ids["WB97X"] = XC_HYB_GGA_XC_WB97X;
    ids["WB97X_D"] = XC_HYB_GGA_XC_WB97X_D;
    // LRC GGA Pieces
    ids["WPBE_X"] = XC_GGA_X_HJS_PBE;
    if (!ids.count(name)) throw std::runtime_error("LibXCFunctionalImpl: Unknown functional name: " + name);
    if (xc_func_init(&func_,ids[name],XC_POLARIZED)) throw std::runtime_error("LibXCFunctionalImpl: Bad functional name: " + name);
    initialized_ = true;
#else
    throw std::runtime_error("LibXCFunctionalImpl: No LibXC - should not construct");
#endif
}
LibXCFunctionalImpl::~LibXCFunctionalImpl()
{
#ifdef HAVE_LIBXC
    if (initialized_) {
        xc_func_end(&func_);
    }
#else
#endif
}
std::string LibXCFunctionalImpl::name() const 
{
#ifdef HAVE_LIBXC
    return std::string(func_.info->name);
#else
    throw std::runtime_error("LibXCFunctionalImpl: No LibXC.");
#endif
}
std::string LibXCFunctionalImpl::citation() const 
{
#ifdef HAVE_LIBXC
    return std::string(func_.info->refs[0]->ref);
#else
    throw std::runtime_error("LibXCFunctionalImpl: No LibXC.");
#endif
}
std::string LibXCFunctionalImpl::string() const
{
    std::string s = "";
    s += "Functional:\n";
    s += sprintf2("  Name   = %11s\n", name().c_str());
    s += sprintf2("  Type   = %11s\n", (type() == 1 ? "GGA" : "LSDA"));
    s += sprintf2("  Deriv  = %11d\n", deriv());
    s += sprintf2("  Alpha  = %11.3E\n", alpha());
    s += sprintf2("  Beta   = %11.3E\n", beta());
    s += sprintf2("  Omega  = %11.3E\n", omega());
    s += sprintf2("  Source = %11s\n", "LibXC");
    return s;
}
bool LibXCFunctionalImpl::has_lsda() const
{
#ifdef HAVE_LIBXC
    bool val = false;    
    val |= (func_.info->family & XC_FAMILY_LDA);
    val |= (func_.info->family & XC_FAMILY_GGA);
    val |= (func_.info->family & XC_FAMILY_MGGA);
    val |= (func_.info->family & XC_FAMILY_HYB_GGA);
    val |= (func_.info->family & XC_FAMILY_HYB_MGGA);
    return val;
#else
    throw std::runtime_error("LibXCFunctionalImpl: No LibXC.");
#endif
}
bool LibXCFunctionalImpl::has_gga() const
{
#ifdef HAVE_LIBXC
    bool val = false;    
    val |= (func_.info->family & XC_FAMILY_GGA);
    val |= (func_.info->family & XC_FAMILY_MGGA);
    val |= (func_.info->family & XC_FAMILY_HYB_GGA);
    val |= (func_.info->family & XC_FAMILY_HYB_MGGA);
    return val;
#else
    throw std::runtime_error("LibXCFunctionalImpl: No LibXC.");
#endif
}
int LibXCFunctionalImpl::deriv() const
{
#ifdef HAVE_LIBXC
    if (func_.info->flags & XC_FLAGS_HAVE_LXC) return 4;
    if (func_.info->flags & XC_FLAGS_HAVE_KXC) return 3;
    if (func_.info->flags & XC_FLAGS_HAVE_FXC) return 2;
    if (func_.info->flags & XC_FLAGS_HAVE_VXC) return 1;
    if (func_.info->flags & XC_FLAGS_HAVE_EXC) return 0;
    throw std::runtime_error("LibXCFunctionalImpl::deriv: Invalid functional");
#else
    throw std::runtime_error("LibXCFunctionalImpl: No LibXC.");
#endif
}
double LibXCFunctionalImpl::alpha() const
{
#ifdef HAVE_LIBXC
    // YAK: Apparently the cam_* parameters are uninitialized unless the functional is hybrid.
    bool hyb = false;
    hyb |= func_.info->family & XC_FAMILY_HYB_GGA;
    hyb |= func_.info->family & XC_FAMILY_HYB_MGGA;
    return (hyb ? func_.cam_alpha + func_.cam_beta : 0.0);
#else
    throw std::runtime_error("LibXCFunctionalImpl: No LibXC.");
#endif
}
double LibXCFunctionalImpl::beta() const
{
#ifdef HAVE_LIBXC
    // YAK: Apparently the cam_* parameters are uninitialized unless the functional is hybrid.
    bool hyb = false;
    hyb |= func_.info->family & XC_FAMILY_HYB_GGA;
    hyb |= func_.info->family & XC_FAMILY_HYB_MGGA;
    return (hyb ? -func_.cam_beta : 0.0);
#else
    throw std::runtime_error("LibXCFunctionalImpl: No LibXC.");
#endif
}
double LibXCFunctionalImpl::omega() const
{
#ifdef HAVE_LIBXC
    // YAK: Apparently the cam_* parameters are uninitialized unless the functional is hybrid.
    bool hyb = false;
    hyb |= func_.info->family & XC_FAMILY_HYB_GGA;
    hyb |= func_.info->family & XC_FAMILY_HYB_MGGA;
    return (hyb ? func_.cam_omega : 0.0);
#else
    throw std::runtime_error("LibXCFunctionalImpl: No LibXC.");
#endif
}
std::shared_ptr<Tensor> LibXCFunctionalImpl::compute(
    const std::shared_ptr<Tensor>& dat,
    int deriv_val) const
{
#ifdef HAVE_LIBXC
    if (deriv_val < 0) throw std::runtime_error("LibXCFunctionalImpl::compute: deriv < 0?");
    if (deriv_val > deriv()) throw std::runtime_error("LibXCFunctionalImpl::compute: requested deriv is not implemented");
    
    dat->ndim_error(2);
    std::vector<size_t> dim;
    dim.push_back(dat->shape()[0]);
    if (has_gga()) dim.push_back(5);
    else dim.push_back(2);
    dat->shape_error(dim);
    size_t npoint = dat->shape()[0];
    const double* datp = dat->data().data();

    int type = (has_gga() ? 1 : 0); // 0 - LDA, 2 - GGA, 3 - MGGA

    // YAK: I hate 32-bit libraries!
    if (npoint > ((1ULL << 31) - 1)) throw std::runtime_error("LibXCFunctionalImpl::compute: maximum number of points in 32-bit LibXC exceeded.");

    // Temporaries for density characteristics
    std::vector<double> rho;
    std::vector<double> sigma;
    if (type == 0) {
        rho.resize(2*npoint);
        for (size_t P = 0; P < npoint; P++) {
            rho[2*P + 0] = datp[2*P + 0];
            rho[2*P + 1] = datp[2*P + 1];
        }
    } else if (type == 1) {
        rho.resize(2*npoint);
        sigma.resize(3*npoint);
        for (size_t P = 0; P < npoint; P++) {
            rho[2*P + 0] = datp[5*P + 0];
            rho[2*P + 1] = datp[5*P + 1];
            sigma[3*P + 0] = datp[5*P + 2];
            sigma[3*P + 1] = datp[5*P + 3];
            sigma[3*P + 2] = datp[5*P + 4];
        }
    } else {
        throw std::runtime_error("LibXCFunctionalImpl::compute: unpacking of density for type is not implemented.");
    }
    const double* rhop = rho.data();
    const double* sigmap = sigma.data();
    
    if (type == 0 && deriv_val == 0) {
        // Temporaries for partials
        std::vector<double> exc(npoint); 
        double* excp = exc.data();
        // Call to LibXC
        xc_lda_exc(&func_,npoint,rhop,excp);
        // Packing into Tensor
        std::vector<size_t> dimr;
        dimr.push_back(npoint);
        dimr.push_back(1);
        std::shared_ptr<Tensor> ret(new Tensor(dimr));
        double* retp = ret->data().data();
        for (size_t P = 0; P < npoint; P++) {
            retp[1*P + 0] = excp[1*P + 0] * (rhop[2*P + 0] + rhop[2*P+ 1]); // YAK: Apparently exc is actually exc / rho
        }
        return ret;
    }
    if (type == 0 && deriv_val == 1) {
        // Temporaries for partials
        std::vector<double> exc(npoint); 
        double* excp = exc.data();
        std::vector<double> vxc(2*npoint); 
        double* vxcp = vxc.data();
        // Call to LibXC
        xc_lda_exc_vxc(&func_,npoint,rhop,excp,vxcp);
        // Packing into Tensor
        std::vector<size_t> dimr;
        dimr.push_back(npoint);
        dimr.push_back(3);
        std::shared_ptr<Tensor> ret(new Tensor(dimr));
        double* retp = ret->data().data();
        for (size_t P = 0; P < npoint; P++) {
            retp[3*P + 0] = excp[1*P + 0] * (rhop[2*P + 0] + rhop[2*P+ 1]); // YAK: Apparently exc is actually exc / rho
            retp[3*P + 1] = vxcp[2*P + 0];
            retp[3*P + 2] = vxcp[2*P + 1];
        }
        return ret;
    }
    if (type == 0 && deriv_val == 2) {
        // Temporaries for partials
        std::vector<double> exc(npoint); 
        double* excp = exc.data();
        std::vector<double> vxc(2*npoint); 
        double* vxcp = vxc.data();
        std::vector<double> fxc(3*npoint); 
        double* fxcp = fxc.data();
        // Call to LibXC
        xc_lda_exc_vxc(&func_,npoint,rhop,excp,vxcp);
        xc_lda_fxc(&func_,npoint,rhop,fxcp);
        // Packing into Tensor
        std::vector<size_t> dimr;
        dimr.push_back(npoint);
        dimr.push_back(6);
        std::shared_ptr<Tensor> ret(new Tensor(dimr));
        double* retp = ret->data().data();
        for (size_t P = 0; P < npoint; P++) {
            retp[6*P + 0] = excp[1*P + 0] * (rhop[2*P + 0] + rhop[2*P+ 1]); // YAK: Apparently exc is actually exc / rho
            retp[6*P + 1] = vxcp[2*P + 0];
            retp[6*P + 2] = vxcp[2*P + 1];
            retp[6*P + 3] = fxcp[3*P + 0];
            retp[6*P + 4] = fxcp[3*P + 1];
            retp[6*P + 5] = fxcp[3*P + 2];
        }
        return ret;
    }
    if (type == 0 && deriv_val == 3) {
        // Temporaries for partials
        std::vector<double> exc(npoint); 
        double* excp = exc.data();
        std::vector<double> vxc(2*npoint); 
        double* vxcp = vxc.data();
        std::vector<double> fxc(3*npoint); 
        double* fxcp = fxc.data();
        std::vector<double> kxc(4*npoint); 
        double* kxcp = kxc.data();
        // Call to LibXC
        xc_lda_exc_vxc(&func_,npoint,rhop,excp,vxcp);
        xc_lda_fxc(&func_,npoint,rhop,fxcp);
        xc_lda_kxc(&func_,npoint,rhop,kxcp);
        // Packing into Tensor
        std::vector<size_t> dimr;
        dimr.push_back(npoint);
        dimr.push_back(10);
        std::shared_ptr<Tensor> ret(new Tensor(dimr));
        double* retp = ret->data().data();
        for (size_t P = 0; P < npoint; P++) {
            retp[10*P + 0] = excp[1*P + 0] * (rhop[2*P + 0] + rhop[2*P+ 1]); // YAK: Apparently exc is actually exc / rho
            retp[10*P + 1] = vxcp[2*P + 0];
            retp[10*P + 2] = vxcp[2*P + 1];
            retp[10*P + 3] = fxcp[3*P + 0];
            retp[10*P + 4] = fxcp[3*P + 1];
            retp[10*P + 5] = fxcp[3*P + 2];
            retp[10*P + 6] = kxcp[4*P + 0];
            retp[10*P + 7] = kxcp[4*P + 1];
            retp[10*P + 8] = kxcp[4*P + 2];
            retp[10*P + 9] = kxcp[4*P + 3];
        }
        return ret;
    }
    if (type == 1 && deriv_val == 0) {
        // Temporaries for partials
        std::vector<double> exc(npoint); 
        double* excp = exc.data();
        // Call to LibXC
        xc_gga_exc(&func_,npoint,rhop,sigmap,excp);
        // Packing into Tensor
        std::vector<size_t> dimr;
        dimr.push_back(npoint);
        dimr.push_back(1);
        std::shared_ptr<Tensor> ret(new Tensor(dimr));
        double* retp = ret->data().data();
        for (size_t P = 0; P < npoint; P++) {
            retp[1*P + 0] = excp[1*P + 0] * (rhop[2*P + 0] + rhop[2*P+ 1]); // YAK: Apparently exc is actually exc / rho
        }
        return ret;
    }
    if (type == 1 && deriv_val == 1) {
        // Temporaries for partials
        std::vector<double> exc(npoint); 
        double* excp = exc.data();
        std::vector<double> vrxc(2*npoint); 
        double* vrxcp = vrxc.data();
        std::vector<double> vsxc(3*npoint); 
        double* vsxcp = vsxc.data();
        // Call to LibXC
        xc_gga_exc_vxc(&func_,npoint,rhop,sigmap,excp,vrxcp,vsxcp);
        // Packing into Tensor
        std::vector<size_t> dimr;
        dimr.push_back(npoint);
        dimr.push_back(6);
        std::shared_ptr<Tensor> ret(new Tensor(dimr));
        double* retp = ret->data().data();
        for (size_t P = 0; P < npoint; P++) {
            retp[6*P + 0] =  excp[1*P + 0] * (rhop[2*P + 0] + rhop[2*P+ 1]); // YAK: Apparently exc is actually exc / rho
            retp[6*P + 1] = vrxcp[2*P + 0];
            retp[6*P + 2] = vrxcp[2*P + 1];
            retp[6*P + 3] = vsxcp[3*P + 0];
            retp[6*P + 4] = vsxcp[3*P + 1];
            retp[6*P + 5] = vsxcp[3*P + 2];
        }
        return ret;
    }
    if (type == 1 && deriv_val == 2) {
        // Temporaries for partials
        std::vector<double> exc(npoint); 
        double* excp = exc.data();
        std::vector<double> vrxc(2*npoint); 
        double* vrxcp = vrxc.data();
        std::vector<double> vsxc(3*npoint); 
        double* vsxcp = vsxc.data();
        std::vector<double> frrxc(3*npoint);
        double* frrxcp = frrxc.data();
        std::vector<double> frsxc(6*npoint);
        double* frsxcp = frsxc.data();
        std::vector<double> fssxc(6*npoint);
        double* fssxcp = fssxc.data();
        // Call to LibXC
        xc_gga_exc_vxc(&func_,npoint,rhop,sigmap,excp,vrxcp,vsxcp);
        xc_gga_fxc(&func_,npoint,rhop,sigmap,frrxcp,frsxcp,fssxcp);
        // Packing into Tensor
        std::vector<size_t> dimr;
        dimr.push_back(npoint);
        dimr.push_back(21);
        std::shared_ptr<Tensor> ret(new Tensor(dimr));
        double* retp = ret->data().data();
        for (size_t P = 0; P < npoint; P++) {
            retp[21*P +  0] =   excp[1*P + 0] * (rhop[2*P + 0] + rhop[2*P+ 1]); // YAK: Apparently exc is actually exc / rho
            retp[21*P +  1] =  vrxcp[2*P + 0];
            retp[21*P +  2] =  vrxcp[2*P + 1];
            retp[21*P +  3] =  vsxcp[3*P + 0];
            retp[21*P +  4] =  vsxcp[3*P + 1];
            retp[21*P +  5] =  vsxcp[3*P + 2];
            retp[21*P +  6] = frrxcp[3*P + 0];
            retp[21*P +  7] = frrxcp[3*P + 1];
            retp[21*P +  8] = frrxcp[3*P + 2];
            retp[21*P +  9] = frsxcp[6*P + 0];
            retp[21*P + 10] = frsxcp[6*P + 1];
            retp[21*P + 11] = frsxcp[6*P + 2];
            retp[21*P + 12] = frsxcp[6*P + 3];
            retp[21*P + 13] = frsxcp[6*P + 4];
            retp[21*P + 14] = frsxcp[6*P + 5];
            retp[21*P + 15] = fssxcp[6*P + 0];
            retp[21*P + 16] = fssxcp[6*P + 1];
            retp[21*P + 17] = fssxcp[6*P + 2];
            retp[21*P + 18] = fssxcp[6*P + 3];
            retp[21*P + 19] = fssxcp[6*P + 4];
            retp[21*P + 20] = fssxcp[6*P + 5];
        }
        return ret;
    }
    
    throw std::runtime_error("LibXCFunctionalImpl::compute: deriv/family combination not implemented.");
#else
    throw std::runtime_error("LibXCFunctionalImpl: No LibXC.");
#endif
}

} // namespace lightspeed
