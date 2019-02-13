#include <lightspeed/lr.hpp>
#include <lightspeed/tensor.hpp>
#include <cmath>

namespace lightspeed {

std::shared_ptr<LaplaceDenom> LaplaceDenom::instance_(new LaplaceDenom());

bool LaplaceDenom::RKThresh::operator< (const RKThresh &rkthresh) const {
    return thresh < rkthresh.thresh;
}

LaplaceDenom::LaplaceDenom()
{
    tabelle_ = build_tabelle();
    omega_ = build_omega();
    alpha_ = build_alpha();
}

LaplaceDenom::Params LaplaceDenom::get_omega_alphas(
    double R, 
    double sc, 
    double max_error
    ) {

    std::shared_ptr<LaplaceDenom> laplace = LaplaceDenom::instance();
    const LaplaceDenom* laplacep = laplace.get();

    LaplaceDenom::TabelleMap tabelle = laplacep->tabelle();
    LaplaceDenom::ParamMap omegam = laplacep->omega();
    LaplaceDenom::ParamMap alpham = laplacep->alpha();

    // Make sure table makes sense
    for (auto it1 = tabelle.begin(); it1 != tabelle.end(); ++it1) {
        for (auto it2 = it1->second.begin(); it2 != it1->second.end(); ++it2) {
                RKThresh rkthresh = (*it2);
                std::string param_file = "1_xk" + rkthresh.k + "_" + rkthresh.R;
                if(!omegam.count(param_file)) printf("Missing omega: %s\n", param_file.c_str());
                if(!alpham.count(param_file)) printf("Missing alpha: %s\n", param_file.c_str());
        }
    }
    
    LaplaceDenom::TabelleMap::iterator it_r = tabelle.upper_bound(R);
    RKThresh rkthresh;
    for (; it_r != tabelle.end(); ++it_r) {
        std::set<RKThresh> rkthreshs = it_r->second;
        rkthresh = *(rkthreshs.begin());
        for (std::set<RKThresh>::iterator it=rkthreshs.begin(); it!=rkthreshs.end(); ++it) {
             // printf("R = %s, k = %s, thresh = %11.3E\n",
             //    it->R, it->k, it->thresh);
             if (it->thresh > max_error) break;
             rkthresh = *it;
        }
        if (rkthresh.thresh <= max_error) break;
    }
    if (rkthresh.thresh > max_error) throw std::runtime_error("LaplaceDenom: No sensible laplace denominator could be found.");
    // printf("Selected: R = %s, k = %s, thresh = %11.3E\n",
    //      rkthresh.R, rkthresh.k, rkthresh.thresh);

    std::string param_file = "1_xk" + rkthresh.k + "_" + rkthresh.R;
    if (!omegam.count(param_file)) throw std::runtime_error("LaplaceDenom: No omega rule found for file " + param_file);
    if (!alpham.count(param_file)) throw std::runtime_error("LaplaceDenom: No alpha rule found for file " + param_file);
    std::vector<double> omega = omegam[param_file];
    std::vector<double> alpha = alpham[param_file];

    for (std::vector<double>::iterator it=omega.begin(); it!= omega.end(); ++it) {
        *it = *it/sc;
    }
    for (std::vector<double>::iterator it=alpha.begin(); it!= alpha.end(); ++it) {
        *it = *it/sc;
    }

    return {omega, alpha};
}

void LaplaceDenom::fill_tau(
    std::shared_ptr<Tensor>& tau, 
    const std::shared_ptr<Tensor>& eps, 
    const LaplaceDenom::Params params, 
    bool is_vir, 
    int order_denom
    ) 
{
    if (params.first.size() != params.second.size()) 
        throw std::runtime_error("LaplaceDenom::fill_tau: omega and alpha vectors have different sizes");

    size_t n_e = eps->shape()[0];
    size_t n_w = params.first.size();
    
    if ((tau->shape()[0] != n_w) || (tau->shape()[1] != n_e))
        throw std::runtime_error("LaplaceDenom::fill_tau: tau tensor has unexpected dimension");

    const double * ep = eps->data().data();
    for (size_t eidx = 0; eidx < n_e; ++eidx) {
        for (size_t widx = 0; widx < n_w; ++widx) {
            double omega = params.first[widx];
            double alpha = params.second[widx];
            double e = ep[eidx];

            double aea = alpha * e;
            if (is_vir) aea = -aea;
            double wpow = pow(omega, 1/((double) 2 * order_denom));
            size_t tau_idx = widx * tau->strides()[0] + eidx * tau->strides()[1];
            tau->data()[tau_idx] = wpow * exp(aea);
        }
    }
}

std::vector< double > LaplaceDenom::get_eps_min_max(
    const std::shared_ptr<Tensor>& eps_occ,
    const std::shared_ptr<Tensor>& eps_vir    
    ) {
    size_t n_occ = eps_occ->shape()[0];
    size_t n_vir = eps_vir->shape()[0];
    
    const double * eop = eps_occ->data().data();
    const double * evp = eps_vir->data().data();

    double eop_max = eop[0];
    double eop_min = eop[0];
    for (size_t i = 0; i < n_occ; i++) {
        eop_min = std::min(eop_min, eop[i]);
        eop_max = std::max(eop_max, eop[i]);
    }

    double evp_max = evp[0];
    double evp_min = evp[0];
    for (size_t i = 0; i < n_occ; i++) {
        evp_min = std::min(evp_min, evp[i]);
        evp_max = std::max(evp_max, evp[i]);
    }

    if (evp_min < eop_max) 
        throw std::runtime_error("LaplaceDenom::o2v2_denom: Min virtual orbital < Max occupied orbital");

    return {eop_min, eop_max, evp_min, evp_max}; 
}

std::vector<std::shared_ptr<Tensor>> LaplaceDenom::o2v2_denom(
    const std::shared_ptr<Tensor>& eps_occ,
    const std::shared_ptr<Tensor>& eps_vir,
    double max_error
    ) 
{
    return onvn_denom(eps_occ, eps_vir, max_error, 2);
}

std::vector<std::shared_ptr<Tensor>> LaplaceDenom::o3v3_denom(
    const std::shared_ptr<Tensor>& eps_occ,
    const std::shared_ptr<Tensor>& eps_vir,
    double max_error
    ) 
{
    return onvn_denom(eps_occ, eps_vir, max_error, 3);
}


std::vector<std::shared_ptr<Tensor>> LaplaceDenom::onvn_denom(
    const std::shared_ptr<Tensor>& eps_occ,
    const std::shared_ptr<Tensor>& eps_vir,
    double max_error,
    int order_denom
    ) 
{
    eps_occ->ndim_error(1);
    eps_vir->ndim_error(1);
    
    size_t n_occ = eps_occ->shape()[0];
    size_t n_vir = eps_vir->shape()[0];
    
    if (n_occ < 1) 
        throw std::runtime_error("LaplaceDenom::o2v2_denom: Number of occupied orbitals < 1");
    if (n_vir < 1) 
        throw std::runtime_error("LaplaceDenom::o2v2_denom: Number of virtual orbitals < 1");

    std::vector< double > eps_min_max = get_eps_min_max(eps_occ, eps_vir);
    double eop_min = eps_min_max[0], eop_max = eps_min_max[1];
    double evp_min = eps_min_max[2], evp_max = eps_min_max[3];

    // R is a bit bigger than it needs to be. 
    double R = (evp_max - eop_min)/(evp_min - eop_max);
    double sc = order_denom * (evp_min - eop_max);
    LaplaceDenom::Params params = get_omega_alphas(R, sc, max_error);
    
    // Compile results
    size_t n_w = params.first.size();
    
    std::shared_ptr<Tensor> tau_occ(new Tensor({n_w, n_occ}, "tau_occ"));
    std::shared_ptr<Tensor> tau_vir(new Tensor({n_w, n_vir}, "tau_vir"));
    
    fill_tau(tau_occ, eps_occ, params, false, order_denom);
    fill_tau(tau_vir, eps_vir, params, true, order_denom);   

    return {tau_occ, tau_vir};
}


} // namespace lightspeed
