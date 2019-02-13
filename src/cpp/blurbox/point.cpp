#include <lightspeed/blurbox.hpp>
#include <lightspeed/gridbox.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/resource_list.hpp>
#include <stdexcept>
#include <cstdio>

namespace lightspeed {

std::shared_ptr<Tensor> BlurBox::pointDensity(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& xyzc,
    const std::shared_ptr<HashedGrid>& grid,
    const std::shared_ptr<Tensor>& trans,
    double alpha,
    double thre,
    const std::shared_ptr<Tensor>& rho
    )
{
    // => Safety Checks <= //

        
    xyzc->ndim_error(2);
    size_t nA = xyzc->shape()[0];    
    std::vector<size_t> dimx;
    dimx.push_back(nA); 
    dimx.push_back(4);
    xyzc->shape_error(dimx);
    const double* xyzcp = xyzc->data().data();

    std::shared_ptr<Tensor> xyz = grid->xyz();
    size_t nP = xyz->shape()[0];
    const double* xyzp = xyz->data().data();
    double R = grid->R();
    const HashedGrid* gridp = grid.get();
    
    trans->ndim_error(2);
    size_t nT = trans->shape()[0];
    std::vector<size_t> dimt;
    dimt.push_back(nT);
    dimt.push_back(3);
    trans->shape_error(dimt);
    const double* transp = trans->data().data();

    // Target
    std::shared_ptr<Tensor> rho2 = rho;
    std::vector<size_t> dimr;
    dimr.push_back(nP);
    if (!rho2) {
        rho2 = std::shared_ptr<Tensor>(new Tensor(dimr));
    }
    rho2->shape_error(dimr);
    double* rhop = rho2->data().data();

    if (alpha == -1.0) throw std::runtime_error("BlurBox::pointDensity: cannot collocate with alpha = -1.0, would be Dirac delta functions");
    double pref = pow(alpha / M_PI, 3.0/2.0);

    #pragma omp parallel for num_threads(resources->nthread()) schedule(dynamic)
    for (size_t A = 0; A < nA; A++) {
        double pref2 = pref * xyzcp[4*A + 3]; 
        double xA2 = xyzcp[4*A + 0];
        double yA2 = xyzcp[4*A + 1];
        double zA2 = xyzcp[4*A + 2];
        // Determine R for this Gaussian     
        double arg = - 1.0 / alpha * log(thre / fabs(pref2));
        if (arg <= 0.0) continue;
        double Rc = sqrt(arg);
        for (size_t T = 0; T < nT; T++) {
            double xA = xA2 + transp[3*T + 0];
            double yA = yA2 + transp[3*T + 1];
            double zA = zA2 + transp[3*T + 2];
            // Determine the hashkey for this Gaussian
            std::tuple<int,int,int> key = gridp->hashkey(xA,yA,zA);
            int nstep = (int) ceil(Rc / R); 
            // Step into relevant boxes
            for (int nx = -nstep; nx <= nstep; nx++) {
            for (int ny = -nstep; ny <= nstep; ny++) {
            for (int nz = -nstep; nz <= nstep; nz++) {
                std::tuple<int,int,int> key2(
                    std::get<0>(key) + nx,
                    std::get<1>(key) + ny,
                    std::get<2>(key) + nz);
                if (!HashedGrid::significant_box(key2, R, xA, yA, zA, Rc)) continue;
                const std::vector<size_t>& inds = gridp->inds(key2); 
                for (size_t ind2 = 0; ind2 < inds.size(); ind2++) {
                    size_t P = inds[ind2];
                    double xP = xyzp[3*P + 0];
                    double yP = xyzp[3*P + 1];
                    double zP = xyzp[3*P + 2];
                    double RPA_2 = 
                        pow(xP-xA,2) + 
                        pow(yP-yA,2) + 
                        pow(zP-zA,2);
                    double val = pref2 * exp(-alpha * RPA_2);
                    if (fabs(val) < thre) continue;
                    #pragma omp atomic
                    rhop[P] -= val; 
                }
            }}}
        }
    }
    
    return rho2;
}

std::shared_ptr<Tensor> BlurBox::pointPotential(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& xyz2,
    const std::shared_ptr<HashedGrid>& grid,
    const std::shared_ptr<Tensor>& v,
    const std::shared_ptr<Tensor>& trans,
    double alpha,
    double thre,
    double vmax,
    const std::shared_ptr<Tensor>& v2
    )
{
    // => Safety Checks <= //

    xyz2->ndim_error(2);
    size_t nA = xyz2->shape()[0];    
    std::vector<size_t> dim;
    dim.push_back(nA); 
    dim.push_back(3);
    xyz2->shape_error(dim);
    const double* xyz2p = xyz2->data().data();

    std::shared_ptr<Tensor> xyz = grid->xyz();
    size_t nP = xyz->shape()[0];
    const double* xyzp = xyz->data().data();
    double R = grid->R();
    const HashedGrid* gridp = grid.get();

    std::vector<size_t> dimv;
    dimv.push_back(nP);
    v->shape_error(dimv);
    const double* vp = v->data().data();

    trans->ndim_error(2);
    size_t nT = trans->shape()[0];
    std::vector<size_t> dimt;
    dimt.push_back(nT);
    dimt.push_back(3);
    trans->shape_error(dimt);
    const double* transp = trans->data().data();

    // Target
    std::shared_ptr<Tensor> v3 = v2;
    std::vector<size_t> dimv2;
    dimv2.push_back(nA);
    if (!v2) {
        v3 = std::shared_ptr<Tensor>(new Tensor(dimv2));
    }
    v3->shape_error(dimv2);
    double* v3p = v3->data().data();

    if (alpha == -1.0) throw std::runtime_error("BlurBox::pointPotential: cannot collocate with alpha = -1.0, would be Dirac delta functions");
    double pref = pow(alpha / M_PI, 3.0/2.0);

    double thre2 = thre / vmax;
    
    #pragma omp parallel for num_threads(resources->nthread()) schedule(dynamic)
    for (size_t A = 0; A < nA; A++) {
        double pref2 = pref;
        double xA2 = xyz2p[3*A + 0];
        double yA2 = xyz2p[3*A + 1];
        double zA2 = xyz2p[3*A + 2];
        // Determine R for this Gaussian     
        double arg = - 1.0 / alpha * log(thre2 / fabs(pref));
        if (arg <= 0.0) continue;
        double Rc = sqrt(arg);
        for (size_t T = 0; T < nT; T++) {
            double xA = xA2 + transp[3*T + 0];
            double yA = yA2 + transp[3*T + 1];
            double zA = zA2 + transp[3*T + 2];
            // Determine the hashkey for this Gaussian
            std::tuple<int,int,int> key = gridp->hashkey(xA,yA,zA);
            int nstep = (int) ceil(Rc / R); 
            // Step into relevant boxes
            for (int nx = -nstep; nx <= nstep; nx++) {
            for (int ny = -nstep; ny <= nstep; ny++) {
            for (int nz = -nstep; nz <= nstep; nz++) {
                std::tuple<int,int,int> key2(
                    std::get<0>(key) + nx,
                    std::get<1>(key) + ny,
                    std::get<2>(key) + nz);
                if (!HashedGrid::significant_box(key2, R, xA, yA, zA, Rc)) continue;
                const std::vector<size_t>& inds = gridp->inds(key2); 
                for (size_t ind2 = 0; ind2 < inds.size(); ind2++) {
                    size_t P = inds[ind2];
                    double xP = xyzp[3*P + 0];
                    double yP = xyzp[3*P + 1];
                    double zP = xyzp[3*P + 2];
                    double vP = vp[P]; // Value of potential on grid
                    double RPA_2 = 
                        pow(xP-xA,2) + 
                        pow(yP-yA,2) + 
                        pow(zP-zA,2);
                    double val = pref * vP * exp(-alpha * RPA_2);
                    if (fabs(val) < thre) continue;
                    v3p[A] += val;
                }
            }}}
        }
    }
    
    return v3;
}

std::shared_ptr<Tensor> BlurBox::pointGrad(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& xyzc,
    const std::shared_ptr<HashedGrid>& grid,
    const std::shared_ptr<Tensor>& v,
    const std::shared_ptr<Tensor>& trans,
    double alpha,
    double thre,
    double vmax,
    const std::shared_ptr<Tensor>& G
    )
{
    // => Safety Checks <= //

    xyzc->ndim_error(2);
    size_t nA = xyzc->shape()[0];
    std::vector<size_t> dim;
    dim.push_back(nA); 
    dim.push_back(4);
    xyzc->shape_error(dim);
    const double* xyzcp = xyzc->data().data();

    std::shared_ptr<Tensor> xyz = grid->xyz();
    size_t nP = xyz->shape()[0];
    const double* xyzp = xyz->data().data();
    double R = grid->R();
    const HashedGrid* gridp = grid.get();

    v->ndim_error(1);
    std::vector<size_t> dimv;
    dimv.push_back(nP);
    v->shape_error(dimv);
    const double* vp = v->data().data();

    trans->ndim_error(2);
    size_t nT = trans->shape()[0];
    std::vector<size_t> dimt;
    dimt.push_back(nT);
    dimt.push_back(3);
    trans->shape_error(dimt);
    const double* transp = trans->data().data();

    // Target
    std::shared_ptr<Tensor> G2 = G;
    std::vector<size_t> dimG;
    dimG.push_back(nA); 
    dimG.push_back(3);
    if (!G2) {
        G2 = std::shared_ptr<Tensor>(new Tensor(dimG));
    }
    G2->shape_error(dimG);
    double* Gp = G2->data().data();

    if (alpha == -1.0) throw std::runtime_error("BlurBox::pointGrad: cannot collocate with alpha = -1.0, would be Dirac delta functions");
    double pref = pow(alpha / M_PI, 3.0/2.0);
        
    double thre2 = thre / vmax;

    #pragma omp parallel for num_threads(resources->nthread()) schedule(dynamic)
    for (size_t A = 0; A < nA; A++) {
        double xA2 = xyzcp[4*A + 0];
        double yA2 = xyzcp[4*A + 1];
        double zA2 = xyzcp[4*A + 2];
        double cA  = xyzcp[4*A + 3];
        // Determine R for this Gaussian     
        double arg = - 1.0 / alpha * log(thre2 / fabs(pref * cA));
        if (arg <= 0.0) continue;
        double Rc = sqrt(arg);
        for (size_t T = 0; T < nT; T++) {
            double xA = xA2 + transp[3*T + 0];
            double yA = yA2 + transp[3*T + 1];
            double zA = zA2 + transp[3*T + 2];
            // Determine the hashkey for this Gaussian
            std::tuple<int,int,int> key = gridp->hashkey(xA,yA,zA);
            int nstep = (int) ceil(Rc / R); 
            // Step into relevant boxes
            for (int nx = -nstep; nx <= nstep; nx++) {
            for (int ny = -nstep; ny <= nstep; ny++) {
            for (int nz = -nstep; nz <= nstep; nz++) {
                std::tuple<int,int,int> key2(
                    std::get<0>(key) + nx,
                    std::get<1>(key) + ny,
                    std::get<2>(key) + nz);
                if (!HashedGrid::significant_box(key2, R, xA, yA, zA, Rc)) continue;
                const std::vector<size_t>& inds = gridp->inds(key2); 
                for (size_t ind2 = 0; ind2 < inds.size(); ind2++) {
                    size_t P = inds[ind2];
                    double xP = xyzp[3*P + 0];
                    double yP = xyzp[3*P + 1];
                    double zP = xyzp[3*P + 2];
                    double vP = vp[P]; // Value of potential on grid
                    double xPA = xP-xA;
                    double yPA = yP-yA;
                    double zPA = zP-zA;
                    double RPA_2 = 
                        pow(xPA,2) + 
                        pow(yPA,2) + 
                        pow(zPA,2);
                    double val = cA * pref * vP * exp(-alpha * RPA_2);
                    if (fabs(val) < thre) continue; // Screen on energy
                    double val2 = 2 * alpha * val;
                    Gp[3*A + 0] += val2 * xPA;
                    Gp[3*A + 1] += val2 * yPA;
                    Gp[3*A + 2] += val2 * zPA;
                }
            }}}
        }
    }
    
    return G2;
}

} // namespace lightspeed
