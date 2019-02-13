#include <lightspeed/gridbox.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/resource_list.hpp>
#include <lightspeed/pure_transform.hpp>
#include <lightspeed/hermite.hpp>
#include <stdexcept>
#include <cstdio>

namespace lightspeed {

std::shared_ptr<Tensor> GridBox::metaDensity(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<HashedGrid>& grid,
    double thre,
    const std::shared_ptr<Tensor>& rho
    )
{
    std::shared_ptr<Tensor> xyz = grid->xyz();
    size_t nP = xyz->shape()[0];
    const double* xyzp = xyz->data().data();
    double R = grid->R();
    const HashedGrid* gridp = grid.get();

    // Target
    std::shared_ptr<Tensor> rho2 = rho;
    std::vector<size_t> dimr;
    dimr.push_back(nP);
    dimr.push_back(10);
    if (!rho2) {
        rho2 = std::shared_ptr<Tensor>(new Tensor(dimr));
    }
    rho2->shape_error(dimr);
    double* rhop = rho2->data().data();
      
    // => Setup <= //

    // Spherical to Cartesian transform
    std::shared_ptr<Tensor> D2 = PureTransform::pureToCart2(
        pairlist->basis1(),
        pairlist->basis2(),
        D);  

    // Cartesian to Hermite transform
    std::shared_ptr<Hermite> Dherm(new Hermite(pairlist)); 
    Dherm->cart_to_herm(D2, false);
    D2.reset();

    const int xmask = (1 << 30) - (1 << 20);
    const int ymask = (1 << 20) - (1 << 10);
    const int zmask = (1 << 10) - (1 <<  0);

    const std::vector<HermiteL>& hermites = Dherm->hermites();
    for (size_t herm_ind = 0; herm_ind < hermites.size(); herm_ind++) {
        const HermiteL& hermite = hermites[herm_ind];
        int H = hermite.H();
        int V = hermite.L() + 1;
        size_t nherm = hermite.npair();
        const double* datap = hermite.data().data();
        const double* geomp = hermite.geom().data();
        // Hermite angular momentum quanta for current H
        std::vector<int> angl = HermiteUtil::angl(H);
        // Max Hermite coefficient per shell
        std::vector<double> dmax = HermiteUtil::dmax_ket(
            hermite);
        #pragma omp parallel for schedule(dynamic) num_threads(resources->nthread())
        for (size_t A = 0; A < nherm; A++) {
            const double* data2p = datap + A * H;
            double bA = geomp[4*A + 0];
            double xA = geomp[4*A + 1];
            double yA = geomp[4*A + 2];
            double zA = geomp[4*A + 3];
            double dA = dmax[A];
            if (dA == 0.0) continue;
            // Determine R for this Gaussian     
            double arg = - 1.0 / bA * log(thre / fabs(dA));
            if (arg <= 0.0) continue;
            double Rc = sqrt(arg);
            //printf("%14.6f\n", Rc);
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
                    double G = exp(-bA * RPA_2);
                    double val = dA * G;
                    if (fabs(val) < thre) continue;
                    double rPA[3] = {
                        xP - xA,
                        yP - yA,
                        zP - zA };
                    double L[3][V+2];
                    for (int d = 0; d < 3; d++) {
                        double* L2 = L[d];
                        double rval = rPA[d];
                        L2[0] = 1.0;
                        L2[1] = 2.0 * bA * rval;
                        for (int t = 1; t < V + 1; t++) {         
                            L2[t+1] = 2.0 * bA * (rval * L2[t] - t * L2[t-1]); 
                        }
                    }
                    double rho_val = 0.0;
                    double rhox_val = 0.0;
                    double rhoy_val = 0.0;
                    double rhoz_val = 0.0;
                    double rhoxx_val = 0.0;
                    double rhoxy_val = 0.0;
                    double rhoxz_val = 0.0;
                    double rhoyy_val = 0.0;
                    double rhoyz_val = 0.0;
                    double rhozz_val = 0.0;
                    for (int t = 0; t < H; t++) {
                        int ttask = angl[t];
                        int tx = (ttask & xmask) >> 20;
                        int ty = (ttask & ymask) >> 10;
                        int tz = (ttask & zmask) >>  0;
                        double Lx = L[0][tx]; 
                        double Ly = L[1][ty];
                        double Lz = L[2][tz]; 
                        double Lxp1 = L[0][tx+1];
                        double Lyp1 = L[1][ty+1];
                        double Lzp1 = L[2][tz+1];
                        double Lxp2 = L[0][tx+2];
                        double Lyp2 = L[1][ty+2];
                        double Lzp2 = L[2][tz+2];
                        double dval = data2p[t] * G;
                        rho_val += dval * Lx * Ly * Lz; 
                        rhox_val -= dval * Lxp1 * Ly * Lz; 
                        rhoy_val -= dval * Lx * Lyp1 * Lz; 
                        rhoz_val -= dval * Lx * Ly * Lzp1; 
                        rhoxx_val += dval * Lxp2 * Ly * Lz;
                        rhoxy_val += dval * Lxp1 * Lyp1 * Lz;
                        rhoxz_val += dval * Lxp1 * Ly * Lzp1;
                        rhoyy_val += dval * Lx * Lyp2 * Lz;
                        rhoyz_val += dval * Lx * Lyp1 * Lzp1;
                        rhozz_val += dval * Lx * Ly * Lzp2;
                    }
                    #pragma omp atomic
                    rhop[10*P + 0] += rho_val;
                    #pragma omp atomic
                    rhop[10*P + 1] += rhox_val;
                    #pragma omp atomic
                    rhop[10*P + 2] += rhoy_val;
                    #pragma omp atomic
                    rhop[10*P + 3] += rhoz_val;
                    #pragma omp atomic
                    rhop[10*P + 4] += rhoxx_val;
                    #pragma omp atomic
                    rhop[10*P + 5] += rhoxy_val;
                    #pragma omp atomic
                    rhop[10*P + 6] += rhoxz_val;
                    #pragma omp atomic
                    rhop[10*P + 7] += rhoyy_val;
                    #pragma omp atomic
                    rhop[10*P + 8] += rhoyz_val;
                    #pragma omp atomic
                    rhop[10*P + 9] += rhozz_val;
                }
            }}}
        }
    }
    
    return rho2;
}


} // namespace lightspeed
