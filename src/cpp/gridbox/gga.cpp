#include <lightspeed/gridbox.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/resource_list.hpp>
#include <lightspeed/pure_transform.hpp>
#include <lightspeed/hermite.hpp>
#include <stdexcept>
#include <cstdio>
#include <cstring>

namespace lightspeed {

std::shared_ptr<Tensor> GridBox::ggaDensity(
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
    dimr.push_back(4);
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
                    double L[3][V+1];
                    for (int d = 0; d < 3; d++) {
                        double* L2 = L[d];
                        double rval = rPA[d];
                        L2[0] = 1.0;
                        L2[1] = 2.0 * bA * rval;
                        for (int t = 1; t < V; t++) {         
                            L2[t+1] = 2.0 * bA * (rval * L2[t] - t * L2[t-1]); 
                        }
                    }
                    double rho_val = 0.0;
                    double rhox_val = 0.0;
                    double rhoy_val = 0.0;
                    double rhoz_val = 0.0;
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
                        double dval = data2p[t] * G;
                        rho_val += dval * Lx * Ly * Lz; 
                        rhox_val -= dval * Lxp1 * Ly * Lz; 
                        rhoy_val -= dval * Lx * Lyp1 * Lz; 
                        rhoz_val -= dval * Lx * Ly * Lzp1; 
                    }
                    #pragma omp atomic
                    rhop[4*P + 0] += rho_val;
                    #pragma omp atomic
                    rhop[4*P + 1] += rhox_val;
                    #pragma omp atomic
                    rhop[4*P + 2] += rhoy_val;
                    #pragma omp atomic
                    rhop[4*P + 3] += rhoz_val;
                }
            }}}
        }
    }
    
    return rho2;
}

std::shared_ptr<Tensor> GridBox::ggaPotential(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<HashedGrid>& grid,
    const std::shared_ptr<Tensor>& v,
    double thre,
    const std::shared_ptr<Tensor>& V
    )
{
    std::shared_ptr<Tensor> xyz = grid->xyz();
    size_t nP = xyz->shape()[0];
    const double* xyzp = xyz->data().data();
    double R = grid->R();
    const HashedGrid* gridp = grid.get();

    std::vector<size_t> dimv;
    dimv.push_back(nP);
    dimv.push_back(4);
    v->shape_error(dimv);
    const double* vp = v->data().data();
      
    double vmax = 0.0;
    for (size_t P = 0; P < nP; P++) {
        vmax = std::max(vmax,fabs(vp[4*P + 0])); // Only seive on grid potentials and not their derivatives
    }
    double thre2 = thre / vmax;

    // => Setup <= //

    // Cartesian to Hermite transform
    std::shared_ptr<Hermite> Vherm(new Hermite(pairlist)); 

    const int xmask = (1 << 30) - (1 << 20);
    const int ymask = (1 << 20) - (1 << 10);
    const int zmask = (1 << 10) - (1 <<  0);

    std::vector<HermiteL>& hermites = Vherm->hermites();
    for (size_t herm_ind = 0; herm_ind < hermites.size(); herm_ind++) {
        HermiteL& hermite = hermites[herm_ind];
        int H = hermite.H();
        int V = hermite.L() + 1;
        size_t nherm = hermite.npair();
        double* datap = hermite.data().data();
        const double* geomp = hermite.geom().data();
        // Hermite angular momentum quanta for current H
        std::vector<int> angl = HermiteUtil::angl(H);
        // Max Hermite coefficient per shell
        std::vector<double> dmax = HermiteUtil::dmax_bra(
            hermite);
        #pragma omp parallel for schedule(dynamic) num_threads(resources->nthread())
        for (size_t A = 0; A < nherm; A++) {
            double bA = geomp[4*A + 0];
            double xA = geomp[4*A + 1];
            double yA = geomp[4*A + 2];
            double zA = geomp[4*A + 3];
            double dA = dmax[A];
            if (dA == 0.0) continue;
            // Determine R for this Gaussian     
            double arg = - 1.0 / bA * log(thre2 / fabs(dA));
            if (arg <= 0.0) continue; 
            double Rc = sqrt(arg);
            //printf("%14.6f\n", Rc);
            double vbuf[H];
            ::memset(vbuf,'\0',sizeof(double)*H);
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
                    double vP  = vp[4*P + 0];
                    double vxP = vp[4*P + 1];
                    double vyP = vp[4*P + 2];
                    double vzP = vp[4*P + 3];
                    double RPA_2 = 
                        pow(xP-xA,2) + 
                        pow(yP-yA,2) + 
                        pow(zP-zA,2);
                    double G = exp(-bA * RPA_2);
                    double val = dA * G * vP;
                    if (fabs(val) < thre) continue;
                    double rPA[3] = {
                        xP - xA,
                        yP - yA,
                        zP - zA };
                    double L[3][V+1];
                    for (int d = 0; d < 3; d++) {
                        double* L2 = L[d];
                        double rval = rPA[d];
                        L2[0] = 1.0;
                        L2[1] = 2.0 * bA * rval;
                        for (int t = 1; t < V; t++) {         
                            L2[t+1] = 2.0 * bA * (rval * L2[t] - t * L2[t-1]); 
                        }
                    }
                    double G20 = G * vP;
                    double G2x = G * vxP;
                    double G2y = G * vyP;
                    double G2z = G * vzP;
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
                        vbuf[t] += G20 * Lx * Ly * Lz; 
                        vbuf[t] -= G2x * Lxp1 * Ly * Lz; 
                        vbuf[t] -= G2y * Lx * Lyp1 * Lz; 
                        vbuf[t] -= G2z * Lx * Ly * Lzp1; 
                    }
                }
            }}}
            double* data2p = datap + A * H;
            for (int t = 0; t < H; t++) {
                data2p[t] += vbuf[t];
            }
        }
    }

    // Get a (ncart1,ncart2) register to accumulate Cartesian integrals
    std::shared_ptr<Tensor> V2 = PureTransform::allocCart2(
        pairlist->basis1(),
        pairlist->basis2(),
        V);

    // Hermite to Cartesian transform
    Vherm->herm_to_cart(V2, false);
    
    // Transform the Cartesian integrals to spherical
    std::shared_ptr<Tensor> V3 = PureTransform::cartToPure2(
        pairlist->basis1(),
        pairlist->basis2(),
        V2,
        V);

    return V3;
}

std::shared_ptr<Tensor> GridBox::ggaGrad(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<HashedGrid>& grid,
    const std::shared_ptr<Tensor>& v,
    double thre,
    const std::shared_ptr<Tensor>& Grad
    )
{
    std::shared_ptr<Tensor> xyz = grid->xyz();
    size_t nP = xyz->shape()[0];
    const double* xyzp = xyz->data().data();
    double R = grid->R();
    const HashedGrid* gridp = grid.get();

    std::vector<size_t> dimv;
    dimv.push_back(nP);
    dimv.push_back(4);
    v->shape_error(dimv);
    const double* vp = v->data().data();
      
    double vmax = 0.0;
    for (size_t P = 0; P < nP; P++) {
        vmax = std::max(vmax,fabs(vp[4*P + 0])); // Only seive on grid potentials and not their derivatives
    }
    double thre2 = thre / vmax;

    // Target
    if (!pairlist->is_symmetric()) throw std::runtime_error("GridBox::ggaGrad: PairList is not symmetric");
    std::shared_ptr<Tensor> Grad2 = Grad;
    std::vector<size_t> dimG;
    dimG.push_back(pairlist->basis1()->natom()); 
    dimG.push_back(3);
    if (!Grad2) {
        Grad2 = std::shared_ptr<Tensor>(new Tensor(dimG));
    }
    Grad2->shape_error(dimG);
    double* Grad2p = Grad2->data().data();

    // => Setup <= //

    // Spherical to Cartesian transform
    std::shared_ptr<Tensor> D2 = PureTransform::pureToCart2(
        pairlist->basis1(),
        pairlist->basis2(),
        D);  

    // Cartesian to Hermite transform
    std::shared_ptr<Hermite> Vherm(new Hermite(pairlist)); 
    std::shared_ptr<Hermite> Dherm(new Hermite(pairlist)); 
    Dherm->cart_to_herm(D2, false);

    const int xmask = (1 << 30) - (1 << 20);
    const int ymask = (1 << 20) - (1 << 10);
    const int zmask = (1 << 10) - (1 <<  0);

    std::vector<HermiteL>& hermites = Vherm->hermites();
    std::vector<HermiteL>& Dhermites = Dherm->hermites();
    for (size_t herm_ind = 0; herm_ind < hermites.size(); herm_ind++) {
        HermiteL& hermite = hermites[herm_ind];
        HermiteL& Dhermite = Dhermites[herm_ind];
        int H = hermite.H();
        int V = hermite.L() + 1;
        size_t nherm = hermite.npair();
        double* datap = hermite.data().data();
        const double* Ddatap = Dhermite.data().data();
        const double* geomp = hermite.geom().data();
        const double* etasp = hermite.etas().data();
        const int* inds = hermite.inds().data();
        // Hermite angular momentum quanta for current H
        std::vector<int> angl = HermiteUtil::angl(H);
        // Max Hermite coefficient per shell
        std::vector<double> dmax = HermiteUtil::dmax_ket(
            Dhermite);
        #pragma omp parallel for schedule(dynamic) num_threads(resources->nthread())
        for (size_t A = 0; A < nherm; A++) {
            const double* Ddata2p = Ddatap + A * H;
            double bA = geomp[4*A + 0];
            double xA = geomp[4*A + 1];
            double yA = geomp[4*A + 2];
            double zA = geomp[4*A + 3];
            double eta1 = etasp[2*A + 0];
            double eta2 = etasp[2*A + 1];
            double dA = dmax[A];
            if (dA == 0.0) continue;
            // Determine R for this Gaussian     
            double arg = - 1.0 / bA * log(thre2 / fabs(dA));
            if (arg <= 0.0) continue; 
            double Rc = sqrt(arg);
            //printf("%14.6f\n", Rc);
            double vbuf[H];
            ::memset(vbuf,'\0',sizeof(double)*H);
            double G1T[3] = {0.0, 0.0, 0.0};
            double G2T[3] = {0.0, 0.0, 0.0};
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
                    double vP  = vp[4*P + 0];
                    double vxP = vp[4*P + 1];
                    double vyP = vp[4*P + 2];
                    double vzP = vp[4*P + 3];
                    double RPA_2 = 
                        pow(xP-xA,2) + 
                        pow(yP-yA,2) + 
                        pow(zP-zA,2);
                    double G = exp(-bA * RPA_2);
                    double val = dA * G * vP;
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
                        for (int t = 1; t < V+1; t++) {         
                            L2[t+1] = 2.0 * bA * (rval * L2[t] - t * L2[t-1]); 
                        }
                    }
                    double G20 = G * vP;
                    double G2x = G * vxP;
                    double G2y = G * vyP;
                    double G2z = G * vzP;
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
                        vbuf[t] += G20 * Lx * Ly * Lz; 
                        vbuf[t] -= G2x * Lxp1 * Ly * Lz; 
                        vbuf[t] -= G2y * Lx * Lyp1 * Lz; 
                        vbuf[t] -= G2z * Lx * Ly * Lzp1; 
                        double dval = Ddata2p[t] * G;
                        // 0
                        G1T[0] += dval * Lxp1 * Ly * Lz * eta1 * vP;
                        G1T[1] += dval * Lx * Lyp1 * Lz * eta1 * vP;
                        G1T[2] += dval * Lx * Ly * Lzp1 * eta1 * vP;
                        G2T[0] += dval * Lxp1 * Ly * Lz * eta2 * vP;
                        G2T[1] += dval * Lx * Lyp1 * Lz * eta2 * vP;
                        G2T[2] += dval * Lx * Ly * Lzp1 * eta2 * vP;
                        // x
                        G1T[0] -= dval * Lxp2 * Ly * Lz * eta1 * vxP;
                        G1T[1] -= dval * Lxp1 * Lyp1 * Lz * eta1 * vxP;
                        G1T[2] -= dval * Lxp1 * Ly * Lzp1 * eta1 * vxP;
                        G2T[0] -= dval * Lxp2 * Ly * Lz * eta2 * vxP;
                        G2T[1] -= dval * Lxp1 * Lyp1 * Lz * eta2 * vxP;
                        G2T[2] -= dval * Lxp1 * Ly * Lzp1 * eta2 * vxP;
                        // y
                        G1T[0] -= dval * Lxp1 * Lyp1 * Lz * eta1 * vyP;
                        G1T[1] -= dval * Lx * Lyp2 * Lz * eta1 * vyP;
                        G1T[2] -= dval * Lx * Lyp1 * Lzp1 * eta1 * vyP;
                        G2T[0] -= dval * Lxp1 * Lyp1 * Lz * eta2 * vyP;
                        G2T[1] -= dval * Lx * Lyp2 * Lz * eta2 * vyP;
                        G2T[2] -= dval * Lx * Lyp1 * Lzp1 * eta2 * vyP;
                        // z
                        G1T[0] -= dval * Lxp1 * Ly * Lzp1 * eta1 * vzP;
                        G1T[1] -= dval * Lx * Lyp1 * Lzp1 * eta1 * vzP;
                        G1T[2] -= dval * Lx * Ly * Lzp2 * eta1 * vzP;
                        G2T[0] -= dval * Lxp1 * Ly * Lzp1 * eta2 * vzP;
                        G2T[1] -= dval * Lx * Lyp1 * Lzp1 * eta2 * vzP;
                        G2T[2] -= dval * Lx * Ly * Lzp2 * eta2 * vzP;
                    }
                }
            }}}
            double* data2p = datap + A * H;
            for (int t = 0; t < H; t++) {
                data2p[t] += vbuf[t];
            }
            int A1 = inds[2*A + 0];
            int A2 = inds[2*A + 1];
            #pragma omp atomic
            Grad2p[3*A1 + 0] += G1T[0];
            #pragma omp atomic
            Grad2p[3*A1 + 1] += G1T[1];
            #pragma omp atomic
            Grad2p[3*A1 + 2] += G1T[2];
            #pragma omp atomic
            Grad2p[3*A2 + 0] += G2T[0];
            #pragma omp atomic
            Grad2p[3*A2 + 1] += G2T[1];
            #pragma omp atomic
            Grad2p[3*A2 + 2] += G2T[2];
        }
    }

    Vherm->grad(D2, Grad2, Grad2, false);
    return Grad2;
}
    
// TODO
#if 0
std::shared_ptr<Tensor> GridBox::ggaGradAdv(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<HashedGrid>& grid,
    const std::shared_ptr<Tensor>& v,
    double thre,
    const std::vector<std::shared_ptr<Tensor> >& G12 
    )
{
}
#endif

} // namespace lightspeed
