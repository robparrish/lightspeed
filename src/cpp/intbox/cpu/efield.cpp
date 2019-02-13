#include <iostream>
#include <lightspeed/intbox.hpp>
#include <lightspeed/am.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/resource_list.hpp>
#include <lightspeed/pair_list.hpp>
#include <lightspeed/pure_transform.hpp>
#include <lightspeed/ewald.hpp>
#include <lightspeed/hermite.hpp>
#include <lightspeed/gh.hpp>
#include <lightspeed/rys.hpp>
#include <cmath>
#include <cstring>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace lightspeed { 

std::shared_ptr<Tensor> IntBox::fieldCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<Tensor>& xyz,
    const std::shared_ptr<Tensor>& F
    )
{

    // Process points on which to probe the field
    size_t npoints = xyz->shape()[0];
    std::vector<size_t> dim;
    dim.push_back(npoints);
    dim.push_back(3);
    xyz->shape_error(dim);
    const double* xyzp = xyz->data().data();

    // The working register of F
    std::shared_ptr<Tensor> F2 = F;
    if (!F) {
        F2 = std::shared_ptr<Tensor>(new Tensor(dim));
    }
    F2->shape_error(dim);
    
    // => Setup <= //
    // Spherical to Cartesian transform
    std::shared_ptr<Tensor> D2 = PureTransform::pureToCart2(
        pairlist->basis1(),
        pairlist->basis2(),
        D);  

    // Hermite representation of basis pairs
    std::shared_ptr<Hermite> Vhermite(new Hermite(pairlist)); 
    std::shared_ptr<Hermite> Dhermite(new Hermite(pairlist)); 

    // Transformation of D from cartesians to Hermite
    Dhermite->cart_to_herm(D2, false);

    std::vector<HermiteL>& Vherms = Vhermite->hermites();
    std::vector<HermiteL>& Dherms = Dhermite->hermites();

    // Grab Rys singleton
    std::shared_ptr<Rys> rys = Rys::instance();
    const Rys* rysp = rys.get();

    // Magic masks for extracting Hermite tx,ty,tz
    const int xmask = (1 << 30) - (1 << 20);
    const int ymask = (1 << 20) - (1 << 10);
    const int zmask = (1 << 10) - (1 <<  0);

    // local copies of F on each thread
    std::vector<std::shared_ptr<Tensor> > F_omp;
    for (int tid = 0; tid < resources->nthread(); tid++) {
        F_omp.push_back(std::shared_ptr<Tensor>(new Tensor(dim)));
    }

    // => Master Loop <= //

    for (size_t ewald_ind = 0; ewald_ind < ewald->scales().size(); ewald_ind++) {
        double alpha = ewald->scales()[ewald_ind];
        double omega = ewald->omegas()[ewald_ind];
        if ((alpha == 0.0) || (omega == 0.0)) continue;

        for (size_t ind_L = 0; ind_L < Vherms.size(); ind_L++) {
            HermiteL& Vherm = Vherms[ind_L];
            HermiteL& Dherm = Dherms[ind_L];
            int L = Vherm.L();
            int H = Vherm.H();
            
            size_t npair = Vherm.npair();
            double* datap = Vherm.data().data();
            const double* Ddatap = Dherm.data().data();
            const double* geomp = Vherm.geom().data();

            int nrys = (L + 1)/2 + 1;
            std::vector<int> angl = HermiteUtil::angl(H);
            
            #pragma omp parallel for schedule(dynamic) num_threads(resources->nthread())
            for (size_t pair_ind = 0; pair_ind < npair; pair_ind++) {

                // raw pointers for each thread
#ifdef _OPENMP
                int tid = omp_get_thread_num();
#else
                int tid = 0;
#endif
                double* Fp = F_omp[tid]->data().data();
                
                // Geometry
                double pP = geomp[4*pair_ind + 0];
                double xP = geomp[4*pair_ind + 1];
                double yP = geomp[4*pair_ind + 2];
                double zP = geomp[4*pair_ind + 3];

                double pref = alpha * 2.0 * M_PI / pP;
                double d2 = 1.0;
                if (omega != -1.0){
                    d2 = omega*omega / (pP + omega*omega);
                    pref *= sqrt(d2);
                }

                double Dbuf[H];
                ::memcpy(Dbuf, Ddatap + pair_ind*H, sizeof(double)*H);

                for (size_t A = 0; A < npoints; A++) {
                    double xA = xyzp[3*A + 0];
                    double yA = xyzp[3*A + 1];
                    double zA = xyzp[3*A + 2];

                    double xPA = xP - xA;
                    double yPA = yP - yA;
                    double zPA = zP - zA;
                    double rPA2 = pow(xPA,2) + pow(yPA,2) + pow(zPA,2);
                    double T = pP * d2 * rPA2;

                    for (int i = 0; i < nrys; i++) {
                        double t = rysp->interpolate_t(nrys, i, T);
                        double w = rysp->interpolate_w(nrys, i, T);
                        double t2 = t*t;
                        double U = -2.0 * pP * d2 * t2;
                        double Hx[L + 2];
                        double Hy[L + 2];
                        double Hz[L + 2];
                        Hx[0] = -pref*w;
                        Hy[0] = 1.0;
                        Hz[0] = 1.0;
                        Hx[1] = U * xPA * Hx[0];
                        Hy[1] = U * yPA;
                        Hz[1] = U * zPA;
                        for (int u = 1; u < L + 1; u++) {
                            Hx[u+1] = U * (xPA*Hx[u] + u*Hx[u-1]);
                            Hy[u+1] = U * (yPA*Hy[u] + u*Hy[u-1]);
                            Hz[u+1] = U * (zPA*Hz[u] + u*Hz[u-1]);
                        }
                        for (int t_ind = 0; t_ind < H; t_ind++) {
                            int t_vec = angl[t_ind];
                            int tx = (t_vec & xmask) >> 20;
                            int ty = (t_vec & ymask) >> 10;
                            int tz = (t_vec & zmask);

                            Fp[3*A  + 0] += Dbuf[t_ind] * Hx[tx+1] * Hy[ty]   * Hz[tz];
                            Fp[3*A  + 1] += Dbuf[t_ind] * Hx[tx]   * Hy[ty+1] * Hz[tz];
                            Fp[3*A  + 2] += Dbuf[t_ind] * Hx[tx]   * Hy[ty]   * Hz[tz+1];
                        }
                    }
                }
            }
        }
    }

    // collect field from each thread
    for (int tid = 0; tid < resources->nthread(); tid++) {
        for (int i = 0; i < F2->size(); i++) {
            F2->data()[i] += F_omp[tid]->data()[i];
        }
    }

    return F2;
}

} //namespace lightspeed
