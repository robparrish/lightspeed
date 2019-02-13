#include <lightspeed/blurbox.hpp>
#include <lightspeed/am.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/resource_list.hpp>
#include <lightspeed/pair_list.hpp>
#include <lightspeed/pure_transform.hpp>
#include <lightspeed/rys.hpp>
#include <lightspeed/hermite.hpp>
#include <cmath>
#include <cstring>
#include <cstdio>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace lightspeed { 

std::shared_ptr<Tensor> BlurBox::potentialGradSR3(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<Tensor>& xyzc,
    double omega,
    double threoe,
    const std::shared_ptr<Tensor>& G12
    )
{
    // Throws if pairlist12 is not symmetric
    if (!pairlist->is_symmetric()) {
        throw std::runtime_error("BlurBox::potentialGradSR3: Pairlist should be symmetric.");
    }

    // Size of G12
    std::vector<size_t> dim;
    dim.push_back(pairlist->basis1()->natom());
    dim.push_back(3);
        
    // The working register of G12
    std::shared_ptr<Tensor> G12T = G12;
    if (!G12) {
        G12T = std::shared_ptr<Tensor>(new Tensor(dim));
    }
    G12T->shape_error(dim);
    
    // Required blur exponent
    double alpha = 2.0 * omega * omega;

    // Spherical to Cartesian transform
    std::shared_ptr<Tensor> D2 = PureTransform::pureToCart2(
        pairlist->basis1(),
        pairlist->basis2(),
        D);  

    // Hermite representation of basis pairs
    std::shared_ptr<Hermite> Dhermite(new Hermite(pairlist)); 
    std::shared_ptr<Hermite> Vhermite(new Hermite(pairlist)); 

    Dhermite->cart_to_herm(D2, false);

    // Grab Rys singleton
    std::shared_ptr<Rys> rys = Rys::instance();
    const Rys * rysp = rys.get();

    // Extract xyzc data
    size_t nA = xyzc->shape()[0];
    const double* xyzcp = xyzc->data().data();

    // Blur exponent for charges (full blur)
    double dAinv = 1.0 / alpha;

    // Magic masks for extracting Hermite tx,ty,tz
    const int xmask = (1 << 30) - (1 << 20);
    const int ymask = (1 << 20) - (1 << 10);
    const int zmask = (1 << 10) - (1 <<  0);

    // local copies of G on each thread
    std::vector<std::shared_ptr<Tensor> > G_omp;
    for (int tid = 0; tid < resources->nthread(); tid++) {
        G_omp.push_back(std::shared_ptr<Tensor>(new Tensor(dim)));
    }

    for (size_t ind_L = 0; ind_L < Dhermite->hermites().size(); ind_L++) {
        HermiteL& herm = Dhermite->hermites()[ind_L];
        int L = herm.L();
        int V = L+1;
        int H = herm.H();
        size_t npair = herm.npair();
        const double* geomp = herm.geom().data();
        const double* Ddatap = herm.data().data();
        const double* etap = herm.etas().data();
        const int* indp = herm.inds().data();
        double* datap = Vhermite->hermites()[ind_L].data().data();
        int nrys = (L+1)/2 + 1;
        std::vector<int> angl = HermiteUtil::angl(L);
        // Max Hermite coefficient per shell
        std::vector<double> dmax = HermiteUtil::dmax_bra(
            herm);

        #pragma omp parallel for schedule(dynamic) num_threads(resources->nthread())
        for (size_t pair_ind = 0; pair_ind < npair; pair_ind++) {

            // raw pointers for each thread
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            double* Gp = G_omp[tid]->data().data();

            double pP = geomp[4*pair_ind + 0];
            double xP = geomp[4*pair_ind + 1];
            double yP = geomp[4*pair_ind + 2];
            double zP = geomp[4*pair_ind + 3];
            double dP = dmax[pair_ind];
            if (dP == 0.0) continue;

            double eta1 = etap[2*pair_ind + 0];
            double eta2 = etap[2*pair_ind + 1];
            int A1 = indp[2*pair_ind + 0];
            int A2 = indp[2*pair_ind + 1];

            // Blur exponent for Hermite Gaussians [tilde type]
            double dPinv = std::max((pP - alpha) / (pP * alpha), 0.0);
            // AP-specific range-separation omega [tilde type]
            double omegaPA = pow(dAinv + dPinv,-0.5);
    
            // Blur exponent for Hermite Gaussians [bar type]
            double aPbar = std::min(pP, alpha);
            // AP-specific range-separation omega [bar type]
            double omegaPAbar = sqrt(aPbar * alpha / (aPbar + alpha));

            // Target
            double Vbuf[H];
            ::memset(Vbuf,'\0',sizeof(double)*H);

            double Dbuf[H];
            ::memcpy(Dbuf, Ddatap + pair_ind*H, sizeof(double)*H);

            for (size_t A = 0; A < nA; A++) {
                double xA = xyzcp[4*A + 0];
                double yA = xyzcp[4*A + 1];
                double zA = xyzcp[4*A + 2];
                double cA = xyzcp[4*A + 3];

                double xPA = xP - xA;
                double yPA = yP - yA;
                double zPA = zP - zA;
                double rPA2 = pow(xPA,2) + pow(yPA,2) + pow(zPA,2);

                // True Thresholding 
                double tPA2 = omegaPAbar * omegaPAbar * rPA2;
                double screenP = dP * pow(M_PI / pP, 1.5) * omegaPAbar;
                double Vest = screenP * cA * exp(-tPA2) * pow(M_PI,-0.5);
                if (fabs(Vest) < threoe * tPA2) continue;
    
                { // Full-range
                    double pref = 2.0 * M_PI / pP;
                    double T = pP * rPA2;
                    for (int i = 0; i < nrys; i++) {
                        double t = rysp->interpolate_t(nrys, i, T);
                        double w = rysp->interpolate_w(nrys, i, T);
                        double t2 = t*t;
                        double U = -2.0 * pP * t2;
                        double Hx[V+1];
                        double Hy[V+1];
                        double Hz[V+1];
                        Hx[0] = -pref*w*cA;
                        Hy[0] = 1.0;
                        Hz[0] = 1.0;
                        Hx[1] = U * xPA * Hx[0];
                        Hy[1] = U * yPA;
                        Hz[1] = U * zPA;
                        for (int u = 1; u < L+1; u++) {
                            Hx[u+1] = U * (xPA*Hx[u] + u*Hx[u-1]);
                            Hy[u+1] = U * (yPA*Hy[u] + u*Hy[u-1]);
                            Hz[u+1] = U * (zPA*Hz[u] + u*Hz[u-1]);

                        }
                        for (int t_ind = 0; t_ind < H; t_ind++) {
                            int t_vec = angl[t_ind];
                            int tx = (t_vec & xmask) >> 20;
                            int ty = (t_vec & ymask) >> 10;
                            int tz = (t_vec & zmask);
                            Vbuf[t_ind] += Hx[tx] * Hy[ty] * Hz[tz];

                            Gp[3*A1 + 0] += Dbuf[t_ind] * Hx[tx+1] * Hy[ty]   * Hz[tz]   * eta1;
                            Gp[3*A1 + 1] += Dbuf[t_ind] * Hx[tx]   * Hy[ty+1] * Hz[tz]   * eta1;
                            Gp[3*A1 + 2] += Dbuf[t_ind] * Hx[tx]   * Hy[ty]   * Hz[tz+1] * eta1;

                            Gp[3*A2 + 0] += Dbuf[t_ind] * Hx[tx+1] * Hy[ty]   * Hz[tz]   * eta2;
                            Gp[3*A2 + 1] += Dbuf[t_ind] * Hx[tx]   * Hy[ty+1] * Hz[tz]   * eta2;
                            Gp[3*A2 + 2] += Dbuf[t_ind] * Hx[tx]   * Hy[ty]   * Hz[tz+1] * eta2;
                        }
                    }
                } // End Full-range
                { // Long-range
                    double d2 = omegaPA*omegaPA / (pP + omegaPA*omegaPA);
                    double pref = 2.0 * M_PI / pP * sqrt(d2); 
                    double T = pP * d2 * rPA2;
                    for (int i = 0; i < nrys; i++) {
                        double t = rysp->interpolate_t(nrys, i, T);
                        double w = rysp->interpolate_w(nrys, i, T);
                        double t2 = t*t;
                        double U = -2.0 * pP * d2 * t2;
                        double Hx[V+1];
                        double Hy[V+1];
                        double Hz[V+1];
                        Hx[0] = -pref*w*cA;
                        Hy[0] = 1.0;
                        Hz[0] = 1.0;
                        Hx[1] = U * xPA * Hx[0];
                        Hy[1] = U * yPA;
                        Hz[1] = U * zPA;
                        for (int u = 1; u < L+1; u++) {
                            Hx[u+1] = U * (xPA*Hx[u] + u*Hx[u-1]);
                            Hy[u+1] = U * (yPA*Hy[u] + u*Hy[u-1]);
                            Hz[u+1] = U * (zPA*Hz[u] + u*Hz[u-1]);

                        }
                        for (int t_ind = 0; t_ind < H; t_ind++) {
                            int t_vec = angl[t_ind];
                            int tx = (t_vec & xmask) >> 20;
                            int ty = (t_vec & ymask) >> 10;
                            int tz = (t_vec & zmask);
                            Vbuf[t_ind] -= Hx[tx] * Hy[ty] * Hz[tz];

                            Gp[3*A1 + 0] -= Dbuf[t_ind] * Hx[tx+1] * Hy[ty]   * Hz[tz]   * eta1;
                            Gp[3*A1 + 1] -= Dbuf[t_ind] * Hx[tx]   * Hy[ty+1] * Hz[tz]   * eta1;
                            Gp[3*A1 + 2] -= Dbuf[t_ind] * Hx[tx]   * Hy[ty]   * Hz[tz+1] * eta1;

                            Gp[3*A2 + 0] -= Dbuf[t_ind] * Hx[tx+1] * Hy[ty]   * Hz[tz]   * eta2;
                            Gp[3*A2 + 1] -= Dbuf[t_ind] * Hx[tx]   * Hy[ty+1] * Hz[tz]   * eta2;
                            Gp[3*A2 + 2] -= Dbuf[t_ind] * Hx[tx]   * Hy[ty]   * Hz[tz+1] * eta2;
                        }
                    }
                } // End Long-range
            }

            double* data2p = datap + pair_ind*H;
            for (int t_ind = 0; t_ind < H; t_ind++) {
                (*data2p++) += Vbuf[t_ind];
            }
        }
    }

    // collect gradient from each thread
    for (int tid = 0; tid < resources->nthread(); tid++) {
        for (int i = 0; i < G12T->size(); i++) {
            G12T->data()[i] += G_omp[tid]->data()[i];
        }
    }

    // gradient contribution from Hermite coefficients
    Vhermite->grad(D2, G12T, G12T, false);

    return G12T;
}

} // namespace lightspeed
