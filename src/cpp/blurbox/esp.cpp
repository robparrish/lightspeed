#include <lightspeed/blurbox.hpp>
#include <lightspeed/intbox.hpp>
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

namespace lightspeed { 

std::shared_ptr<Tensor> BlurBox::espSR3(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<Tensor>& xyz,
    double omega,
    double threoe,
    const std::shared_ptr<Tensor>& w
    )
{
    // Required blur exponent
    double alpha = 2.0 * omega * omega;

    // process grid points
    size_t nA = xyz->shape()[0];
    std::vector<size_t> dim3;
    dim3.push_back(nA);
    dim3.push_back(3);
    xyz->shape_error(dim3);
    const double* xyzp = xyz->data().data();

    // process w
    std::vector<size_t> dim;
    dim.push_back(nA);
    std::shared_ptr<Tensor> w2;
    if (w) {
        w->shape_error(dim);
        w2 = w;
    } else {
        w2 = std::shared_ptr<Tensor>(new Tensor(dim));
    }

    double* w2p = w2->data().data();

    // Build Hermite representation of basis pairs to accumulate into
    std::shared_ptr<Hermite> hermite(new Hermite(pairlist));

    // Pure-to-Cart transform of D
    std::shared_ptr<Tensor> D2 = PureTransform::pureToCart2(
        pairlist->basis1(),
        pairlist->basis2(),
        D);

    // Cart-to-Hermite transform of D
    hermite->cart_to_herm(D2, false);

    // Grab Rys singleton
    std::shared_ptr<Rys> rys = Rys::instance();
    const Rys* rysp = rys.get();

    // Blur exponent for charges (full blur)
    double dAinv = 1.0 / alpha;

    // Magic masks for extracting Hermite tx,ty,tz
    const int xmask = (1 << 30) - (1 << 20);
    const int ymask = (1 << 20) - (1 << 10);
    const int zmask = (1 << 10) - (1 <<  0);

    for (size_t ind_L = 0; ind_L < hermite->hermites().size(); ind_L++) {
        HermiteL& herm = hermite->hermites()[ind_L];
        int L = herm.L();
        int V = L+1;
        int H = herm.H();
        size_t npair = herm.npair();
        const double* geomp = herm.geom().data();
        double* datap = herm.data().data();
        int nrys = L/2 + 1;
        std::vector<int> angl = HermiteUtil::angl(L);
        // Max Hermite coefficient per shell
        std::vector<double> dmax = HermiteUtil::dmax_ket(
            herm);

        #pragma omp parallel for schedule(dynamic) num_threads(resources->nthread())
        for (size_t pair_ind = 0; pair_ind < npair; pair_ind++) {
            double pP = geomp[4*pair_ind + 0];
            double xP = geomp[4*pair_ind + 1];
            double yP = geomp[4*pair_ind + 2];
            double zP = geomp[4*pair_ind + 3];
            double dP = dmax[pair_ind];
            if (dP == 0.0) continue;

            // Blur exponent for Hermite Gaussians [tilde type]
            double dPinv = std::max((pP - alpha) / (pP * alpha), 0.0);
            // AP-specific range-separation omega [tilde type]
            double omegaPA = pow(dAinv + dPinv,-0.5);
    
            // Blur exponent for Hermite Gaussians [bar type]
            double aPbar = std::min(pP, alpha);
            // AP-specific range-separation omega [bar type]
            double omegaPAbar = sqrt(aPbar * alpha / (aPbar + alpha));

            double dp[H];
            ::memcpy(dp, datap + pair_ind * H, sizeof(double) * H);

            for (size_t A = 0; A < nA; A++) {

                double xA = xyzp[3*A + 0];
                double yA = xyzp[3*A + 1];
                double zA = xyzp[3*A + 2];

                double xPA = xP - xA;
                double yPA = yP - yA;
                double zPA = zP - zA;
                double rPA2 = pow(xPA,2) + pow(yPA,2) + pow(zPA,2);

                // True Thresholding 
                double tPA2 = omegaPAbar * omegaPAbar * rPA2;
                double screenP = dP * pow(M_PI / pP, 1.5) * omegaPAbar;
                double Vest = screenP * exp(-tPA2) * pow(M_PI,-0.5);
                if (fabs(Vest) < threoe * tPA2) continue;

                { // Full-range
                    double pref = 2.0 * M_PI / pP;
                    double T = pP * rPA2;
                    for (int i = 0; i < nrys; i++) {
                        double t = rysp->interpolate_t(nrys, i, T);
                        double w = rysp->interpolate_w(nrys, i, T);
                        double t2 = t*t;
                        double U = -2.0 * pP * t2;
                        double Hx[V];
                        double Hy[V];
                        double Hz[V];
                        Hx[0] = pref*w;
                        Hy[0] = 1.0;
                        Hz[0] = 1.0;
                        if (L > 0) {
                            Hx[1] = U * xPA * Hx[0];
                            Hy[1] = U * yPA;
                            Hz[1] = U * zPA;
                        } 
                        for (int u = 1; u < L; u++) {
                            Hx[u+1] = U * (xPA*Hx[u] + u*Hx[u-1]);
                            Hy[u+1] = U * (yPA*Hy[u] + u*Hy[u-1]);
                            Hz[u+1] = U * (zPA*Hz[u] + u*Hz[u-1]);

                        }
                        for (int t_ind = 0; t_ind < H; t_ind++) {
                            int t_vec = angl[t_ind];
                            int tx = (t_vec & xmask) >> 20;
                            int ty = (t_vec & ymask) >> 10;
                            int tz = (t_vec & zmask);
                            #pragma omp atomic
                            w2p[A] += Hx[tx] * Hy[ty] * Hz[tz] * dp[t_ind];
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
                        double Hx[V];
                        double Hy[V];
                        double Hz[V];
                        Hx[0] = pref*w;
                        Hy[0] = 1.0;
                        Hz[0] = 1.0;
                        if (L > 0) {
                            Hx[1] = U * xPA * Hx[0];
                            Hy[1] = U * yPA;
                            Hz[1] = U * zPA;
                        } 
                        for (int u = 1; u < L; u++) {
                            Hx[u+1] = U * (xPA*Hx[u] + u*Hx[u-1]);
                            Hy[u+1] = U * (yPA*Hy[u] + u*Hy[u-1]);
                            Hz[u+1] = U * (zPA*Hz[u] + u*Hz[u-1]);

                        }
                        for (int t_ind = 0; t_ind < H; t_ind++) {
                            int t_vec = angl[t_ind];
                            int tx = (t_vec & xmask) >> 20;
                            int ty = (t_vec & ymask) >> 10;
                            int tz = (t_vec & zmask);
                            #pragma omp atomic
                            w2p[A] -= Hx[tx] * Hy[ty] * Hz[tz] * dp[t_ind];
                        }
                    }
                } // End Long-range
            }
        }
    }

    return w2;
}

} // namespace lightspeed
