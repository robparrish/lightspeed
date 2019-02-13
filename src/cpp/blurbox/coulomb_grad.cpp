#include <lightspeed/blurbox.hpp>
#include <lightspeed/intbox.hpp>
#include <lightspeed/basis.hpp>
#include <lightspeed/pair_list.hpp>
#include <lightspeed/pure_transform.hpp>
#include <lightspeed/resource_list.hpp>
#include <lightspeed/rys.hpp>
#include <lightspeed/hermite.hpp>
#include <cmath>
#include <cstring>
#include <cstdio>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace lightspeed {
        
std::shared_ptr<Tensor> BlurBox::coulombGradSR3(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist12,
    const std::shared_ptr<PairList>& pairlist34,
    const std::shared_ptr<Tensor>& D12,
    const std::shared_ptr<Tensor>& D34,
    double omega,
    double threte,
    const std::shared_ptr<Tensor>& G12
    )
{
    // Throws if pairlist12 is not symmetric
    if (!pairlist12->is_symmetric()) {
        throw std::runtime_error("BlurBox::coulombGradSR3: Pairlist12 should be symmetric.");
    }

    // size of G12
    std::vector<size_t> dim;
    dim.push_back(pairlist12->basis1()->natom());
    dim.push_back(3);

    // The working register of G
    std::shared_ptr<Tensor> G12T = G12;
    if (!G12) {
        G12T = std::shared_ptr<Tensor>(new Tensor(dim));
    }
    G12T->shape_error(dim);

    // Required blur exponent
    double alpha = 2.0 * omega * omega;

    // Cartesian representation of D34
    std::shared_ptr<Tensor> D34T = PureTransform::pureToCart2(
        pairlist34->basis1(),
        pairlist34->basis2(),
        D34);

    // Cartesian representation of D12
    std::shared_ptr<Tensor> D12T = PureTransform::pureToCart2(
        pairlist12->basis1(),
        pairlist12->basis2(),
        D12);

    // Find the max element in the bra
    float max12 = PairListUtil::max_bound(pairlist12);

    // Truncate the ket (including density)
    std::shared_ptr<PairList> pairlist34T = PairListUtil::truncate_ket(
        pairlist34,
        D34T,
        threte / max12);
        
    // Find the max element in the ket (including density)
    float max34 = PairListUtil::max_bound(pairlist34T);

    // Truncate the bra
    std::shared_ptr<PairList> pairlist12T = PairListUtil::truncate_bra(
        pairlist12,
        threte / max34);

    // Hermite representations
    std::shared_ptr<Hermite> Dherms12(new Hermite(pairlist12T));
    std::shared_ptr<Hermite> Vherms12(new Hermite(pairlist12T));
    std::shared_ptr<Hermite> Dherms34(new Hermite(pairlist34T));

    // Cart-to-Hermite transform of D34
    // Will manually take care of phases
    Dherms12->cart_to_herm(D12T, false);
    Dherms34->cart_to_herm(D34T, false);

    // Grab Rys singleton
    std::shared_ptr<Rys> rys = Rys::instance();
    const Rys * rysp = rys.get();
      
    // Magic masks for extracting Hermite tx,ty,tz
    const int xmask = (1 << 30) - (1 << 20);
    const int ymask = (1 << 20) - (1 << 10);
    const int zmask = (1 << 10) - (1 <<  0);

    // local copies of G on each thread
    std::vector<std::shared_ptr<Tensor> > G_omp;
    for (int tid = 0; tid < resources->nthread(); tid++) {
        G_omp.push_back(std::shared_ptr<Tensor>(new Tensor(dim)));
    }

    // Ket loop
    for (size_t ind_L34 = 0; ind_L34 < Dherms34->hermites().size(); ind_L34++) {
        HermiteL& herm34 = Dherms34->hermites()[ind_L34];
        int L34 = herm34.L();
        int V34 = L34+1;
        int H34 = herm34.H();
        size_t npair34 = herm34.npair();
        if (npair34 == 0) continue;
        const double* geom34p = herm34.geom().data();
        const double* Ddata34p = herm34.data().data();
        const float* bound34p = herm34.bounds().data();
        std::vector<double> dmax34 = HermiteUtil::dmax_ket(
            herm34);
    
        double p34min = alpha;
        double eta34max = 0.0;
        std::vector<double> eta34 = dmax34;
        for (size_t ind34 = 0; ind34 < npair34; ind34++) {
            eta34[ind34] *= pow(M_PI / geom34p[4*ind34 + 0],3.0/2.0);
            eta34max = std::max(eta34max, eta34[ind34]);
            p34min = std::min(p34min,geom34p[4*ind34 + 0]);
        }

    // Bra loop
    for (size_t ind_L12 = 0; ind_L12 < Dherms12->hermites().size(); ind_L12++) {
        HermiteL& herm12 = Dherms12->hermites()[ind_L12];
        int L12 = herm12.L();
        int V12 = L12+1;
        int H12 = herm12.H();
        size_t npair12 = herm12.npair();
        if (npair12 == 0) continue;
        const double* geom12p = herm12.geom().data();
        const double* Ddata12p = herm12.data().data();
        double* data12p = Vherms12->hermites()[ind_L12].data().data();
        const float* bound12p = herm12.bounds().data();
        const int* index12p = herm12.inds().data();
        const double* eta12p = herm12.etas().data();
        std::vector<double> dmax12 = HermiteUtil::dmax_bra(
            herm12);

        double p12min = alpha;
        double eta12max = 0.0;
        std::vector<double> eta12 = dmax12;
        for (size_t ind12 = 0; ind12 < npair12; ind12++) {
            eta12[ind12] *= pow(M_PI / geom12p[4*ind12 + 0],3.0/2.0);
            eta12max = std::max(eta12max, eta12[ind12]);
            p12min = std::min(p12min,geom12p[4*ind12 + 0]);
        }
    
        int L = L12 + L34;
        int V = L12 + L34 + 1;
        int nrys = (L12 + L34 + 1)/2 + 1;
        std::vector<int> angl = HermiteUtil::angl(std::max(L12,L34));

        #pragma omp parallel for schedule(dynamic) num_threads(resources->nthread())
        for (size_t pair12_ind = 0; pair12_ind < npair12; pair12_ind++) {

            // raw pointers for each thread
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            double* Gp = G_omp[tid]->data().data();

            double p12 = geom12p[4*pair12_ind + 0];
            double x12 = geom12p[4*pair12_ind + 1];
            double y12 = geom12p[4*pair12_ind + 2];
            double z12 = geom12p[4*pair12_ind + 3];
            double d12 = dmax12[pair12_ind];
            if (d12 == 0.0) continue;

            int A1 = index12p[2*pair12_ind + 0];
            int A2 = index12p[2*pair12_ind + 1];
            double e1 = eta12p[2*pair12_ind + 0];
            double e2 = eta12p[2*pair12_ind + 1];

            // Target
            double j12p[H12];
            ::memset(j12p,'\0',sizeof(double)*H12);

            double d12p[H12];
            ::memcpy(d12p, Ddata12p + pair12_ind*H12, H12*sizeof(double));

            // Blur exponent for Hermite Gaussians [tilde type]
            double dP12inv = std::max((p12 - alpha) / (p12 * alpha), 0.0);
            // Blur exponent for Hermite Gaussians [bar type]
            double p12bar = std::min(p12, alpha);

            for (size_t pair34_ind = 0; pair34_ind < npair34; pair34_ind++) {
                double p34 = geom34p[4*pair34_ind + 0];
                double x34 = geom34p[4*pair34_ind + 1];
                double y34 = geom34p[4*pair34_ind + 2];
                double z34 = geom34p[4*pair34_ind + 3];

                double xPQ = x12 - x34;
                double yPQ = y12 - y34;
                double zPQ = z12 - z34;
                double rPQ_2 = pow(xPQ,2) + pow(yPQ,2) + pow(zPQ,2);

                // wFTC Covers diffuse-diffuse interactions
                if (p12 < alpha && p34 < alpha) continue;

                // Blur exponent for Hermite Gaussians [tilde type]
                double dP34inv = std::max((p34 - alpha) / (p34 * alpha), 0.0);
                // Blur exponent for Hermite Gaussians [bar type]
                double p34bar = std::min(p34, alpha);

                // Range-separation factor [bar type]
                double omegaPQbar = sqrt(p12bar * p34bar / (p12bar + p34bar));
                
                // True Thresholding
                double tPQ2 = omegaPQbar * omegaPQbar * rPQ_2;
                double Vest = eta12[pair12_ind] * eta34[pair34_ind] * omegaPQbar * exp(-tPQ2) * pow(M_PI,-0.5);
                if (fabs(Vest) < threte * tPQ2) continue;

                double omegaPQtilde = pow(dP12inv + dP34inv,-0.5);

                double d34p[H34];
                ::memcpy(d34p, Ddata34p + pair34_ind*H34, H34*sizeof(double));
    
                { // Full-range 
                    double pref = 2.0 * pow(M_PI, 2.5) / (p12 * p34 * sqrt(p12 + p34));
                    double rho = p12 * p34 / (p12 + p34);
                    double T = rho * rPQ_2;
                    for (int i = 0; i < nrys; i++) {
                        double t = rysp->interpolate_t(nrys, i, T);
                        double w = rysp->interpolate_w(nrys, i, T);
                        double t2 = t*t;
                        double U = -2.0 * rho * t2;
                        double Hx[V+1];
                        double Hy[V+1];
                        double Hz[V+1];
                        Hx[0] = pref*w;
                        Hy[0] = 1.0;
                        Hz[0] = 1.0;
                        Hx[1] = U * xPQ * Hx[0];
                        Hy[1] = U * yPQ;
                        Hz[1] = U * zPQ;
                        for (int u = 1; u < L+1; u++) {
                            Hx[u+1] = U * (xPQ*Hx[u] + u*Hx[u-1]);
                            Hy[u+1] = U * (yPQ*Hy[u] + u*Hy[u-1]);
                            Hz[u+1] = U * (zPQ*Hz[u] + u*Hz[u-1]);
                        }
                        for (int t_ind = 0; t_ind < H12; t_ind++) {
                            int t_vec = angl[t_ind];
                            int tx = (t_vec & xmask) >> 20;
                            int ty = (t_vec & ymask) >> 10;
                            int tz = (t_vec & zmask);
                            for (int u_ind = 0; u_ind < H34; u_ind++) {
                                int u_vec = angl[u_ind];
                                int ux = (u_vec & xmask) >> 20;
                                int uy = (u_vec & ymask) >> 10;
                                int uz = (u_vec & zmask);

                                // phase: (-1)^u
                                int phase = 1 - 2*((ux+uy+uz)&1);

                                j12p[t_ind] += phase * Hx[tx+ux] * Hy[ty+uy] * Hz[tz+uz] * d34p[u_ind];

                                Gp[3*A1 + 0] += phase * d12p[t_ind] * Hx[tx+ux+1] * Hy[ty+uy]   * Hz[tz+uz]   * d34p[u_ind] * e1;
                                Gp[3*A1 + 1] += phase * d12p[t_ind] * Hx[tx+ux]   * Hy[ty+uy+1] * Hz[tz+uz]   * d34p[u_ind] * e1;
                                Gp[3*A1 + 2] += phase * d12p[t_ind] * Hx[tx+ux]   * Hy[ty+uy]   * Hz[tz+uz+1] * d34p[u_ind] * e1;

                                Gp[3*A2 + 0] += phase * d12p[t_ind] * Hx[tx+ux+1] * Hy[ty+uy]   * Hz[tz+uz]   * d34p[u_ind] * e2;
                                Gp[3*A2 + 1] += phase * d12p[t_ind] * Hx[tx+ux]   * Hy[ty+uy+1] * Hz[tz+uz]   * d34p[u_ind] * e2;
                                Gp[3*A2 + 2] += phase * d12p[t_ind] * Hx[tx+ux]   * Hy[ty+uy]   * Hz[tz+uz+1] * d34p[u_ind] * e2;
                            }
                        }
                    } 
                } // End Full-range
                { // Long-range 
                    double pref = 2.0 * pow(M_PI, 2.5) / (p12 * p34 * sqrt(p12 + p34));
                    double rho = p12 * p34 / (p12 + p34);
                    double d2 = omegaPQtilde * omegaPQtilde / (rho + omegaPQtilde * omegaPQtilde);
                    pref *= sqrt(d2);
                    double T = rho * d2 * rPQ_2;
                    for (int i = 0; i < nrys; i++) {
                        double t = rysp->interpolate_t(nrys, i, T);
                        double w = rysp->interpolate_w(nrys, i, T);
                        double t2 = t*t;
                        double U = -2.0 * rho * d2 * t2;
                        double Hx[V+1];
                        double Hy[V+1];
                        double Hz[V+1];
                        Hx[0] = pref*w;
                        Hy[0] = 1.0;
                        Hz[0] = 1.0;
                        Hx[1] = U * xPQ * Hx[0];
                        Hy[1] = U * yPQ;
                        Hz[1] = U * zPQ;
                        for (int u = 1; u < L+1; u++) {
                            Hx[u+1] = U * (xPQ*Hx[u] + u*Hx[u-1]);
                            Hy[u+1] = U * (yPQ*Hy[u] + u*Hy[u-1]);
                            Hz[u+1] = U * (zPQ*Hz[u] + u*Hz[u-1]);
                        }
                        for (int t_ind = 0; t_ind < H12; t_ind++) {
                            int t_vec = angl[t_ind];
                            int tx = (t_vec & xmask) >> 20;
                            int ty = (t_vec & ymask) >> 10;
                            int tz = (t_vec & zmask);
                            for (int u_ind = 0; u_ind < H34; u_ind++) {
                                int u_vec = angl[u_ind];
                                int ux = (u_vec & xmask) >> 20;
                                int uy = (u_vec & ymask) >> 10;
                                int uz = (u_vec & zmask);

                                // phase: (-1)^u
                                int phase = 1 - 2*((ux+uy+uz)&1);

                                j12p[t_ind] -= phase * Hx[tx+ux] * Hy[ty+uy] * Hz[tz+uz] * d34p[u_ind]; 

                                Gp[3*A1 + 0] -= phase * d12p[t_ind] * Hx[tx+ux+1] * Hy[ty+uy]   * Hz[tz+uz]   * d34p[u_ind] * e1;
                                Gp[3*A1 + 1] -= phase * d12p[t_ind] * Hx[tx+ux]   * Hy[ty+uy+1] * Hz[tz+uz]   * d34p[u_ind] * e1;
                                Gp[3*A1 + 2] -= phase * d12p[t_ind] * Hx[tx+ux]   * Hy[ty+uy]   * Hz[tz+uz+1] * d34p[u_ind] * e1;

                                Gp[3*A2 + 0] -= phase * d12p[t_ind] * Hx[tx+ux+1] * Hy[ty+uy]   * Hz[tz+uz]   * d34p[u_ind] * e2;
                                Gp[3*A2 + 1] -= phase * d12p[t_ind] * Hx[tx+ux]   * Hy[ty+uy+1] * Hz[tz+uz]   * d34p[u_ind] * e2;
                                Gp[3*A2 + 2] -= phase * d12p[t_ind] * Hx[tx+ux]   * Hy[ty+uy]   * Hz[tz+uz+1] * d34p[u_ind] * e2;
                            }
                        }
                    } 
                } // End Long-range
            }

            double* data12Tp = data12p + pair12_ind*H12;
            for (int t_ind = 0; t_ind < H12; t_ind++) {
                (*data12Tp++) += j12p[t_ind];
            }
        }

    }} // Bra/Ket Loops

    // collect gradient from each thread
    for (int tid = 0; tid < resources->nthread(); tid++) {
        for (int i = 0; i < G12T->size(); i++) {
            G12T->data()[i] += G_omp[tid]->data()[i];
        }
    }
    
    // gradient contribution from Hermite coefficients
    Vherms12->grad(D12T, G12T, G12T, false);

    return G12T;
}

} // namespace lightspeed
