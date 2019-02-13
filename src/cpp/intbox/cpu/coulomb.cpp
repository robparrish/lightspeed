#include <lightspeed/intbox.hpp>
#include <lightspeed/basis.hpp>
#include <lightspeed/pair_list.hpp>
#include <lightspeed/pure_transform.hpp>
#include <lightspeed/resource_list.hpp>
#include <lightspeed/ewald.hpp>
#include <lightspeed/rys.hpp>
#include <lightspeed/hermite.hpp>
#include <cmath>
#include <cstring>

namespace lightspeed {
        
std::shared_ptr<Tensor> IntBox::coulombCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist12,
    const std::shared_ptr<PairList>& pairlist34,
    const std::shared_ptr<Tensor>& D34,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& J12
    )
{
    // Cartesian representation of D34
    std::shared_ptr<Tensor> D34T = PureTransform::pureToCart2(
        pairlist34->basis1(),
        pairlist34->basis2(),
        D34);

    // Cartesian representation of J12
    std::shared_ptr<Tensor> J12T = PureTransform::allocCart2(
        pairlist12->basis1(),
        pairlist12->basis2(),
        J12);

    // Find the max element in the bra
    float max12 = PairListUtil::max_bound(pairlist12);

    // Truncate the ket (including density)
    std::shared_ptr<PairList> pairlist34T = PairListUtil::truncate_ket(
        pairlist34,
        D34T,
        thresp / max12);
        
    // Find the max element in the ket (including density)
    float max34 = PairListUtil::max_bound(pairlist34T);

    // Truncate the bra
    std::shared_ptr<PairList> pairlist12T = PairListUtil::truncate_bra(
        pairlist12,
        thresp / max34);

    // Hermite representations
    std::shared_ptr<Hermite> herms12(new Hermite(pairlist12T));
    std::shared_ptr<Hermite> herms34(new Hermite(pairlist34T));

    // Cart-to-Hermite transform of D34
    herms34->cart_to_herm(D34T, true);

    // Grab Rys singleton
    std::shared_ptr<Rys> rys = Rys::instance();
    const Rys * rysp = rys.get();
      
    // Magic masks for extracting Hermite tx,ty,tz
    const int xmask = (1 << 30) - (1 << 20);
    const int ymask = (1 << 20) - (1 << 10);
    const int zmask = (1 << 10) - (1 <<  0);

    for (size_t ewald_ind = 0; ewald_ind < ewald->scales().size(); ewald_ind++) {
        double alpha = ewald->scales()[ewald_ind];
        double omega = ewald->omegas()[ewald_ind];
        if ((alpha == 0.0) || (omega == 0.0)) continue;

        // Bra loop
        for (size_t ind_L12 = 0; ind_L12 < herms12->hermites().size(); ind_L12++) {
            HermiteL& herm12 = herms12->hermites()[ind_L12];
            int L12 = herm12.L();
            int V12 = L12+1;
            int H12 = herm12.H();
            size_t npair12 = herm12.npair();
            const double* geom12p = herm12.geom().data();
            double* data12p = herm12.data().data();
            const float* bound12p = herm12.bounds().data();;

        // Ket loop
        for (size_t ind_L34 = 0; ind_L34 < herms34->hermites().size(); ind_L34++) {
            HermiteL& herm34 = herms34->hermites()[ind_L34];
            int L34 = herm34.L();
            int V34 = L34+1;
            int H34 = herm34.H();
            size_t npair34 = herm34.npair();
            const double* geom34p = herm34.geom().data();
            const double* data34p = herm34.data().data();
            const float* bound34p = herm34.bounds().data();;
        
            int L = L12 + L34;
            int V = L12 + L34 + 1;
            int nrys = (L12 + L34)/2 + 1;
            std::vector<int> angl = HermiteUtil::angl(std::max(L12,L34));

            #pragma omp parallel for schedule(dynamic) num_threads(resources->nthread())
            for (size_t pair12_ind = 0; pair12_ind < npair12; pair12_ind++) {
                float B12 = bound12p[pair12_ind];
                double p12 = geom12p[4*pair12_ind + 0];
                double x12 = geom12p[4*pair12_ind + 1];
                double y12 = geom12p[4*pair12_ind + 2];
                double z12 = geom12p[4*pair12_ind + 3];

                double jp[H12];
                ::memset(jp,'\0',sizeof(double)*H12);

                for (size_t pair34_ind = 0; pair34_ind < npair34; pair34_ind++) {
                    float B34 = bound34p[pair34_ind];
                    if (B12 * B34 < thresp) break;
                    double p34 = geom34p[4*pair34_ind + 0];
                    double x34 = geom34p[4*pair34_ind + 1];
                    double y34 = geom34p[4*pair34_ind + 2];
                    double z34 = geom34p[4*pair34_ind + 3];

                    double dp[H34];
                    ::memcpy(dp, data34p + pair34_ind*H34,H34*sizeof(double));

                    double pref = alpha * 2.0 * pow(M_PI, 2.5) / (p12 * p34 * sqrt(p12 + p34));
                    double rho = p12 * p34 / (p12 + p34);
                    double d2 = 1.0;
                    if (omega != -1.0){
                        d2 = omega*omega / (rho + omega*omega);
                        pref *= sqrt(d2);
                    }

                    double xPQ = x12 - x34;
                    double yPQ = y12 - y34;
                    double zPQ = z12 - z34;
                    double rPQ_2 = pow(xPQ,2) + pow(yPQ,2) + pow(zPQ,2);
                    double T = rho * d2 * rPQ_2;

                    for (int i = 0; i < nrys; i++) {
                        double t = rysp->interpolate_t(nrys, i, T);
                        double w = rysp->interpolate_w(nrys, i, T);
                        double t2 = t*t;
                        double U = -2.0 * rho * d2 * t2;
                        double Hx[V];
                        double Hy[V];
                        double Hz[V];
                        Hx[0] = pref*w;
                        Hy[0] = 1.0;
                        Hz[0] = 1.0;
                        if (L > 0) {
                            Hx[1] = U * xPQ * Hx[0];
                            Hy[1] = U * yPQ;
                            Hz[1] = U * zPQ;
                        } 
                        for (int u = 1; u < L; u++) {
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
                                jp[t_ind] += Hx[tx+ux] * Hy[ty+uy] * Hz[tz+uz] * dp[u_ind]; 
                            }
                        }
                    } 
                }

                double* data12Tp = data12p + pair12_ind*H12;
                for (int t_ind = 0; t_ind < H12; t_ind++) {
                    (*data12Tp++) += jp[t_ind];
                }
            }
    
        }} // Bra/Ket loop
            
    } // Ewald loop
    
    // Hermite-to-cart transform of J12
    herms12->herm_to_cart(J12T, false);
    
    // Cartesian-to-pure transform of J12
    std::shared_ptr<Tensor> J12U = PureTransform::cartToPure2(
        pairlist12->basis1(),
        pairlist12->basis2(),
        J12T,
        J12);
    
    return J12U;
}

} // namespace lightspeed
