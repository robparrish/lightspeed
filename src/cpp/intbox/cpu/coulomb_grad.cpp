#include <lightspeed/intbox.hpp>
#include <lightspeed/basis.hpp>
#include <lightspeed/pair_list.hpp>
#include <lightspeed/pure_transform.hpp>
#include <lightspeed/resource_list.hpp>
#include <lightspeed/ewald.hpp>
#include <lightspeed/boys.hpp>
#include <lightspeed/rys.hpp>
#include <lightspeed/hermite.hpp>
#include <cmath>
#include <cstring>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace lightspeed {

std::shared_ptr<Tensor> IntBox::coulombGradSymmCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& G
    )
{
    // Throws if pairlist is not symmetric
    if (!pairlist->is_symmetric()) {
        throw std::runtime_error("IntBox::coulombGradSymmCPU: Pairlist should be symmetric.");
    }

    // size of G
    std::vector<size_t> dim;
    dim.push_back(pairlist->basis1()->natom());
    dim.push_back(3);

    // New working register G2
    std::shared_ptr<Tensor> G2 (new Tensor(dim));
    G2->shape_error(dim);

    // Cartesian representation of D
    std::shared_ptr<Tensor> DT = PureTransform::pureToCart2(
        pairlist->basis1(),
        pairlist->basis2(),
        D);

    // Find the max element in the bra
    float max12 = PairListUtil::max_bound(pairlist);

    // Truncate the ket (including density)
    std::shared_ptr<PairList> pairlist34T = PairListUtil::truncate_ket(
        pairlist,
        DT,
        thresp / max12);

    // Find the max element in the ket (including density)
    float max34 = PairListUtil::max_bound(pairlist34T);

    // Truncate the bra
    std::shared_ptr<PairList> pairlist12T = PairListUtil::truncate_bra(
        pairlist,
        thresp / max34);

    // Hermite representations
    std::shared_ptr<Hermite> Dherms12(new Hermite(pairlist12T));
    std::shared_ptr<Hermite> Vherms12(new Hermite(pairlist12T));
    std::shared_ptr<Hermite> Dherms34(new Hermite(pairlist34T));

    // Cart-to-Hermite transform
    // Will manually take care of phases
    Dherms12->cart_to_herm(DT, false);
    Dherms34->cart_to_herm(DT, false);

    // Grab Rys singleton
    std::shared_ptr<Rys> rys = Rys::instance();
    const Rys* rysp = rys.get();

    // Magic masks for extracting Hermite tx,ty,tz
    const int xmask = (1 << 30) - (1 << 20);
    const int ymask = (1 << 20) - (1 << 10);
    const int zmask = (1 << 10) - (1 <<  0);

    // local copies of G on each thread
    std::vector<std::shared_ptr<Tensor> > G_omp;
    for (int tid = 0; tid < resources->nthread(); tid++) {
        G_omp.push_back(std::shared_ptr<Tensor>(new Tensor(dim)));
    }

    // ==> Master Loop <== //

    for (size_t ewald_ind = 0; ewald_ind < ewald->scales().size(); ewald_ind++) {
        double alpha = ewald->scales()[ewald_ind];
        double omega = ewald->omegas()[ewald_ind];
        if ((alpha == 0.0) || (omega == 0.0)) continue;

        // Bra loop
        for (size_t ind_L12 = 0; ind_L12 < Dherms12->hermites().size(); ind_L12++) {
            HermiteL& herm12 = Dherms12->hermites()[ind_L12];
            int L12 = herm12.L();
            int V12 = L12+1;
            int H12 = herm12.H();
            size_t npair12 = herm12.npair();
            const double* geom12p = herm12.geom().data();
            const float* bound12p = herm12.bounds().data();
            const int* index12p = herm12.inds().data();
            const double* eta12p = herm12.etas().data();

            const double* Ddata12p = herm12.data().data();
            double* data12p = Vherms12->hermites()[ind_L12].data().data();

        // Ket loop
        for (size_t ind_L34 = 0; ind_L34 < Dherms34->hermites().size(); ind_L34++) {
            HermiteL& herm34 = Dherms34->hermites()[ind_L34];
            int L34 = herm34.L();
            int V34 = L34+1;
            int H34 = herm34.H();
            size_t npair34 = herm34.npair();
            const double* geom34p = herm34.geom().data();
            const float* bound34p = herm34.bounds().data();
            const int* index34p = herm34.inds().data();
            const double* eta34p = herm34.etas().data();

            const double* Ddata34p = herm34.data().data();
        
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

                float B12 = bound12p[pair12_ind];
                double p12 = geom12p[4*pair12_ind + 0];
                double x12 = geom12p[4*pair12_ind + 1];
                double y12 = geom12p[4*pair12_ind + 2];
                double z12 = geom12p[4*pair12_ind + 3];

                int A1 = index12p[2*pair12_ind + 0];
                int A2 = index12p[2*pair12_ind + 1];
                double e1 = eta12p[2*pair12_ind + 0];
                double e2 = eta12p[2*pair12_ind + 1];

                double j12p[H12];
                ::memset(j12p,'\0',sizeof(double)*H12);

                double d12p[H12];
                ::memcpy(d12p, Ddata12p + pair12_ind*H12, H12*sizeof(double));

                for (size_t pair34_ind = 0; pair34_ind < npair34; pair34_ind++) {
                    float B34 = bound34p[pair34_ind];
                    if (B12 * B34 < thresp) break;
                    double p34 = geom34p[4*pair34_ind + 0];
                    double x34 = geom34p[4*pair34_ind + 1];
                    double y34 = geom34p[4*pair34_ind + 2];
                    double z34 = geom34p[4*pair34_ind + 3];

                    double d34p[H34];
                    ::memcpy(d34p, Ddata34p + pair34_ind*H34, H34*sizeof(double));

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
                        double U = -2.0 * rho * d2 * t * t;
                        double Hx[V+1];
                        double Hy[V+1];
                        double Hz[V+1];
                        Hx[0] = 1.0;
                        Hy[0] = 1.0;
                        Hz[0] = 1.0;
                        Hx[1] = U * xPQ;
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

                                j12p[t_ind] += pref * w * phase * Hx[tx+ux] * Hy[ty+uy] * Hz[tz+uz] * d34p[u_ind];

                                Gp[3*A1 + 0] += pref * w * phase * d12p[t_ind] * Hx[tx+ux+1] * Hy[ty+uy]   * Hz[tz+uz]   * d34p[u_ind] * e1;
                                Gp[3*A1 + 1] += pref * w * phase * d12p[t_ind] * Hx[tx+ux]   * Hy[ty+uy+1] * Hz[tz+uz]   * d34p[u_ind] * e1;
                                Gp[3*A1 + 2] += pref * w * phase * d12p[t_ind] * Hx[tx+ux]   * Hy[ty+uy]   * Hz[tz+uz+1] * d34p[u_ind] * e1;

                                Gp[3*A2 + 0] += pref * w * phase * d12p[t_ind] * Hx[tx+ux+1] * Hy[ty+uy]   * Hz[tz+uz]   * d34p[u_ind] * e2;
                                Gp[3*A2 + 1] += pref * w * phase * d12p[t_ind] * Hx[tx+ux]   * Hy[ty+uy+1] * Hz[tz+uz]   * d34p[u_ind] * e2;
                                Gp[3*A2 + 2] += pref * w * phase * d12p[t_ind] * Hx[tx+ux]   * Hy[ty+uy]   * Hz[tz+uz+1] * d34p[u_ind] * e2;
                            } // u_ind
                        } // t_ind
                    } // nrys
                } // pair34_ind

                double* data12Tp = data12p + pair12_ind*H12;
                for (int t_ind = 0; t_ind < H12; t_ind++) {
                    (*data12Tp++) += j12p[t_ind];
                }

            } // pair12_ind
        }} // Bra/Ket loop
    } // Ewald loop

    // collect gradient from each thread
    for (int tid = 0; tid < resources->nthread(); tid++) {
        for (int i = 0; i < G2->size(); i++) {
            G2->data()[i] += G_omp[tid]->data()[i];
        }
    }
    
    // gradient contribution from Hermite coefficients
    Vherms12->grad(DT, G2, G2, false);

    // taking advantage of symmetry!
    G2->scale(2.0);

    // accumulate G2 to G
    std::shared_ptr<Tensor> G3 = G;
    if (!G) {
        G3 = std::shared_ptr<Tensor>(new Tensor(dim));
    }
    G3->shape_error(dim);
    for (size_t i = 0; i < G3->size(); i++) {
        G3->data()[i] += G2->data()[i];
    }
    return G3;
}

std::vector<std::shared_ptr<Tensor> > IntBox::coulombGradAdvCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist12, // symm
    const std::shared_ptr<PairList>& pairlist34, // symm
    const std::shared_ptr<Tensor>& D12,
    const std::shared_ptr<Tensor>& D34,
    float thresp,
    float thredp,
    const std::vector<std::shared_ptr<Tensor> >& G1234
    )
{
    // Throws if pairlist is not symmetric
    if (!pairlist12->is_symmetric() && !pairlist34->is_symmetric()) {
        throw std::runtime_error("IntBox::coulombGradCPUAdv: Pairlist12 and Pairlist34 should be symmetric.");
    }

    std::vector<size_t> natom(4);
    natom[0] = pairlist12->basis1()->natom();
    natom[1] = pairlist12->basis2()->natom();
    natom[2] = pairlist34->basis1()->natom();
    natom[3] = pairlist34->basis2()->natom();

    // sizes of G1,G2,G3,G4
    std::vector<std::vector<size_t> > dim(4);
    for (int a = 0; a < 4; a++) {
        dim[a].push_back(natom[a]);
        dim[a].push_back(3);
    }

    // working register of G1,G2,G3,G4
    std::vector<std::shared_ptr<Tensor> > G1234T;
    if (G1234.size() == 0) {
        // The user has not allocated G1234 and we should allocate this
        for (int a = 0; a < 4; a++) {
            G1234T.push_back(std::shared_ptr<Tensor>(new Tensor(dim[a])));
        }
    } else if (G1234.size() == 4) {
        // The user has allocated G1234 and we should check that this is the right size
        G1234T = G1234;
        for (int a = 0; a < 4; a++) {
            G1234T[a]->shape_error(dim[a]);
        }
    } else {
        // The user doesn't know how many perturbations there are
        throw std::runtime_error("IntBox::coulombGradAdvCPU: G1234 should be size 0 or size 4");
    }

    // Cartesian representation of D12
    std::shared_ptr<Tensor> D12T = PureTransform::pureToCart2(
        pairlist12->basis1(),
        pairlist12->basis2(),
        D12);

    // Cartesian representation of D34
    std::shared_ptr<Tensor> D34T = PureTransform::pureToCart2(
        pairlist34->basis1(),
        pairlist34->basis2(),
        D34);

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
    std::shared_ptr<Hermite> Dherms12(new Hermite(pairlist12T));
    std::shared_ptr<Hermite> Vherms12(new Hermite(pairlist12T));
    std::shared_ptr<Hermite> Dherms34(new Hermite(pairlist34T));
    std::shared_ptr<Hermite> Vherms34(new Hermite(pairlist34T));

    // Cart-to-Hermite transform
    // Will manually take care of phases
    Dherms12->cart_to_herm(D12T, false);
    Dherms34->cart_to_herm(D34T, false);

    // Grab Rys singleton
    std::shared_ptr<Rys> rys = Rys::instance();
    const Rys* rysp = rys.get();

    // Magic masks for extracting Hermite tx,ty,tz
    const int xmask = (1 << 30) - (1 << 20);
    const int ymask = (1 << 20) - (1 << 10);
    const int zmask = (1 << 10) - (1 <<  0);

    // local copies of G on each thread
    std::vector<std::vector<std::shared_ptr<Tensor> > > G_omp(4);
    for (int a = 0; a < 4; a++) {
        for (int tid = 0; tid < resources->nthread(); tid++) {
            G_omp[a].push_back(std::shared_ptr<Tensor>(new Tensor(dim[a])));
        }
    }

    // ==> Master Loop <== //

    for (size_t ewald_ind = 0; ewald_ind < ewald->scales().size(); ewald_ind++) {
        double alpha = ewald->scales()[ewald_ind];
        double omega = ewald->omegas()[ewald_ind];
        if ((alpha == 0.0) || (omega == 0.0)) continue;

        // Bra loop
        for (size_t ind_L12 = 0; ind_L12 < Dherms12->hermites().size(); ind_L12++) {
            HermiteL& herm12 = Dherms12->hermites()[ind_L12];
            int L12 = herm12.L();
            int V12 = L12+1;
            int H12 = herm12.H();
            size_t npair12 = herm12.npair();
            const double* geom12p = herm12.geom().data();
            const float* bound12p = herm12.bounds().data();
            const int* index12p = herm12.inds().data();
            const double* eta12p = herm12.etas().data();

            const double* Ddata12p = herm12.data().data();
            double* data12p = Vherms12->hermites()[ind_L12].data().data();

        // Ket loop
        for (size_t ind_L34 = 0; ind_L34 < Dherms34->hermites().size(); ind_L34++) {
            HermiteL& herm34 = Dherms34->hermites()[ind_L34];
            int L34 = herm34.L();
            int V34 = L34+1;
            int H34 = herm34.H();
            size_t npair34 = herm34.npair();
            const double* geom34p = herm34.geom().data();
            const float* bound34p = herm34.bounds().data();
            const int* index34p = herm34.inds().data();
            const double* eta34p = herm34.etas().data();

            const double* Ddata34p = herm34.data().data();
            double* data34p = Vherms34->hermites()[ind_L34].data().data();
        
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
                double* G1p = G_omp[0][tid]->data().data();
                double* G2p = G_omp[1][tid]->data().data();
                double* G3p = G_omp[2][tid]->data().data();
                double* G4p = G_omp[3][tid]->data().data();

                float B12 = bound12p[pair12_ind];
                double p12 = geom12p[4*pair12_ind + 0];
                double x12 = geom12p[4*pair12_ind + 1];
                double y12 = geom12p[4*pair12_ind + 2];
                double z12 = geom12p[4*pair12_ind + 3];

                int A1 = index12p[2*pair12_ind + 0];
                int A2 = index12p[2*pair12_ind + 1];
                double e1 = eta12p[2*pair12_ind + 0];
                double e2 = eta12p[2*pair12_ind + 1];

                double j12p[H12];
                ::memset(j12p,'\0',sizeof(double)*H12);

                double d12p[H12];
                ::memcpy(d12p, Ddata12p + pair12_ind*H12, H12*sizeof(double));

                for (size_t pair34_ind = 0; pair34_ind < npair34; pair34_ind++) {
                    float B34 = bound34p[pair34_ind];
                    if (B12 * B34 < thresp) break;
                    double p34 = geom34p[4*pair34_ind + 0];
                    double x34 = geom34p[4*pair34_ind + 1];
                    double y34 = geom34p[4*pair34_ind + 2];
                    double z34 = geom34p[4*pair34_ind + 3];

                    int A3 = index34p[2*pair34_ind + 0];
                    int A4 = index34p[2*pair34_ind + 1];
                    double e3 = eta34p[2*pair34_ind + 0];
                    double e4 = eta34p[2*pair34_ind + 1];

                    double j34p[H34];
                    ::memset(j34p,'\0',sizeof(double)*H34);

                    double d34p[H34];
                    ::memcpy(d34p, Ddata34p + pair34_ind*H34, H34*sizeof(double));

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
                        double U = -2.0 * rho * d2 * t * t;
                        double Hx[V+1];
                        double Hy[V+1];
                        double Hz[V+1];
                        Hx[0] = 1.0;
                        Hy[0] = 1.0;
                        Hz[0] = 1.0;
                        Hx[1] = U * xPQ;
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

                                j12p[t_ind] += pref * w * phase * Hx[tx+ux] * Hy[ty+uy] * Hz[tz+uz] * d34p[u_ind];
                                j34p[u_ind] += pref * w * phase * Hx[tx+ux] * Hy[ty+uy] * Hz[tz+uz] * d12p[t_ind];

                                G1p[3*A1 + 0] += pref * w * phase * d12p[t_ind] * Hx[tx+ux+1] * Hy[ty+uy]   * Hz[tz+uz]   * d34p[u_ind] * e1;
                                G1p[3*A1 + 1] += pref * w * phase * d12p[t_ind] * Hx[tx+ux]   * Hy[ty+uy+1] * Hz[tz+uz]   * d34p[u_ind] * e1;
                                G1p[3*A1 + 2] += pref * w * phase * d12p[t_ind] * Hx[tx+ux]   * Hy[ty+uy]   * Hz[tz+uz+1] * d34p[u_ind] * e1;

                                G2p[3*A2 + 0] += pref * w * phase * d12p[t_ind] * Hx[tx+ux+1] * Hy[ty+uy]   * Hz[tz+uz]   * d34p[u_ind] * e2;
                                G2p[3*A2 + 1] += pref * w * phase * d12p[t_ind] * Hx[tx+ux]   * Hy[ty+uy+1] * Hz[tz+uz]   * d34p[u_ind] * e2;
                                G2p[3*A2 + 2] += pref * w * phase * d12p[t_ind] * Hx[tx+ux]   * Hy[ty+uy]   * Hz[tz+uz+1] * d34p[u_ind] * e2;

                                G3p[3*A3 + 0] -= pref * w * phase * d12p[t_ind] * Hx[tx+ux+1] * Hy[ty+uy]   * Hz[tz+uz]   * d34p[u_ind] * e3;
                                G3p[3*A3 + 1] -= pref * w * phase * d12p[t_ind] * Hx[tx+ux]   * Hy[ty+uy+1] * Hz[tz+uz]   * d34p[u_ind] * e3;
                                G3p[3*A3 + 2] -= pref * w * phase * d12p[t_ind] * Hx[tx+ux]   * Hy[ty+uy]   * Hz[tz+uz+1] * d34p[u_ind] * e3;

                                G4p[3*A4 + 0] -= pref * w * phase * d12p[t_ind] * Hx[tx+ux+1] * Hy[ty+uy]   * Hz[tz+uz]   * d34p[u_ind] * e4;
                                G4p[3*A4 + 1] -= pref * w * phase * d12p[t_ind] * Hx[tx+ux]   * Hy[ty+uy+1] * Hz[tz+uz]   * d34p[u_ind] * e4;
                                G4p[3*A4 + 2] -= pref * w * phase * d12p[t_ind] * Hx[tx+ux]   * Hy[ty+uy]   * Hz[tz+uz+1] * d34p[u_ind] * e4;
                            } // u_ind
                        } // t_ind
                    } // nrys

                    double* data34Tp = data34p + pair34_ind*H34;
                    for (int u_ind = 0; u_ind < H34; u_ind++) {
                        #pragma omp atomic
                        (*data34Tp++) += j34p[u_ind];
                    }

                } // pair34_ind

                double* data12Tp = data12p + pair12_ind*H12;
                for (int t_ind = 0; t_ind < H12; t_ind++) {
                    (*data12Tp++) += j12p[t_ind];
                }

            } // pair12_ind
        }} // Bra/Ket loop
    } // Ewald loop

    // collect gradient from each thread
    for (int a = 0; a < 4; a++) {
        for (int tid = 0; tid < resources->nthread(); tid++) {
            for (int i = 0; i < G1234T[a]->size(); i++) {
                G1234T[a]->data()[i] += G_omp[a][tid]->data()[i];
            }
        }
    }
    
    // gradient contribution from Hermite coefficients
    Vherms12->grad(D12T, G1234T[0], G1234T[1], false);
    Vherms34->grad(D34T, G1234T[2], G1234T[3], false);

    return G1234T;
}

} // namespace lightspeed
