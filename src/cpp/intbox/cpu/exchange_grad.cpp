#include <lightspeed/intbox.hpp>
#include <lightspeed/basis.hpp>
#include <lightspeed/pair_list.hpp>
#include <lightspeed/pure_transform.hpp>
#include <lightspeed/resource_list.hpp>
#include <lightspeed/ewald.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/rys.hpp>
#include <lightspeed/gh.hpp>
#include "../../core/kvec.hpp"
#include <cmath>
#include <cstring>
#include <cstdio>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace lightspeed {

std::shared_ptr<Tensor> IntBox::exchangeGradSymmCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& G
    )
{
    // size of G
    std::vector<size_t> dim;
    dim.push_back(pairlist->basis1()->natom());
    dim.push_back(3);

    // working register of G
    std::shared_ptr<Tensor> G2 = G;
    if (!G) {
        G2 = std::shared_ptr<Tensor>(new Tensor(dim));
    }
    G2->shape_error(dim);

    // Cartesian representation of D13
    std::shared_ptr<Tensor> D2 = PureTransform::pureToCart2(
        pairlist->basis1(),
        pairlist->basis1(),
        D);

    const double* Dp = D2->data().data();

    size_t ncart = pairlist->basis1()->ncart();

    double Dmax = 0.0;
    for (size_t ind = 0; ind < ncart*ncart; ind++){
        Dmax = std::max(Dmax, fabs(Dp[ind]));
    }

    std::shared_ptr<KBlock> kblock12 = 
        KBlock::build21(
                pairlist, 
                thresp/(PairListUtil::max_bound(pairlist)*Dmax)
                );
    std::shared_ptr<KBlock> kblock34 = kblock12;

    // Grab Rys singleton
    std::shared_ptr<Rys> rys = Rys::instance();
    const Rys* rysp = rys.get();
      
    // Gauss-Hermite quadrature rules
    std::shared_ptr<GH> gh = GH::instance();

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

    for (size_t indl12 = 0; indl12 < kblock12->kblocks().size(); indl12++){
        const KBlockL& kblockl12 = kblock12->kblocks()[indl12];
    for (size_t indl34 = 0; indl34 < kblock34->kblocks().size(); indl34++){
        const KBlockL& kblockl34 = kblock34->kblocks()[indl34];

        // Note: K-Invert
        int L1 = kblockl12.L2();
        int L2 = kblockl12.L1();
        int L3 = kblockl34.L2();
        int L4 = kblockl34.L1();
        int V1 = L1 + 1;
        int V2 = L2 + 1;
        int V3 = L3 + 1;
        int V4 = L4 + 1;

        int nrys = (L1+L2+L3+L4+1)/2 + 1;
        const double* tgh = gh->ts()[nrys].data();
        const double* wgh = gh->ws()[nrys].data();

        int N1 = (L1+1)*(L1+2)/2;
        int N2 = (L2+1)*(L2+2)/2;
        int N3 = (L3+1)*(L3+2)/2;
        int N4 = (L4+1)*(L4+2)/2;

        const std::vector<KVecL>& kvecs12 = kblockl12.kvecs();
        const std::vector<KVecL>& kvecs34 = kblockl34.kvecs();

        #pragma omp parallel for num_threads(resources->nthread()) schedule(dynamic)
        for (size_t ind24 = 0; ind24 < kvecs12.size() * kvecs34.size(); ind24++) {

            // raw pointers for each thread
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            double* Gp = G_omp[tid]->data().data();

            size_t ind2 = ind24 / kvecs34.size();
            size_t ind4 = ind24 % kvecs34.size();
            const KVecL& kvec12 = kvecs12[ind2];
            const KVecL& kvec34 = kvecs34[ind4];
            // Note: K-Invert
            const Primitive& prim2 = kvec12.prim1();
            const Primitive& prim4 = kvec34.prim1();

            // offsets from global access
            int o2 = prim2.cartIdx();
            int o4 = prim4.cartIdx();

            int K2 = prim2.primIdx();
            int K4 = prim4.primIdx();
        
            double D24V[N2*N4];
            double D24Vmax = 0.0;
            for (int p2 = 0; p2 < N2; p2++){
            for (int p4 = 0; p4 < N4; p4++){
                double Dval = Dp[(p2+o2)*ncart + (p4+o4)];
                D24V[p2*N4 + p4] = Dval;
                D24Vmax = std::max(D24Vmax, fabs(Dval));
            }}

            // Note: K-Invert
            size_t nprim1 = kvec12.nprim2();
            size_t nprim3 = kvec34.nprim2();
            const double* geom1 = kvec12.geom2().data();
            const double* geom3 = kvec34.geom2().data();
            const int* idx1 = kvec12.idx2().data();
            const int* idx3 = kvec34.idx2().data();
            const float* bounds12 = kvec12.bounds12().data();
            const float* bounds34 = kvec34.bounds12().data();

            double e2 = prim2.e();
            double e4 = prim4.e();
            double x2[3] = { prim2.x(), prim2.y(), prim2.z() };
            double x4[3] = { prim4.x(), prim4.y(), prim4.z() };

            for (size_t ind1 = 0; ind1 < nprim1; ind1++){
                float bound12 = bounds12[ind1];

                double pref12 = alpha*geom1[5*ind1 + 0]*2.0*pow(M_PI,-0.5);
                double e1 = geom1[5*ind1 + 1];
                double x1[3] = { 
                    geom1[5*ind1 + 2], 
                    geom1[5*ind1 + 3], 
                    geom1[5*ind1 + 4] 
                };

                int K1 = idx1[3*ind1 + 0];
                int o1 = idx1[3*ind1 + 1];

                double e12 = e1 + e2;
                double e12_m1 = 1.0/e12;

                bool did_work = false;
                for (size_t ind3 = 0; ind3 < nprim3; ind3++){
                    float bound34 = bounds34[ind3];

                    int K3 = idx3[3*ind3 + 0];
                    int o3 = idx3[3*ind3 + 1];

                    double D13V[N1*N3];
                    for (int p1 = 0; p1 < N1; p1++){
                    for (int p3 = 0; p3 < N3; p3++){
                        double Dval = Dp[(p1+o1)*ncart + (p3+o3)];
                        D13V[p1*N3 + p3] = Dval;
                    }}

                    if (bound12*bound34*D24Vmax < thresp) { break; }
                    did_work = true;

                    double D14V[N1*N4];
                    for (int p1 = 0; p1 < N1; p1++){
                    for (int p4 = 0; p4 < N4; p4++){
                        D14V[p1*N4 + p4] = Dp[(p1+o1)*ncart + (p4+o4)];
                    }}

                    double D23V[N2*N3];
                    for (int p2 = 0; p2 < N2; p2++){
                    for (int p3 = 0; p3 < N3; p3++){
                        D23V[p2*N3 + p3] = Dp[(p2+o2)*ncart + (p3+o3)];
                    }}

                    // Full symmetry sieving
                    // unique ERI: (12|12), (11|12), (22|12), (11|22), (11|11), (22|22)
                    // sieved ERI: (21|12), (21|11), (21|22), (22|11),
                    //             (21|21), (11|21), (22|21),
                    //             (12|21), (12|11), (12|22)

                    if (K1>K2 || K3>K4 || (K1<K2 && K3==K4) || (K1==K2 && K3==K4 && K2>K4)) {
                        continue;
                    }

                    // ==> Integral Kernel <== //

                    double pref1234 = pref12*geom3[5*ind3 + 0];
                    double e3 = geom3[5*ind3 + 1];
                    double x3[3] = { 
                        geom3[5*ind3 + 2], 
                        geom3[5*ind3 + 3], 
                        geom3[5*ind3 + 4] 
                    };

                    double e34 = e3 + e4;
                    double e34_m1 = 1.0/e34;

                    pref1234 *= e12_m1*e34_m1*pow(e12+e34, -0.5);

                    double xP[3], xQ[3], rPQ_2 = 0.0;
                    for (int d = 0; d < 3; d++) {
                        xP[d] = (e1*x1[d] + e2*x2[d])*e12_m1;
                        xQ[d] = (e3*x3[d] + e4*x4[d])*e34_m1;
                        rPQ_2 += pow(xP[d] - xQ[d], 2);
                    }
                    double rho = e12*e34/(e12+e34);

                    double d2 = 1.0;
                    if (omega != -1.0){
                        d2 = omega*omega / (rho + omega*omega);
                        pref1234 *= sqrt(d2);
                    }
                    double T = rho * d2 * rPQ_2;

                    // Rys quadrature
                    for (int i = 0; i < nrys; i++) {
                        double t = rysp->interpolate_t(nrys, i, T);
                        double w = rysp->interpolate_w(nrys, i, T);
                        double t2 = t*t*d2;
                        
                        double J2[3][V1*V2*V3*V4];
                        double P2[3][V1*V2*V3*V4];
                        double Q2[3][V1*V2*V3*V4];
                        double R2[3][V1*V2*V3*V4];
                        double S2[3][V1*V2*V3*V4];
                        ::memset(J2[0], '\0', sizeof(double)*3*V1*V2*V3*V4);
                        ::memset(P2[0], '\0', sizeof(double)*3*V1*V2*V3*V4);
                        ::memset(Q2[0], '\0', sizeof(double)*3*V1*V2*V3*V4);
                        ::memset(R2[0], '\0', sizeof(double)*3*V1*V2*V3*V4);
                        ::memset(S2[0], '\0', sizeof(double)*3*V1*V2*V3*V4);

                        // Coordinate transform
                        double gamma = rho * t2 / (1.0 - t2);
                        double delta_m1 = (e34 - e12) / (2.0 * gamma);
                        double Qval = delta_m1 * pow(1.0 + delta_m1 * delta_m1, -0.5);
                        double c2val = 0.5 + 0.5 * Qval;
                        double s2val = 0.5 - 0.5 * Qval;
                        double csval = 0.5 * pow(1.0 + delta_m1 * delta_m1, -0.5);
                        double cval = sqrt(c2val);
                        double sval = sqrt(s2val);
                        double a1 = c2val * (e12 + gamma) + s2val * (e34 + gamma) - 2.0 * csval * gamma;
                        double a2 = s2val * (e12 + gamma) + c2val * (e34 + gamma) + 2.0 * csval * gamma;
                        double a1_m12 = pow(a1, -0.5);
                        double a2_m12 = pow(a2, -0.5);
                        // Coordinate transforms of GH
                        double X11 =  cval * a1_m12;
                        double X12 = -sval * a2_m12;
                        double X21 =  sval * a1_m12;
                        double X22 =  cval * a2_m12;

                        double xA[3], xB[3];
                        for (int d = 0; d < 3; d++) {
                            xA[d] = xP[d] + rho*t2*e12_m1*(xQ[d]-xP[d]);
                            xB[d] = xQ[d] + rho*t2*e34_m1*(xP[d]-xQ[d]);
                        }

                        // 2D Gauss-Hermite quadrature
                        for (int v1 = 0; v1 < nrys; v1++){
                        for (int v2 = 0; v2 < nrys; v2++){
                            double zeta1 = X11*tgh[v1]+X12*tgh[v2];
                            double zeta2 = X21*tgh[v1]+X22*tgh[v2];
                            double wgh12 = wgh[v1]*wgh[v2];

                            for (int d = 0; d < 3; d++) {
                                double* J2x = J2[d];
                                double* P2x = P2[d];
                                double* Q2x = Q2[d];
                                double* R2x = R2[d];
                                double* S2x = S2[d];

                                double x1d = zeta1+xA[d]-x1[d];
                                double x2d = zeta1+xA[d]-x2[d];
                                double x3d = zeta2+xB[d]-x3[d];
                                double x4d = zeta2+xB[d]-x4[d];

                                double x1lm = 0.0;
                                double x1l = 1.0;
                                double x1lp = x1d;

                            for (int l1 = 0; l1 < V1; l1++){
                                double x2lm = 0.0;
                                double x2l = 1.0;
                                double x2lp = x2d;

                            for (int l2 = 0; l2 < V2; l2++){
                                double x3lm = 0.0;
                                double x3l = 1.0;
                                double x3lp = x3d;

                            for (int l3 = 0; l3 < V3; l3++){
                                double x4lm = 0.0;
                                double x4l = 1.0;
                                double x4lp = x4d;

                            for (int l4 = 0; l4 < V4; l4++){
                                (*J2x++) += wgh12*x1l*x2l*x3l*x4l;
                                (*P2x++) -= wgh12*(l1*x1lm - 2.0*e1*x1lp)*x2l*x3l*x4l;
                                (*Q2x++) -= wgh12*x1l*(l2*x2lm - 2.0*e2*x2lp)*x3l*x4l;
                                (*R2x++) -= wgh12*x1l*x2l*(l3*x3lm - 2.0*e3*x3lp)*x4l;
                                (*S2x++) -= wgh12*x1l*x2l*x3l*(l4*x4lm - 2.0*e4*x4lp);

                                x4lm = x4l;
                                x4l = x4lp;
                                x4lp *= x4d;
                            } // l4

                                x3lm = x3l;
                                x3l = x3lp;
                                x3lp *= x3d;
                            } // l3

                                x2lm = x2l;
                                x2l = x2lp;
                                x2lp *= x2d;
                            } // l2

                                x1lm = x1l;
                                x1l = x1lp;
                                x1lp *= x1d;
                            } // l1
                            } // d
                        }} // v1,v2

                        // => Gradient on atoms <= //

                        // AM1 
                        for (int i1 = 0, index1 = 0; i1 <= L1; i1++) {
                            int l1 = L1 - i1;
                            int l1V = l1*V2*V3*V4;
                        for (int j1 = 0; j1 <= i1; j1++, index1++) {
                            int m1 = i1 - j1;
                            int n1 = j1;
                            int m1V = m1*V2*V3*V4;
                            int n1V = n1*V2*V3*V4;

                        // AM2
                        for (int i2 = 0, index2 = 0; i2 <= L2; i2++) {
                            int l2 = L2 - i2;
                            int l12V = l1V + l2*V3*V4;
                        for (int j2 = 0; j2 <= i2; j2++, index2++) {
                            int m2 = i2 - j2;
                            int n2 = j2;
                            int m12V = m1V + m2*V3*V4;
                            int n12V = n1V + n2*V3*V4;

                        // AM3
                        for (int i3 = 0, index3 = 0; i3 <= L3; i3++) {
                            int l3 = L3 - i3;
                            int l123V = l12V + l3*V4;
                        for (int j3 = 0; j3 <= i3; j3++, index3++) {
                            int m3 = i3 - j3;
                            int n3 = j3;
                            int m123V = m12V + m3*V4;
                            int n123V = n12V + n3*V4;

                        // AM4
                        for (int i4 = 0, index4 = 0; i4 <= L4; i4++) {
                            int l4 = L4 - i4;
                            int l1234V = l123V + l4;
                        for (int j4 = 0; j4 <= i4; j4++, index4++) {
                            int m4 = i4 - j4;
                            int n4 = j4;
                            int m1234V = m123V + m4;
                            int n1234V = n123V + n4;

                            double PX[3] = {
                                pref1234 * w * P2[0][l1234V] * J2[1][m1234V] * J2[2][n1234V],
                                pref1234 * w * J2[0][l1234V] * P2[1][m1234V] * J2[2][n1234V],
                                pref1234 * w * J2[0][l1234V] * J2[1][m1234V] * P2[2][n1234V]
                                };

                            double QX[3] = {
                                pref1234 * w * Q2[0][l1234V] * J2[1][m1234V] * J2[2][n1234V],
                                pref1234 * w * J2[0][l1234V] * Q2[1][m1234V] * J2[2][n1234V],
                                pref1234 * w * J2[0][l1234V] * J2[1][m1234V] * Q2[2][n1234V]
                                };

                            double RX[3] = {
                                pref1234 * w * R2[0][l1234V] * J2[1][m1234V] * J2[2][n1234V],
                                pref1234 * w * J2[0][l1234V] * R2[1][m1234V] * J2[2][n1234V],
                                pref1234 * w * J2[0][l1234V] * J2[1][m1234V] * R2[2][n1234V]
                                };

                            double SX[3] = {
                                pref1234 * w * S2[0][l1234V] * J2[1][m1234V] * J2[2][n1234V],
                                pref1234 * w * J2[0][l1234V] * S2[1][m1234V] * J2[2][n1234V],
                                pref1234 * w * J2[0][l1234V] * J2[1][m1234V] * S2[2][n1234V]
                                };

                            double Dval = D13V[index1*N3 + index3] * D24V[index2*N4 + index4];
                            int PA = idx1[3*ind1 + 2];
                            int PB = prim2.atomIdx();
                            int PC = idx3[3*ind3 + 2];
                            int PD = prim4.atomIdx();

                            // four-fold ERI: (12|12), (11|12), (22|12)
                            // two-fold ERI:  (11|22)
                            // one-fold ERI:  (11|11), (22|22)

                            if (K1 != K2 || K3 != K4) {
                                Dval += D23V[index2*N3 + index3] * D14V[index1*N4 + index4];
                                Dval *= 2.0;
                            } else if (K2 != K4) {
                                Dval *= 2.0;
                            }

                            for (int d = 0; d < 3; d++) {
                                Gp[3*PA + d] += Dval * PX[d];
                                Gp[3*PB + d] += Dval * QX[d];
                                Gp[3*PC + d] += Dval * RX[d];
                                Gp[3*PD + d] += Dval * SX[d];
                            }

                        }}}}}}}} // AM (L^8)

                    } // nrys

                    // ==> End Integral Kernel <== //

                } // ind3
                if (!did_work) { break; }
            } // ind1
        } // ind24 (OpenMP)
    }} // indl12,indl34
    } // ewald_ind

    // collect gradient from each thread
    for (int tid = 0; tid < resources->nthread(); tid++) {
        for (int i = 0; i < G2->size(); i++) {
            G2->data()[i] += G_omp[tid]->data()[i];
        }
    }

    return G2;
}

std::vector<std::shared_ptr<Tensor> > IntBox::exchangeGradAdvCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist12,
    const std::shared_ptr<PairList>& pairlist34,
    const std::shared_ptr<Tensor>& D13,
    const std::shared_ptr<Tensor>& D24,
    bool D13symm,
    bool D24symm,
    bool Dsame,
    float thresp,
    float thredp,
    const std::vector<std::shared_ptr<Tensor> >& G1234
    )
{
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
        throw std::runtime_error("IntBox::exchangeGradAdvCPU: G1234 should be size 0 or size 4");
    }

    // Cartesian representation of D13
    std::shared_ptr<Tensor> D13T = PureTransform::pureToCart2(
        pairlist12->basis1(),
        pairlist34->basis1(),
        D13);

    // Cartesian representation of D24
    std::shared_ptr<Tensor> D24T = PureTransform::pureToCart2(
        pairlist12->basis2(),
        pairlist34->basis2(),
        D24);

    const double* D13p = D13T->data().data();
    const double* D24p = D24T->data().data();

    size_t ncart1 = pairlist12->basis1()->ncart();
    size_t ncart2 = pairlist12->basis2()->ncart();
    size_t ncart3 = pairlist34->basis1()->ncart();
    size_t ncart4 = pairlist34->basis2()->ncart();

    double D24max = 0.0;
    for (size_t ind = 0; ind < ncart2*ncart4; ind++){
        D24max = std::max(D24max, fabs(D24p[ind]));
    }

    std::shared_ptr<KBlock> kblock12;
    std::shared_ptr<KBlock> kblock34;
    if (true){
        kblock12 = KBlock::build21(
                pairlist12, 
                thresp/(PairListUtil::max_bound(pairlist34)*D24max)
                );
    }

    // Check same pointer
    if (pairlist12 == pairlist34){
        kblock34 = kblock12;
    } else{
        kblock34 = KBlock::build21(
                pairlist34, 
                thresp/(PairListUtil::max_bound(pairlist12)*D24max)
                );
    }
    
    // Check if D13symm and D24symm flags make sense
    if (D13symm && !Basis::equivalent(pairlist12->basis1(),pairlist34->basis1())) {
        throw std::runtime_error("IntBox::exchangeCPU: D13symm implied, but basis sets 1 and 3 not the same");
    }
    if (D24symm && !Basis::equivalent(pairlist12->basis2(),pairlist34->basis2())) {
        throw std::runtime_error("IntBox::exchangeCPU: D24symm implied, but basis sets 2 and 4 not the same");
    }

    // We can only profitably use the Dsymm flag when the basis are symmetric also
    if (D13symm) {
        D13symm &= Basis::equivalent(pairlist12->basis2(),pairlist34->basis2());
    }
    if (D24symm) {
        D24symm &= Basis::equivalent(pairlist12->basis1(),pairlist34->basis1());
    }

    // check if Dsame flag makes sense
    if (Dsame) {
        if (!Basis::equivalent(pairlist12->basis1(),pairlist12->basis2())) {
            throw std::runtime_error("IntBox::exchangeCPU: Dsame implied, but basis sets 1 and 2 not the same");
        }
        if (!Basis::equivalent(pairlist34->basis1(),pairlist34->basis2())) {
            throw std::runtime_error("IntBox::exchangeCPU: Dsame implied, but basis sets 3 and 4 not the same");
        }
        if (D13symm != D24symm) {
            throw std::runtime_error("IntBox::exchangeCPU: Dsame implied, but D13symm and D24symm not the same");
        }
    }

    // Grab Rys singleton
    std::shared_ptr<Rys> rys = Rys::instance();
    const Rys* rysp = rys.get();
      
    // Gauss-Hermite quadrature rules
    std::shared_ptr<GH> gh = GH::instance();

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

    for (size_t indl12 = 0; indl12 < kblock12->kblocks().size(); indl12++){
        const KBlockL& kblockl12 = kblock12->kblocks()[indl12];
    for (size_t indl34 = 0; indl34 < kblock34->kblocks().size(); indl34++){
        const KBlockL& kblockl34 = kblock34->kblocks()[indl34];

        // Note: K-Invert
        int L1 = kblockl12.L2();
        int L2 = kblockl12.L1();
        int L3 = kblockl34.L2();
        int L4 = kblockl34.L1();
        int V1 = L1 + 1;
        int V2 = L2 + 1;
        int V3 = L3 + 1;
        int V4 = L4 + 1;

        int nrys = (L1+L2+L3+L4+1)/2 + 1;
        const double* tgh = gh->ts()[nrys].data();
        const double* wgh = gh->ws()[nrys].data();

        int N1 = (L1+1)*(L1+2)/2;
        int N2 = (L2+1)*(L2+2)/2;
        int N3 = (L3+1)*(L3+2)/2;
        int N4 = (L4+1)*(L4+2)/2;

        const std::vector<KVecL>& kvecs12 = kblockl12.kvecs();
        const std::vector<KVecL>& kvecs34 = kblockl34.kvecs();

        #pragma omp parallel for num_threads(resources->nthread()) schedule(dynamic)
        for (size_t ind24 = 0; ind24 < kvecs12.size() * kvecs34.size(); ind24++) {

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

            size_t ind2 = ind24 / kvecs34.size();
            size_t ind4 = ind24 % kvecs34.size();
            const KVecL& kvec12 = kvecs12[ind2];
            const KVecL& kvec34 = kvecs34[ind4];
            // Note: K-Invert
            const Primitive& prim2 = kvec12.prim1();
            const Primitive& prim4 = kvec34.prim1();

            // offsets from global access
            int o2 = prim2.cartIdx();
            int o4 = prim4.cartIdx();

            int K2 = prim2.primIdx();
            int K4 = prim4.primIdx();
        
            // D24symm sieve
            if (D24symm && K2 > K4) { continue; }
            bool flip24 = (D24symm && K2 != K4);

            double D24V[N2*N4];
            double D24Vmax = 0.0;
            double D24VT[N4*N2];
            for (int p2 = 0; p2 < N2; p2++){
            for (int p4 = 0; p4 < N4; p4++){
                double Dval = D24p[(p2+o2)*ncart4 + (p4+o4)];
                D24V[p2*N4 + p4] = Dval;
                D24Vmax = std::max(D24Vmax, fabs(Dval));
                D24VT[p4*N2 + p2] = D24p[(p4+o4)*ncart4 + (p2+o2)];
            }}

            // Note: K-Invert
            size_t nprim1 = kvec12.nprim2();
            size_t nprim3 = kvec34.nprim2();
            const double* geom1 = kvec12.geom2().data();
            const double* geom3 = kvec34.geom2().data();
            const int* idx1 = kvec12.idx2().data();
            const int* idx3 = kvec34.idx2().data();
            const float* bounds12 = kvec12.bounds12().data();
            const float* bounds34 = kvec34.bounds12().data();

            double e2 = prim2.e();
            double e4 = prim4.e();
            double x2[3] = { prim2.x(), prim2.y(), prim2.z() };
            double x4[3] = { prim4.x(), prim4.y(), prim4.z() };

            for (size_t ind1 = 0; ind1 < nprim1; ind1++){
                float bound12 = bounds12[ind1];

                double pref12 = alpha*geom1[5*ind1 + 0]*2.0*pow(M_PI,-0.5);
                double e1 = geom1[5*ind1 + 1];
                double x1[3] = { 
                    geom1[5*ind1 + 2], 
                    geom1[5*ind1 + 3], 
                    geom1[5*ind1 + 4] 
                };

                int K1 = idx1[3*ind1 + 0];
                int o1 = idx1[3*ind1 + 1];

                double e12 = e1 + e2;
                double e12_m1 = 1.0/e12;

                bool did_work = false;
                for (size_t ind3 = 0; ind3 < nprim3; ind3++){
                    float bound34 = bounds34[ind3];

                    int K3 = idx3[3*ind3 + 0];
                    int o3 = idx3[3*ind3 + 1];

                    double D13V[N1*N3];
                    double D13VT[N3*N1];
                    for (int p1 = 0; p1 < N1; p1++){
                    for (int p3 = 0; p3 < N3; p3++){
                        double Dval = D13p[(p1+o1)*ncart3 + (p3+o3)];
                        D13V[p1*N3 + p3] = Dval;
                        D13VT[p3*N1 + p1] = D13p[(p3+o3)*ncart3 + (p1+o1)];
                    }}

                    if (bound12*bound34*D24Vmax < thresp) { break; }
                    did_work = true;

                    // D13symm sieve
                    // In case of D24symm, only sieve K2==K4 && K1>K3 case.
                    // Otherwise sieve all K1>K3.
                    bool flag_24_symm = (D24symm && K2 == K4) || (!D24symm);
                    if (flag_24_symm && D13symm && K1 > K3) { continue; }
                    bool flip13 = (flag_24_symm && D13symm && K1 != K3);

                    // Dsame sieve, without D24symm or D13symm
                    if (!D24symm && !D13symm && Dsame && (K3>K4 || (K3==K4 && K1>K2))) { continue; }
                    bool flip12 = (!D24symm && !D13symm && Dsame && (K1!=K2 || K3!=K4));

                    // ==> Integral Kernel <== //

                    double pref1234 = pref12*geom3[5*ind3 + 0];
                    double e3 = geom3[5*ind3 + 1];
                    double x3[3] = { 
                        geom3[5*ind3 + 2], 
                        geom3[5*ind3 + 3], 
                        geom3[5*ind3 + 4] 
                    };

                    double e34 = e3 + e4;
                    double e34_m1 = 1.0/e34;

                    pref1234 *= e12_m1*e34_m1*pow(e12+e34, -0.5);

                    double xP[3], xQ[3], rPQ_2 = 0.0;
                    for (int d = 0; d < 3; d++) {
                        xP[d] = (e1*x1[d] + e2*x2[d])*e12_m1;
                        xQ[d] = (e3*x3[d] + e4*x4[d])*e34_m1;
                        rPQ_2 += pow(xP[d] - xQ[d], 2);
                    }
                    double rho = e12*e34/(e12+e34);

                    double d2 = 1.0;
                    if (omega != -1.0){
                        d2 = omega*omega / (rho + omega*omega);
                        pref1234 *= sqrt(d2);
                    }
                    double T = rho * d2 * rPQ_2;

                    // Rys quadrature
                    for (int i = 0; i < nrys; i++) {
                        double t = rysp->interpolate_t(nrys, i, T);
                        double w = rysp->interpolate_w(nrys, i, T);
                        double t2 = t*t*d2;
                        
                        double J2[3][V1*V2*V3*V4];
                        double P2[3][V1*V2*V3*V4];
                        double Q2[3][V1*V2*V3*V4];
                        double R2[3][V1*V2*V3*V4];
                        double S2[3][V1*V2*V3*V4];
                        ::memset(J2[0], '\0', sizeof(double)*3*V1*V2*V3*V4);
                        ::memset(P2[0], '\0', sizeof(double)*3*V1*V2*V3*V4);
                        ::memset(Q2[0], '\0', sizeof(double)*3*V1*V2*V3*V4);
                        ::memset(R2[0], '\0', sizeof(double)*3*V1*V2*V3*V4);
                        ::memset(S2[0], '\0', sizeof(double)*3*V1*V2*V3*V4);

                        // Coordinate transform
                        double gamma = rho * t2 / (1.0 - t2);
                        double delta_m1 = (e34 - e12) / (2.0 * gamma);
                        double Qval = delta_m1 * pow(1.0 + delta_m1 * delta_m1, -0.5);
                        double c2val = 0.5 + 0.5 * Qval;
                        double s2val = 0.5 - 0.5 * Qval;
                        double csval = 0.5 * pow(1.0 + delta_m1 * delta_m1, -0.5);
                        double cval = sqrt(c2val);
                        double sval = sqrt(s2val);
                        double a1 = c2val * (e12 + gamma) + s2val * (e34 + gamma) - 2.0 * csval * gamma;
                        double a2 = s2val * (e12 + gamma) + c2val * (e34 + gamma) + 2.0 * csval * gamma;
                        double a1_m12 = pow(a1, -0.5);
                        double a2_m12 = pow(a2, -0.5);
                        // Coordinate transforms of GH
                        double X11 =  cval * a1_m12;
                        double X12 = -sval * a2_m12;
                        double X21 =  sval * a1_m12;
                        double X22 =  cval * a2_m12;

                        double xA[3], xB[3];
                        for (int d = 0; d < 3; d++) {
                            xA[d] = xP[d] + rho*t2*e12_m1*(xQ[d]-xP[d]);
                            xB[d] = xQ[d] + rho*t2*e34_m1*(xP[d]-xQ[d]);
                        }

                        // 2D Gauss-Hermite quadrature
                        for (int v1 = 0; v1 < nrys; v1++){
                        for (int v2 = 0; v2 < nrys; v2++){
                            double zeta1 = X11*tgh[v1]+X12*tgh[v2];
                            double zeta2 = X21*tgh[v1]+X22*tgh[v2];
                            double wgh12 = wgh[v1]*wgh[v2];

                            for (int d = 0; d < 3; d++) {
                                double* J2x = J2[d];
                                double* P2x = P2[d];
                                double* Q2x = Q2[d];
                                double* R2x = R2[d];
                                double* S2x = S2[d];

                                double x1d = zeta1+xA[d]-x1[d];
                                double x2d = zeta1+xA[d]-x2[d];
                                double x3d = zeta2+xB[d]-x3[d];
                                double x4d = zeta2+xB[d]-x4[d];

                                double x1lm = 0.0;
                                double x1l = 1.0;
                                double x1lp = x1d;

                            for (int l1 = 0; l1 < V1; l1++){
                                double x2lm = 0.0;
                                double x2l = 1.0;
                                double x2lp = x2d;

                            for (int l2 = 0; l2 < V2; l2++){
                                double x3lm = 0.0;
                                double x3l = 1.0;
                                double x3lp = x3d;

                            for (int l3 = 0; l3 < V3; l3++){
                                double x4lm = 0.0;
                                double x4l = 1.0;
                                double x4lp = x4d;

                            for (int l4 = 0; l4 < V4; l4++){
                                (*J2x++) += wgh12*x1l*x2l*x3l*x4l;
                                (*P2x++) -= wgh12*(l1*x1lm - 2.0*e1*x1lp)*x2l*x3l*x4l;
                                (*Q2x++) -= wgh12*x1l*(l2*x2lm - 2.0*e2*x2lp)*x3l*x4l;
                                (*R2x++) -= wgh12*x1l*x2l*(l3*x3lm - 2.0*e3*x3lp)*x4l;
                                (*S2x++) -= wgh12*x1l*x2l*x3l*(l4*x4lm - 2.0*e4*x4lp);

                                x4lm = x4l;
                                x4l = x4lp;
                                x4lp *= x4d;
                            } // l4

                                x3lm = x3l;
                                x3l = x3lp;
                                x3lp *= x3d;
                            } // l3

                                x2lm = x2l;
                                x2l = x2lp;
                                x2lp *= x2d;
                            } // l2

                                x1lm = x1l;
                                x1l = x1lp;
                                x1lp *= x1d;
                            } // l1
                            } // d
                        }} // v1,v2

                        // => Gradient on atoms <= //

                        // AM1 
                        for (int i1 = 0, index1 = 0; i1 <= L1; i1++) {
                            int l1 = L1 - i1;
                            int l1V = l1*V2*V3*V4;
                        for (int j1 = 0; j1 <= i1; j1++, index1++) {
                            int m1 = i1 - j1;
                            int n1 = j1;
                            int m1V = m1*V2*V3*V4;
                            int n1V = n1*V2*V3*V4;

                        // AM2
                        for (int i2 = 0, index2 = 0; i2 <= L2; i2++) {
                            int l2 = L2 - i2;
                            int l12V = l1V + l2*V3*V4;
                        for (int j2 = 0; j2 <= i2; j2++, index2++) {
                            int m2 = i2 - j2;
                            int n2 = j2;
                            int m12V = m1V + m2*V3*V4;
                            int n12V = n1V + n2*V3*V4;

                        // AM3
                        for (int i3 = 0, index3 = 0; i3 <= L3; i3++) {
                            int l3 = L3 - i3;
                            int l123V = l12V + l3*V4;
                        for (int j3 = 0; j3 <= i3; j3++, index3++) {
                            int m3 = i3 - j3;
                            int n3 = j3;
                            int m123V = m12V + m3*V4;
                            int n123V = n12V + n3*V4;

                        // AM4
                        for (int i4 = 0, index4 = 0; i4 <= L4; i4++) {
                            int l4 = L4 - i4;
                            int l1234V = l123V + l4;
                        for (int j4 = 0; j4 <= i4; j4++, index4++) {
                            int m4 = i4 - j4;
                            int n4 = j4;
                            int m1234V = m123V + m4;
                            int n1234V = n123V + n4;

                            double PX[3] = {
                                pref1234 * w * P2[0][l1234V] * J2[1][m1234V] * J2[2][n1234V],
                                pref1234 * w * J2[0][l1234V] * P2[1][m1234V] * J2[2][n1234V],
                                pref1234 * w * J2[0][l1234V] * J2[1][m1234V] * P2[2][n1234V]
                                };

                            double QX[3] = {
                                pref1234 * w * Q2[0][l1234V] * J2[1][m1234V] * J2[2][n1234V],
                                pref1234 * w * J2[0][l1234V] * Q2[1][m1234V] * J2[2][n1234V],
                                pref1234 * w * J2[0][l1234V] * J2[1][m1234V] * Q2[2][n1234V]
                                };

                            double RX[3] = {
                                pref1234 * w * R2[0][l1234V] * J2[1][m1234V] * J2[2][n1234V],
                                pref1234 * w * J2[0][l1234V] * R2[1][m1234V] * J2[2][n1234V],
                                pref1234 * w * J2[0][l1234V] * J2[1][m1234V] * R2[2][n1234V]
                                };

                            double SX[3] = {
                                pref1234 * w * S2[0][l1234V] * J2[1][m1234V] * J2[2][n1234V],
                                pref1234 * w * J2[0][l1234V] * S2[1][m1234V] * J2[2][n1234V],
                                pref1234 * w * J2[0][l1234V] * J2[1][m1234V] * S2[2][n1234V]
                                };

                            double Dval = D13V[index1*N3 + index3] * D24V[index2*N4 + index4];
                            int PA = idx1[3*ind1 + 2];
                            int PB = prim2.atomIdx();
                            int PC = idx3[3*ind3 + 2];
                            int PD = prim4.atomIdx();
                            for (int d = 0; d < 3; d++) {
                                // (12|34)
                                G1p[3*PA + d] += Dval * PX[d];
                                G2p[3*PB + d] += Dval * QX[d];
                                G3p[3*PC + d] += Dval * RX[d];
                                G4p[3*PD + d] += Dval * SX[d];
                            }

                            if (flip24 || flip13) {
                                if (flip24) { Dval = D13VT[index3*N1 + index1] * D24V[index2*N4 + index4]; }
                                if (flip13) { Dval = D13V[index1*N3 + index3] * D24VT[index4*N2 + index2]; }
                                for (int d = 0; d < 3; d++) {
                                    // (34|12)
                                    G1p[3*PC + d] += Dval * RX[d];
                                    G2p[3*PD + d] += Dval * SX[d];
                                    G3p[3*PA + d] += Dval * PX[d];
                                    G4p[3*PB + d] += Dval * QX[d];
                                }
                            }

                            else if (flip12) {
                                for (int d = 0; d < 3; d++) {
                                    // (21|43)
                                    G1p[3*PB + d] += Dval * QX[d];
                                    G2p[3*PA + d] += Dval * PX[d];
                                    G3p[3*PD + d] += Dval * SX[d];
                                    G4p[3*PC + d] += Dval * RX[d];
                                }
                            }

                        }}}}}}}} // AM (L^8)

                    } // nrys

                    // ==> End Integral Kernel <== //

                } // ind3
                if (!did_work) { break; }
            } // ind1
        } // ind24 (OpenMP)
    }} // indl12,indl34
    } // ewald_ind

    // collect gradient from each thread
    for (int a = 0; a < 4; a++) {
        for (int tid = 0; tid < resources->nthread(); tid++) {
            for (int i = 0; i < G1234T[a]->size(); i++) {
                G1234T[a]->data()[i] += G_omp[a][tid]->data()[i];
            }
        }
    }

    return G1234T;
}

} // namespace lightspeed
