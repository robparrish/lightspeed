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

namespace lightspeed {

std::shared_ptr<Tensor> IntBox::exchangeCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist12,
    const std::shared_ptr<PairList>& pairlist34,
    const std::shared_ptr<Tensor>& D24,
    bool D24symm,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& K13
    )
{
    // Cartesian representation of D24
    std::shared_ptr<Tensor> D24T = PureTransform::pureToCart2(
        pairlist12->basis2(),
        pairlist34->basis2(),
        D24);

    // Cartesian representation of K13
    std::shared_ptr<Tensor> K13T = PureTransform::allocCart2(
        pairlist12->basis1(),
        pairlist34->basis1(),
        K13);

    const double * D24p = D24T->data().data();
    double* K13p = K13T->data().data();

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
    
    // Check if D24symm flag makes sense
    if (D24symm && !Basis::equivalent(pairlist12->basis2(),pairlist34->basis2())) {
        throw std::runtime_error("IntBox::exchangeCPU: D24symm implied, but basis sets 2 and 4 not the same");
    }

    // We can only profitably use this flag if basis1 and basis3 are symmetric also
    if (D24symm) {
        D24symm &= Basis::equivalent(pairlist12->basis1(),pairlist34->basis1());
    }

    // Grab Rys singleton
    std::shared_ptr<Rys> rys = Rys::instance();
    const Rys * rysp = rys.get();
      
    // Gauss-Hermite quadrature rules
    std::shared_ptr<GH> gh = GH::instance();

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

        int nrys = (L1+L2+L3+L4)/2 + 1;
        const double * tgh = gh->ts()[nrys].data();
        const double * wgh = gh->ws()[nrys].data();

        int N1 = (L1+1)*(L1+2)/2;
        int N2 = (L2+1)*(L2+2)/2;
        int N3 = (L3+1)*(L3+2)/2;
        int N4 = (L4+1)*(L4+2)/2;

        const std::vector<KVecL>& kvecs12 = kblockl12.kvecs();
        const std::vector<KVecL>& kvecs34 = kblockl34.kvecs();

        #pragma omp parallel for num_threads(resources->nthread()) schedule(dynamic)
        for (size_t ind24 = 0; ind24 < kvecs12.size() * kvecs34.size(); ind24++) {
            size_t ind2 = ind24 / kvecs34.size();
            size_t ind4 = ind24 % kvecs34.size();
            const KVecL& kvec12 = kvecs12[ind2];
            const KVecL& kvec34 = kvecs34[ind4];
            // Note: K-Invert
            const Primitive& prim2 = kvec12.prim1();
            const Primitive& prim4 = kvec34.prim1();

            int K2 = prim2.primIdx();
            int K4 = prim4.primIdx();

            // Factor of 2 sieve if D24symm
            if (D24symm && K2 > K4) { continue; }
            bool flip24 = (D24symm && K2 != K4);

            // offsets from global access
            int o2 = prim2.cartIdx();
            int o4 = prim4.cartIdx();
        
            double D24V[N2*N4];
            double D24Vmax = 0.0;
            for (int p2 = 0; p2 < N2; p2++){
            for (int p4 = 0; p4 < N4; p4++){
                double Dval = D24p[(p2+o2)*ncart4 + (p4+o4)];
                D24V[p2*N4 + p4] = Dval;
                D24Vmax = std::max(D24Vmax, fabs(Dval));
            }}

            // Note: K-Invert
            size_t nprim1 = kvec12.nprim2();
            size_t nprim3 = kvec34.nprim2();
            const double * geom1 = kvec12.geom2().data();
            const double * geom3 = kvec34.geom2().data();
            const int * idx1 = kvec12.idx2().data();
            const int * idx3 = kvec34.idx2().data();
            const float * bounds12 = kvec12.bounds12().data();
            const float * bounds34 = kvec34.bounds12().data();

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

                double e12 = e1 + e2;
                double e12_m1 = 1.0/e12;

                bool did_work = false;
                for (size_t ind3 = 0; ind3 < nprim3; ind3++){
                    float bound34 = bounds34[ind3];
                    if (bound12*bound34*D24Vmax < thresp) { break; }
                    did_work = true;

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

                    double K13V[N1*N3];
                    ::memset(K13V, '\0', sizeof(double)*N1*N3);

                    // Rys quadrature
                    for (int i = 0; i < nrys; i++) {
                        double t = rysp->interpolate_t(nrys, i, T);
                        double w = rysp->interpolate_w(nrys, i, T);
                        double t2 = t*t*d2;
                        
                        double J2[3][V1*V2*V3*V4];
                        ::memset(J2[0], '\0', sizeof(double)*3*V1*V2*V3*V4);

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
                                double x1l = 1.0;
                            for (int l1 = 0; l1 < V1; l1++){
                                double x2l = 1.0;
                            for (int l2 = 0; l2 < V2; l2++){
                                double x3l = 1.0;
                            for (int l3 = 0; l3 < V3; l3++){
                                double x4l = 1.0;
                            for (int l4 = 0; l4 < V4; l4++){
                                (*J2x++) += wgh12*x1l*x2l*x3l*x4l;
                                x4l *= zeta2+xB[d]-x4[d];
                            } // l4
                                x3l *= zeta2+xB[d]-x3[d];
                            } // l3
                                x2l *= zeta1+xA[d]-x2[d];
                            } // l2
                                x1l *= zeta1+xA[d]-x1[d];
                            } // l1
                            } // d
                        }} // v1,v2

                        // => 6D Integrals <= //

                        // AM1 
                        for (int i1 = 0, index1 = 0; i1 <= L1; i1++) {
                            int l1 = L1 - i1;
                        for (int j1 = 0; j1 <= i1; j1++, index1++) {
                            int m1 = i1 - j1;
                            int n1 = j1;

                        // AM2
                        for (int i2 = 0, index2 = 0; i2 <= L2; i2++) {
                            int l2 = L2 - i2;
                        for (int j2 = 0; j2 <= i2; j2++, index2++) {
                            int m2 = i2 - j2;
                            int n2 = j2;

                        // AM3
                        for (int i3 = 0, index3 = 0; i3 <= L3; i3++) {
                            int l3 = L3 - i3;
                        for (int j3 = 0; j3 <= i3; j3++, index3++) {
                            int m3 = i3 - j3;
                            int n3 = j3;

                        // AM4
                        for (int i4 = 0, index4 = 0; i4 <= L4; i4++) {
                            int l4 = L4 - i4;
                        for (int j4 = 0; j4 <= i4; j4++, index4++) {
                            int m4 = i4 - j4;
                            int n4 = j4;

                            double I = pref1234*w*
                                J2[0][l1*V2*V3*V4 + l2*V3*V4 + l3*V4 + l4]*
                                J2[1][m1*V2*V3*V4 + m2*V3*V4 + m3*V4 + m4]*
                                J2[2][n1*V2*V3*V4 + n2*V3*V4 + n3*V4 + n4];
                            double D = D24V[index2*N4 + index4];
                            K13V[index1*N3 + index3] += I*D;

                        }}}}}}}} // AM (L^8)

                    } // nrys

                    int o1 = idx1[3*ind1 + 1];
                    int o3 = idx3[3*ind3 + 1];
                    for (int p1 = 0; p1 < N1; p1++){
                    for (int p3 = 0; p3 < N3; p3++){
                        #pragma omp atomic
                        K13p[(p1+o1)*ncart3 + (p3+o3)] += K13V[p1*N3 + p3];
                        if (flip24) {
                            #pragma omp atomic
                            K13p[(p3+o3)*ncart3 + (p1+o1)] += K13V[p1*N3 + p3];
                        }
                    }}

                    // ==> End Integral Kernel <== //

                } // ind3
                if (!did_work) { break; }
            } // ind1
        } // ind24 (OpenMP)
    }} // indl12,indl34
    } // ewald_ind

    // Cartesian-to-pure transform of K13
    std::shared_ptr<Tensor> K13U = PureTransform::cartToPure2(
        pairlist12->basis1(),
        pairlist34->basis1(),
        K13T,
        K13);
    
    return K13U;
}

} // namespace lightspeed
