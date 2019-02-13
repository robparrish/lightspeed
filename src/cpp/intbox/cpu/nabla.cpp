#include <lightspeed/intbox.hpp>
#include <lightspeed/am.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/resource_list.hpp>
#include <lightspeed/pair_list.hpp>
#include <lightspeed/pure_transform.hpp>
#include <lightspeed/gh.hpp>
#include <cmath>
#include <cstring>

namespace lightspeed { 

std::vector< std::shared_ptr<Tensor> > IntBox::nablaCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::vector< std::shared_ptr<Tensor> >& P
    )
{
    // Get a (ncart1,ncart2) register to accumulate Cartesian integrals
    std::vector< std::shared_ptr<Tensor> > P2;
    if (P.empty()) {
        for (int d = 0; d < 3; d++) {
            P2.push_back(
                    PureTransform::allocCart2(
                        pairlist->basis1(), 
                        pairlist->basis2()));
        }
    } else {
        for (int d = 0; d < 3; d++) {
            P2.push_back(
                    PureTransform::allocCart2(
                        pairlist->basis1(), 
                        pairlist->basis2(),
                        P[d]));
        }
    }

    // Target for Cartesian integral accumulation
    double* P2p[3];
    for (int d = 0; d < 3; d++) {
        P2p[d] = P2[d]->data().data();
    }

    // Sizes of target
    //int ncart1 = pairlist->basis1()->ncart();
    int ncart2 = pairlist->basis2()->ncart();

    // Gauss-Hermite quadrature rules
    std::shared_ptr<GH> gh = GH::instance();

    // Are we exploiting 1<->2 permutational symmetry?
    bool symm = pairlist->is_symmetric();
    
    // Loop over angular quanta pairs in pairlist
    for (int ang_ind = 0; ang_ind < pairlist->pairlists().size(); ang_ind++) {

        // reference to PairListL
        const PairListL& pairL = pairlist->pairlists()[ang_ind];
        // reference to actual pairs
        const std::vector<Pair>& pairsL = pairL.pairs();

        // number of GH nodes (+1 for nabla)
        const int ngh = (pairL.L1() + pairL.L2() + 1) / 2 + 1;
        // GH nodes needed
        const double* tgh = gh->ts()[ngh].data();
        const double* wgh = gh->ws()[ngh].data();

        int L1 = pairL.L1();
        int L2 = pairL.L2();
        int V1 = pairL.L1() + 1;
        int V2 = pairL.L2() + 1;

        #pragma omp parallel for num_threads(resources->nthread())
        for (size_t ind12 = 0; ind12 < pairsL.size(); ind12++) {
            const Pair& pair = pairsL[ind12];
            const Primitive& prim1 = pair.prim1();
            const Primitive& prim2 = pair.prim2();

            int off1 = prim1.cartIdx();
            int off2 = prim2.cartIdx();
            int K1 = prim1.primIdx();
            int K2 = prim2.primIdx();

            double eta = prim1.e() + prim2.e();
            double eta_m12 = pow(eta, -0.5);
 
            double X12 = (prim1.e() * prim1.x() + prim2.e() * prim2.x()) / eta;
            double Y12 = (prim1.e() * prim1.y() + prim2.e() * prim2.y()) / eta;
            double Z12 = (prim1.e() * prim1.z() + prim2.e() * prim2.z()) / eta;

            double px1[3] = { X12 - prim1.x(), Y12 - prim1.y(), Z12 - prim1.z() };
            double px2[3] = { X12 - prim2.x(), Y12 - prim2.y(), Z12 - prim2.z() };

            double A12 = prim1.e() * prim2.e() / eta;
            double r12_2 = pow(prim1.x() - prim2.x(), 2) +
                           pow(prim1.y() - prim2.y(), 2) +
                           pow(prim1.z() - prim2.z(), 2);
            double K12 = prim1.c() * prim2.c() * exp(-A12 * r12_2);

            // 1D Gaussian Integrals
            double S1D[3][V1*V2];
            double P1D[3][V1*V2];
            ::memset(S1D[0],'\0',sizeof(double)*3*V1*V2);
            ::memset(P1D[0],'\0',sizeof(double)*3*V1*V2);
            for (int v = 0; v < ngh; v++) {
                double xe = tgh[v] * eta_m12;
                double we = wgh[v] * eta_m12;
                for (int d = 0; d < 3; d ++) {
                    double* S1Dx = S1D[d];
                    double* P1Dx = P1D[d];
                    double x1 = xe + px1[d];
                    double x2 = xe + px2[d];
                    double pow_x1 = 1.0;
                    for (int i = 0; i < V1; i++) {
                        double pow_x2m = 0.0;
                        double pow_x2  = 1.0;
                        double pow_x2p = x2;
                        for (int j = 0; j < V2; j++) {
                            double Sval = we * pow_x1 * pow_x2;
                            double Pval = we * pow_x1 * (pow_x2m * j - pow_x2p * 2.0 * prim2.e());
                            (*S1Dx++) += Sval;
                            (*P1Dx++) += Pval;
                            pow_x2m = pow_x2;
                            pow_x2  = pow_x2p;
                            pow_x2p *= x2;
                        } // V2
                        pow_x1 *= x1;
                    } // V1
                } // d
            } // GH points

            double P2val[3];

            // 3D Gaussian Integrals
            int ind1 = 0;
            for (int i1 = 0; i1 <= L1; i1++) {
                int l1 = L1 - i1;
            for (int j1 = 0; j1 <= i1; j1++) {
                int m1 = i1 - j1;
                int n1 = j1;

            int ind2 = 0;
            for (int i2 = 0; i2 <= L2; i2++) {
                int l2 = L2 - i2;
            for (int j2 = 0; j2 <= i2; j2++) {
                int m2 = i2 - j2;
                int n2 = j2;

                // X
                P2val[0] = K12 * P1D[0][l1 * V2 + l2] * 
                                 S1D[1][m1 * V2 + m2] * 
                                 S1D[2][n1 * V2 + n2];
                // Y
                P2val[1] = K12 * S1D[0][l1 * V2 + l2] * 
                                 P1D[1][m1 * V2 + m2] * 
                                 S1D[2][n1 * V2 + n2];
                // Z
                P2val[2] = K12 * S1D[0][l1 * V2 + l2] * 
                                 S1D[1][m1 * V2 + m2] * 
                                 P1D[2][n1 * V2 + n2];

                for (int d = 0; d < 3; d++) {
                    #pragma omp atomic
                    P2p[d][(ind1 + off1) * ncart2 + (ind2 + off2)] += P2val[d];
                }
                if (symm && (K1 != K2)) {
                    for (int d = 0; d < 3; d++) {
                        #pragma omp atomic
                        P2p[d][(ind2 + off2) * ncart2 + (ind1 + off1)] -= P2val[d];
                    }
                }

                ind2++;
                }}
            ind1++;
            }} // End 3D Integrals
             
        } // pairL
    } // pairlist

    // Transform the Cartesian integrals to spherical
    std::vector< std::shared_ptr<Tensor> > P3;
    if (P.empty()) {
        for (int d = 0; d < 3; d++) {
            P3.push_back(
                    PureTransform::cartToPure2(
                        pairlist->basis1(), 
                        pairlist->basis2(), 
                        P2[d]));
        }
    } else {
        for (int d = 0; d < 3; d++) {
            P3.push_back(
                    PureTransform::cartToPure2(
                        pairlist->basis1(), 
                        pairlist->basis2(), 
                        P2[d],
                        P[d]));
        }
    }

    return P3;
}

} // namespace lightspeed
