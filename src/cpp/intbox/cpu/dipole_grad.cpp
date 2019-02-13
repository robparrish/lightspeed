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

std::vector< std::vector< std::shared_ptr<Tensor> >> IntBox::dipoleGradAdvCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    double x0,
    double y0,
    double z0,
    const std::shared_ptr<Tensor> & D,
    const std::vector< std::vector<std::shared_ptr<Tensor>> >& Dipole
    )
{

    // Size of the Dipole matrix
    std::vector<size_t> dim1;
    dim1.push_back(pairlist->basis1()->natom());
    dim1.push_back(3);
    std::vector<size_t> dim2;
    dim2.push_back(pairlist->basis1()->natom());
    dim2.push_back(3);

    // The working register of Dipole
    std::vector< std::vector<std::shared_ptr<Tensor>> > Dipole2 = Dipole;
    if (Dipole.size() == 0) {
        // The user has not allocated G12. We allocate it here.
        for (int d = 0; d < 3; d++) {
            Dipole2.push_back(std::vector< std::shared_ptr<Tensor> >());
            Dipole2[d].push_back(std::shared_ptr<Tensor>(new Tensor(dim1)));
            Dipole2[d].push_back(std::shared_ptr<Tensor>(new Tensor(dim2)));
        }
    } else if (Dipole.size() == 3) {
        // Dipole has been allocated. Check that it is of the right size
        for (int d = 0; d <3; d++) {
            Dipole2[d][0]->shape_error(dim1);
            Dipole2[d][1]->shape_error(dim2);
        }
    } else {
        throw std::runtime_error("IntBox::dipoleGradAdvCPU: G12 should be of size 0 or size 2");
    }

    double* Dip1p[3];
    double* Dip2p[3];
    for (int d = 0; d < 3; d++) {
        Dip1p[d] = Dipole2[d][0]->data().data();  
        Dip2p[d] = Dipole2[d][1]->data().data();  
    }
    // Pure-to-Cart transform of the density matrix
    std::shared_ptr<Tensor> D2 = PureTransform::pureToCart2(
        pairlist->basis1(),
        pairlist->basis2(),
        D);

    double* D2p = D2->data().data();
   
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

        // number of GH nodes (+1 for dipole, +1 for gradient)
        const int ngh = (pairL.L1() + pairL.L2() + 2) / 2 + 1;
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

            double px0[3] = { X12 - x0, Y12 - y0, Z12 - z0 };

            double A12 = prim1.e() * prim2.e() / eta;
            double r12_2 = pow(prim1.x() - prim2.x(), 2) +
                           pow(prim1.y() - prim2.y(), 2) +
                           pow(prim1.z() - prim2.z(), 2);
            double K12 = prim1.c() * prim2.c() * exp(-A12 * r12_2);

            // 1D Gaussian Integrals
            double S1D[3][V1*V2];
            double P1D[3][V1*V2];
            double Q1D[3][V1*V2];
            double D1D[3][V1*V2];
            double PD1D[3][V1*V2];
            double QD1D[3][V1*V2];
            ::memset(S1D[0],'\0',sizeof(double)*3*V1*V2);
            ::memset(D1D[0],'\0',sizeof(double)*3*V1*V2);
            ::memset(P1D[0],'\0',sizeof(double)*3*V1*V2);
            ::memset(Q1D[0],'\0',sizeof(double)*3*V1*V2);
            ::memset(PD1D[0],'\0',sizeof(double)*3*V1*V2);
            ::memset(QD1D[0],'\0',sizeof(double)*3*V1*V2);
            for (int v = 0; v < ngh; v++) {
                double xe = tgh[v] * eta_m12;
                double we = wgh[v] * eta_m12;
                for (int d = 0; d < 3; d ++) {
                    double* S1Dx  = S1D[d];
                    double* D1Dx  = D1D[d];
                    double* P1Dx  = P1D[d];
                    double* Q1Dx  = Q1D[d];
                    double* PD1Dx = PD1D[d];
                    double* QD1Dx = QD1D[d];
                    double x1 = xe + px1[d];
                    double x2 = xe + px2[d];
                    double x0v = xe + px0[d];
                    double x1lm = 0.0;
                    double x1l  = 1.0;
                    double x1lp = x1;
                    for (int i = 0; i < V1; i++) {
                        double x2lm = 0.0;
                        double x2l  = 1.0;
                        double x2lp = x2;
                        for (int j = 0; j < V2; j++) {
                            double S1Dval = we * x1l * x2l;
                            double P1Dval = we * (i * x1lm - 2.0 * prim1.e() * x1lp) * x2l; 
                            double Q1Dval = we * (j * x2lm - 2.0 * prim2.e() * x2lp) * x1l; 
                            double Dval = S1Dval * x0v;
                            double PD1Dval = P1Dval * x0v;
                            double QD1Dval = Q1Dval * x0v;
                            (*S1Dx++)  += S1Dval;
                            (*D1Dx++)  += Dval;
                            (*P1Dx++)  -= P1Dval;
                            (*Q1Dx++)  -= Q1Dval;
                            (*PD1Dx++) -= PD1Dval;
                            (*QD1Dx++) -= QD1Dval;
                            x2lm = x2l;
                            x2l  = x2lp;
                            x2lp *= x2;
                        } // V2
                        x1lm = x1l;
                        x1l  = x1lp;
                        x1lp *= x1;
                    } // V1
                } // d
            } // GH points

            double PD[3][3];
            double QD[3][3];

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
            
            // Bra center
                // Dipole X - derivative X
                PD[0][0] = K12 * PD1D[0][l1 * V2 + l2] * 
                                 S1D[1][m1 * V2 + m2] * 
                                 S1D[2][n1 * V2 + n2];
                // Dipole X - derivative Y
                PD[0][1] = K12 * D1D[0][l1 * V2 + l2] * 
                                 P1D[1][m1 * V2 + m2] * 
                                 S1D[2][n1 * V2 + n2];
                // Dipole X - derivative Z
                PD[0][2] = K12 * D1D[0][l1 * V2 + l2] * 
                                 S1D[1][m1 * V2 + m2] * 
                                 P1D[2][n1 * V2 + n2];

                // Dipole Y - derivative X
                PD[1][0] = K12 * P1D[0][l1 * V2 + l2] * 
                                 D1D[1][m1 * V2 + m2] * 
                                 S1D[2][n1 * V2 + n2];
                // Dipole Y - derivative Y
                PD[1][1] = K12 * S1D[0][l1 * V2 + l2] * 
                                 PD1D[1][m1 * V2 + m2] * 
                                 S1D[2][n1 * V2 + n2];
                // Dipole Y - derivative Z
                PD[1][2] = K12 * S1D[0][l1 * V2 + l2] * 
                                 D1D[1][m1 * V2 + m2] * 
                                 P1D[2][n1 * V2 + n2];

                // Dipole Z - derivative X
                PD[2][0] = K12 * P1D[0][l1 * V2 + l2] * 
                                 S1D[1][m1 * V2 + m2] * 
                                 D1D[2][n1 * V2 + n2];
                // Dipole Z - derivative Y
                PD[2][1] = K12 * S1D[0][l1 * V2 + l2] * 
                                 P1D[1][m1 * V2 + m2] * 
                                 D1D[2][n1 * V2 + n2];
                // Dipole Z - derivative Z
                PD[2][2] = K12 * S1D[0][l1 * V2 + l2] * 
                                 S1D[1][m1 * V2 + m2] * 
                                 PD1D[2][n1 * V2 + n2];
            // Ket center
                // Dipole X - derivative X
                QD[0][0] = K12 * QD1D[0][l1 * V2 + l2] * 
                                 S1D[1][m1 * V2 + m2] * 
                                 S1D[2][n1 * V2 + n2];
                // Dipole X - derivative Y
                QD[0][1] = K12 * D1D[0][l1 * V2 + l2] * 
                                 Q1D[1][m1 * V2 + m2] * 
                                 S1D[2][n1 * V2 + n2];
                // Dipole X - derivative Z
                QD[0][2] = K12 * D1D[0][l1 * V2 + l2] * 
                                 S1D[1][m1 * V2 + m2] * 
                                 Q1D[2][n1 * V2 + n2];

                // Dipole Y - derivative X
                QD[1][0] = K12 * Q1D[0][l1 * V2 + l2] * 
                                 D1D[1][m1 * V2 + m2] * 
                                 S1D[2][n1 * V2 + n2];
                // Dipole Y - derivative Y
                QD[1][1] = K12 * S1D[0][l1 * V2 + l2] * 
                                 QD1D[1][m1 * V2 + m2] * 
                                 S1D[2][n1 * V2 + n2];
                // Dipole Y - derivative Z
                QD[1][2] = K12 * S1D[0][l1 * V2 + l2] * 
                                 D1D[1][m1 * V2 + m2] * 
                                 Q1D[2][n1 * V2 + n2];

                // Dipole Z - derivative X
                QD[2][0] = K12 * Q1D[0][l1 * V2 + l2] * 
                                 S1D[1][m1 * V2 + m2] * 
                                 D1D[2][n1 * V2 + n2];
                // Dipole Z - derivative Y
                QD[2][1] = K12 * S1D[0][l1 * V2 + l2] * 
                                 Q1D[1][m1 * V2 + m2] * 
                                 D1D[2][n1 * V2 + n2];
                // Dipole Z - derivative Z
                QD[2][2] = K12 * S1D[0][l1 * V2 + l2] * 
                                 S1D[1][m1 * V2 + m2] * 
                                 QD1D[2][n1 * V2 + n2];

                double Dval = D2p[(ind1 + off1) * ncart2 + (ind2 + off2)];  
                int PA = prim1.atomIdx();
                int PB = prim2.atomIdx();

                double * Matp;
                int idx = (ind1 + off1) * ncart2 + (ind2 + off2);
                for (int d = 0; d < 3; d++) {
                    #pragma omp atomic
                    Dip1p[d][3 * PA + 0] += Dval * PD[d][0]; 
                    #pragma omp atomic
                    Dip1p[d][3 * PA + 1] += Dval * PD[d][1];
                    #pragma omp atomic
                    Dip1p[d][3 * PA + 2] += Dval * PD[d][2];

                    #pragma omp atomic
                    Dip2p[d][3 * PB + 0] += Dval * QD[d][0];
                    #pragma omp atomic
                    Dip2p[d][3 * PB + 1] += Dval * QD[d][1];
                    #pragma omp atomic
                    Dip2p[d][3 * PB + 2] += Dval * QD[d][2];
                }

                if (symm && (K1 != K2)) {
                    int idx = (ind2 + off2) * ncart2 + (ind1 + off1);
                    Dval = D2p[(ind2 + off2) * ncart2 + (ind1 + off1)];
                    PA = prim2.atomIdx();
                    PB = prim1.atomIdx();

                    for (int d = 0; d < 3; d++) {
                        #pragma omp atomic
                        Dip1p[d][3 * PA + 0] += Dval * QD[d][0];
                        #pragma omp atomic
                        Dip1p[d][3 * PA + 1] += Dval * QD[d][1];
                        #pragma omp atomic
                        Dip1p[d][3 * PA + 2] += Dval * QD[d][2];

                        #pragma omp atomic
                        Dip2p[d][3 * PB + 0] += Dval * PD[d][0];
                        #pragma omp atomic
                        Dip2p[d][3 * PB + 1] += Dval * PD[d][1];
                        #pragma omp atomic
                        Dip2p[d][3 * PB + 2] += Dval * PD[d][2];
                    }
                }

                ind2++;
                }}

            ind1++;
            }} // End 3D Integrals

        } // pairL
    } // pairlist

    return Dipole2;
}

} // namespace lightspeed
