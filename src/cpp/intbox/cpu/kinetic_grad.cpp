#include <lightspeed/intbox.hpp>
#include <lightspeed/am.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/resource_list.hpp>
#include <lightspeed/pair_list.hpp>
#include <lightspeed/pure_transform.hpp>
#include <lightspeed/gh.hpp>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace lightspeed { 
  
std::vector<std::shared_ptr<Tensor> > IntBox::kineticGradAdvCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::vector<std::shared_ptr<Tensor> >& G12
    )
{
    // sizes of G1 and G2
    std::vector<size_t> dim1;
    dim1.push_back(pairlist->basis1()->natom());
    dim1.push_back(3);
    std::vector<size_t> dim2;
    dim2.push_back(pairlist->basis2()->natom());
    dim2.push_back(3);

    // The working register of G1 and G2
    std::vector<std::shared_ptr<Tensor> > G12T;
    if (G12.size() == 0) {
        // The user has not allocated G12 and we should allocate this
        G12T.push_back(std::shared_ptr<Tensor>(new Tensor(dim1)));
        G12T.push_back(std::shared_ptr<Tensor>(new Tensor(dim2)));
    } else if (G12.size() == 2) {
        // The user has allocated G12 and we should check that this is the right size
        G12T = G12;
        G12T[0]->shape_error(dim1);
        G12T[1]->shape_error(dim2);
    } else {
        // The user doesn't know how many perturbations there are
        throw std::runtime_error("IntBox::overlapGradCPU: G12 should be size 0 or size 2");
    }

    double* G1p = G12T[0]->data().data();
    double* G2p = G12T[1]->data().data();

    // Pure-to-Cart transform of D
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

        // number of GH nodes (+3 for gradient of kinetic energy)
        const int ngh = (pairL.L1() + pairL.L2() + 3) / 2 + 1;
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

            // 1D Integrals
            double S1D[3][V1*V2];
            double P1D[3][V1*V2];
            double Q1D[3][V1*V2];
            double T1D[3][V1*V2];
            double U1D[3][V1*V2];
            double V1D[3][V1*V2];
            ::memset(S1D[0],'\0',3*V1*V2*sizeof(double)); 
            ::memset(P1D[0],'\0',3*V1*V2*sizeof(double)); 
            ::memset(Q1D[0],'\0',3*V1*V2*sizeof(double)); 
            ::memset(T1D[0],'\0',3*V1*V2*sizeof(double)); 
            ::memset(U1D[0],'\0',3*V1*V2*sizeof(double)); 
            ::memset(V1D[0],'\0',3*V1*V2*sizeof(double)); 
            for (int v = 0; v < ngh; v++) {
                double xe = tgh[v] * eta_m12;
                double we = wgh[v] * eta_m12;
                for (int d = 0; d < 3; d ++) {
                    double* S1Dx = S1D[d];
                    double* P1Dx = P1D[d];
                    double* Q1Dx = Q1D[d];
                    double* T1Dx = T1D[d];
                    double* U1Dx = U1D[d];
                    double* V1Dx = V1D[d];
                    double x1 = xe + px1[d];
                    double x2 = xe + px2[d];
                    double x1lm2 = 0.0;
                    double x1lm = 0.0;
                    double x1l = 1.0;
                    double x1lp = x1;
                    double x1lp2 = x1*x1;
                    for (int i = 0; i < V1; i++) {
                        double x2lm2 = 0.0;
                        double x2lm = 0.0;
                        double x2l = 1.0;
                        double x2lp = x2;
                        double x2lp2 = x2*x2;
                        for (int j = 0; j < V2; j++) {
                            double S1Dval = we * x1l * x2l;
                            double P1Dval = we * (i * x1lm - 2.0 * prim1.e() * x1lp) * x2l;
                            double Q1Dval = we * x1l * (j * x2lm - 2.0 * prim2.e() * x2lp);
                            double T1Dval = we * 
                                (i * x1lm - 2.0 * prim1.e() * x1lp) * 
                                (j * x2lm - 2.0 * prim2.e() * x2lp);
                            double U1Dval = we * 
                                (i * (i-1) * x1lm2 - 2.0 * prim1.e() * (2*i+1) * x1l + 4.0 * pow(prim1.e(),2) * x1lp2) * 
                                (j * x2lm - 2.0 * prim2.e() * x2lp);
                            double V1Dval = we * 
                                (i * x1lm - 2.0 * prim1.e() * x1lp) * 
                                (j * (j-1) * x2lm2 - 2.0 * prim2.e() * (2*j+1) * x2l + 4.0 * pow(prim2.e(),2) * x2lp2);
                            (*S1Dx++) += S1Dval;
                            (*P1Dx++) -= P1Dval;
                            (*Q1Dx++) -= Q1Dval;
                            (*T1Dx++) += T1Dval;
                            (*U1Dx++) -= U1Dval;
                            (*V1Dx++) -= V1Dval;
                            x2lm2 = x2lm;
                            x2lm = x2l;
                            x2l = x2lp;
                            x2lp = x2lp2;
                            x2lp2 *= x2;
                        } // V2
                        x1lm2 = x1lm;
                        x1lm = x1l;
                        x1l = x1lp;
                        x1lp = x1lp2;
                        x1lp2 *= x1;
                    } // V1
                } // d
            } // GH points

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

                    double PXval = K12 * 0.5 * (
                                   U1D[0][l1 * V2 + l2] * 
                                   S1D[1][m1 * V2 + m2] * 
                                   S1D[2][n1 * V2 + n2] +
                                   P1D[0][l1 * V2 + l2] * 
                                   T1D[1][m1 * V2 + m2] * 
                                   S1D[2][n1 * V2 + n2] +
                                   P1D[0][l1 * V2 + l2] * 
                                   S1D[1][m1 * V2 + m2] * 
                                   T1D[2][n1 * V2 + n2]);

                    double PYval = K12 * 0.5 * (
                                   T1D[0][l1 * V2 + l2] * 
                                   P1D[1][m1 * V2 + m2] * 
                                   S1D[2][n1 * V2 + n2] +
                                   S1D[0][l1 * V2 + l2] * 
                                   U1D[1][m1 * V2 + m2] * 
                                   S1D[2][n1 * V2 + n2] +
                                   S1D[0][l1 * V2 + l2] * 
                                   P1D[1][m1 * V2 + m2] * 
                                   T1D[2][n1 * V2 + n2]);

                    double PZval = K12 * 0.5 * (
                                   T1D[0][l1 * V2 + l2] * 
                                   S1D[1][m1 * V2 + m2] * 
                                   P1D[2][n1 * V2 + n2] +
                                   S1D[0][l1 * V2 + l2] * 
                                   T1D[1][m1 * V2 + m2] * 
                                   P1D[2][n1 * V2 + n2] +
                                   S1D[0][l1 * V2 + l2] * 
                                   S1D[1][m1 * V2 + m2] * 
                                   U1D[2][n1 * V2 + n2]);

                    double QXval = K12 * 0.5 * (
                                   V1D[0][l1 * V2 + l2] * 
                                   S1D[1][m1 * V2 + m2] * 
                                   S1D[2][n1 * V2 + n2] +
                                   Q1D[0][l1 * V2 + l2] * 
                                   T1D[1][m1 * V2 + m2] * 
                                   S1D[2][n1 * V2 + n2] +
                                   Q1D[0][l1 * V2 + l2] * 
                                   S1D[1][m1 * V2 + m2] * 
                                   T1D[2][n1 * V2 + n2]);

                    double QYval = K12 * 0.5 * (
                                   T1D[0][l1 * V2 + l2] * 
                                   Q1D[1][m1 * V2 + m2] * 
                                   S1D[2][n1 * V2 + n2] +
                                   S1D[0][l1 * V2 + l2] * 
                                   V1D[1][m1 * V2 + m2] * 
                                   S1D[2][n1 * V2 + n2] +
                                   S1D[0][l1 * V2 + l2] * 
                                   Q1D[1][m1 * V2 + m2] * 
                                   T1D[2][n1 * V2 + n2]);

                    double QZval = K12 * 0.5 * (
                                   T1D[0][l1 * V2 + l2] * 
                                   S1D[1][m1 * V2 + m2] * 
                                   Q1D[2][n1 * V2 + n2] +
                                   S1D[0][l1 * V2 + l2] * 
                                   T1D[1][m1 * V2 + m2] * 
                                   Q1D[2][n1 * V2 + n2] +
                                   S1D[0][l1 * V2 + l2] * 
                                   S1D[1][m1 * V2 + m2] * 
                                   V1D[2][n1 * V2 + n2]);
		
                    double Dval = D2p[(ind1 + off1) * ncart2 + (ind2 + off2)];
                    int PA = prim1.atomIdx();
                    int PB = prim2.atomIdx();
		
                    #pragma omp atomic
                    G1p[3 * PA + 0] += Dval * PXval;  
                    #pragma omp atomic
                    G1p[3 * PA + 1] += Dval * PYval;  
                    #pragma omp atomic
                    G1p[3 * PA + 2] += Dval * PZval;  
		
                    #pragma omp atomic
                    G2p[3 * PB + 0] += Dval * QXval;  
                    #pragma omp atomic
                    G2p[3 * PB + 1] += Dval * QYval;  
                    #pragma omp atomic
                    G2p[3 * PB + 2] += Dval * QZval;  
		
                    if (symm && (K1 != K2)) {
                        Dval = D2p[(ind2 + off2) * ncart2 + (ind1 + off1)]; 
                        PA = prim2.atomIdx();
                        PB = prim1.atomIdx();

                        #pragma omp atomic
                        G1p[3 * PA + 0] += Dval * QXval;  
                        #pragma omp atomic
                        G1p[3 * PA + 1] += Dval * QYval;  
                        #pragma omp atomic
                        G1p[3 * PA + 2] += Dval * QZval;  
		
                        #pragma omp atomic
                        G2p[3 * PB + 0] += Dval * PXval;  
                        #pragma omp atomic
                        G2p[3 * PB + 1] += Dval * PYval;  
                        #pragma omp atomic
                        G2p[3 * PB + 2] += Dval * PZval;  
                    }  

                    ind2++;
                }}

                ind1++;
            }} // End 3D Integrals
             
        } // pairL
    }// pairlist
    
    // Now we've got a (natom1,3) Tensor at G12T[0] and a (natom2,3) Tensor at G12T[1]
    // Return the vector of gradients to the user
    return G12T;
}

} // namespace lightspeed
