#include <lightspeed/hermite.hpp>
#include <lightspeed/gh.hpp>
#include <cmath>
#include <cstring>
#include <cstdio>

namespace lightspeed {
    
void HermiteL::cart_to_herm(
    const std::shared_ptr<Tensor>& Dcart,
    bool phases
    )
{
    int L1 = L1_;    
    int L2 = L2_;    
    int L = L_;    
    int H = H_;

    int N1 = (L1 + 1) * (L1 + 2) / 2;
    int N2 = (L2 + 1) * (L2 + 2) / 2;

    int V1 = L1 + 1;
    int V2 = L2 + 1;
    int V = L + 1;

    const std::vector<Pair>& pairlist = pairs_->pairs();
    size_t nao2 = Dcart->shape()[1];
    bool symm = pairs_->is_symmetric();

    std::shared_ptr<GH> gh = GH::instance();
    const std::vector<double>& ts = gh->ts()[V];
    const std::vector<double>& ws = gh->ws()[V];

    double* DA = data_.data();
    const double* D12 = Dcart->data().data();

    #pragma omp parallel for
    for (size_t ind = 0; ind < pairlist.size(); ind++) {

        // Pair extraction
        const Pair& pair = pairlist[ind];
        const Primitive& prim1 = pair.prim1();
        const Primitive& prim2 = pair.prim2();

        // Geometry
        double c1 = prim1.c();
        double e1 = prim1.e();
        double r1[3] = {
            prim1.x(),
            prim1.y(),
            prim1.z() };

        double c2 = prim2.c();
        double e2 = prim2.e();
        double r2[3] = {
            prim2.x(),
            prim2.y(),
            prim2.z() };

        double p = e1 + e2;
        double pm1 = 1.0 / p;
        double pm12 = pow(p, -0.5);

        double rAP[3] = {
            e2 * (r1[0] - r2[0]) * pm1,
            e2 * (r1[1] - r2[1]) * pm1,
            e2 * (r1[2] - r2[2]) * pm1 };

        double rPA[3] = {
            e2 * (r2[0] - r1[0]) * pm1,
            e2 * (r2[1] - r1[1]) * pm1,
            e2 * (r2[2] - r1[2]) * pm1 };

        double rPB[3] = {
            e1 * (r1[0] - r2[0]) * pm1,
            e1 * (r1[1] - r2[1]) * pm1,
            e1 * (r1[2] - r2[2]) * pm1 };
    
        double K = c1 * c2 * exp(-e1 * e2 * pm1 * (
            (r1[0] - r2[0]) * (r1[0] - r2[0]) +
            (r1[1] - r2[1]) * (r1[1] - r2[1]) +
            (r1[2] - r2[2]) * (r1[2] - r2[2])));

        // Hermite expansion coefficients
        double E[3][V*V1*V2];
        ::memset(&E[0][0],'\0',3*V*V1*V2*sizeof(double));
        for (int d = 0; d < 3; d++) {
            double xPA = rPA[d];
            double xPB = rPB[d];
            double* Ex = E[d];
            for (int t = 0; t < V; t++) {
                
                // Quadrature points
                double tgh = ts[t];
                double wgh = ws[t];

                // Change of variable
                double xgh = tgh * pm12;
                double ugh = wgh * pm12;
                double zP = xgh; 
                double zPA = xgh + xPA; 
                double zPB = xgh + xPB; 

                // Auxiliary Hermite Polynomials
                double chi[V];
                chi[0] = ugh * sqrt(p / M_PI);
                if (V > 1) chi[1] = zP * chi[0];
                for (int t = 1; t < V - 1; t++) {
                    chi[t+1] = 1.0 / (t + 1) * (zP * chi[t] - 0.5 * pm1 * chi[t-1]);
                }

                // Cartesian Monomials
                double X1[V1];
                X1[0] = 1.0;
                for (int l1 = 1; l1 < V1; l1++) {
                    X1[l1] = X1[l1-1] * zPA;
                }
                double X2[V2];
                X2[0] = 1.0;
                for (int l2 = 1; l2 < V2; l2++) {
                    X2[l2] = X2[l2-1] * zPB;
                }

                // Assembly
                for (int L = 0; L < V; L++) {
                    for (int l1 = 0; l1 < V1; l1++) {
                        for (int l2 = 0; l2 < V2; l2++) {
                            if (L > (l1 + l2)) continue;
                            Ex[L * V1 * V2 + l1 * V2 + l2] += chi[L] * X1[l1] * X2[l2];
                        }
                    } 
                }
            }
        }

        const double* Ex = E[0];
        const double* Ey = E[1];
        const double* Ez = E[2];

        // Temporary Source
        double D12T[N1 * N2];

        // Density matrix extraction
        int off1 = prim1.cartIdx(); 
        int off2 = prim2.cartIdx(); 
        if (symm) {
            double scale = (prim1.primIdx() == prim2.primIdx() ? 0.5 : 1.0);
            for (int p = 0; p < N1; p++) {
                for (int q = 0; q < N2; q++) {
                    D12T[p * N2 + q] = D12[(p + off1) * nao2 + (q + off2)];
                    D12T[p * N2 + q] += D12[(q + off2) * nao2 + (p + off1)];
                    D12T[p * N2 + q] *= scale;
                }
            }
        } else { 
            for (int p = 0; p < N1; p++) {
                for (int q = 0; q < N2; q++) {
                    D12T[p * N2 + q] = D12[(p + off1) * nao2 + (q + off2)];
                }
            }
        }

        // Target
        double* DAT = DA + ind * H;
        
        // Basis function 1
        for (int i1 = 0, index1 = 0; i1 <= L1; i1++) {
            int l1 = L1 - i1;
        for (int j1 = 0; j1 <= i1; j1++, index1++) {
            int m1 = i1 - j1;
            int n1 = j1; 

        // Basis function 2
        for (int i2 = 0, index2 = 0; i2 <= L2; i2++) {
            int l2 = L2 - i2;
        for (int j2 = 0; j2 <= i2; j2++, index2++) {
            int m2 = i2 - j2;
            int n2 = j2; 

            // Cartesian basis density element
            double Dval = D12T[index1 * N2 + index2];

            // Hermite Gaussian L
            for (int LH = 0, indexH = 0; LH < V; LH++) {
            
                // Hermite Gaussian shell
                for (int iH = 0; iH <= LH; iH++) {
                    int lH = LH - iH;
                for (int jH = 0; jH <= iH; jH++, indexH++) {
                    int mH = iH - jH;
                    int nH = jH; 
                    double sign = 1.0;
                    if (phases) {
                        sign = ((lH + mH + nH) % 2 ? -1.0 : 1.0);
                    }
                    DAT[indexH] += sign * K * 
                        Ex[lH * V1 * V2 + l1 * V2 + l2] *
                        Ey[mH * V1 * V2 + m1 * V2 + m2] *
                        Ez[nH * V1 * V2 + n1 * V2 + n2] *
                        Dval;
                }}
            }
        }}
        }}
    }
}
void HermiteL::herm_to_cart(
    const std::shared_ptr<Tensor>& Dcart,
    bool phases
    ) const
{
    int L1 = L1_;    
    int L2 = L2_;    
    int L = L_;    
    int H = H_;

    int N1 = (L1 + 1) * (L1 + 2) / 2;
    int N2 = (L2 + 1) * (L2 + 2) / 2;

    int V1 = L1 + 1;
    int V2 = L2 + 1;
    int V = L + 1;

    const std::vector<Pair>& pairlist = pairs_->pairs();
    size_t nao2 = Dcart->shape()[1];
    bool symm = pairs_->is_symmetric();

    std::shared_ptr<GH> gh = GH::instance();
    const std::vector<double>& ts = gh->ts()[V];
    const std::vector<double>& ws = gh->ws()[V];

    const double* DA = data_.data();
    double* D12 = Dcart->data().data();

    #pragma omp parallel for
    for (size_t ind = 0; ind < pairlist.size(); ind++) {

        // Pair extraction
        const Pair& pair = pairlist[ind];
        const Primitive& prim1 = pair.prim1();
        const Primitive& prim2 = pair.prim2();

        // Geometry
        double c1 = prim1.c();
        double e1 = prim1.e();
        double r1[3] = {
            prim1.x(),
            prim1.y(),
            prim1.z() };

        double c2 = prim2.c();
        double e2 = prim2.e();
        double r2[3] = {
            prim2.x(),
            prim2.y(),
            prim2.z() };

        double p = e1 + e2;
        double pm1 = 1.0 / p;
        double pm12 = pow(p, -0.5);

        double rAP[3] = {
            e2 * (r1[0] - r2[0]) * pm1,
            e2 * (r1[1] - r2[1]) * pm1,
            e2 * (r1[2] - r2[2]) * pm1 };

        double rPA[3] = {
            e2 * (r2[0] - r1[0]) * pm1,
            e2 * (r2[1] - r1[1]) * pm1,
            e2 * (r2[2] - r1[2]) * pm1 };

        double rPB[3] = {
            e1 * (r1[0] - r2[0]) * pm1,
            e1 * (r1[1] - r2[1]) * pm1,
            e1 * (r1[2] - r2[2]) * pm1 };
    
        double K = c1 * c2 * exp(-e1 * e2 * pm1 * (
            (r1[0] - r2[0]) * (r1[0] - r2[0]) +
            (r1[1] - r2[1]) * (r1[1] - r2[1]) +
            (r1[2] - r2[2]) * (r1[2] - r2[2])));

        // Hermite expansion coefficients
        double E[3][V*V1*V2];
        ::memset(&E[0][0],'\0',3*V*V1*V2*sizeof(double));
        for (int d = 0; d < 3; d++) {
            double xPA = rPA[d];
            double xPB = rPB[d];
            double* Ex = E[d];
            for (int t = 0; t < V; t++) {
                
                // Quadrature points
                double tgh = ts[t];
                double wgh = ws[t];

                // Change of variable
                double xgh = tgh * pm12;
                double ugh = wgh * pm12;
                double zP = xgh; 
                double zPA = xgh + xPA; 
                double zPB = xgh + xPB; 

                // Auxiliary Hermite Polynomials
                double chi[V];
                chi[0] = ugh * sqrt(p / M_PI);
                if (V > 1) chi[1] = zP * chi[0];
                for (int t = 1; t < V - 1; t++) {
                    chi[t+1] = 1.0 / (t + 1) * (zP * chi[t] - 0.5 * pm1 * chi[t-1]);
                }

                // Cartesian Monomials
                double X1[V1];
                X1[0] = 1.0;
                for (int l1 = 1; l1 < V1; l1++) {
                    X1[l1] = X1[l1-1] * zPA;
                }
                double X2[V2];
                X2[0] = 1.0;
                for (int l2 = 1; l2 < V2; l2++) {
                    X2[l2] = X2[l2-1] * zPB;
                }

                // Assembly
                for (int L = 0; L < V; L++) {
                    for (int l1 = 0; l1 < V1; l1++) {
                        for (int l2 = 0; l2 < V2; l2++) {
                            if (L > (l1 + l2)) continue;
                            Ex[L * V1 * V2 + l1 * V2 + l2] += chi[L] * X1[l1] * X2[l2];
                        }
                    } 
                }
            }
        }

        const double* Ex = E[0];
        const double* Ey = E[1];
        const double* Ez = E[2];

        // Source 
        const double* DAT = DA + ind * H;

        // Temp Target
        double D12T[N1 * N2];
        
        // Basis function 1
        for (int i1 = 0, index1 = 0; i1 <= L1; i1++) {
            int l1 = L1 - i1;
        for (int j1 = 0; j1 <= i1; j1++, index1++) {
            int m1 = i1 - j1;
            int n1 = j1; 

        // Basis function 2
        for (int i2 = 0, index2 = 0; i2 <= L2; i2++) {
            int l2 = L2 - i2;
        for (int j2 = 0; j2 <= i2; j2++, index2++) {
            int m2 = i2 - j2;
            int n2 = j2; 

            // Cartesian basis density element
            double Dval = 0.0;

            // Hermite Gaussian L
            for (int LH = 0, indexH = 0; LH < V; LH++) {
            
                // Hermite Gaussian shell
                for (int iH = 0; iH <= LH; iH++) {
                    int lH = LH - iH;
                for (int jH = 0; jH <= iH; jH++, indexH++) {
                    int mH = iH - jH;
                    int nH = jH; 
                    double sign = 1.0;
                    if (phases) {
                        sign = ((lH + mH + nH) % 2 ? -1.0 : 1.0);
                    }
                    Dval += sign * K * 
                        Ex[lH * V1 * V2 + l1 * V2 + l2] *
                        Ey[mH * V1 * V2 + m1 * V2 + m2] *
                        Ez[nH * V1 * V2 + n1 * V2 + n2] *
                        DAT[indexH];
                }}

            }

            D12T[index1 * N2 + index2] = Dval;

        }}
        }}

        // Density matrix placement of Cartesian functions
        int off1 = prim1.cartIdx();
        int off2 = prim2.cartIdx();
        if (symm) {
            double scale = (prim1.primIdx() == prim2.primIdx() ? 0.5 : 1.0);
            for (int p = 0; p < N1; p++) {
                for (int q = 0; q < N2; q++) {
                    D12T[p * N2 + q] *= scale;
                    #pragma omp atomic
                    D12[(p + off1) * nao2 + (q + off2)] += D12T[p * N2 + q];
                    #pragma omp atomic
                    D12[(q + off2) * nao2 + (p + off1)] += D12T[p * N2 + q];
                }
            }
        } else {
            for (int p = 0; p < N1; p++) {
                for (int q = 0; q < N2; q++) {
                    #pragma omp atomic
                    D12[(p + off1) * nao2 + (q + off2)] += D12T[p * N2 + q];
                }
            }
        }
    }
}
void HermiteL::grad(
    const std::shared_ptr<Tensor>& Dcart,
    const std::shared_ptr<Tensor>& G1,
    const std::shared_ptr<Tensor>& G2,
    bool phases
    ) const
{
    int L1 = L1_;    
    int L2 = L2_;    
    int L = L_;    
    int H = H_;

    int N1 = (L1 + 1) * (L1 + 2) / 2;
    int N2 = (L2 + 1) * (L2 + 2) / 2;

    int V1 = L1 + 1;
    int V2 = L2 + 1;
    int V = L + 1;

    const std::vector<Pair>& pairlist = pairs_->pairs();
    size_t nao2 = Dcart->shape()[1];
    bool symm = pairs_->is_symmetric();

    std::shared_ptr<GH> gh = GH::instance();
    const std::vector<double>& ts = gh->ts()[V];
    const std::vector<double>& ws = gh->ws()[V];

    const double* DA = data_.data();
    const double* D12 = Dcart->data().data();
    double* G1p = G1->data().data();
    double* G2p = G2->data().data();

    #pragma omp parallel for
    for (size_t ind = 0; ind < pairlist.size(); ind++) {

        // Pair extraction
        const Pair& pair = pairlist[ind];
        const Primitive& prim1 = pair.prim1();
        const Primitive& prim2 = pair.prim2();

        // Geometry
        double c1 = prim1.c();
        double e1 = prim1.e();
        double r1[3] = {
            prim1.x(),
            prim1.y(),
            prim1.z() };

        double c2 = prim2.c();
        double e2 = prim2.e();
        double r2[3] = {
            prim2.x(),
            prim2.y(),
            prim2.z() };

        double p = e1 + e2;
        double pm1 = 1.0 / p;
        double pm12 = pow(p, -0.5);

        double rAP[3] = {
            e2 * (r1[0] - r2[0]) * pm1,
            e2 * (r1[1] - r2[1]) * pm1,
            e2 * (r1[2] - r2[2]) * pm1 };

        double rPA[3] = {
            e2 * (r2[0] - r1[0]) * pm1,
            e2 * (r2[1] - r1[1]) * pm1,
            e2 * (r2[2] - r1[2]) * pm1 };

        double rPB[3] = {
            e1 * (r1[0] - r2[0]) * pm1,
            e1 * (r1[1] - r2[1]) * pm1,
            e1 * (r1[2] - r2[2]) * pm1 };
    
        double K = c1 * c2 * exp(-e1 * e2 * pm1 * (
            (r1[0] - r2[0]) * (r1[0] - r2[0]) +
            (r1[1] - r2[1]) * (r1[1] - r2[1]) +
            (r1[2] - r2[2]) * (r1[2] - r2[2])));

        double eta1 = e1 * pm1;
        double eta2 = e2 * pm1;

        int A1 = prim1.atomIdx();
        int A2 = prim2.atomIdx();

        // Hermite expansion coefficients
        double E[3][V*V1*V2];
        double E1[3][V*V1*V2];
        double E2[3][V*V1*V2];
        ::memset(&E[0][0],'\0',3*V*V1*V2*sizeof(double));
        ::memset(&E1[0][0],'\0',3*V*V1*V2*sizeof(double));
        ::memset(&E2[0][0],'\0',3*V*V1*V2*sizeof(double));
        for (int d = 0; d < 3; d++) {
            double xPA = rPA[d];
            double xPB = rPB[d];
            double* Ex = E[d];
            double* E1x = E1[d];
            double* E2x = E2[d];
            for (int t = 0; t < V; t++) {
                
                // Quadrature points
                double tgh = ts[t];
                double wgh = ws[t];

                // Change of variable
                double xgh = tgh * pm12;
                double ugh = wgh * pm12;
                double zP = xgh; 
                double zPA = xgh + xPA; 
                double zPB = xgh + xPB; 

                // Auxiliary Hermite Polynomials
                double chi[V];
                chi[0] = ugh * sqrt(p / M_PI);
                if (V > 1) chi[1] = zP * chi[0];
                for (int t = 1; t < V - 1; t++) {
                    chi[t+1] = 1.0 / (t + 1) * (zP * chi[t] - 0.5 * pm1 * chi[t-1]);
                }
                  
                // Gradient Auxiliary Hermite Polynomials
                double chip[V];
                chip[0] = 0.0;
                if (V > 1) chip[1] = chi[0];
                for (int t = 1; t < V - 1; t++) {
                    chip[t+1] = 1.0 / (t + 1) * (chi[t] + zP * chip[t] - 0.5 * pm1 * chip[t-1]);
                }

                // Cartesian Monomials
                double X1[V1+1];
                X1[0] = 1.0;
                for (int l1 = 1; l1 < V1+1; l1++) {
                    X1[l1] = X1[l1-1] * zPA;
                }
                double X2[V2+1];
                X2[0] = 1.0;
                for (int l2 = 1; l2 < V2+1; l2++) {
                    X2[l2] = X2[l2-1] * zPB;
                }

                // Cartesian Monomial Derivs
                double dX1[V1];
                dX1[0] = 2.0 * e1 * zPA;
                for (int l1 = 1; l1 < V1; l1++) {
                    dX1[l1] = -l1 * X1[l1-1] + 2.0 * e1 * X1[l1+1];
                }
                double dX2[V2];
                dX2[0] = 2.0 * e2 * zPB;
                for (int l2 = 1; l2 < V2; l2++) {
                    dX2[l2] = -l2 * X2[l2-1] + 2.0 * e2 * X2[l2+1];
                }

                // Assembly
                for (int L = 0; L < V; L++) {
                    for (int l1 = 0; l1 < V1; l1++) {
                        for (int l2 = 0; l2 < V2; l2++) {
                            if (L > (l1 + l2)) continue;
                            Ex[L * V1 * V2 + l1 * V2 + l2] += chi[L] * X1[l1] * X2[l2];
                            E1x[L * V1 * V2 + l1 * V2 + l2] += chi[L] * dX1[l1] * X2[l2];
                            E1x[L * V1 * V2 + l1 * V2 + l2] -= eta1 * chip[L] * X1[l1] * X2[l2];
                            E2x[L * V1 * V2 + l1 * V2 + l2] += chi[L] * X1[l1] * dX2[l2];
                            E2x[L * V1 * V2 + l1 * V2 + l2] -= eta2 * chip[L] * X1[l1] * X2[l2];
                        }
                    } 
                }
            }
        }

        const double* Ex = E[0];
        const double* Ey = E[1];
        const double* Ez = E[2];
        const double* E1x = E1[0];
        const double* E1y = E1[1];
        const double* E1z = E1[2];
        const double* E2x = E2[0];
        const double* E2y = E2[1];
        const double* E2z = E2[2];

        // Temporary Source
        double D12T[N1 * N2];
        // Target
        const double* DAT = DA + ind * H;

        // Density matrix extraction
        int off1 = prim1.cartIdx(); 
        int off2 = prim2.cartIdx(); 
        for (int p = 0; p < N1; p++) {
            for (int q = 0; q < N2; q++) {
                D12T[p * N2 + q] = D12[(p + off1) * nao2 + (q + off2)];
            }
        }
        if (symm && (prim1.primIdx() != prim2.primIdx())) {
            for (int p = 0; p < N1; p++) {
                for (int q = 0; q < N2; q++) {
                    D12T[p * N2 + q] += D12[(q + off2) * nao2 + (p + off1)];
                }
            }
        }

        double G1T[3] = {0.0, 0.0, 0.0};
        double G2T[3] = {0.0, 0.0, 0.0};
        
        // Basis function 1
        for (int i1 = 0, index1 = 0; i1 <= L1; i1++) {
            int l1 = L1 - i1;
        for (int j1 = 0; j1 <= i1; j1++, index1++) {
            int m1 = i1 - j1;
            int n1 = j1; 

        // Basis function 2
        for (int i2 = 0, index2 = 0; i2 <= L2; i2++) {
            int l2 = L2 - i2;
        for (int j2 = 0; j2 <= i2; j2++, index2++) {
            int m2 = i2 - j2;
            int n2 = j2; 

            // Cartesian basis density element
            double Dval = D12T[index1 * N2 + index2];

            // Hermite Gaussian L
            for (int LH = 0, indexH = 0; LH < V; LH++) {
            
                // Hermite Gaussian shell
                for (int iH = 0; iH <= LH; iH++) {
                    int lH = LH - iH;
                for (int jH = 0; jH <= iH; jH++, indexH++) {
                    int mH = iH - jH;
                    int nH = jH; 
                    double sign = 1.0;
                    if (phases) {
                        sign = ((lH + mH + nH) % 2 ? -1.0 : 1.0);
                    }
                    double VA = DAT[indexH];
                    double Eval = sign * K * Dval * VA;
                    G1T[0] += Eval * 
                        E1x[lH * V1 * V2 + l1 * V2 + l2] *
                        Ey[mH * V1 * V2 + m1 * V2 + m2] *
                        Ez[nH * V1 * V2 + n1 * V2 + n2];
                    G1T[1] += Eval * 
                        Ex[lH * V1 * V2 + l1 * V2 + l2] *
                        E1y[mH * V1 * V2 + m1 * V2 + m2] *
                        Ez[nH * V1 * V2 + n1 * V2 + n2];
                    G1T[2] += Eval * 
                        Ex[lH * V1 * V2 + l1 * V2 + l2] *
                        Ey[mH * V1 * V2 + m1 * V2 + m2] *
                        E1z[nH * V1 * V2 + n1 * V2 + n2];
                    G2T[0] += Eval * 
                        E2x[lH * V1 * V2 + l1 * V2 + l2] *
                        Ey[mH * V1 * V2 + m1 * V2 + m2] *
                        Ez[nH * V1 * V2 + n1 * V2 + n2];
                    G2T[1] += Eval * 
                        Ex[lH * V1 * V2 + l1 * V2 + l2] *
                        E2y[mH * V1 * V2 + m1 * V2 + m2] *
                        Ez[nH * V1 * V2 + n1 * V2 + n2];
                    G2T[2] += Eval * 
                        Ex[lH * V1 * V2 + l1 * V2 + l2] *
                        Ey[mH * V1 * V2 + m1 * V2 + m2] *
                        E2z[nH * V1 * V2 + n1 * V2 + n2];
                }}
            }
        }}
        }}

        #pragma omp atomic
        G1p[3*A1 + 0] += G1T[0];
        #pragma omp atomic
        G1p[3*A1 + 1] += G1T[1];
        #pragma omp atomic
        G1p[3*A1 + 2] += G1T[2];
    
        #pragma omp atomic
        G2p[3*A2 + 0] += G2T[0];
        #pragma omp atomic
        G2p[3*A2 + 1] += G2T[1];
        #pragma omp atomic
        G2p[3*A2 + 2] += G2T[2];

    }
}

} // namespace lightspeed
