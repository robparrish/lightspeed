#include "potential4c.hpp"
#include <lightspeed/gh.hpp>
#include <lightspeed/rys.hpp>
#include <lightspeed/basis.hpp>
#include <math.h>
#include <string.h>
#include <stdexcept>

namespace lightspeed {

PotentialInt4C::PotentialInt4C(
    const std::vector<double>& alphas,
    const std::vector<double>& omegas,
    int L1,
    int L2,
    int L3,
    int L4,
    int deriv) :
    alphas_(alphas),
    omegas_(omegas),
    L1_(L1),
    L2_(L2),
    L3_(L3),
    L4_(L4),    
    deriv_(deriv)
{
    size_t size;
    if (deriv_ == 0) {
        size = 1L * chunk_size();
    } else {
        throw std::runtime_error("PotentialInt4C: deriv too high");
    }
    data1_.resize(size);
    data2_.resize(size);

    size_t n2D = 
        (L1_ + 1) * 
        (L2_ + 1) * 
        (L3_ + 1) * 
        (L4_ + 1);
    J_.resize(3);
    J_[0].resize(n2D);
    J_[1].resize(n2D);
    J_[2].resize(n2D);
    
    am_info_ = AngularMomentum::build(max_am());
    gh_ = GH::instance();

    rys_ = Rys::instance();
    ts_.resize(total_am() / 2 + 1);
    ws_.resize(total_am() / 2 + 1);
}
void PotentialInt4C::compute_shell0(
    const Shell& sh1,
    const Shell& sh2,
    const Shell& sh3,
    const Shell& sh4)
{
    // => Pre-Requisites <= //

    int L1 = sh1.L();
    int L2 = sh2.L();
    int L3 = sh3.L();
    int L4 = sh4.L();
    int V1 = L1 + 1;
    int V2 = L2 + 1;
    int V3 = L3 + 1;
    int V4 = L4 + 1;
    int I1 = sh1.ncart();
    int I2 = sh2.ncart();
    int I3 = sh3.ncart();
    int I4 = sh4.ncart();
    int K1 = sh1.nprim();
    int K2 = sh2.nprim();
    int K3 = sh3.nprim();
    int K4 = sh4.nprim();
    const std::vector<double>& c1s = sh1.cs();
    const std::vector<double>& c2s = sh2.cs();
    const std::vector<double>& c3s = sh3.cs();
    const std::vector<double>& c4s = sh4.cs();
    const std::vector<double>& e1s = sh1.es();
    const std::vector<double>& e2s = sh2.es();
    const std::vector<double>& e3s = sh3.es();
    const std::vector<double>& e4s = sh4.es();

    if (L1 > L1_) throw std::runtime_error("L1 is too high");
    if (L2 > L2_) throw std::runtime_error("L2 is too high");
    if (L3 > L3_) throw std::runtime_error("L3 is too high");
    if (L4 > L4_) throw std::runtime_error("L4 is too high");

    double X1[3]; 
    X1[0] = sh1.x();
    X1[1] = sh1.y();
    X1[2] = sh1.z();
    double X2[3]; 
    X2[0] = sh2.x();
    X2[1] = sh2.y();
    X2[2] = sh2.z();
    double X3[3]; 
    X3[0] = sh3.x();
    X3[1] = sh3.y();
    X3[2] = sh3.z();
    double X4[3]; 
    X4[0] = sh4.x();
    X4[1] = sh4.y();
    X4[2] = sh4.z();

    double x12_2 = 0.0;
    x12_2 += (X1[0] - X2[0]) * (X1[0] - X2[0]); 
    x12_2 += (X1[1] - X2[1]) * (X1[1] - X2[1]); 
    x12_2 += (X1[2] - X2[2]) * (X1[2] - X2[2]); 
    double x34_2 = 0.0;
    x34_2 += (X3[0] - X4[0]) * (X3[0] - X4[0]); 
    x34_2 += (X3[1] - X4[1]) * (X3[1] - X4[1]); 
    x34_2 += (X3[2] - X4[2]) * (X3[2] - X4[2]); 

    int nv = (L1 + L2 + L3 + L4) / 2 + 1;
    const std::vector<double>& xs = gh_->ts()[nv];
    const std::vector<double>& ws = gh_->ws()[nv];

    // => Integral Generation <= //

    memset(data1_.data(),'\0',sizeof(double)*I1*I2*I3*I4);

    for (size_t wind = 0; wind < omegas_.size(); wind++) {

        double alpha = alphas_[wind];
        double omega = omegas_[wind];

    for (int k1 = 0; k1 < K1; k1++) {
    for (int k2 = 0; k2 < K2; k2++) {
    
        double c1 = c1s[k1];
        double e1 = e1s[k1];
        double c2 = c2s[k2];
        double e2 = e2s[k2];

        // => Gaussian Product Theorem <= //

        double e12 = e1 + e2;
        double e12_m1 = 1.0 / e12;
        double K12 = exp(- e1 * e2 * e12_m1 * x12_2);
        double X12[3];
        X12[0] = (e1 * X1[0] + e2 * X2[0]) * e12_m1;
        X12[1] = (e1 * X1[1] + e2 * X2[1]) * e12_m1;
        X12[2] = (e1 * X1[2] + e2 * X2[2]) * e12_m1;
    
    for (int k3 = 0; k3 < K3; k3++) {
    for (int k4 = 0; k4 < K4; k4++) {

        double c3 = c3s[k3];
        double e3 = e3s[k3];
        double c4 = c4s[k4];
        double e4 = e4s[k4];

        // => Gaussian Product Theorem <= //

        double e34 = e3 + e4;
        double e34_m1 = 1.0 / e34;
        double K34 = exp(- e3 * e4 * e34_m1 * x34_2);
        double X34[3];
        X34[0] = (e3 * X3[0] + e4 * X4[0]) * e34_m1;
        X34[1] = (e3 * X3[1] + e4 * X4[1]) * e34_m1;
        X34[2] = (e3 * X3[2] + e4 * X4[2]) * e34_m1;

        // => Merging <= //

        double rho = e12 * e34 / (e12 + e34);
        double pref = c1 * c2 * c3 * c4 * K12 * K34;
        pref *= 2.0 / (sqrt(M_PI) * e12 * e34 * sqrt(e12 + e34)); // ERI

        // => Ewald Operator <= //
        
        double dval  = 1.0;
        double dval2 = 1.0;
        if (omega != -1.0) {
            dval2 = omega * omega / (rho + omega * omega);    
            dval = sqrt(dval2);
        }
        pref *= alpha * dval;

        // => Rys Quadrature <= //

        double r1234_2 = 0.0;
        r1234_2 += (X12[0] - X34[0]) * (X12[0] - X34[0]);
        r1234_2 += (X12[1] - X34[1]) * (X12[1] - X34[1]);
        r1234_2 += (X12[2] - X34[2]) * (X12[2] - X34[2]);
        double T = dval2 * rho * r1234_2;
        for (int i = 0; i < nv; i++) {
            ts_[i] = rys_->interpolate_t(nv, i, T);
            ws_[i] = rys_->interpolate_w(nv, i, T);
        }

        // => Rys Points <= //

        for (int i = 0; i < nv; i++) {

            double trys = ts_[i] * dval;
            double wrys = ws_[i];   

            // Eigenstructure considerations
            double gamma = rho * trys * trys / (1.0 - trys * trys);
            double delta_inv = (e34 - e12) / (2.0 * gamma);
            double Qval = delta_inv / (2.0 * sqrt(1.0 + delta_inv * delta_inv));
            double cval = sqrt(0.5 + Qval);
            double sval = sqrt(0.5 - Qval);
            double a1 = cval * cval * (e12 + gamma) + sval * sval * (e34 + gamma) - 2.0 * cval * sval * gamma;
            double a2 = sval * sval * (e12 + gamma) + cval * cval * (e34 + gamma) + 2.0 * cval * sval * gamma;
            double a1_m12 = pow(a1,-1.0/2.0);
            double a2_m12 = pow(a2,-1.0/2.0);
    
            double mUd1[3];
            double fact1 = (rho * trys * trys / e12);
            mUd1[0] = X12[0] + fact1 * (X34[0] - X12[0]);         
            mUd1[1] = X12[1] + fact1 * (X34[1] - X12[1]);         
            mUd1[2] = X12[2] + fact1 * (X34[2] - X12[2]);         
    
            double mUd2[3];
            double fact2 = (rho * trys * trys / e34);
            mUd2[0] = X34[0] + fact2 * (X12[0] - X34[0]);         
            mUd2[1] = X34[1] + fact2 * (X12[1] - X34[1]);         
            mUd2[2] = X34[2] + fact2 * (X12[2] - X34[2]);         

            memset(J_[0].data(),'\0',sizeof(double) * V1 * V2 * V3 * V4);
            memset(J_[1].data(),'\0',sizeof(double) * V1 * V2 * V3 * V4);
            memset(J_[2].data(),'\0',sizeof(double) * V1 * V2 * V3 * V4);

            for (int u = 0; u < nv; u++) {
            for (int v = 0; v < nv; v++) {
                
                // Gauss-Hermite Quadrature nodes
                double zeta1 = xs[u];
                double zeta2 = xs[v];
                double w1 = ws[u];
                double w2 = ws[v];

                // Coordinate transformation
                double z1 = cval * a1_m12 * zeta1 - sval * a2_m12 * zeta2;
                double z2 = sval * a1_m12 * zeta1 + cval * a2_m12 * zeta2;

                // Total Gauss-Hermite weight
                double we = w1 * w2;

                // Primitive-specific sampling points
                double XQ1[3];
                XQ1[0] = z1 + mUd1[0] - X1[0];
                XQ1[1] = z1 + mUd1[1] - X1[1];
                XQ1[2] = z1 + mUd1[2] - X1[2];
                double XQ2[3];
                XQ2[0] = z1 + mUd1[0] - X2[0];
                XQ2[1] = z1 + mUd1[1] - X2[1];
                XQ2[2] = z1 + mUd1[2] - X2[2];
                double XQ3[3];
                XQ3[0] = z2 + mUd2[0] - X3[0];
                XQ3[1] = z2 + mUd2[1] - X3[1];
                XQ3[2] = z2 + mUd2[2] - X3[2];
                double XQ4[3];
                XQ4[0] = z2 + mUd2[0] - X4[0];
                XQ4[1] = z2 + mUd2[1] - X4[1];
                XQ4[2] = z2 + mUd2[2] - X4[2];

                // 2D overlap integrals
                for (int d = 0; d < 3; d++) {
                    double X1_l = 1.0;
                    for (int l1 = 0; l1 < V1; l1++) {
                        double X2_l = 1.0;
                        for (int l2 = 0; l2 < V2; l2++) {
                            double X3_l = 1.0;
                            for (int l3 = 0; l3 < V3; l3++) {
                                double X4_l = 1.0;
                                for (int l4 = 0; l4 < V4; l4++) {
                                    double Sval = we * X1_l * X2_l * X3_l * X4_l;
                                    J_[d][l1 * (V2 * V3 * V4) + l2 * (V3 * V4) + l3 * V4 + l4] += Sval;
                                    X4_l *= XQ4[d];
                                }
                                X3_l *= XQ3[d];
                            }
                            X2_l *= XQ2[d];
                        }
                        X1_l *= XQ1[d];
                    } // 2D AM quanta (L^4)
                } // d (3)

            }} // Gauss-Hermite points (L^2)

            // => 6D Integrals <= //

            int bufind = 0;

            // AM1 
            for (int i1 = 0; i1 <= L1; i1++) {
                int l1 = L1 - i1;
            for (int j1 = 0; j1 <= i1; j1++) {
                int m1 = i1 - j1;
                int n1 = j1;

            // AM2
            for (int i2 = 0; i2 <= L2; i2++) {
                int l2 = L2 - i2;
            for (int j2 = 0; j2 <= i2; j2++) {
                int m2 = i2 - j2;
                int n2 = j2;

            // AM3
            for (int i3 = 0; i3 <= L3; i3++) {
                int l3 = L3 - i3;
            for (int j3 = 0; j3 <= i3; j3++) {
                int m3 = i3 - j3;
                int n3 = j3;

            // AM4
            for (int i4 = 0; i4 <= L4; i4++) {
                int l4 = L4 - i4;
            for (int j4 = 0; j4 <= i4; j4++) {
                int m4 = i4 - j4;
                int n4 = j4;

                data1_[bufind++] += pref * wrys * 
                    J_[0][l1 * (V2 * V3 * V4) + l2 * (V3 * V4) + l3 * (V4) + l4] * 
                    J_[1][m1 * (V2 * V3 * V4) + m2 * (V3 * V4) + m3 * (V4) + m4] * 
                    J_[2][n1 * (V2 * V3 * V4) + n2 * (V3 * V4) + n3 * (V4) + n4];

            }}}}}}}} // AM (L^8)

        } // Rys points (L)

    }}
    }} // Primitives (p^4)

    } // Omegas

    // => Spherical Harmonics <= //

    bool s1 = sh1.is_pure();
    bool s2 = sh2.is_pure();
    bool s3 = sh3.is_pure();
    bool s4 = sh4.is_pure();
    if (is_spherical_) apply_spherical(L1, L2, L3, L4, s1, s2, s3, s4, data1_.data(), data2_.data());
}

void PotentialInt4C::apply_spherical(
        int L1,
        int L2,
        int L3,
        int L4,
        bool S1,
        bool S2,
        bool S3,
        bool S4,
        double *target,
        double *scratch)
{
    if (!S1 && !S2 && !S3 && !S4) return;

    size_t ncart1 = (L1 + 1) * (L1 + 2) / 2;
    size_t ncart2 = (L2 + 1) * (L2 + 2) / 2;
    size_t ncart3 = (L3 + 1) * (L3 + 2) / 2;
    size_t ncart4 = (L4 + 1) * (L4 + 2) / 2;

    size_t npure1 = 2 * L1 + 1;
    size_t npure2 = 2 * L2 + 1;
    size_t npure3 = 2 * L3 + 1;
    size_t npure4 = 2 * L4 + 1;

    size_t nfun1 = (S1 ? npure1 : ncart1);
    size_t nfun2 = (S2 ? npure2 : ncart2);
    size_t nfun3 = (S3 ? npure3 : ncart3);
    size_t nfun4 = (S4 ? npure4 : ncart4);

    if (S4 && L4 > 0) {
        memset(scratch, '\0', sizeof(double) * ncart1 * ncart2 * ncart3 * npure4);
        const AngularMomentum &trans = am_info_[L4];
        size_t ncoef = trans.ncoef();
        const std::vector<int> &cart_inds = trans.cart_inds();
        const std::vector<int> &pure_inds = trans.pure_inds();
        const std::vector<double> &cart_coefs = trans.cart_coefs();
        for (size_t p = 0L; p < ncart1 * ncart2 * ncart3; p++) {
            double *cartp = target + p * ncart4;
            double *purep = scratch + p * npure4;
            for (size_t ind = 0L; ind < ncoef; ind++) {
                *(purep + pure_inds[ind]) += cart_coefs[ind] * *(cartp + cart_inds[ind]);
            }
        }
    } else {
        memcpy(scratch, target, sizeof(double) * ncart1 * ncart2 * ncart3 * ncart4);
    }

    if (S3 && L3 > 0) {
        memset(target, '\0', sizeof(double) * ncart1 * ncart2 * npure3 * nfun4);
        const AngularMomentum &trans = am_info_[L3];
        size_t ncoef = trans.ncoef();
        const std::vector<int> &cart_inds = trans.cart_inds();
        const std::vector<int> &pure_inds = trans.pure_inds();
        const std::vector<double> &cart_coefs = trans.cart_coefs();
        for (size_t p = 0L; p < ncart1 * ncart2; p++) {
            double *cart2p = scratch + p * ncart3 * nfun4;
            double *pure2p = target + p * npure3 * nfun4;
            for (size_t ind = 0L; ind < ncoef; ind++) {
                double *cartp = cart2p + cart_inds[ind] * nfun4;
                double *purep = pure2p + pure_inds[ind] * nfun4;
                double coef = cart_coefs[ind];
                for (size_t p2 = 0L; p2 < nfun4; p2++) {
                    *purep++ += coef * *cartp++;
                }
            }
        }
    } else {
        memcpy(target, scratch, sizeof(double) * ncart1 * ncart2 * ncart3 * nfun4);
    }

    if (S2 && L2 > 0) {
        memset(scratch, '\0', sizeof(double) * ncart1 * npure2 * nfun3 * nfun4);
        const AngularMomentum &trans = am_info_[L2];
        size_t ncoef = trans.ncoef();
        const std::vector<int> &cart_inds = trans.cart_inds();
        const std::vector<int> &pure_inds = trans.pure_inds();
        const std::vector<double> &cart_coefs = trans.cart_coefs();
        for (size_t p = 0L; p < ncart1; p++) {
            double *cart2p = target + p * ncart2 * nfun3 * nfun4;
            double *pure2p = scratch + p * npure2 * nfun3 * nfun4;
            for (size_t ind = 0L; ind < ncoef; ind++) {
                double *cartp = cart2p + cart_inds[ind] * nfun3 * nfun4;
                double *purep = pure2p + pure_inds[ind] * nfun3 * nfun4;
                double coef = cart_coefs[ind];
                for (size_t p2 = 0L; p2 < nfun3 * nfun4; p2++) {
                    *purep++ += coef * *cartp++;
                }
            }
        }
    } else {
        memcpy(scratch, target, sizeof(double) * ncart1 * ncart2 * nfun3 * nfun4);
    }

    if (S1 && L1 > 0) {
        memset(target, '\0', sizeof(double) * npure1 * nfun2 * nfun3 * nfun4);
        const AngularMomentum &trans = am_info_[L1];
        size_t ncoef = trans.ncoef();
        const std::vector<int> &cart_inds = trans.cart_inds();
        const std::vector<int> &pure_inds = trans.pure_inds();
        const std::vector<double> &cart_coefs = trans.cart_coefs();
        for (size_t ind = 0L; ind < ncoef; ind++) {
            double *cartp = scratch + cart_inds[ind] * nfun2 * nfun3 * nfun4;
            double *purep = target + pure_inds[ind] * nfun2 * nfun3 * nfun4;
            double coef = cart_coefs[ind];
            for (size_t p2 = 0L; p2 < nfun2 * nfun3 * nfun4; p2++) {
                *purep++ += coef * *cartp++;
            }
        }
    } else {
        memcpy(target, scratch, sizeof(double) * ncart1 * nfun2 * nfun3 * nfun4);
    }
}
size_t PotentialInt4C::chunk_size() const 
{
    size_t ncart1 = (L1_ + 1) * (L1_ + 2) / 2;
    size_t ncart2 = (L2_ + 1) * (L2_ + 2) / 2;
    size_t ncart3 = (L3_ + 1) * (L3_ + 2) / 2;
    size_t ncart4 = (L4_ + 1) * (L4_ + 2) / 2;
    return ncart1 * ncart2 * ncart3 * ncart4;
}

} // namespace lightspeed
