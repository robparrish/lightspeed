#include <lightspeed/intbox.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/ewald.hpp>
#include <lightspeed/resource_list.hpp>
#include <cmath>

namespace lightspeed {
    
double IntBox::chargeEnergySelf(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<Tensor>& xyzc
    )
{
    return 0.5 * IntBox::chargeEnergyOther(
        resources,
        ewald,
        xyzc,
        xyzc);
}

std::shared_ptr<Tensor> IntBox::chargeFieldOther(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<Tensor>& xyzc1,
    const std::shared_ptr<Tensor>& xyz2,
    const std::shared_ptr<Tensor>& F
    )
{

    xyzc1->ndim_error(2);
    std::vector<size_t> dim1;
    dim1.push_back(xyzc1->shape()[0]);
    dim1.push_back(4);
    
    xyz2->ndim_error(2);
    std::vector<size_t> dim2;
    dim2.push_back(xyz2->shape()[0]);
    dim2.push_back(3);
    xyz2->shape_error(dim2);
    
    size_t nA = xyzc1->shape()[0];
    size_t nB = xyz2->shape()[0];
    const double* xyzc1p = xyzc1->data().data();
    const double* xyz2p = xyz2->data().data();

    // Size of F
    std::vector<size_t> dim;
    dim.push_back(nB);
    dim.push_back(3);

    std::shared_ptr<Tensor> F2 = F;
    if (!F) {
        F2 = std::shared_ptr<Tensor>(new Tensor(dim));
    } 
    F2->shape_error(dim);
    double * F2p = F2->data().data();

    // prepare ewald operator
    const double* alpha_p = ewald->scales().data();
    const double* omega_p = ewald->omegas().data();
    size_t w_size = ewald->scales().size();

    double* sr_alpha_p = NULL;
    double* sr_omega_p = NULL;
    bool sr_ewald = ewald->is_sr();
    if (sr_ewald) {
        double sr_alpha = ewald->sr_scale();
        double sr_omega = ewald->sr_omega();
        sr_alpha_p = &sr_alpha;
        sr_omega_p = &sr_omega;
    }

    // Compute electric field from A point charges on point B
    for (size_t B = 0; B < nB; B++) {
        double xB = xyz2p[3*B + 0];
        double yB = xyz2p[3*B + 1];
        double zB = xyz2p[3*B + 2];

        for (size_t A = 0; A < nA; A++) {
            double xA = xyzc1p[4*A + 0];
            double yA = xyzc1p[4*A + 1];
            double zA = xyzc1p[4*A + 2];
            double cA = xyzc1p[4*A + 3];

            double rAB2 = (
                pow(xA-xB,2) +
                pow(yA-yB,2) +
                pow(zA-zB,2)
                );
            double xAB = xA - xB;
            double yAB = yA - yB;
            double zAB = zA - zB;
            if (rAB2 == 0.0) continue; // Avoid self-interaction
            double rAB = sqrt(rAB2);

            if (sr_ewald) {
                double alpha = *sr_alpha_p;
                double omega = *sr_omega_p;
                double fac = erfc(omega * rAB)/rAB2 + 2.0/sqrt(M_PI)*omega*exp(-pow(omega, 2.0)*rAB2)/rAB;
                F2p[3*B + 0] -= alpha * cA * fac * xAB / rAB;
                F2p[3*B + 1] -= alpha * cA * fac * yAB / rAB;
                F2p[3*B + 2] -= alpha * cA * fac * zAB / rAB; 
            } else {
                for (size_t wind = 0; wind < w_size; wind++) {
                    double alpha = alpha_p[wind];
                    double omega = omega_p[wind];
                    double fac = (
                        omega == -1.0 ? 1.0/rAB2 :
                        erfc(omega * rAB)/rAB2 + 2.0/sqrt(M_PI)*omega*exp(-pow(omega, 2.0)*rAB2)/rAB
                        );
                        F2p[3*B + 0] -= alpha * cA * fac * xAB / rAB;
                        F2p[3*B + 1] -= alpha * cA * fac * yAB / rAB;
                        F2p[3*B + 2] -= alpha * cA * fac * zAB / rAB;
                }
            }
        }
    }
    return F2; 
}
    

double IntBox::chargeEnergyOther(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<Tensor>& xyzc1,
    const std::shared_ptr<Tensor>& xyzc2
    )
{
    xyzc1->ndim_error(2);
    std::vector<size_t> dim1;
    dim1.push_back(xyzc1->shape()[0]);
    dim1.push_back(4);
    xyzc1->shape_error(dim1);

    xyzc2->ndim_error(2);
    std::vector<size_t> dim2;
    dim2.push_back(xyzc2->shape()[0]);
    dim2.push_back(4);
    xyzc2->shape_error(dim2);

    size_t nA = xyzc1->shape()[0];
    size_t nB = xyzc2->shape()[0];
    const double* xyzc1p = xyzc1->data().data();
    const double* xyzc2p = xyzc2->data().data();

    // prepare ewald operator
    const double* alpha_p = ewald->scales().data();
    const double* omega_p = ewald->omegas().data();
    size_t w_size = ewald->scales().size();

    double* sr_alpha_p = NULL;
    double* sr_omega_p = NULL;
    bool sr_ewald = ewald->is_sr();
    if (sr_ewald) {
        double sr_alpha = ewald->sr_scale();
        double sr_omega = ewald->sr_omega();
        sr_alpha_p = &sr_alpha;
        sr_omega_p = &sr_omega;
    }

    double val = 0.0;

    for (size_t A = 0; A < nA; A++) {
        double xA = xyzc1p[4*A + 0];
        double yA = xyzc1p[4*A + 1];
        double zA = xyzc1p[4*A + 2];
        double cA = xyzc1p[4*A + 3];

        for (size_t B = 0; B < nB; B++) {
            double xB = xyzc2p[4*B + 0];
            double yB = xyzc2p[4*B + 1];
            double zB = xyzc2p[4*B + 2];
            double cB = xyzc2p[4*B + 3];

            double rAB2 = (
                pow(xA-xB,2) +
                pow(yA-yB,2) +
                pow(zA-zB,2)
                );
            if (rAB2 == 0.0) continue; // Avoid self-interaction
            double rAB = sqrt(rAB2);

            if (sr_ewald) {
                double alpha = *sr_alpha_p;
                double omega = *sr_omega_p;
                val += alpha * cA * cB * erfc(omega * rAB) / rAB;
            } else {
                for (size_t wind = 0; wind < w_size; wind++) {
                    double alpha = alpha_p[wind];
                    double omega = omega_p[wind];
                    double num = (omega == -1.0 ? 1.0 : erf(omega * rAB));
                    val += alpha * cA * cB * num / rAB;
                }
            }
        }
    }

    return val; 
}
std::shared_ptr<Tensor> IntBox::chargeGradSelf(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<Tensor>& xyzc,
    const std::shared_ptr<Tensor>& G
    )
{
    xyzc->ndim_error(2);
    std::vector<size_t> dim1;
    dim1.push_back(xyzc->shape()[0]);
    dim1.push_back(4);
    xyzc->shape_error(dim1);
   
    std::vector<size_t> dim2;
    dim2.push_back(xyzc->shape()[0]);
    dim2.push_back(3);
    std::shared_ptr<Tensor> G2 = G;
    if (!G) {
        G2 = std::shared_ptr<Tensor>(new Tensor(dim2));
    }
    G2->shape_error(dim2);

    G2->scale(2.0);
    
    std::vector<std::shared_ptr<Tensor> > Gtemp;
    Gtemp.push_back(G2);
    Gtemp.push_back(G2);
    
    IntBox::chargeGradOther(
        resources,
        ewald,
        xyzc,
        xyzc,
        Gtemp);
    
    G2->scale(0.5);
    
    return G2;
}
std::vector<std::shared_ptr<Tensor> > IntBox::chargeGradOther(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<Tensor>& xyzc1,
    const std::shared_ptr<Tensor>& xyzc2,
    const std::vector<std::shared_ptr<Tensor> >& G12
    )
{
    // sizes of xyzc1 and xyzc2
    xyzc1->ndim_error(2);
    std::vector<size_t> dim1;
    dim1.push_back(xyzc1->shape()[0]);
    dim1.push_back(4);
    xyzc1->shape_error(dim1);

    xyzc2->ndim_error(2);
    std::vector<size_t> dim2;
    dim2.push_back(xyzc2->shape()[0]);
    dim2.push_back(4);
    xyzc2->shape_error(dim2);

    size_t nA = xyzc1->shape()[0];
    size_t nB = xyzc2->shape()[0];
    const double* xyzc1p = xyzc1->data().data();
    const double* xyzc2p = xyzc2->data().data();

    // sizes of G1 and G2
    std::vector<size_t> dim_g1;
    dim_g1.push_back(nA);
    dim_g1.push_back(3);
    std::vector<size_t> dim_g2;
    dim_g2.push_back(nB);
    dim_g2.push_back(3);

    std::vector<std::shared_ptr<Tensor> > G12T;
    if (G12.size() == 0) {
        G12T.push_back(std::shared_ptr<Tensor>(new Tensor(dim_g1)));
        G12T.push_back(std::shared_ptr<Tensor>(new Tensor(dim_g2)));
    } else if (G12.size() == 2) {
        G12T = G12;
        G12T[0]->shape_error(dim_g1);
        G12T[1]->shape_error(dim_g2);
    } else {
        throw std::runtime_error("IntBox::chargeGrad: G12 should be size 0 or size 2");
    }

    double* G1p = G12T[0]->data().data();
    double* G2p = G12T[1]->data().data();

    // prepare ewald operator
    const double* alpha_p = ewald->scales().data();
    const double* omega_p = ewald->omegas().data();
    size_t w_size = ewald->scales().size();

    double* sr_alpha_p = NULL;
    double* sr_omega_p = NULL;
    bool sr_ewald = ewald->is_sr();
    if (sr_ewald) {
        double sr_alpha = ewald->sr_scale();
        double sr_omega = ewald->sr_omega();
        sr_alpha_p = &sr_alpha;
        sr_omega_p = &sr_omega;
    }

    for (size_t A = 0; A < nA; A++) {
        double xA = xyzc1p[4*A + 0];
        double yA = xyzc1p[4*A + 1];
        double zA = xyzc1p[4*A + 2];
        double cA = xyzc1p[4*A + 3];

        for (size_t B = 0; B < nB; B++) {
            double xB = xyzc2p[4*B + 0];
            double yB = xyzc2p[4*B + 1];
            double zB = xyzc2p[4*B + 2];
            double cB = xyzc2p[4*B + 3];

            double xAB = xA - xB;
            double yAB = yA - yB;
            double zAB = zA - zB;
            double rAB2 = (pow(xAB,2) + pow(yAB,2) + pow(zAB,2));
            if (rAB2 == 0.0) continue; // Avoid self-interaction
            double rAB = sqrt(rAB2);

            if (sr_ewald) {
                // short-range case
                double alpha = *sr_alpha_p;
                double omega = *sr_omega_p;
                double num = erfc(omega*rAB)/rAB2 + 2.0/sqrt(M_PI)*exp(-pow(omega,2)*rAB2)*omega/rAB;
                G1p[A*3+0] -= alpha * cA * cB * num * (xAB / rAB);
                G1p[A*3+1] -= alpha * cA * cB * num * (yAB / rAB);
                G1p[A*3+2] -= alpha * cA * cB * num * (zAB / rAB);
                G2p[B*3+0] += alpha * cA * cB * num * (xAB / rAB);
                G2p[B*3+1] += alpha * cA * cB * num * (yAB / rAB);
                G2p[B*3+2] += alpha * cA * cB * num * (zAB / rAB);
            } else {
                // full- or long-range case
                for (size_t wind = 0; wind < w_size; wind++) {
                    double alpha = alpha_p[wind];
                    double omega = omega_p[wind];
                    double num = (
                        omega == -1.0 ? 
                        1.0 / rAB2 : 
                        erf(omega*rAB)/rAB2 - 2.0/sqrt(M_PI)*exp(-pow(omega,2)*rAB2)*omega/rAB
                        );
                    G1p[A*3+0] -= alpha * cA * cB * num * (xAB / rAB);
                    G1p[A*3+1] -= alpha * cA * cB * num * (yAB / rAB);
                    G1p[A*3+2] -= alpha * cA * cB * num * (zAB / rAB);
                    G2p[B*3+0] += alpha * cA * cB * num * (xAB / rAB);
                    G2p[B*3+1] += alpha * cA * cB * num * (yAB / rAB);
                    G2p[B*3+2] += alpha * cA * cB * num * (zAB / rAB);
                }
            }
        }
    }

    return G12T;
}
std::shared_ptr<Tensor> IntBox::chargePotential(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<Tensor>& xyz1,
    const std::shared_ptr<Tensor>& xyzc2,
    const std::shared_ptr<Tensor>& v
    )
{
    // sizes of xyzc1 and xyzc2
    xyz1->ndim_error(2);
    std::vector<size_t> dim1;
    dim1.push_back(xyz1->shape()[0]);
    dim1.push_back(3);
    xyz1->shape_error(dim1);

    xyzc2->ndim_error(2);
    std::vector<size_t> dim2;
    dim2.push_back(xyzc2->shape()[0]);
    dim2.push_back(4);
    xyzc2->shape_error(dim2);

    size_t nA = xyz1->shape()[0];
    size_t nB = xyzc2->shape()[0];
    const double* xyz1p = xyz1->data().data();
    const double* xyzc2p = xyzc2->data().data();

    std::shared_ptr<Tensor> v2 = v;
    std::vector<size_t> dimA;
    dimA.push_back(nA);
    if (!v) {
        v2 = std::shared_ptr<Tensor>(new Tensor(dimA));
    }
    v2->shape_error(dimA);
    double* v2p = v2->data().data();

    // prepare ewald operator
    const double* alpha_p = ewald->scales().data();
    const double* omega_p = ewald->omegas().data();
    size_t w_size = ewald->scales().size();

    double sr_alpha = 0.0;
    double sr_omega = 0.0;
    bool sr_ewald = ewald->is_sr();
    if (sr_ewald) {
        sr_alpha = ewald->sr_scale();
        sr_omega = ewald->sr_omega();
    }

    for (size_t A = 0; A < nA; A++) {
        double xA = xyz1p[3*A + 0];
        double yA = xyz1p[3*A + 1];
        double zA = xyz1p[3*A + 2];

        double val = 0.0;
        for (size_t B = 0; B < nB; B++) {
            double xB = xyzc2p[4*B + 0];
            double yB = xyzc2p[4*B + 1];
            double zB = xyzc2p[4*B + 2];
            double cB = xyzc2p[4*B + 3];

            double xAB = xA - xB;
            double yAB = yA - yB;
            double zAB = zA - zB;
            double rAB2 = (pow(xAB,2) + pow(yAB,2) + pow(zAB,2));
            if (rAB2 == 0.0) continue; // Avoid self-interaction
            double rAB = sqrt(rAB2);

            if (sr_ewald) {
                val += sr_alpha * cB * erfc(sr_omega * rAB) / rAB;
            } else {
                for (int wind = 0; wind < w_size; wind++) {
                    double alpha = alpha_p[wind];
                    double omega = omega_p[wind];
                    double num = (omega == -1.0 ? 1.0 : erf(omega * rAB));
                    val += alpha * cB * num / rAB;
                }
            }
        }
        v2p[A] += val;
    }

    return v2;
}

} // namespace lightspeed
