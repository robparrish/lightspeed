#include "cpu_aiscat.hpp"
#include <lightspeed/resource_list.hpp>
#include <lightspeed/tensor.hpp>
#include <omp.h>
#include <functional>

namespace lightspeed {

// => Spherical Bessel Functions <= //

inline double J0(double x)
{
    if (x == 0.0) return 1.0;
    else return sin(x) / x;
}
inline double J1_x(double x)
{
    if (x == 0.0) return 1.0 / 3.0; 
    else return sin(x) / pow(x,3) - cos(x) / pow(x,2);
}
inline double J2(double x)
{
    if (x == 0.0) return 0.0;
    else return (3.0 / pow(x,2) - 1.0) * sin(x) / x - 3.0 * cos(x) / pow(x, 2);
}

// => Moment Kernels <= //

inline std::pair<double, double> isotropic_kernel(
    double L,
    double s,
    double dx,
    double dy,
    double dz)
{
    double r = sqrt(dx*dx + dy*dy + dz*dz);
    double sr = s*r;
    return std::pair<double, double>(J0(sr), 0.0);
}

inline std::pair<double, double> parallel_kernel(
    double L,
    double s,
    double dx,
    double dy,
    double dz)
{
    double r = sqrt(dx*dx + dy*dy + dz*dz);
    double sr = s*r;
    if (sr == 0.0) return std::pair<double, double>(1.0/3.0, 0.0);
    double sg2 = (dx*dx + dy*dy) / (dx*dx + dy*dy + dz*dz); 
     
    double j1sr = J1_x(sr);
    double j2 = J2(sr); 

    double B = (2.0 - 3.0 * sg2) * pow(s*L / (4.0 * M_PI), 2);

    return std::pair<double, double>(
        j1sr - 0.5 * (sg2 + B) * j2,
        0.0
        );
}

inline std::pair<double, double> perpendicular_kernel(
    double L,
    double s,
    double dx,
    double dy,
    double dz)
{
    double r = sqrt(dx*dx + dy*dy + dz*dz);
    double sr = s*r;
    if (sr == 0.0) return std::pair<double, double>(1.0/3.0, 0.0);
    double sg2 = (dx*dx + dy*dy) / (dx*dx + dy*dy + dz*dz); 
     
    double j1sr = J1_x(sr);
    double j2 = J2(sr); 

    double A = 0.5 * (2.0 - 3.0 * sg2) * (1.0 - pow(s*L / (4.0 * M_PI), 2));

    return std::pair<double, double>(
        j1sr - 0.5 * (sg2 + A) * j2,
        - 0.5 * A * j2
        );
}

// => General Moment Code <= //

std::shared_ptr<Tensor> moments(
    std::function<std::pair<double, double>(double, double, double, double, double)> kernel,
    const std::shared_ptr<ResourceList>& resources,
    double L,
    const std::shared_ptr<Tensor> s,
    const std::shared_ptr<Tensor> xyzq
    )
{
    // Validity checks
    s->ndim_error(1);
    xyzq->ndim_error(2);
    xyzq->shape_error({xyzq->shape()[0], 4});
    
    // Sizes
    size_t nS = s->shape()[0];
    size_t nP = xyzq->shape()[0];

    // Pointers
    const double* sp = s->data().data();
    const double* xyzqp = xyzq->data().data();

    // Working thread copies of moments
    int nthread = resources->nthread();
    std::vector<std::shared_ptr<Tensor>> Itemp;
    std::vector<double*> Ips;
    for (int t = 0; t < nthread; t++) {
        Itemp.push_back(std::shared_ptr<Tensor>(new Tensor({s->shape()[0], 2})));
        Ips.push_back(Itemp[t]->data().data());
    }

    #pragma omp parallel for num_threads(nthread) schedule(dynamic, 8)
    for (size_t P = 0; P < nP; P++) {
        double* Ip = Ips[omp_get_thread_num()];
        double xP = xyzqp[4*P + 0];
        double yP = xyzqp[4*P + 1];
        double zP = xyzqp[4*P + 2];
        double qP = xyzqp[4*P + 3];
        for (size_t Q = 0; Q < nP; Q++) {
            if (P < Q) continue;
            double perm = (P == Q ? 1.0 : 2.0);
            double xQ = xyzqp[4*Q + 0];
            double yQ = xyzqp[4*Q + 1];
            double zQ = xyzqp[4*Q + 2];
            double qQ = xyzqp[4*Q + 3];
            double xPQ = xP - xQ;
            double yPQ = yP - yQ;
            double zPQ = zP - zQ;
            double qPQ = qP * qQ;
            for (size_t S = 0; S < nS; S++) {    
                double sval = sp[S];
                std::pair<double, double> K = kernel(L, sval, xPQ, yPQ, zPQ);
                Ip[2*S + 0] += perm * qPQ * K.first;
                Ip[2*S + 1] += perm * qPQ * K.second;
            }
        }
    } 

    // Target (reduction over thread copies)
    std::shared_ptr<Tensor> I(new Tensor({s->shape()[0], 2}));
    for (int t = 0; t < nthread; t++) {
        I->axpby(Itemp[t], 1.0, 1.0);
    }

    return I;
}

// => Moment Specialization <= //

std::shared_ptr<Tensor> isotropic_moments(
    const std::shared_ptr<ResourceList>& resources,
    double L,
    const std::shared_ptr<Tensor> s,
    const std::shared_ptr<Tensor> xyzq
    )
{
    return moments(
        isotropic_kernel,
        resources,
        L,
        s,
        xyzq);
}
    
std::shared_ptr<Tensor> parallel_moments(
    const std::shared_ptr<ResourceList>& resources,
    double L,
    const std::shared_ptr<Tensor> s,
    const std::shared_ptr<Tensor> xyzq
    )
{
    return moments(
        parallel_kernel,
        resources,
        L,
        s,
        xyzq);
}

std::shared_ptr<Tensor> perpendicular_moments(
    const std::shared_ptr<ResourceList>& resources,
    double L,
    const std::shared_ptr<Tensor> s,
    const std::shared_ptr<Tensor> xyzq
    )
{
    return moments(
        perpendicular_kernel,
        resources,
        L,
        s,
        xyzq);
}

std::shared_ptr<Tensor> aligned_diffraction(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor> sxyz,
    const std::shared_ptr<Tensor> xyzq
    )
{
    // Validity checks
    sxyz->ndim_error(2);
    sxyz->shape_error({sxyz->shape()[0], 3});
    xyzq->ndim_error(2);
    xyzq->shape_error({xyzq->shape()[0], 4});
    
    // Sizes
    size_t nS = sxyz->shape()[0];
    size_t nP = xyzq->shape()[0];

    // Pointers
    const double* sxyzp = sxyz->data().data();
    const double* xyzqp = xyzq->data().data();

    // Working thread copies of moments
    int nthread = resources->nthread();
    std::vector<std::shared_ptr<Tensor>> Itemp;
    std::vector<double*> Ips;
    for (int t = 0; t < nthread; t++) {
        Itemp.push_back(std::shared_ptr<Tensor>(new Tensor({sxyz->shape()[0]})));
        Ips.push_back(Itemp[t]->data().data());
    }

    #pragma omp parallel for num_threads(nthread) schedule(dynamic, 8)
    for (size_t P = 0; P < nP; P++) {
        double* Ip = Ips[omp_get_thread_num()];
        double xP = xyzqp[4*P + 0];
        double yP = xyzqp[4*P + 1];
        double zP = xyzqp[4*P + 2];
        double qP = xyzqp[4*P + 3];
        for (size_t Q = 0; Q < nP; Q++) {
            if (P < Q) continue;
            double perm = (P == Q ? 1.0 : 2.0);
            double xQ = xyzqp[4*Q + 0];
            double yQ = xyzqp[4*Q + 1];
            double zQ = xyzqp[4*Q + 2];
            double qQ = xyzqp[4*Q + 3];
            double xPQ = xP - xQ;
            double yPQ = yP - yQ;
            double zPQ = zP - zQ;
            double qPQ = qP * qQ;
            for (size_t S = 0; S < nS; S++) {    
                double sx = sxyzp[3*S + 0];
                double sy = sxyzp[3*S + 1];
                double sz = sxyzp[3*S + 2];
                Ip[S] += perm * qPQ * cos(sx * xPQ + sy * yPQ + sz * zPQ);
            }
        }
    } 

    // Target (reduction over thread copies)
    std::shared_ptr<Tensor> I(new Tensor({sxyz->shape()[0]}));
    for (int t = 0; t < nthread; t++) {
        I->axpby(Itemp[t], 1.0, 1.0);
    }

    return I;
}

std::shared_ptr<Tensor> aligned_diffraction2(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor> sxyz,
    const std::shared_ptr<Tensor> xyzq
    )
{
    // Validity checks
    sxyz->ndim_error(2);
    sxyz->shape_error({sxyz->shape()[0], 3});
    xyzq->ndim_error(2);
    xyzq->shape_error({xyzq->shape()[0], 4});
    
    // Sizes
    size_t nS = sxyz->shape()[0];
    size_t nP = xyzq->shape()[0];

    // Pointers
    const double* sxyzp = sxyz->data().data();
    const double* xyzqp = xyzq->data().data();

    // Structure factor
    std::shared_ptr<Tensor> FR(new Tensor({nS}));
    std::shared_ptr<Tensor> FI(new Tensor({nS}));
    double* FRp = FR->data().data();
    double* FIp = FI->data().data();

    int nthread = resources->nthread();

    #pragma omp parallel for num_threads(nthread) schedule(dynamic, 8)
    for (size_t S = 0; S < nS; S++) {    
        double sx = sxyzp[3*S + 0];
        double sy = sxyzp[3*S + 1];
        double sz = sxyzp[3*S + 2];
        for (size_t P = 0; P < nP; P++) {
            double xP = xyzqp[4*P + 0];
            double yP = xyzqp[4*P + 1];
            double zP = xyzqp[4*P + 2];
            double qP = xyzqp[4*P + 3];
            double arg = sx * xP + sy * yP + sz * zP;
            FRp[S] += qP * cos(arg); 
            FIp[S] += qP * sin(arg); 
        }
    } 

    // Target 
    std::shared_ptr<Tensor> I(new Tensor({sxyz->shape()[0]}));
    double* Ip = I->data().data();
    for (size_t S = 0; S < nS; S++) {
        Ip[S] = pow(FRp[S],2) + pow(FIp[S],2);
    }

    return I;
}

} // namespace lightspeed
