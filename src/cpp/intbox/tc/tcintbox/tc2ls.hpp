#ifndef LS_TC2LS_HPP
#define LS_TC2LS_HPP

#include <cstddef>
#include <vector>
#include <cstdio>
#include <cstring>
#include <memory>

namespace lightspeed {

class Tensor;
class Basis;
class Ewald;

class TCTransform {

public:

/**
 * Convert Ewald object to a vector of tuples of (scalfr,scallr,omega) tasks as
 * TeraChem expects.
 * @param ewald the LS-type Ewald operator
 * @return a vector of (scalfr,scallr,omega) tasks, each of which requires one
 * TeraChem IntBox function call.
 **/
static std::vector<std::tuple<double,double,double> > ewaldTC(
    const std::shared_ptr<Ewald>& ewald);

static std::shared_ptr<Tensor> LStoTC(
    const std::shared_ptr<Tensor>& T1,
    const std::shared_ptr<Basis>& primary);

static std::shared_ptr<Tensor> TCtoLS(
    const std::shared_ptr<Tensor>& T2,
    const std::shared_ptr<Basis>& primary);

static 
void LStoTC(
    int L1,
    int L2,
    double* target,
    double* scratch)
{
    if (L1 < 2 && L2 < 2) return;

    const double three13 = 1.7320508075688772E+00; //sqrt(3.0);

    int ncart1 = (L1 + 1) * (L1 + 2) / 2;
    int ncart2 = (L2 + 1) * (L2 + 2) / 2;

    if (L1 == 2) {
        for (int j = 0; j < ncart2; j++) {
            scratch[3*ncart2 + j] = target[0*ncart2 + j] * three13; // xx
            scratch[0*ncart2 + j] = target[1*ncart2 + j]; // xy
            scratch[1*ncart2 + j] = target[2*ncart2 + j]; // xz
            scratch[4*ncart2 + j] = target[3*ncart2 + j] * three13; // yy
            scratch[2*ncart2 + j] = target[4*ncart2 + j]; // yz
            scratch[5*ncart2 + j] = target[5*ncart2 + j] * three13; // zz
        } 
    } else {
        memcpy(scratch,target,sizeof(double)*ncart1*ncart2);
    }

    if (L2 == 2) {
        for (int i = 0; i < ncart1; i++) {
            target[i*ncart2 + 3] = scratch[i*ncart2 + 0] * three13; // xx
            target[i*ncart2 + 0] = scratch[i*ncart2 + 1]; // xy
            target[i*ncart2 + 1] = scratch[i*ncart2 + 2]; // xz
            target[i*ncart2 + 4] = scratch[i*ncart2 + 3] * three13; // yy
            target[i*ncart2 + 2] = scratch[i*ncart2 + 4]; // yz
            target[i*ncart2 + 5] = scratch[i*ncart2 + 5] * three13; // zz
        } 
    } else {
        memcpy(target,scratch,sizeof(double)*ncart1*ncart2);
    }
}

static 
void TCtoLS(
    int L1,
    int L2,
    double* target,
    double* scratch)
{
    if (L1 < 2 && L2 < 2) return;

    const double three13 = 1.7320508075688772E+00; //sqrt(3.0);

    int ncart1 = (L1 + 1) * (L1 + 2) / 2;
    int ncart2 = (L2 + 1) * (L2 + 2) / 2;

    if (L1 == 2) {
        for (int j = 0; j < ncart2; j++) {
            scratch[0*ncart2 + j] = target[3*ncart2 + j] * three13; // xx
            scratch[1*ncart2 + j] = target[0*ncart2 + j]; // xy
            scratch[2*ncart2 + j] = target[1*ncart2 + j]; // xz
            scratch[3*ncart2 + j] = target[4*ncart2 + j] * three13; // yy
            scratch[4*ncart2 + j] = target[2*ncart2 + j]; // yz
            scratch[5*ncart2 + j] = target[5*ncart2 + j] * three13; // zz
        } 
    } else {
        memcpy(scratch,target,sizeof(double)*ncart1*ncart2);
    }

    if (L2 == 2) {
        for (int i = 0; i < ncart1; i++) {
            target[i*ncart2 + 0] = scratch[i*ncart2 + 3] * three13; // xx
            target[i*ncart2 + 1] = scratch[i*ncart2 + 0]; // xy
            target[i*ncart2 + 2] = scratch[i*ncart2 + 1]; // xz
            target[i*ncart2 + 3] = scratch[i*ncart2 + 4] * three13; // yy
            target[i*ncart2 + 4] = scratch[i*ncart2 + 2]; // yz
            target[i*ncart2 + 5] = scratch[i*ncart2 + 5] * three13; // zz
        } 
    } else {
        memcpy(target,scratch,sizeof(double)*ncart1*ncart2);
    }
}

};

} // namespace lightspeed

#endif

