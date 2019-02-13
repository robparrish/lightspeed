#ifndef LS_PURE_TRANSFORM_UTIL_HPP
#define LS_PURE_TRANSFORM_UTIL_HPP

#include <lightspeed/am.hpp>
#include <cstring>
#include <vector>

namespace lightspeed {

class Primitive;

class PureTransformUtil {

public:

static
void cartToPure2(
    const std::vector<AngularMomentum>& am_info,
    int L1,
    int L2,
    bool S1,
    bool S2,
    double* target,
    double* scratch)
{
    if (!S1 && !S2) return;
    
    int ncart1 = (L1 + 1) * (L1 + 2) / 2;
    int ncart2 = (L2 + 1) * (L2 + 2) / 2;

    int npure1 = 2 * L1 + 1;
    int npure2 = 2 * L2 + 1;

    //int nfun1 = (S1 ? npure1 : ncart1);
    int nfun2 = (S2 ? npure2 : ncart2);

    if (S2 && L2 > 0) {
        ::memset(scratch,'\0',sizeof(double)*ncart1*npure2);
        const AngularMomentum& trans = am_info[L2];
        size_t ncoef = trans.ncoef();
        const std::vector<int>& cart_inds  = trans.cart_inds();
        const std::vector<int>& pure_inds  = trans.pure_inds();
        const std::vector<double>& cart_coefs = trans.cart_coefs();
        for (int p1 = 0; p1 < ncart1; p1++) {
            double* cartp = target  + p1 * ncart2;
            double* purep = scratch + p1 * npure2;
            for (size_t ind = 0L; ind < ncoef; ind++) {
                *(purep + pure_inds[ind]) += cart_coefs[ind] * *(cartp + cart_inds[ind]);
            }
        } 
    } else {
        ::memcpy(scratch,target,sizeof(double)*ncart1*ncart2);
    }

    if (S1 && L1 > 0) {
        ::memset(target,'\0',sizeof(double)*npure1*nfun2);
        const AngularMomentum& trans = am_info[L1];
        size_t ncoef = trans.ncoef();
        const std::vector<int>& cart_inds  = trans.cart_inds();
        const std::vector<int>& pure_inds  = trans.pure_inds();
        const std::vector<double>& cart_coefs = trans.cart_coefs();
        for (size_t ind = 0L; ind < ncoef; ind++) {
            double* cartp = scratch + cart_inds[ind] * nfun2;
            double* purep = target  + pure_inds[ind] * nfun2;
            double coef = cart_coefs[ind];
            for (int p2 = 0; p2 < nfun2; p2++) {
                *purep++ += coef * *cartp++;
            }
        } 
    } else {
        ::memcpy(target,scratch,sizeof(double)*ncart1*nfun2);
    }
}

static
void pureToCart2(
    const std::vector<AngularMomentum>& am_info,
    int L1,
    int L2,
    bool S1,
    bool S2,
    double* target,
    double* scratch)
{
    if (!S1 && !S2) return;
    
    int ncart1 = (L1 + 1) * (L1 + 2) / 2;
    int ncart2 = (L2 + 1) * (L2 + 2) / 2;

    int npure1 = 2 * L1 + 1;
    int npure2 = 2 * L2 + 1;

    int nfun1 = (S1 ? npure1 : ncart1);
    //int nfun2 = (S2 ? npure2 : ncart2);

    if (S2 && L2 > 0) {
        ::memset(scratch,'\0',sizeof(double)*nfun1*ncart2);
        const AngularMomentum& trans = am_info[L2];
        size_t ncoef = trans.ncoef();
        const std::vector<int>& cart_inds  = trans.cart_inds();
        const std::vector<int>& pure_inds  = trans.pure_inds();
        const std::vector<double>& cart_coefs = trans.cart_coefs();
        for (int p1 = 0; p1 < nfun1; p1++) {
            double* cartp = scratch + p1 * ncart2;
            double* purep = target  + p1 * npure2;
            for (size_t ind = 0L; ind < ncoef; ind++) {
                *(cartp + cart_inds[ind]) += cart_coefs[ind] * *(purep + pure_inds[ind]);
            }
        } 
    } else {
        ::memcpy(scratch,target,sizeof(double)*nfun1*ncart2);
    }

    if (S1 && L1 > 0) {
        ::memset(target,'\0',sizeof(double)*ncart1*ncart2);
        const AngularMomentum& trans = am_info[L1];
        size_t ncoef = trans.ncoef();
        const std::vector<int>& cart_inds  = trans.cart_inds();
        const std::vector<int>& pure_inds  = trans.pure_inds();
        const std::vector<double>& cart_coefs = trans.cart_coefs();
        for (size_t ind = 0L; ind < ncoef; ind++) {
            double* cartp = target  + cart_inds[ind] * ncart2;
            double* purep = scratch + pure_inds[ind] * ncart2;
            double coef = cart_coefs[ind];
            for (int p2 = 0; p2 < ncart2; p2++) {
                *cartp++ += coef * *purep++;
            }
        } 
    } else {
        ::memcpy(target,scratch,sizeof(double)*ncart1*ncart2);
    }
}

static
void pureToCart1(
    const std::vector<AngularMomentum>& am_info,
    int L1,
    bool S1,
    double* target,
    const double* source,
    int norb)
{
    int ncart1 = (L1 + 1) * (L1 + 2) / 2;
    int npure1 = 2 * L1 + 1;
    if (!S1) {
        ::memcpy(target,source,sizeof(double)*ncart1*norb);    
        return;
    }
    
    //::memset(target,'\0',sizeof(double)*ncart1*norb); // Always pre-allocated
    const AngularMomentum& trans = am_info[L1];
    size_t ncoef = trans.ncoef();
    const std::vector<int>& cart_inds  = trans.cart_inds();
    const std::vector<int>& pure_inds  = trans.pure_inds();
    const std::vector<double>& cart_coefs = trans.cart_coefs();
    for (size_t ind = 0L; ind < ncoef; ind++) {
        double* cartp = target + cart_inds[ind] * norb; 
        const double* purep = source + pure_inds[ind] * norb; 
        double coef = cart_coefs[ind];
        for (int i = 0; i < norb; i++) {
            *cartp++ += coef * *purep++;
        }
    }
}

};
    
} // namespace lightspeed 

#endif
