#ifndef LS_BOYS_HPP
#define LS_BOYS_HPP

#include <memory>
#include <vector>
#include <cmath>

namespace lightspeed {

class Tensor;
    
/**
 * Class Boys provides CPU-based Boys table, using seventh-order Chebyshev interpolation. 
 *
 * Defined for nboys from 0 to 17
 *
 * Notation: 
 *  nboys - order of Boys function.
 *  Tc - T argument at which to switch to asymptotic rule (Tc = 30 for absolute
 *      eps of < 10^-14).
 * 
 * For each Boys nboys, the domain from 0 to Tc is partitioned into nunit
 * equally spaced bins. 
 * 
 * 0|------|------|------|(nunit)|------|Tc
 * 
 * The bin spacing is dT = Tc / nunit. For argument T, the correct bin can be
 * located as floor(T * dTinv);
 *
 * Within each bin the 7 Chebyshev coefficients a_k are stored in reverse
 * order, to facilitate evaluation by the Clenshaw algorithm.
 *
 * Users should obtain a pointer to the singleton instance by using the
 * static instance() method below.
 **/ 
class Boys {

public:

// => Computational Routines <= //

/**
 * Compute the Boys functions with indices [0,nboys] at argument T
 * @param nboys - the highest order Boys function to compute
 * @param T - the argument to the Boys function
 * @param F - pointer where Boys functions will be written in ascending order 
 * @result F is overwritten with F_0, F_1, ... F_nboys.
 **/
inline void interpolate_F(
    int nboys,
    double T,
    double* F) const
{
    // T = 0 asymptotic limit
    if (T == 0) {
        for (int k = 0; k <= nboys; k++) {
            F[k] = inv_odds_[k];
        }
    }
    // T -> \infty asymptotic limit
    if (T >= Tc_) {
        // F[0] = 0.5 * sqrt(pi / T)
        F[0] = 8.8622692545275794E-01 / sqrt(T);
        // Early exit
        if (nboys == 0) return; 
        // Upward recursion is stable
        double expT = exp(-T);
        double invT = 0.5 / T;
        for (int k = 1; k <= nboys; k++) {
            F[k] = invT * ((2*k-1) * F[k-1] - expT);
        }
        return;
    }
    
    // Chebyshev interpolating bin
    double x = dTinv_[nboys] * T;
    int bin = (int) x;
    x = 2.0 * (x - bin) - 1.0;
    const double* as = boys_Fs_[nboys].data()
        + bin * 7;

    // Clenshaw algorithm (N = 7)
    double b0, b1, b2;
     
    b0 = (*as++); 
    b1 = b0;

    b0 = (*as++) + 2.0 * x * b1;
    b2 = b1; b1 = b0;

    b0 = (*as++) + 2.0 * x * b1 - b2;
    b2 = b1; b1 = b0;

    b0 = (*as++) + 2.0 * x * b1 - b2;
    b2 = b1; b1 = b0;

    b0 = (*as++) + 2.0 * x * b1 - b2;
    b2 = b1; b1 = b0;

    b0 = (*as++) + 2.0 * x * b1 - b2;
    b2 = b1; b1 = b0;

    b0 = 2.0 * (*as) + 2.0 * x * b1 - b2; 
    F[nboys] = exp(0.5 * (b0 - b2));
    
    // Early exit
    if (nboys == 0) return; 

    // Downward recursion
    double expT = exp(-T);
    for (int k = nboys; k > 0; k--) {
        F[k-1] = (2.0 * T * F[k] + expT) * inv_odds_[k-1];
    }
}

// => Member Data Accessors <= //
    
// Critical T to crossover to asymptotic formula (Tc = 30 herein)
double Tc() const { return Tc_; }
// Number of bins for each nboys
const std::vector<int>& nunit() const { return nunit_; }
// The inverse spacing of interpolation units for each nrys rule (nunit / Tc)
const std::vector<double>& dTinv() const { return dTinv_; }
// 1/1, 1/3, 1/5, ... (T=0 limit of F_n)
const std::vector<double>& inv_odds() const { return inv_odds_; }

/**
 * Boys Chebyshev coefficients for each nboys
 *
 * Within each nboys, the arrays are striped as
 * (nunit,(a6,a5,a4,a3,a2,a1,a0))
 **/
const std::vector<std::vector<double> >& boys_Fs() const { return boys_Fs_; }
    
private:

// => Boys Table Data <= //

double Tc_;
std::vector<int> nunit_;
std::vector<double> dTinv_;
std::vector<std::vector<double> > boys_Fs_;
std::vector<double> inv_odds_;

static std::vector<int> build_nunit();
static std::vector<std::vector<double> > build_boys_Fs();

// => Singleton Pattern <= //

public:
  
// The pointer to the singleton instance of this class
static std::shared_ptr<Boys> instance() {
    return Boys::instance_;
}

private:

static std::shared_ptr<Boys> instance_;

/**
 * Constructor, fills in data from source files. Users of this class
 * should use the singleton instance instead.
 **/
Boys();

public:

// => Python-Accessible Use Functions <= //

/**
 * Compute the Boys functions at arguments T
 *
 * @param nboys the maximum Boys function to compute (nboys+1 total Boys functions)
 * @param T an (npoint,) Tensor containing the T arguments to evaluate the quadrature at
 * @return an (npoint,nboys+1) Tensor with the desired Boys functions
 **/
static
std::shared_ptr<Tensor> compute(
    int nboys,
    const std::shared_ptr<Tensor>& T);
};

} // namespace lightspeed

#endif
