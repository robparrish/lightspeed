#ifndef LS_RYS_HPP
#define LS_RYS_HPP

#include <memory>
#include <vector>
#include <cmath>

namespace lightspeed {

class Tensor;
    
/**
 * Class Rys provides a CPU-based Rys table, using seventh-order Chebyshev
 * interpolation. 
 *
 * Currently defined for nrys from 1 to 9
 *
 * Notation: 
 *  nrys - the number of points in a given Rys quadrature
 *  mrys - the index of a point within a given Rys quadrature, in [0,nrys).
 *     The quadrature nodes are sorted from least to greatest in mrys. 
 *  t - the quadrature nodes
 *  w - the quadrature weights
 *  gl - Gauss-Legendre quadrature (the T = 0 limit)
 *  gh - Gauss-Hermite quadrature (the T = \infty limit)
 *  Tc - T argument at which to switch to asymptotic Gauss-Hermite rule.
 * 
 * For each Rys quadrature (nrys) and index (mrys), the domain from 0 to Tc is
 * partitioned into nunit equally spaced bins. 
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
class Rys {

public:

// => Computational Routines <= //


/**
 * Compute the m-th Rys root for the n-th Rys quadrature rule at argument T
 * @param nrys - the number of points in the desired Rys quadrature
 * @param mrys - the specific index of the point in [0,nrys)
 * @param T - the argument to the Rys quadrature
 * @return t - the desired Rys root
 **/
inline double interpolate_t(
    int nrys,
    int mrys,
    double T) const
{
    // T = 0 asymptotic limit
    if (T == 0.0) {
        return pegl_ts_[nrys][mrys];
    }
    // T -> \infty asymptotic limit
    if (T >= Tc_[nrys]) {
        return pegh_ts_[nrys][mrys] / sqrt(T);
    }
    // Chebyshev interpolating bin
    double x = dTinv_[nrys] * T;
    int bin = (int) x;
    x = 2.0 * (x - bin) - 1.0;
    const double* as = rys_ts_[nrys].data()
        + mrys * (nunit_[nrys] * 7)
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
    return 0.5 * (b0 - b2);
}

/**
 * Compute the m-th Rys weight for the n-th Rys quadrature rule at argument T
 * @param nrys - the number of points in the desired Rys quadrature
 * @param mrys - the specific index of the point in [0,nrys)
 * @param T - the argument to the Rys quadrature
 * @return t - the desired Rys weight
 **/
inline double interpolate_w(
    int nrys,
    int mrys,
    double T) const
{
    // T = 0 asymptotic limit
    if (T == 0.0) {
        return pegl_ws_[nrys][mrys];
    }
    // T -> \infty asymptotic limit
    if (T >= Tc_[nrys]) {
        return pegh_ws_[nrys][mrys] / sqrt(T);
    }
    // Chebyshev interpolating bin
    double x = dTinv_[nrys] * T;
    int bin = (int) x;
    x = 2.0 * (x - bin) - 1.0;
    const double* as = rys_ws_[nrys].data()
        + mrys * (nunit_[nrys] * 7)
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
    return exp(0.5 * (b0 - b2));
}

// => Member Data Accessors <= //
    
// The values of T where each nrys rule goes to the asymptotic Gauss-Hermite quadrature
const std::vector<double>& Tc() const { return Tc_; }
// The number of interpolation units for each nrys rule
const std::vector<int>& nunit() const { return nunit_; }
// The inverse spacing of interpolation units for each nrys rule (nunit / Tc)
const std::vector<double>& dTinv() const { return dTinv_; }

/**
 * Rys Chebyshev coefficients for each nrys root values
 *
 * Within each nrys, the arrays are striped as
 * (mrys,nunit,(a6,a5,a4,a3,a2,a1,a0))
 **/
const std::vector<std::vector<double> >& rys_ts() const { return rys_ts_; }
/**
 * Rys Chebyshev coefficients for each nrys weight values
 *
 * Within each nrys, the arrays are striped as
 * (mrys,nunit,(a6,a5,a4,a3,a2,a1,a0))
 *
 * The natural log of the weights are interpolated, then the exponential is
 * taken. This helps improve relative accuracy for small weights.
 **/
const std::vector<std::vector<double> >& rys_ws() const { return rys_ws_; }
// Positive even Gauss-Legendre quadrature roots (T=0)
const std::vector<std::vector<double> >& pegl_ts() const { return pegl_ts_; }
// Positive even Gauss-Legendre quadrature weights (T=0)
const std::vector<std::vector<double> >& pegl_ws() const { return pegl_ws_; }
// Positive even Gauss-Hermite quadrature roots (T->\infty)
const std::vector<std::vector<double> >& pegh_ts() const { return pegh_ts_; }
// Positive even Gauss-Hermite quadrature weights (T->\infty)
const std::vector<std::vector<double> >& pegh_ws() const { return pegh_ws_; }
    
private:

// => Rys Table Data <= //

// Asymptotic crossovers for nrys
std::vector<double> Tc_; 
// Number of bins for nrys
std::vector<int> nunit_;
// nunit / Tc for nrys
std::vector<double> dTinv_;

/**
 * Rys Chebyshev coefficients for each nrys
 *
 * Within each nrys, the arrays are striped as
 * (mrys,nunit,(a6,a5,a4,a3,a2,a1,a0))
 **/
std::vector<std::vector<double> > rys_ts_;
std::vector<std::vector<double> > rys_ws_;

// => Gauss-Legendre/Gauss-Hermite Data <= //

// Positive even Gauss-Legendre quadrature nodes
std::vector<std::vector<double> > pegl_ts_;
// Positive even Gauss-Legendre quadrature weights
std::vector<std::vector<double> > pegl_ws_;
// Positive even Gauss-Hermite quadrature nodes
std::vector<std::vector<double> > pegh_ts_;
// Positive even Gauss-Hermite quadrature weights
std::vector<std::vector<double> > pegh_ws_;

static std::vector<double> build_Tc();
static std::vector<int> build_nunit();
static std::vector<std::vector<double> > build_rys_ts();
static std::vector<std::vector<double> > build_rys_ws();
static std::vector<std::vector<double> > build_pegl_ts();
static std::vector<std::vector<double> > build_pegl_ws();
static std::vector<std::vector<double> > build_pegh_ts();
static std::vector<std::vector<double> > build_pegh_ws();

// => Singleton Pattern <= //

public:
  
// The pointer to the singleton instance of this class
static std::shared_ptr<Rys> instance() {
    return Rys::instance_;
}

private:

static std::shared_ptr<Rys> instance_;

/**
 * Constructor, fills in quadrature data from source files. Users of this class
 * should use the singleton instance instead.
 **/
Rys();

public:

// => Python-Accessible Use Functions <= //

/**
 * Compute the Rys roots for the nrys-point Rys table at arguments T
 *
 * @param nrys the number of points in the Rys table
 * @param T an (npoint,) Tensor containing the T arguments to evaluate the quadrature at
 * @return an (npoint,nrys) Tensor with the desired Rys roots
 **/
static
std::shared_ptr<Tensor> compute_t(
    int nrys,
    const std::shared_ptr<Tensor>& T);

/**
 * Compute the Rys weights for the nrys-point Rys table at arguments T
 *
 * @param nrys the number of points in the Rys table
 * @param T an (npoint,) Tensor containing the T arguments to evaluate the quadrature at
 * @return an (npoint,nrys) Tensor with the desired Rys weights
 **/
static
std::shared_ptr<Tensor> compute_w(
    int nrys,
    const std::shared_ptr<Tensor>& T);

};

} // namespace lightspeed

#endif
