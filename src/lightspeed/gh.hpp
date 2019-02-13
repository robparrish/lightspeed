#ifndef LS_GH_HPP
#define LS_GH_HPP

#include <vector>
#include <memory>

namespace lightspeed {

class Tensor;

/**
 * Class GH is a singleton class providing the standard Gauss-Hermite
 * quadrature rules. These exactly integrate:
 *
 * I = \int_{\mathbb{R}} \mathrm{d} x \ P_{2n-1} (x) \exp(-x) 
 *   = \sum_{i=1}^{n} w_i P_{2n-1} (t_i)
 *
 * Rules are provided for n = 1 to n = 20.
 *
 * Users should obtain a pointer to the singleton instance by using the
 * static instance() method below.
 **/
class GH {

public:

// The vector of n quadrature nodes t_i for the n-point quadrature rule
const std::vector<std::vector<double> >& ts() const { return ts_; }
// The vector of n quadrature weights t_i for the n-point quadrature rule
const std::vector<std::vector<double> >& ws() const { return ws_; }

private:

GH() : ts_(GH::build_ts()), ws_(GH::build_ws()) {}

std::vector<std::vector<double> > ts_;
std::vector<std::vector<double> > ws_;

public:

static std::vector<std::vector<double> > build_ts();
static std::vector<std::vector<double> > build_ws();

// => Singleton Pattern <= //

public:

// The pointer to the singleton instance of this class
static std::shared_ptr<GH> instance() {
    return GH::instance_;
}

private:

static std::shared_ptr<GH> instance_;

public:

// => Python-Accessible Use Functions <= //

/**
 * Build the n-point Gauss-Hermite rule
 *
 * @param n the number of points in the Gauss-Hermite rule
 * @return (n,2) Tensor with (t,w) on each row
 **/
static
std::shared_ptr<Tensor> compute(int n);
    
};

} // namespace lightspeed

#endif
