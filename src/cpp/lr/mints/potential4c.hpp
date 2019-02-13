#ifndef LS_POTENTIAL_INT4C_HPP
#define LS_POTENTIAL_INT4C_HPP

#include <memory>
#include <lightspeed/am.hpp>
#include <cstddef>
#include <memory>
#include <vector>

namespace lightspeed {

class Shell;
class Basis;
class PairList;
class GH;
class Rys;

/**!
 * Class Int4C provides a common interface for the low-level computation of
 * electron repulsion integrals.
 *
 * => Additional Notes <=
 *
 * These objects are not thread-safe due to internal scratch arrays! The best
 * policy is to make one object for each thread.
 *
 * - Rob Parrish, 21 August, 2015
 **/
class PotentialInt4C {

public:

// => Constructors <= //

/**
 * Construct a PotentialInt4C to compute shell-based integrals of the type:
 *
 * (12|O(r_12)|34)
 *
 * O(r_12) = \sum_{i} a_i erf(w_i r_12) / r_12
 *
 * I.e., a linear combination of long-range Ewald operators. 
 *
 * A value of w_i = -1.0 is used to signal w_i = \infty, i.e., the usual 
 *  1 / r_12 operator.
 * 
 * @param alphas scales of the various Ewald terms in the operator O(r_12)
 * @param omegas omegas of the various Ewald terms in the operator O(r_12). -1.0 is
 *  used to signal the usual 1 / r_12 operator.
 * @param L1 max angular momentum in shell 1
 * @param L2 max angular momentum in shell 2
 * @param L3 max angular momentum in shell 3
 * @param L4 max angular momentum in shell 4
 **/
PotentialInt4C(
    const std::vector<double>& alphas,
    const std::vector<double>& omegas,
    int L1,
    int L2,
    int L3,
    int L4,
    int deriv=0);

// => Accessors <= //

/// scales of the various Ewald terms in the operator O(r_12) 
const std::vector<double>& alphas() const { return alphas_; }
/// omegas of the various Ewald terms in the operator O(r_12). -1.0 is used to
/// signal the usual 1 / r_12 operator.
const std::vector<double>& omegas() const { return omegas_; }
/// Maximum angular momentum in shell 1
int L1() const { return L1_; }
/// Maximum angular momentum in shell 2
int L2() const { return L2_; }
/// Maximum angular momentum in shell 3
int L3() const { return L3_; }
/// Maximum angular momentum in shell 4
int L4() const { return L4_; }
/// Maximum derivative level enabled
int deriv() const { return deriv_; }

/// Buffer of output integrals or integral derivatives
const std::vector<double>& data() const { return data1_; }

/// Should this object apply spherical transformations if present in the basis sets (defaults to true)
bool is_spherical() const { return is_spherical_; }

/// Single maximum angular momentum present across the basis sets
int max_am() const { return std::max(std::max(L1_,L2_), std::max(L3_,L4_)); }
/// Total maximum angular momentum across the basis sets
int total_am() const { return L1_ + L2_ + L3_ + L4_; }
/// Return the chunk size (max_ncart1 x max_ncart2 x max_ncart3 x max_ncart4)
size_t chunk_size() const;

// => Setters <= //

void set_is_spherical(bool is_spherical) { is_spherical_ = is_spherical; }

// => Low-Level Computers <= //

/**
 * Compute the integrals for 4 shells (12|O(r_12)|34)
 * @param shell1 shell1
 * @param shell2 shell2
 * @param shell3 shell3
 * @param shell4 shell4
 * @result the integrals are placed in the "data" scratch vector in C order. 
 *  If is_spherical is set to true, any needed spherical transformations will
 *  be applied according to the is_pure field of each shell.
 **/
void compute_shell0(
    const Shell& shell1,
    const Shell& shell2,
    const Shell& shell3,
    const Shell& shell4);

protected:

/// Verbatim fields, see constructor
std::vector<double> alphas_;
std::vector<double> omegas_;
int L1_;
int L2_;
int L3_;
int L4_;
int deriv_;

/// Scratch space for integrals
std::vector<double> data1_; 
/// Scratch space for spherical transformation
std::vector<double> data2_; 

/// Reference to Gauss-Hermite quadrature
std::shared_ptr<GH> gh_;
    
/// Reference to Rys quadrature
std::shared_ptr<Rys> rys_;
/// Scratch space for Rys quadrature
std::vector<double> ts_;
std::vector<double> ws_;

/// 1D Rys Integrals
std::vector<std::vector<double>> J_;

// => Spherical Transforms <= //

/// Are spherical transforms to be applied (if in input shells?)
/// Setting to false forces integrals to occur in cartesian
bool is_spherical_ = true;
/// Internal CO->SO transformation information
std::vector<AngularMomentum> am_info_;

/**!
 * Helper to apply spherical transformations to the cartesian integrals
 *
 * This should be called separately for each chunk
 *
 * Does not check is_spherical, the calling code is responsible for this
 *
 * Uses buffer2 for scratch space
 *
 * @param L1 angular momentum of shell1
 * @param L2 angular momentum of shell2
 * @param L3 angular momentum of shell3
 * @param L4 angular momentum of shell4
 * @param S1 perform transform for shell1
 * @param S2 perform transform for shell2
 * @param S3 perform transform for shell3
 * @param S4 perform transform for shell4
 * @param target pointer in buffer1 to start of chunk
 * @param scratch pointer (at least chunk size)
 * @result target is updated with the transformed integrals
 **/
void apply_spherical(
        int L1,
        int L2,
        int L3,
        int L4,
        bool S1,
        bool S2,
        bool S3,
        bool S4,
        double *target,
        double *scratch);

};

} // namespace lightspeed

#endif
