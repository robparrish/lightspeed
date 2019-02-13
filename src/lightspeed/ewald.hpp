#ifndef LS_EWALD_HPP
#define LS_EWALD_HPP

#include <memory>
#include <vector>
#include <stdexcept>

namespace lightspeed {

/**
 * Class Ewald represents a generalized Ewald operator:
 *
 * O(r_12) = \sum_{i} a_i erf(w_i r_12) / r_12
 *
 * I.e., a linear combination of long-range Ewald operators. 
 *
 * A value of w_i = -1.0 is used to signal w_i = \infty, i.e., the usual 
 *  1 / r_12 operator.
 * 
 * Some limited characteristics are provided to check if the operator is of the
 * form 1 / r_12 or erfc(w r_12) / r_12. This can allow certain specialized
 * algorithms to achieve better sparsity, or use alternative code with fewer
 * steps.
 **/ 
class Ewald {

public:

/**
 * Verbatim Constructor: fills fields below (see accessor fields for details)
 **/
Ewald(
    const std::vector<double>& scales,
    const std::vector<double>& omegas) :
    scales_(scales),
    omegas_(omegas) 
    {
        if (scales.size() != omegas_.size()) throw std::runtime_error("Ewald: scales and omegas must be same size.");
        if (scales.size() == 0) throw std::runtime_error("Ewald: scales/omegas must have >= 1 elements.");
    }

/**
 * Convenience routine to grab the Coulomb operator.
 * @return the Ewald class representation of the Coulomb operator
 **/
static
std::shared_ptr<Ewald> coulomb() { 
    std::vector<double> p1;
    p1.push_back(1.0);
    std::vector<double> m1;
    m1.push_back(-1.0);
    return std::shared_ptr<Ewald>(new Ewald(p1,m1));
}

// => Accessors <= //

// The scales a_i
const std::vector<double>& scales() const { return scales_; }
// The cutoff parameters w_i (-1.0 is used to signal w_i = \infty)
const std::vector<double>& omegas() const { return omegas_; }

// => Extra Characteristics <= //

// Is this operator a scaled version of the usual 1 / r_12 operator?
bool is_coulomb() const { return omegas_.size() == 1 && omegas_[0] == -1.0; }
  
// Is this operator a scaled version of the common erfc(w r_12) / r_12 operator?
bool is_sr() const { 
    if (omegas_.size() != 2) return false;
    if (scales_[0] != -scales_[1]) return false;
    if ((omegas_[0] == -1.0) ^ (omegas_[1] == -1.0)) return true; // opposite scales with two 1/r operators is pathological
    return false;
}
// What is the short-range omega in a erfc(w r_12) / r_12? Throws if is_sr() is false
double sr_omega() const {
    if (!is_sr()) throw std::runtime_error("Ewald: sr_omega does not make sense: is_sr() is false");
    return (omegas_[0] == -1.0 ? omegas_[1] : omegas_[0]);
}
// What is the short-range scaling in a erfc(w r_12) / r_12? Throws if is_sr() is false
double sr_scale() const {
    if (!is_sr()) throw std::runtime_error("Ewald: sr_omega does not make sense: is_sr() is false");
    return (omegas_[0] == -1.0 ? scales_[0] : scales_[1]);
}

// Is this operator a scaled version of the common erf(w r_12) / r_12 operator?
bool is_lr() const { 
    if (omegas_.size() != 1) return false;
    if (omegas_[0] == -1.0) return false;
    return true;
}
// What is the short-range omega in a erf(w r_12) / r_12? Throws if is_lr() is false
double lr_omega() const {
    if (!is_lr()) throw std::runtime_error("Ewald: lr_omega does not make sense: is_lr() is false");
    return omegas_[0];
}
// What is the short-range scaling in a erf(w r_12) / r_12? Throws if is_lr() is false
double lr_scale() const {
    if (!is_lr()) throw std::runtime_error("Ewald: lr_omega does not make sense: is_lr() is false");
    return scales_[0];
}

private:

// => Fields <= //

std::vector<double> scales_;
std::vector<double> omegas_;

};

} // namespace lightspeed 

#endif

