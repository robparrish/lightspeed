#ifndef LS_LOCAL_HPP
#define LS_LOCAL_HPP

#include <memory>

namespace lightspeed {

class Basis;
class Tensor;

class Local {

public:

/**
 * Localize a set of orbitals by maximizing the sums of squares or quartic
 * powers of the orbital atomic charges. The orbital atomic charges are
 * computed as the inner product of overlaps between the orbitals and a set of
 * "indicator orbitals" (IAOs in IBO), where the indicator orbitals are labeled
 * by atomic center.
 *
 * This subroutine can be used for IBO or Pipek-Mezey localization.
 *
 * @param power the power in the localization metric (2 or 4, 4 recommended)
 * @param maxiter the maximum number of Jacobi sweep iterations to user
 * @param convergence the convergence criterion in the 2-norm of the orbital
 *  gradient
 * @param basis Basis object with nmin AOs, whose shells define which indicator
 *  orbitals go with which atoms.
 * @param L [input/output] (norb, nmin) Tensor containing the orbital -
 *  indicator orbital overlaps. 
 * @param U [output] (norb, norb) Tensor which is overwritten with the rotation
 *  matrix to transform from the input orbitals to the localized orbitals. The
 *  localized orbitals are obtained as Clocal_pj = Corig_pi U_ij.
 * @return (niter,2) Tensor with (metric,gradient) for each iteration on each
 *  row. Convergence was obtained if the gradient norm in the last row is <=
 *  convergence.
 **/
static
std::shared_ptr<Tensor> localize(
    int power,
    int maxiter,
    double convergence,
    const std::shared_ptr<Basis>& basis,
    const std::shared_ptr<Tensor>& L,
    const std::shared_ptr<Tensor>& U);

};

} // namespace lightspeed

#endif
