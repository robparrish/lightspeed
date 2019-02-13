#ifndef LS_BLUR_BOX_HPP
#define LS_BLUR_BOX_HPP

#include <memory>

namespace lightspeed {

class ResourceList;
class Tensor;
class HashedGrid;
class PairList;

/**
 * Class BlurBox provides routines for Gaussian-smoothed ("blurred") evaluation
 * of grid properties of both point charges and Cartesian Gaussian charge
 * densities, including arbitrary lattice translation vectors.
 **/
class BlurBox {
    
public:

// => Point Routines <= //

/**
 * Compute the grid-collocated density values of a set of blurred point
 * charges:
 *      
 *  rho_P += \sum_{AT} -Z_A (alpha / pi)^(3/2) \exp(-alpha * r_ATP^2)
 *
 * The centers \vec r_A are translated according to a set of user-provided
 * translation vectors \vec r_T, e.g., \vec r_AT = \vec r_A + \vec r_T
 *
 * The computation is sieved so that only contributions with 
 * |rho_{ATP}| >= thre are retained. 
 *
 * Implementation: CPU only.
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param xyzc the point charge field, an (nA, 4) Tensor with (x,y,z,c) on each
 *  row
 * @param grid a HashedGrid to collocate on, collocation points P taken from
 *  grid->xyz() 
 * @param trans (nT,3) Tensor with translation vectors for the centers r_A. If
 *  you do not want translations, provide <0.0,0.0,0.0>.
 * @param alpha the blur exponent (-1.0 indicates no blurring)
 * @param thre cutoff value below which collocation values are clamped to zero.
 * @param rho [optional] an (nP,) Tensor to accumulate into. If not provided,
 *  this will be allocated.
 * @return a (nP,) Tensor with the result accumulated. 
 **/
static
std::shared_ptr<Tensor> pointDensity(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& xyzc,
    const std::shared_ptr<HashedGrid>& grid,
    const std::shared_ptr<Tensor>& trans,
    double alpha,
    double thre,
    const std::shared_ptr<Tensor>& rho = std::shared_ptr<Tensor>()
    );

/**
 * Compute the point potentials of a blurred grid-based potential 
 *      
 *  v2_A += \sum_{PT} v_P (alpha / pi)^(3/2) \exp(-alpha * r_ATP^2)
 *
 * The centers \vec r_A are translated according to a set of user-provided
 * translation vectors \vec r_T, e.g., \vec r_AT = \vec r_A + \vec r_T
 *
 * The computation is sieved so that only contributions with 
 * |v2_{ATP}| >= thre are retained. 
 *
 * Implementation: CPU only.
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param xyz the point charge field, an (nA, 3) Tensor with (x,y,z) on each
 *  row
 * @param grid a HashedGrid to collocate on, collocation points P taken from
 *  grid->xyz() 
 * @param v an (nP,) Tensor with the grid potential values
 * @param trans (nT,3) Tensor with translation vectors for the centers r_A. If
 *  you do not want translations, provide <0.0,0.0,0.0>.
 * @param alpha the blur exponent (-1.0 indicates no blurring)
 * @param thre cutoff value below which collocation values are clamped to zero.
 * @param vmax the maximum absolute value of v, as it is formally O(N^2) to
 *  compute on the fly.
 * @param v2 [optional] an (nA,) Tensor to accumulate into. If not provided,
 *  this will be allocated.
 * @return a (nA,) Tensor with the result accumulated. 
 **/
static
std::shared_ptr<Tensor> pointPotential(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& xyz,
    const std::shared_ptr<HashedGrid>& grid,
    const std::shared_ptr<Tensor>& v,
    const std::shared_ptr<Tensor>& trans,
    double alpha,
    double thre,
    double vmax,
    const std::shared_ptr<Tensor>& v2 = std::shared_ptr<Tensor>()
    );

/**
 * Compute the gradient of the energy from the interaction of a set of point
 * charges with a blurred grid-based potential. The energy is defined as:
 *  
 *  E = \sum_{ATP} - Z_A v_P (alpha / pi)^(3/2) \exp(-alpha * r_ATP^2)
 *      
 * and the gradient is then, 
 *
 *  G_A += \sum_{TP} - Z_A v_P (alpha / pi)^(3/2) \partial_{A} 
 *      \exp(-alpha * r_ATP^2)
 *
 * The centers \vec r_A are translated according to a set of user-provided
 * translation vectors \vec r_T, e.g., \vec r_AT = \vec r_A + \vec r_T
 *
 * The computation is sieved so that only energy contributions with 
 * |E_{ATP}| >= thre are retained. 
 *
 * Implementation: CPU only.
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param xyzc the point charge field, an (nA, 4) Tensor with (x,y,z,c) on each
 *  row
 * @param grid a HashedGrid to collocate on, collocation points P taken from
 *  grid->xyz() 
 * @param v an (nP,) Tensor with the grid potential values
 * @param trans (nT,3) Tensor with translation vectors for the centers r_A. If
 *  you do not want translations, provide <0.0,0.0,0.0>.
 * @param alpha the blur exponent (-1.0 indicates no blurring)
 * @param thre cutoff value below which collocation values are clamped to zero.
 * @param vmax the maximum absolute value of v, as it is formally O(N^2) to
 *  compute on the fly.
 * @param G [optional] an (nA,3) Tensor to accumulate the gradient into. If not
 *  provided, this will be allocated.
 * @return a (nA,3) Tensor with the result accumulated. 
 **/
static
std::shared_ptr<Tensor> pointGrad(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& xyzc,
    const std::shared_ptr<HashedGrid>& grid,
    const std::shared_ptr<Tensor>& v,
    const std::shared_ptr<Tensor>& trans,
    double alpha,
    double thre,
    double vmax,
    const std::shared_ptr<Tensor>& G = std::shared_ptr<Tensor>()
    );

// => Version 1: Constant Blurring <= //

// => LDA Routines <= //

/**
 * Compute the blurred Gaussian density field (LDA) on a grid:
 *
 *  rho_P += BLUR1[D_pq \phi_p^P \phi_q^P]
 *
 * This BLUR1 kernel blurs the Hermite Gaussians by a constant blur exponent
 * alpha.
 *
 * The geometric centers \vec r_A are translated according to a set of
 * user-provided translation vectors \vec r_T, e.g., \vec r_AT = \vec r_A +
 * \vec r_T
 *
 * The computation is sieved so that only contributions with 
 * |rho_ATP| >= thre are retained. 
 *
 * A 2x reduction in work is obtained if pairlist is symmetric.
 *
 * Implementation: CPU only (Hermite).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param pairlist the PairList object defining basis1 and basis2
 * @param D an (nao1,nao2) Tensor containing the OPDM  
 * @param grid a HashedGrid to collocate on, collocation points P taken from
 *  grid->xyz() 
 * @param trans (nT,3) Tensor with translation vectors for the centers r_A. If
 *  you do not want translations, provide <0.0,0.0,0.0>.
 * @param alpha the blur exponent (-1.0 indicates no blurring)
 * @param thre cutoff value below which collocation values are clamped to zero.
 * @param rho [optional] an (nP,) Tensor to accumulate the result into. If not
 *  provided, this will be allocated.
 * @return a (nP,) Tensor with the result accumulated. 
 **/
static
std::shared_ptr<Tensor> ldaDensity(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<HashedGrid>& grid,
    const std::shared_ptr<Tensor>& trans,
    double alpha,
    double thre,
    const std::shared_ptr<Tensor>& rho = std::shared_ptr<Tensor>()
    );

/**
 * Compute the blurred Gaussian potential (LDA) from a grid potential:
 *
 *  V_pq += BLUR1[\phi_p^P \phi_q^P v_P]
 *
 * This BLUR1 kernel blurs the Hermite Gaussians by a constant blur exponent
 * alpha.
 *
 * The geometric centers \vec r_A are translated according to a set of
 * user-provided translation vectors \vec r_T, e.g., \vec r_AT = \vec r_A +
 * \vec r_T
 *
 * The computation is sieved so that only contributions with 
 * |v_ATP| >= thre are retained. 
 *
 * A 2x reduction in work is obtained if pairlist is symmetric.
 *
 * Implementation: CPU only (Hermite).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param pairlist the PairList object defining basis1 and basis2
 * @param grid a HashedGrid to collocate on, collocation points P taken from
 *  grid->xyz() 
 * @param v an (nP,) Tensor with the grid potential values
 * @param trans (nT,3) Tensor with translation vectors for the centers r_A. If
 *  you do not want translations, provide <0.0,0.0,0.0>.
 * @param alpha the blur exponent (-1.0 indicates no blurring)
 * @param thre cutoff value below which collocation values are clamped to zero.
 * @param vmax the maximum absolute value of v, as it is formally O(N^2) to
 *  compute on the fly.
 * @param V [optional] an (nao1,nao2) Tensor to accumulate the result into. If not
 *  provided, this will be allocated.
 * @return a (nao1,nao2) Tensor with the result accumulated. 
 **/
static
std::shared_ptr<Tensor> ldaPotential(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<HashedGrid>& grid,
    const std::shared_ptr<Tensor>& v,
    const std::shared_ptr<Tensor>& trans,
    double alpha,
    double thre,
    double vmax,
    const std::shared_ptr<Tensor>& V = std::shared_ptr<Tensor>()
    );

/**
 * Compute the gradient of a blurred Gaussian charge distribution (LDA)
 * interacting with a grid potential:
 *
 *  G_C += \partial_{C} BLUR1[D_{pq} \phi_p^P \phi_q^P v_P]
 *
 * This BLUR1 kernel blurs the Hermite Gaussians by a constant blur exponent
 * alpha.
 *
 * The geometric centers \vec r_A are translated according to a set of
 * user-provided translation vectors \vec r_T, e.g., \vec r_AT = \vec r_A +
 * \vec r_T
 *
 * The computation is sieved so that only contributions with 
 * |E_ATP| >= thre are retained. 
 *
 * This is the simple version of the gradient routine, and sums the
 * gradient contributions from \phi_p and \phi_q.
 *
 * Throws if pairlist is not symmetric.
 *
 * Implementation: CPU only (Hermite).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param pairlist the PairList object defining basis1 and basis2
 * @param D an (nao1,nao2) Tensor containing the OPDM  
 * @param grid a HashedGrid to collocate on, collocation points P taken from
 *  grid->xyz() 
 * @param v an (nP,) Tensor with the grid potential values
 * @param trans (nT,3) Tensor with translation vectors for the centers r_A. If
 *  you do not want translations, provide <0.0,0.0,0.0>.
 * @param alpha the blur exponent (-1.0 indicates no blurring)
 * @param thre cutoff value below which collocation values are clamped to zero.
 * @param vmax the maximum absolute value of v, as it is formally O(N^2) to
 *  compute on the fly.
 * @param V [optional] an (natom1,3) Tensor to accumulate the result into. If not
 *  provided, this will be allocated.
 * @return a (natom1,3) Tensor with the result accumulated. 
 **/
static
std::shared_ptr<Tensor> ldaGrad(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<HashedGrid>& grid,
    const std::shared_ptr<Tensor>& v,
    const std::shared_ptr<Tensor>& trans,
    double alpha,
    double thre,
    double vmax,
    const std::shared_ptr<Tensor>& G = std::shared_ptr<Tensor>()
    );

// => Version 2: Minimal Blurring <= //

// => LDA Routines <= //

/**
 * Compute the blurred Gaussian density field (LDA) on a grid:
 *
 *  rho_P += BLUR2[D_pq \phi_p^P \phi_q^P]
 *
 * This BLUR2 kernel blurs the Hermite Gaussians out to a mimimum blur of alpha,
 * but does not affect Hermite Gaussians with exponents more-diffuse than
 * alpha. E.g., the blur exponent is Hermite-Gaussian specific between [alpha,
 * \infty).
 *
 * The geometric centers \vec r_A are translated according to a set of
 * user-provided translation vectors \vec r_T, e.g., \vec r_AT = \vec r_A +
 * \vec r_T
 *
 * The computation is sieved so that only contributions with 
 * |rho_ATP| >= thre are retained. 
 *
 * A 2x reduction in work is obtained if pairlist is symmetric.
 *
 * Implementation: CPU only (Hermite).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param pairlist the PairList object defining basis1 and basis2
 * @param D an (nao1,nao2) Tensor containing the OPDM  
 * @param grid a HashedGrid to collocate on, collocation points P taken from
 *  grid->xyz() 
 * @param trans (nT,3) Tensor with translation vectors for the centers r_A. If
 *  you do not want translations, provide <0.0,0.0,0.0>.
 * @param alpha the blur exponent (-1.0 indicates no blurring)
 * @param thre cutoff value below which collocation values are clamped to zero.
 * @param rho [optional] an (nP,) Tensor to accumulate the result into. If not
 *  provided, this will be allocated.
 * @return a (nP,) Tensor with the result accumulated. 
 **/
static
std::shared_ptr<Tensor> ldaDensity2(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<HashedGrid>& grid,
    const std::shared_ptr<Tensor>& trans,
    double alpha,
    double thre,
    const std::shared_ptr<Tensor>& rho = std::shared_ptr<Tensor>()
    );

/**
 * Compute the blurred Gaussian potential (LDA) from a grid potential:
 *
 *  V_pq += BLUR2[\phi_p^P \phi_q^P v_P]
 *
 * This BLUR2 kernel blurs the Hermite Gaussians out to a mimimum blur of alpha,
 * but does not affect Hermite Gaussians with exponents more-diffuse than
 * alpha. E.g., the blur exponent is Hermite-Gaussian specific between [alpha,
 * \infty).
 *
 * The geometric centers \vec r_A are translated according to a set of
 * user-provided translation vectors \vec r_T, e.g., \vec r_AT = \vec r_A +
 * \vec r_T
 *
 * The computation is sieved so that only contributions with 
 * |v_ATP| >= thre are retained. 
 *
 * A 2x reduction in work is obtained if pairlist is symmetric.
 *
 * Implementation: CPU only (Hermite).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param pairlist the PairList object defining basis1 and basis2
 * @param grid a HashedGrid to collocate on, collocation points P taken from
 *  grid->xyz() 
 * @param v an (nP,) Tensor with the grid potential values
 * @param trans (nT,3) Tensor with translation vectors for the centers r_A. If
 *  you do not want translations, provide <0.0,0.0,0.0>.
 * @param alpha the blur exponent (-1.0 indicates no blurring)
 * @param thre cutoff value below which collocation values are clamped to zero.
 * @param vmax the maximum absolute value of v, as it is formally O(N^2) to
 *  compute on the fly.
 * @param V [optional] an (nao1,nao2) Tensor to accumulate the result into. If not
 *  provided, this will be allocated.
 * @return a (nao1,nao2) Tensor with the result accumulated. 
 **/
static
std::shared_ptr<Tensor> ldaPotential2(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<HashedGrid>& grid,
    const std::shared_ptr<Tensor>& v,
    const std::shared_ptr<Tensor>& trans,
    double alpha,
    double thre,
    double vmax,
    const std::shared_ptr<Tensor>& V = std::shared_ptr<Tensor>()
    );

/**
 * Compute the gradient of a blurred Gaussian charge distribution (LDA)
 * interacting with a grid potential:
 *
 *  G_C += \partial_{C} BLUR2[D_{pq} \phi_p^P \phi_q^P v_P]
 *
 * This BLUR2 kernel blurs the Hermite Gaussians out to a mimimum blur of alpha,
 * but does not affect Hermite Gaussians with exponents more-diffuse than
 * alpha. E.g., the blur exponent is Hermite-Gaussian specific between [alpha,
 * \infty).
 *
 * The geometric centers \vec r_A are translated according to a set of
 * user-provided translation vectors \vec r_T, e.g., \vec r_AT = \vec r_A +
 * \vec r_T
 *
 * The computation is sieved so that only contributions with 
 * |E_ATP| >= thre are retained. 
 *
 * This is the simple version of the gradient routine, and sums the
 * gradient contributions from \phi_p and \phi_q.
 *
 * Throws if pairlist is not symmetric.
 *
 * Implementation: CPU only (Hermite).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param pairlist the PairList object defining basis1 and basis2
 * @param D an (nao1,nao2) Tensor containing the OPDM  
 * @param grid a HashedGrid to collocate on, collocation points P taken from
 *  grid->xyz() 
 * @param v an (nP,) Tensor with the grid potential values
 * @param trans (nT,3) Tensor with translation vectors for the centers r_A. If
 *  you do not want translations, provide <0.0,0.0,0.0>.
 * @param alpha the blur exponent (-1.0 indicates no blurring)
 * @param thre cutoff value below which collocation values are clamped to zero.
 * @param vmax the maximum absolute value of v, as it is formally O(N^2) to
 *  compute on the fly.
 * @param V [optional] an (natom1,3) Tensor to accumulate the result into. If not
 *  provided, this will be allocated.
 * @return a (natom1,3) Tensor with the result accumulated. 
 **/
static
std::shared_ptr<Tensor> ldaGrad2(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<HashedGrid>& grid,
    const std::shared_ptr<Tensor>& v,
    const std::shared_ptr<Tensor>& trans,
    double alpha,
    double thre,
    double vmax,
    const std::shared_ptr<Tensor>& G = std::shared_ptr<Tensor>()
    );

// => SR Correction Integrals <= //

/**
 * Compute the V (potential) matrix:
 *
 *  V_pq += \sum_{A}
 *          \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          \phi_p^1 \phi_q^1
 *          SR(r_1A) (-c_A)
 *
 * No screening beyond pair cache screening is used. A 2x reduction in work is
 * achieved if pairlist is symmetric. 
 *
 * TODO: Documentation needs to be updated
 * 
 * Implementation: CPU only (Hermite via Rys-Hermite).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param omega the Ewald parameter for the long-range Ewald interaction
 * @param pairlist the list of significant pairs describing basis1 and basis2
 * @param xyzc the point charge field, an (nA, 4) Tensor with (x,y,z,c) on each
 *  row
 * @param V [optional] an (nao1, nao2) Tensor to accumulate into. If not
 *  provided, this will be allocated.
 * @return an (nao1, nao2) Tensor with the integrals accumulated
 **/
static 
std::shared_ptr<Tensor> potentialSR3(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& xyzc,
    double omega, // Ewald range-separate parameter
    double threoe,
    const std::shared_ptr<Tensor>& V = 
          std::shared_ptr<Tensor>()
    );

/**
 * Compute the w (electrostatic potential) matrix:
 *
 *  w_A  += \sum_{pq} D_{pq}
 *          \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          \phi_p^1 \phi_q^1
 *          O(r_1A) 
 *
 * TODO: Documentation needs to be updated
 * 
 * No screening beyond pair cache screening is used. A 2x reduction in work is
 * achieved if pairlist is symmetric. 
 * 
 * Implementation: CPU only (Hermite via Rys-Hermite).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param ewald the interaction operator O(r_1A)
 * @param pairlist the list of significant pairs describing basis1 and basis2
 * @param D the OPDM, an (nao1, nao2) Tensor
 * @param xyz the grid to evaluate the ESP on, an (nA, 3) Tensor with (x,y,z)
 *  on each row
 * @param w [optional] an (nA) Tensor to accumulate into. If not
 *  provided, this will be allocated.
 * @return an (nA) Tensor with the integrals accumulated
 **/
static
std::shared_ptr<Tensor> espSR3(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<Tensor>& xyz,
    double omega, // Ewald range-separate parameter
    double threoe,
    const std::shared_ptr<Tensor>& w = std::shared_ptr<Tensor>()
    );

/**
 * Compute the J (Coulomb) matrix:
 *
 *  J_pq += \sum_{rs}
 *          \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          \phi_p^1 \phi_q^1
 *          O(r_12) 
 *          \phi_r^2 \phi_s^2
 *          D_{rs}
 *
 * TODO: Documentation needs to be updated
 *
 * Density-based sieving is used. Integrals are sieved (by whole shell
 *  quartet) according to the estimate Jest = B_{pq} B_{rs} D_{rs} < thre.
 *  Here D_{rs} is the maximum density matrix element in the rs shell pair
 *  (both upper and lower triangle are checked if pairlist34 is symmetric).
 * Mixed precision is possible (though not presently implemented on CPU).
 *  In mixed precision, integrals are computed in double precision if Jest >=
 *  thredp, in single precision if Jest >= thresp, and neglected otherwise. For
 *  algorithms not using mixed precision, integrals are computed in double
 *  precision if Jest >= thresp, and neglected otherwise - thredp is not used.
 *  For good accuracy, one should set thresp << thredp. Note that all
 *  accumulation is performed in double precision.
 * A 2x reduction in work is obtained for each pairlist that is symmetric (up
 *  to 4x total).
 * 
 * Implementation: CPU only (Hermite via Rys-Hermite).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param ewald the interaction operator O(r_12)
 * @param pairlist12 the list of significant pairs describing basis1 and basis2
 * @param pairlist34 the list of significant pairs describing basis3 and basis4
 * @param D34 the OPDM, an (nao3, nao4) Tensor
 * @param thresp the value of Jest for which integral contributions are
 *  guaranteed to be computed in at least single precision.
 * @param thredp the value of Jest for which integral contributions are
 *  guaranteed to be computed in at least double precision.
 * @param J [optional] an (nao1, nao2) Tensor to accumulate into. If not
 *  provided, this will be allocated.
 * @return an (nao1, nao2) Tensor with the integrals accumulated
 **/
static 
std::shared_ptr<Tensor> coulombSR3(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist12,
    const std::shared_ptr<PairList>& pairlist34,
    const std::shared_ptr<Tensor>& D34,
    double omega, // Ewald range-separate parameter
    double threte,
    const std::shared_ptr<Tensor>& J12 = std::shared_ptr<Tensor>()
    );

static
std::shared_ptr<Tensor> potentialGradSR3(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<Tensor>& xyzc,
    double omega,
    double threoe,
    const std::shared_ptr<Tensor>& G12 = std::shared_ptr<Tensor>()
    );

static
std::shared_ptr<Tensor> espGradSR3(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<Tensor>& xyzc,
    double omega,
    double threoe,
    const std::shared_ptr<Tensor>& GA = std::shared_ptr<Tensor>()
    );

static
std::shared_ptr<Tensor> coulombGradSR3(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist12,
    const std::shared_ptr<PairList>& pairlist34,
    const std::shared_ptr<Tensor>& D12,
    const std::shared_ptr<Tensor>& D34,
    double omega,
    double threte,
    const std::shared_ptr<Tensor>& G12 = std::shared_ptr<Tensor>()
    );

};

} // namespace lightspeed 

#endif
