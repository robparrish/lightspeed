#ifndef LS_INTBOX_HPP
#define LS_INTBOX_HPP

#include <memory>
#include <vector>

namespace lightspeed {

class Tensor;
class ResourceList;
class PairList;
class ECPBasis;
class Ewald;
    
/**
 * Class IntBox computes many types of Gaussian integrals, using CPU or GPU
 * resources.
 *
 * All input/output matrices should have basis dimensions referring to nao
 * (possibly including spherical harmonics), rather than ncart. Spherical
 * harmonic transformations/backtransformations are applied under the hood.
 * All routines will throw std::exception if input/output matrices have the
 * wrong dimensions.
 **/
class IntBox {

public:

// ==> Best-Case Routines (Call these unless in debug mode) <== //

// => Point-Charge Interactions <= //

/**
 * Compute the electrostatic energy of a set of point charges interacting with
 * itself (e.g., the nuclear-nuclear repulsion energy)
 *      
 *  E = 1/2 \sum_{A,B}' Z_A Z_B V(r_{AB})
 *
 * Here the ' in the sum means the term is excluded if r_{AB} == 0
 *
 * Implementation: CPU only, calls chargeEnergyOther.
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param ewald the Ewald operator defining the interaction operator V(r_{AB})
 * @param xyzc the (x,y,z,c) fields of the set of charges, an (nA,4) Tensor
 * @return the electrostatic energy
 **/
static 
double chargeEnergySelf(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<Tensor>& xyzc
    );

/**
 * Compute the electrostatic energy gradient of a set of point charges
 * interacting with itself (e.g., the nuclear-nuclear repulsion energy
 * gradient)
 *      
 *  G_{A} += \sum_{B}' \partial_{A} Z_{A} Z_{B} V(r_{AB})
 *
 * Here the ' in the sum means the term is excluded if r_{AB} == 0.
 *
 * Implementation: CPU only, calls chargeGradOther.
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param ewald the Ewald operator defining the interaction operator V(r_{AB})
 * @param xyzc the (x,y,z,c) fields of the set of charges, an (nA,4) Tensor
 * @param G12 [optional] a vector with an (nA,3) and an (nB,3) Tensor to
 *  accumulate the gradient contributions into. If not provided, this will be
 *  allocated.
 * @return the electrostatic energy gradient accumulated into G12
 **/
static
std::shared_ptr<Tensor> chargeGradSelf(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<Tensor>& xyzc,
    const std::shared_ptr<Tensor>& G =
          std::shared_ptr<Tensor>()
    );

/**
 * Compute the electrostatic energy between two sets of point charges, defined
 * as:
 *      
 *  E = \sum_{A,B}' Z_A Z_B V(r_{AB})
 *
 * Here the ' in the sum means the term is excluded if r_{AB} == 0
 *
 * The call:
 *      
 *  double E = IntBox::chargeEnergyOther(
 *      resources,
 *      Ewald::coulomb(),
 *      mol.xyzZ(),
 *      mol.xyzZ());
 *  
 * Returns the usual nuclear repulsion energy of the molecule. However, this
 * routine is much more general, and can compute the nuclear-nuclear
 * interaction for two different molecules and/or use an Ewald operator to
 * modify the nature of the interaction.
 *
 * Implementation: CPU only.
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param ewald the Ewald operator defining the interaction operator V(r_{AB})
 * @param xyzc1 the (x,y,z,c) fields of the first set of charges, an (nA,4) Tensor
 * @param xyzc2 the (x,y,z,c) fields of the second set of charges, an (nB,4) Tensor
 * @return the electrostatic energy
 **/
static 
double chargeEnergyOther(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<Tensor>& xyzc1,
    const std::shared_ptr<Tensor>& xyzc2
    );

/**
 * Compute the electric field created by a set of points charges on a set of points
 * , defined as:
 *      
 *  G1_{B} += \sum_{A}' \partial_{A} Z_{A} V(r_{AB})
 *
 * Here the ' in the sum means the term is excluded if r_{AB} == 0.
 *
 *
 * Implementation: CPU only.
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param ewald the Ewald operator defining the interaction operator V(r_{AB})
 * @param xyzc1 the (x,y,z,c) fields of the first set of charges, an (nA,4) Tensor
 * @param xyz2 the (x,y,z) fields of the set of points on which to probe the 
 * electric field, an (nB,3) Tensor
 * @param F [optional] a vector with an (nA,3) Tensor to
 *  accumulate the field contributions into. If not provided, this will be
 *  allocated.
 * @return the electric field accumulated into F
 **/
static
std::shared_ptr<Tensor> chargeFieldOther(           
    const std::shared_ptr<ResourceList>& resources,         
    const std::shared_ptr<Ewald>& ewald,                    
    const std::shared_ptr<Tensor>& xyzc1,                   
    const std::shared_ptr<Tensor>& xyz2,
    const std::shared_ptr<Tensor> & F =  
          std::shared_ptr<Tensor>()                    
    );                                                             

/**
 * Compute the gradient of the electrostatic energy between two sets of point
 * charges, defined as:
 *      
 *  G1_{A} += \sum_{B}' \partial_{A} Z_{A} Z_{B} V(r_{AB})
 *  G2_{B} += \sum_{A}' \partial_{B} Z_{A} Z_{B} V(r_{AB})
 *
 * Here the ' in the sum means the term is excluded if r_{AB} == 0.
 *
 * The call:
 *
 *  auto G12 = IntBox::chargeGradOther(
 *      resources,
 *      Ewald::coulomb(),
 *      mol.xyzZ(),
 *      mol.xyzZ());
 * 
 * Returns the usual nuclear repulsion gradient of the molecule (in the sum of
 * G1 and G2). However, this routine is much more general, and can compute the
 * gradient of the nuclear-nuclear interaction for two different molecules
 * and/or use an Ewald operator to modify the nature of the interaction.
 *
 * Implementation: CPU only.
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param ewald the Ewald operator defining the interaction operator V(r_{AB})
 * @param xyzc1 the (x,y,z,c) fields of the first set of charges, an (nA,4) Tensor
 * @param xyzc2 the (x,y,z,c) fields of the second set of charges, an (nB,4) Tensor
 * @param G12 [optional] a vector with an (nA,3) and an (nB,3) Tensor to
 *  accumulate the gradient contributions into. If not provided, this will be
 *  allocated.
 * @return the electrostatic energy gradient accumulated into G12
 **/
static
std::vector<std::shared_ptr<Tensor> > chargeGradOther(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<Tensor>& xyzc1,
    const std::shared_ptr<Tensor>& xyzc2,
    const std::vector<std::shared_ptr<Tensor> >& G12 =
          std::vector<std::shared_ptr<Tensor> >()
    );

/**
 * Compute the electrostatic potential of a set of point charges, defined
 * as:
 *      
 *  v_A = \sum_{A,B}' Z_B V(r_{AB})
 *
 * Here the ' in the sum means the term is excluded if r_{AB} == 0
 *
 * Implementation: CPU only.
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param ewald the Ewald operator defining the interaction operator V(r_{AB})
 * @param xyz1 the (x,y,z) fields of the first set of charges, an (nA,3) Tensor
 * @param xyzc2 the (x,y,z,c) fields of the second set of charges, an (nB,4) Tensor
 * @param v [optional] an (nA,) Tensor to accumulate into. If not provided,
 *  this will be allocated.
 * @return the electrostatic potential, an (nA,) Tensor
 **/
static 
std::shared_ptr<Tensor> chargePotential(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<Tensor>& xyz1,
    const std::shared_ptr<Tensor>& xyzc2,
    const std::shared_ptr<Tensor>& v =
          std::shared_ptr<Tensor>()
    );

// => Overlap-Type Integrals <= //

/**
 * Compute the S (overlap) matrix:
 *
 *  S_pq += \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          \phi_p^1 \phi_q^1
 *
 * No screening beyond pair cache screening is used. A 2x reduction in work is
 * achieved if pairlist is symmetric. 
 * 
 * Implementation: CPU only (Cartesian via Gauss-Hermite quadrature).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param pairlist the list of significant pairs describing basis1 and basis2
 * @param S [optional] an (nao1, nao2) Tensor to accumulate into. If not
 *  provided, this will be allocated.
 * @return an (nao1, nao2) Tensor with the integrals accumulated
 **/
static
std::shared_ptr<Tensor> overlap(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& S = 
          std::shared_ptr<Tensor>()
    );

/**
 * Compute the X (dipole) matrices:
 *
 *  X_pq += \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          \phi_p^1 [x_1 - x0] \phi_q^1
 *
 * The dipole matrices are ordered X, Y, Z.
 *
 * No screening beyond pair cache screening is used. A 2x reduction in work is
 * achieved if pairlist is symmetric. 
 * 
 * Implementation: CPU only (Cartesian via Gauss-Hermite quadrature).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param pairlist the list of significant pairs describing basis1 and basis2
 * @param x0 the x coordinate of the property origin 
 * @param y0 the y coordinate of the property origin 
 * @param z0 the z coordinate of the property origin 
 * @param X [optional] a vector of 3x (nao1, nao2) Tensors to accumulate into. If not
 *  provided, this will be allocated.
 * @return a vector of 3x (nao1, nao2) Tensors with the integrals accumulated,
 *  ordered X, Y, Z.
 **/
static
std::vector< std::shared_ptr<Tensor> > dipole(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    double x0,
    double y0,
    double z0,
    const std::vector< std::shared_ptr<Tensor> >& X = 
          std::vector< std::shared_ptr<Tensor> >()
    );

/**
 * Compute the Q (quadrupole) matrices:
 *
 *  Q_pq += \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          \phi_p^1 [x_1 - x0][y_1 - y0] \phi_q^1
 *
 * The quadrupole matrices are ordered XX, XY, XZ, YY, YZ, ZZ.
 *
 * No screening beyond pair cache screening is used. A 2x reduction in work is
 * achieved if pairlist is symmetric. 
 * 
 * Implementation: CPU only (Cartesian via Gauss-Hermite quadrature).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param pairlist the list of significant pairs describing basis1 and basis2
 * @param x0 the x coordinate of the property origin 
 * @param y0 the y coordinate of the property origin 
 * @param z0 the z coordinate of the property origin 
 * @param Q [optional] a vector of 6x (nao1, nao2) Tensors to accumulate into. If not
 *  provided, this will be allocated.
 * @return a vector of 6x (nao1, nao2) Tensors with the integrals accumulated,
 *  ordered XX, XY, XZ, YY, YZ, ZZ.
 **/
static
std::vector< std::shared_ptr<Tensor> > quadrupole(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    double x0,
    double y0,
    double z0,
    const std::vector< std::shared_ptr<Tensor> >& Q = 
          std::vector< std::shared_ptr<Tensor> >()
    );

/**
 * Compute the P (nabla) matrices:
 *
 *  P_pq += \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          \phi_p^1 [\nabla_1] \phi_q^1
 *
 * The nabla matrices are ordered PX, PY, PZ.
 *
 * No screening beyond pair cache screening is used. A 2x reduction in work is
 * achieved if pairlist is symmetric. 
 * 
 * Implementation: CPU only (Cartesian via Gauss-Hermite quadrature).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param pairlist the list of significant pairs describing basis1 and basis2
 * @param P [optional] a vector of 3x (nao1, nao2) Tensors to accumulate into. If not
 *  provided, this will be allocated.
 * @return a vector of 3x (nao1, nao2) Tensors with the integrals accumulated,
 *  ordered PX, PY, PZ.
 **/
static
std::vector< std::shared_ptr<Tensor> > nabla(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::vector< std::shared_ptr<Tensor> >& P = 
          std::vector< std::shared_ptr<Tensor> >()
    );
    
/**
 * Compute the L (angular momentum) matrices:
 *
 *  L_pq += \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          \phi_p^1 [\vec r_1 \times \nabla_1] \phi_q^1
 *
 * The nabla matrices are ordered LX, LY, LZ.
 *
 * No screening beyond pair cache screening is used. A 2x reduction in work is
 * achieved if pairlist is symmetric. 
 * 
 * Implementation: CPU only (Cartesian via Gauss-Hermite quadrature).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param pairlist the list of significant pairs describing basis1 and basis2
 * @param x0 the x coordinate of the property origin 
 * @param y0 the y coordinate of the property origin 
 * @param z0 the z coordinate of the property origin 
 * @param L [optional] a vector of 3x (nao1, nao2) Tensors to accumulate into. If not
 *  provided, this will be allocated.
 * @return a vector of 3x (nao1, nao2) Tensors with the integrals accumulated,
 *  ordered LX, LY, LZ.
 **/
static
std::vector< std::shared_ptr<Tensor> > angularMomentum(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    double x0,
    double y0,
    double z0,
    const std::vector< std::shared_ptr<Tensor> >& L = 
          std::vector< std::shared_ptr<Tensor> >()
    );

/**
 * Compute the T (kinetic) matrix:
 *
 *  T_pq += \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          \phi_p^1 [-\nabla_1^2 / 2] \phi_q^1
 *
 * No screening beyond pair cache screening is used. A 2x reduction in work is
 * achieved if pairlist is symmetric. 
 * 
 * Implementation: CPU only (Cartesian via Gauss-Hermite quadrature).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param pairlist the list of significant pairs describing basis1 and basis2
 * @param T [optional] an (nao1, nao2) Tensor to accumulate into. If not
 *  provided, this will be allocated.
 * @return an (nao1, nao2) Tensor with the integrals accumulated
 **/
static
std::shared_ptr<Tensor> kinetic(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& T = 
          std::shared_ptr<Tensor>()
    );

/**
 * Compute the S (overlap) gradient - simple version:
 *
 *  G_{A} += \sum_{pq} W_{pq}
 *          \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          \partial_{A} (\phi_p^1 \phi_q^1)
 *
 * Throws if pairlist is not symmetric.
 *
 * No screening beyond pair cache screening is used.
 * 
 * Implementation: CPU only (Cartesian via Gauss-Hermite quadrature).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param pairlist the list of significant pairs describing basis1 and basis2
 * @param W the energy-weighted OPDM, an (nao1, nao2) Tensor
 * @param G [optional] an (natom1, 3) Tensor to accumulate into. If not
 *  provided, this will be allocated.
 * @return an (natom1, 3) Tensor with the gradient accumulated
 **/
static
std::shared_ptr<Tensor> overlapGrad(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& W,
    const std::shared_ptr<Tensor>& G =
          std::shared_ptr<Tensor>()
    );

/**
 * Compute the T (kinetic) gradient - simple version:
 *
 *  G_{A} += \sum_{pq} W_{pq}
 *          \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          \partial_{A} (\phi_p^1 [\nabla_1^2 / 2] \phi_q^1)
 *
 * Throws if pairlist is not symmetric.
 *
 * No screening beyond pair cache screening is used.
 * 
 * Implementation: CPU only (Cartesian via Gauss-Hermite quadrature).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param pairlist the list of significant pairs describing basis1 and basis2
 * @param D the OPDM, an (nao1, nao2) Tensor
 * @param G [optional] an (natom1, 3) Tensor to accumulate into. If not
 *  provided, this will be allocated.
 * @return an (natom1, 3) Tensor with the gradient accumulated
 **/
static
std::shared_ptr<Tensor> kineticGrad(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<Tensor>& G =
          std::shared_ptr<Tensor>()
    );

/**
 * Compute the dipole gradient - simple version:
 *
 *  G_{A} += \sum_{pq} D_{pq}
 *          \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          \partial_{A} (\phi_p^1 [x_1 - x0] \phi_q^1)
 *
 * Throws if pairlist is not symmetric.
 *
 * Dipole matrices are ordered X, Y and Z.
 *
 * No screening beyond pair cache screening is used.
 * 
 * Implementation: CPU only (Cartesian via Gauss-Hermite quadrature).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param pairlist the list of significant pairs describing basis1 and basis2
 * @param x0 the x coordinate of the property origin 
 * @param y0 the y coordinate of the property origin 
 * @param z0 the z coordinate of the property origin 
 * @param D the OPDM, an (nao1, nao2) Tensor
 * @param Dgrad [optional] a vector of 3x (natom1, 3) Tensors to accumulate into. If not
 *  provided, this will be allocated.
 * @return a vector of 3x (natom1, 3). Vector of tensors with the gradient accumulated, 
 *  ordered X, Y and Z
 **/

static
std::vector< std::shared_ptr<Tensor>> dipoleGrad(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    double x0,
    double y0,
    double z0,
    const std::shared_ptr<Tensor>& D,
    const std::vector< std::shared_ptr<Tensor>>& Dgrad =
          std::vector< std::shared_ptr<Tensor>>()
    );

/**
 * Compute the S (overlap) gradient - advanced version:
 *
 *  G1_{A} += \sum_{pq} W_{pq}
 *          \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          \partial_{A} \phi_p^1 \phi_q^1
 *  G2_{B} += \sum_{pq} W_{pq}
 *          \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          \phi_q^1 \partial_{B} \phi_q^1
 *
 * No screening beyond pair cache screening is used. A 2x reduction in work is
 * achieved if pairlist is symmetric. 
 * 
 * Implementation: CPU only (Cartesian via Gauss-Hermite quadrature).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param pairlist the list of significant pairs describing basis1 and basis2
 * @param W the energy-weighted OPDM, an (nao1, nao2) Tensor
 * @param G12 [optional] a vector with an (natom1, 3) and an (natom2, 3) Tensor
 *  to accumulate into. If not provided, this will be allocated.
 * @return a vector with an (natom1, 3) and an (natom2, 3) Tensor with the
 *  integrals accumulated
 **/
static
std::vector<std::shared_ptr<Tensor> > overlapGradAdv(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& W,
    const std::vector<std::shared_ptr<Tensor> >& G12 =
          std::vector<std::shared_ptr<Tensor> >()
    );

/**
 * Compute the T (kinetic) gradient - advanced version:
 *
 *  G1_{A} += \sum_{pq} W_{pq}
 *          \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          \partial_{A} \phi_p^1 [\nabla_1^2 / 2] \phi_q^1
 *  G2_{B} += \sum_{pq} W_{pq}
 *          \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          \phi_p^1 [\nabla_1^2 / 2] \partial_{B} \phi_q^1
 *
 * No screening beyond pair cache screening is used. A 2x reduction in work is
 * achieved if pairlist is symmetric. 
 * 
 * Implementation: CPU only (Cartesian via Gauss-Hermite quadrature).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param pairlist the list of significant pairs describing basis1 and basis2
 * @param D the OPDM, an (nao1, nao2) Tensor
 * @param G12 [optional] a vector with an (natom1, 3) and an (natom2, 3) Tensor
 *  to accumulate into. If not provided, this will be allocated.
 * @return a vector with an (natom1, 3) and an (natom2, 3) Tensor with the
 *  integrals accumulated
 **/
static
std::vector<std::shared_ptr<Tensor> > kineticGradAdv(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::vector<std::shared_ptr<Tensor> >& G12 =
          std::vector<std::shared_ptr<Tensor> >()
    );

/**
 * Compute the dipole gradient - advanced version:
 *
 *  For each Cartesian direction d:
 *
 *      Dgrad12[d]_{A} += \sum_{pq} D_{pq}
 *              \int_{\mathbb{R}^3} 
 *              \mathrm{d}^3 r_1
 *              \partial_{A} \phi_p^1 [d1 - d0]\phi_q^1
 *      Dgrad12[d]_{B} += \sum_{pq} D_{pq}
 *              \int_{\mathbb{R}^3} 
 *              \mathrm{d}^3 r_1
 *              \phi_q^1 \partial_{B} [d1 - d0] \phi_q^1
 *
 * Dipole matrices are ordered X, Y and Z. 
 *
 * No screening beyond pair cache screening is used. A 2x reduction in work is
 * achieved if pairlist is symmetric. 
 * 
 * Implementation: CPU only (Cartesian via Gauss-Hermite quadrature).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param pairlist the list of significant pairs describing basis1 and basis2
 * @param W the energy-weighted OPDM, an (nao1, nao2) Tensor
 * @param Dgrad12 [optional] a 3-vector with 2-vectors with an (natom1, 3) and an (natom2, 3) Tensor
 *  to accumulate into. If not provided, this will be allocated.
 * @return a 3-vector with 2-vectors with an (natom1, 3) and an (natom2, 3) Tensor with the
 *  integrals accumulated, ordered X, Y and Z.
 **/
static
std::vector<std::vector< std::shared_ptr<Tensor> >> dipoleGradAdv(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    double x0,
    double y0,
    double z0,
    const std::shared_ptr<Tensor>& D,
    const std::vector< std::vector<std::shared_ptr<Tensor> >>& Dgrad12 =
          std::vector< std::vector<std::shared_ptr<Tensor> >>()
    );

// => Potential-Type Integrals <= //

/**
 * Compute the V (potential) matrix:
 *
 *  V_pq += \sum_{A}
 *          \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          \phi_p^1 \phi_q^1
 *          O(r_1A) (-c_A)
 *
 * No screening beyond pair cache screening is used. A 2x reduction in work is
 * achieved if pairlist is symmetric. 
 * 
 * Implementation: CPU only (Hermite via Rys-Hermite).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param ewald the interaction operator O(r_1A)
 * @param pairlist the list of significant pairs describing basis1 and basis2
 * @param xyzc the point charge field, an (nA, 4) Tensor with (x,y,z,c) on each
 *  row
 * @param V [optional] an (nao1, nao2) Tensor to accumulate into. If not
 *  provided, this will be allocated.
 * @return an (nao1, nao2) Tensor with the integrals accumulated
 **/
static 
std::shared_ptr<Tensor> potential(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& xyzc,
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
std::shared_ptr<Tensor> esp(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<Tensor>& xyz,
    const std::shared_ptr<Tensor>& w = std::shared_ptr<Tensor>()
    );

/**
 * Compute the electric field on a set of points:
 *
 *  F_A  += \sum_{pq} D_{pq}
 *          \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          \phi_p^1 \phi_q^1
 *          [(r1-rA)/r_1A^3]
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
 * @param xyz the points on which to evaluate the electric field, an (nA, 3) 
 * Tensor with (x,y,z) on each row
 * @param F [optional] an (nA, 3) Tensor to accumulate into. If not
 *  provided, this will be allocated.
 * @return an (nA,3) Tensor with the integrals accumulated
 **/
static
std::shared_ptr<Tensor> field(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<Tensor>& xyz,
    const std::shared_ptr<Tensor>& F = std::shared_ptr<Tensor>()
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
std::shared_ptr<Tensor> coulomb(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist12,
    const std::shared_ptr<PairList>& pairlist34,
    const std::shared_ptr<Tensor>& D34,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& J12 = std::shared_ptr<Tensor>()
    );

/**
 * Compute the K (Exchange) matrix:
 *
 *  K_pq += \sum_{rs}
 *          \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          \phi_p^1 \phi_r^1
 *          O(r_12) 
 *          \phi_q^2 \phi_s^2
 *          D_{rs}
 *
 * Density-based sieving is used. Integrals are sieved (by whole shell
 *  quartet) according to the estimate Kest = B_{pr} B_{qs} D_{rs} < thre.
 *  Here D_{rs} is the maximum density matrix element in the rs shell pair
 * Mixed precision is possible (though not presently implemented on CPU).
 *  In mixed precision, integrals are computed in double precision if Kest >=
 *  thredp, in single precision if Kest >= thresp, and neglected otherwise. For
 *  algorithms not using mixed precision, integrals are computed in double
 *  precision if Kest >= thresp, and neglected otherwise - thredp is not used.
 *  For good accuracy, one should set thresp << thredp. Note that all
 *  accumulation is performed in double precision.
 * This is presently implemented using the "K-invert" technique, where lists of
 *  significant p are constructed for each r (likewise q for s), allowing the
 *  significant contributions for a given D_{rs} entry to be efficiently
 *  determined.
 * 
 * Implementation: CPU only (Cartesian via Rys-Gauss-Hermite quadrature).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param ewald the interaction operator O(r_12)
 * @param pairlist12 the list of significant pairs describing basis1 and basis2
 * @param pairlist34 the list of significant pairs describing basis3 and basis4
 * @param D24 the OPDM, an (nao2, nao4) Tensor
 * @param D24symm is the D24 matrix symmetric (2x speedup possible if so)?
 *  Setting D24symm to true only makes sense if basis2 and basis4 are the same -
 *  the code will throw if this is not satisfied.
 * @param thresp the value of Kest for which integral contributions are
 *  guaranteed to be computed in at least single precision.
 * @param thredp the value of Kest for which integral contributions are
 *  guaranteed to be computed in at least double precision.
 * @param K [optional] an (nao1, nao3) Tensor to accumulate into. If not
 *  provided, this will be allocated.
 * @return an (nao1, nao3) Tensor with the integrals accumulated
 **/
static 
std::shared_ptr<Tensor> exchange(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist12,
    const std::shared_ptr<PairList>& pairlist34,
    const std::shared_ptr<Tensor>& D24,
    bool D24symm,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& K13 = std::shared_ptr<Tensor>()
    );

/**
 * Compute the V (potential) gradient - simple version:
 *
 *  G_{C} += \sum_{Apq} D_{pq}
 *          \partial_{C}
 *          \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          \phi_p^1 \phi_q^1
 *          O(r_1A) (-c_A)
 *
 * Throws if pairlist is not symmetric or if the number of point charges does
 * not equal the number of atoms in the basis sets.
 * 
 * No screening beyond pair cache screening is used. 
 *
 * Implementation: CPU only (Hermite via Rys-Hermite).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param ewald the interaction operator O(r_1A)
 * @param pairlist the list of significant pairs describing basis1 and basis2
 * @param D the OPDM, an (nao1, nao2) Tensor
 * @param xyzc the point charge field, an (nA, 4) Tensor with (x,y,z,c) on each
 *  row
 * @param G [optional] an (natom1, 3) Tensor to accumulate into. If not
 *  provided, this will be allocated.
 * @return an (natom1, 3) Tensor with the gradient accumulated
 **/
static
std::shared_ptr<Tensor> potentialGrad(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<Tensor>& xyzc,
    const std::shared_ptr<Tensor>& G = 
          std::shared_ptr<Tensor>()
    );

/**
 * Compute the V (potential) gradient - advanced version:
 *
 *  G1_{C} += \sum_{Apq} D_{pq}
 *          \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          \partial_{C} \phi_p^1 \phi_q^1
 *          O(r_1A) (-c_A)
 *  G2_{C} += \sum_{Apq} D_{pq}
 *          \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          \phi_p^1 \partial_{C} \phi_q^1
 *          O(r_1A) (-c_A)
 *  GA_{A} += \sum_{Apq} D_{pq}
 *          \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          \phi_p^1 \phi_q^1
 *          \partial_{A} O(r_1A) (-c_A)
 *
 * No screening beyond pair cache screening is used. 
 *
 * This routine requires pairlist to not be symmetric, throws otherwise.
 *
 * Implementation: CPU only (Hermite via Rys-Hermite).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param ewald the interaction operator O(r_1A)
 * @param pairlist the list of significant pairs describing basis1 and basis2
 * @param D the OPDM, an (nao1, nao2) Tensor
 * @param xyzc the point charge field, an (nA, 4) Tensor with (x,y,z,c) on each
 *  row
 * @param G12A [optional] a vector of {(natom1, 3), (natom2, 3), (nA, 3)}
 *  Tensors to accumulate into. If not provided, this will be allocated.
 * @return a vector of {(natom1, 3), (natom2, 3), (nA, 3)} Tensors with the
 *  gradients accumulated
 **/
static
std::vector<std::shared_ptr<Tensor> > potentialGradAdv(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<Tensor>& xyzc,
    const std::vector<std::shared_ptr<Tensor> >& G12A = 
          std::vector<std::shared_ptr<Tensor> >()
    );

/**
 * Compute the V (potential) gradient - advanced version 2:
 *
 *  G12_{C} += \sum_{Apq} D_{pq}
 *          \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          (\partial_{C} \phi_p^1 \phi_q^1)
 *          O(r_1A) (-c_A)
 *  GA_{A} += \sum_{Apq} D_{pq}
 *          \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          \phi_p^1 \phi_q^1
 *          \partial_{A} O(r_1A) (-c_A)
 *
 * No screening beyond pair cache screening is used. 
 *
 * This routine requires pairlist to be symmetric, throws otherwise.
 *
 * Implementation: CPU only (Hermite via Rys-Hermite).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param ewald the interaction operator O(r_1A)
 * @param pairlist the list of significant pairs describing basis1 and basis2
 * @param D the OPDM, an (nao1, nao2) Tensor
 * @param xyzc the point charge field, an (nA, 4) Tensor with (x,y,z,c) on each
 *  row
 * @param G12A [optional] a vector of {(natom1, 3), (nA, 3)}
 *  Tensors to accumulate into. If not provided, this will be allocated.
 * @return a vector of {(natom1, 3), (nA, 3)} Tensors with the
 *  gradients accumulated
 **/
static
std::vector<std::shared_ptr<Tensor> > potentialGradAdv2(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<Tensor>& xyzc,
    const std::vector<std::shared_ptr<Tensor> >& G12A = 
          std::vector<std::shared_ptr<Tensor> >()
    );

/**
 * Compute the J (Coulomb) gradient - simple version:
 *
 *  G_{C} += \sum_{pqrs} D_{pq}
 *          \partial_{C}
 *          \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          \phi_p^1 \phi_q^1
 *          O(r_12) 
 *          \phi_r^2 \phi_s^2
 *          D_{rs}
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
 * A 2x speedup is obtained if the two density matrices are identical.
 * 
 * Implementation: CPU only (Hermite via Rys-Hermite).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param ewald the interaction operator O(r_12)
 * @param pairlist the list of significant pairs (must be symmetric)
 * @param D12 the OPDM, an (nao, nao) Tensor
 * @param D34 the OPDM, an (nao, nao) Tensor
 * @param thresp the value of Jest for which integral contributions are
 *  guaranteed to be computed in at least single precision.
 * @param thredp the value of Jest for which integral contributions are
 *  guaranteed to be computed in at least double precision.
 * @param G [optional] an (natom, 3) Tensor to accumulate into. If not
 *  provided, this will be allocated.
 * @return an (natom, 3) Tensor with the gradients accumulated
 **/
static
std::shared_ptr<Tensor> coulombGrad(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist, // symm
    const std::shared_ptr<Tensor>& D12,
    const std::shared_ptr<Tensor>& D34,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& G =
          std::shared_ptr<Tensor>()
    );

/**
 * Compute the J (Coulomb) gradient - advanced version:
 *
 *  G1_{C} += \sum_{pqrs} D_{pq}
 *          \int_{\mathbb{R}^3} \mathrm{d}^3 r_1
 *          (\partial_{C} \phi_p^1) \phi_q^1 O(r_12) \phi_r^2 \phi_s^2
 *          D_{rs}
 *  G2_{C} += \sum_{pqrs} D_{pq}
 *          \int_{\mathbb{R}^3} \mathrm{d}^3 r_1
 *          \phi_p^1 (\partial_{C} \phi_q^1) O(r_12) \phi_r^2 \phi_s^2
 *          D_{rs}
 *  G3_{C} += \sum_{pqrs} D_{pq}
 *          \int_{\mathbb{R}^3} \mathrm{d}^3 r_1
 *          \phi_p^1 \phi_q^1 O(r_12) (\partial_{C} \phi_r^2) \phi_s^2
 *          D_{rs}
 *  G4_{C} += \sum_{pqrs} D_{pq}
 *          \int_{\mathbb{R}^3} \mathrm{d}^3 r_1
 *          \phi_p^1 \phi_q^1 O(r_12) \phi_r^2 (\partial_{C} \phi_s^2)
 *          D_{rs}
 *
 * Density-based sieving is used. Mixed precision is possible (though not
 * presently implemented on CPU).  
 *
 * Implementation: CPU only (Hermite via Rys-Hermite).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param ewald the interaction operator O(r_12)
 * @param pairlist12 the list of significant pairs describing basis1 and
 *  basis2, must be symmetric
 * @param pairlist34 the list of significant pairs describing basis3 and
 *  basis4, must be symmetric
 * @param D12 the OPDM, an (nao1, nao2) Tensor
 * @param D34 the OPDM, an (nao3, nao4) Tensor
 * @param thresp the value of Jest for which integral contributions are
 *  guaranteed to be computed in at least single precision.
 * @param thredp the value of Jest for which integral contributions are
 *  guaranteed to be computed in at least double precision.
 * @param G1234 [optional] a vector of {(natom1, 3), (natom2, 3), (natom3, 3),
 *  (natom4, 3)} Tensors to accumulate into. If not provided, this will be
 *  allocated.
 * @return a vector of {(natom1, 3), (natom2, 3), (natom3, 3), (natom4, 3)}
 *  Tensors with the gradients accumulated
 **/
static
std::vector<std::shared_ptr<Tensor> > coulombGradAdv(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist12, // symm
    const std::shared_ptr<PairList>& pairlist34, // symm
    const std::shared_ptr<Tensor>& D12,
    const std::shared_ptr<Tensor>& D34,
    float thresp,
    float thredp,
    const std::vector<std::shared_ptr<Tensor> >& G1234 =
          std::vector<std::shared_ptr<Tensor> >()
    );

static
std::vector<std::shared_ptr<Tensor> > coulombGradDF1(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist12, // not symm
    const std::shared_ptr<PairList>& pairlist34, // not symm
    const std::shared_ptr<Tensor>& D12,
    const std::shared_ptr<Tensor>& D34,
    float thresp,
    float thredp,
    const std::vector<std::shared_ptr<Tensor> >& G1234 =
          std::vector<std::shared_ptr<Tensor> >()
    );

static
std::vector<std::shared_ptr<Tensor> > coulombGradDF2(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist12, // not symm
    const std::shared_ptr<PairList>& pairlist34, // symm
    const std::shared_ptr<Tensor>& D12,
    const std::shared_ptr<Tensor>& D34,
    float thresp,
    float thredp,
    const std::vector<std::shared_ptr<Tensor> >& G123 =
          std::vector<std::shared_ptr<Tensor> >()
    );

/**
 * Compute the K (Exchange) gradient - simple version:
 *
 *  G_{C} += \sum_{pqrs} D_{pq}
 *          \partial_{C}
 *          \int_{\mathbb{R}^3}
 *          \mathrm{d}^3 r_1
 *          \phi_p^1 \phi_r^1
 *          O(r_12)
 *          \phi_q^2 \phi_s^2
 *          D_{rs}
 *
 * Density-based sieving is used. Integrals are sieved (by whole shell quartet)
 *  according to the estimate Kest = D_{pq} B_{pr} B_{qs} D_{rs} < thre.  Here
 *  D_{pq} and D_{rs} are the maximum density matrix elements in the pq and rs
 *  shell pairs, respectively.
 * Mixed precision is possible (though not presently implemented on CPU).
 *  In mixed precision, integrals are computed in double precision if Kest >=
 *  thredp, in single precision if Kest >= thresp, and neglected otherwise. For
 *  algorithms not using mixed precision, integrals are computed in double
 *  precision if Kest >= thresp, and neglected otherwise - thredp is not used.
 *  For good accuracy, one should set thresp << thredp. Note that all
 *  accumulation is performed in double precision.
 * This is presently implemented using the "K-invert" technique, where lists of
 *  significant p are constructed for each r (likewise q for s), allowing the
 *  significant contributions for a given D_{rs} entry to be efficiently
 *  determined.
 *
 * Implementation: CPU only (Cartesian via Rys-Gauss-Hermite quadrature).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param ewald the interaction operator O(r_12)
 * @param pairlist the list of significant pairs (must be symmetric)
 * @param D13 the OPDM, an (nao, nao) Tensor
 * @param D24 the OPDM, an (nao, nao) Tensor
 * @param D13symm is the D13 matrix symmetric (1-2x speedup possible)
 *  Setting D13symm to true only makes sense if basis1 and basis3 are the same
 *  - the code will throw if this is not satisfied.
 * @param D24symm is the D24 matrix symmetric (1-2x speedup possible)
 *  Setting D24symm to true only makes sense if basis2 and basis4 are the same
 *  - the code will throw if this is not satisfied.
 * @param Dsame are D13 and D24 the same matrix (2x speedup possible)
 *  If D13symm, D24symm and Dsame are all set to Ture, 4x speedup is possible.
 * @param thresp the value of Kest for which integral contributions are
 *  guaranteed to be computed in at least single precision.
 * @param thredp the value of Kest for which integral contributions are
 *  guaranteed to be computed in at least double precision.
 * @param G [optional] an (natom, 3) Tensor to accumulate into. If not
 *  provided, this will be allocated.
 * @return an (natom, 3) Tensor with the gradients accumulated
 **/
static
std::shared_ptr<Tensor> exchangeGrad(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist, 
    const std::shared_ptr<Tensor>& D13,
    const std::shared_ptr<Tensor>& D24,
    bool D13symm,
    bool D24symm,
    bool Dsame,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& G =
          std::shared_ptr<Tensor>()
    );

/**
 * Compute the K (Exchange) gradient - advanced version:
 *
 *  G1_{C} += \sum_{pqrs} D_{pq}
 *          \int_{\mathbb{R}^3} \mathrm{d}^3 r_1
 *          (\partial_{C} \phi_p^1) \phi_r^1 O(r_12) \phi_q^2 \phi_s^2
 *          D_{rs}
 *  G2_{C} += \sum_{pqrs} D_{pq}
 *          \int_{\mathbb{R}^3} \mathrm{d}^3 r_1
 *          \phi_p^1 (\partial_{C} \phi_r^1) O(r_12) \phi_q^2 \phi_s^2
 *          D_{rs}
 *  G3_{C} += \sum_{pqrs} D_{pq}
 *          \int_{\mathbb{R}^3} \mathrm{d}^3 r_1
 *          \phi_p^1 \phi_r^1 O(r_12) (\partial_{C} \phi_q^2) \phi_s^2
 *          D_{rs}
 *  G4_{C} += \sum_{pqrs} D_{pq}
 *          \int_{\mathbb{R}^3} \mathrm{d}^3 r_1
 *          \phi_p^1 \phi_r^1 O(r_12) \phi_q^2 (\partial_{C} \phi_s^2)
 *          D_{rs}
 *
 * Density-based sieving is used. Mixed precision is possible (though not
 * presently implemented on CPU). 
 *
 * This is presently implemented using the "K-invert" technique.
 *
 * Implementation: CPU only (Cartesian via Rys-Gauss-Hermite quadrature).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param ewald the interaction operator O(r_12)
 * @param pairlist12 the list of significant pairs describing basis1 and basis2
 * @param pairlist34 the list of significant pairs describing basis3 and basis4
 * @param D13 the OPDM, an (nao1, nao3) Tensor
 * @param D24 the OPDM, an (nao2, nao4) Tensor
 * @param D13symm is the D13 matrix symmetric (1-2x speedup possible)
 *  Setting D13symm to true only makes sense if basis1 and basis3 are the same
 *  - the code will throw if this is not satisfied.
 * @param D24symm is the D24 matrix symmetric (1-2x speedup possible)
 *  Setting D24symm to true only makes sense if basis2 and basis4 are the same
 *  - the code will throw if this is not satisfied.
 * @param Dsame are D13 and D24 the same matrix (2x speedup possible)
 * @param thresp the value of Kest for which integral contributions are
 *  guaranteed to be computed in at least single precision.
 * @param thredp the value of Kest for which integral contributions are
 *  guaranteed to be computed in at least double precision.
 * @param G1234 [optional] a vector of {(natom1, 3), (natom2, 3), (natom3, 3),
 *  (natom4, 3)} Tensors to accumulate into. If not provided, this will be
 *  allocated.
 * @return a vector of {(natom1, 3), (natom2, 3), (natom3, 3), (natom4, 3)}
 *  Tensors with the gradients accumulated
 **/
static
std::vector<std::shared_ptr<Tensor> > exchangeGradAdv(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist12,
    const std::shared_ptr<PairList>& pairlist34,
    const std::shared_ptr<Tensor>& D13,
    const std::shared_ptr<Tensor>& D24,
    bool D13symm,
    bool D24symm,
    bool Dsame,
    float thresp,
    float thredp,
    const std::vector<std::shared_ptr<Tensor> >& G1234 =
          std::vector<std::shared_ptr<Tensor> >()
    );

// => ECP Integrals <= //

/**
 * Compute the V (ECP) matrix:
 *
 *  V_pq =  \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          \phi_p^1 
 *          [U_{Lmax,A}^1 + \sum_{lm} 
 *          | S_{lm} >[ U_{l,A}^1 - U_{Lmax,A}^1] < S_{lm} |]
 *          \phi_q^1
 *
 * No screening beyond pair cache screening is used. A 2x reduction in work is
 * achieved if pairlist is symmetric. 
 *
 * NOTE: Due to restrictions in TeraChem, the routine will currently throw an
 * exception if the pairlist is not symmetric or if the basis and ecp basis do
 * not have the same number of atoms.
 *
 * Implementation: GPU only.
 * 
 * @param resources the list of CPU and/or GPU resources to use
 * @param pairlist the list of significant pairs describing basis1 and basis2
 * @param ecp an ECPBasisSet object
 * @param threoe the threshold to discard integrals below
 * @param V [optional] an (nao1, nao2) Tensor to accumulate into. If not
 *  provided, this will be allocated.
 * @return an (nao1, nao2) Tensor with the integrals accumulated
 **/
static
std::shared_ptr<Tensor> ecp(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<ECPBasis>& ecp,
    float threoe,
    const std::shared_ptr<Tensor>& V = 
          std::shared_ptr<Tensor>()
    );

/**
 * Compute the V (ECP) gradient - simple version:
 *
 *  G_{C} += \sum_{pqA} D_{pq}
 *          \int_{\mathbb{R}^3} 
 *          \mathrm{d}^3 r_1
 *          (\partial_C) \phi_p^1 
 *          [U_{Lmax,A}^1 + \sum_{lm} 
 *          (\partial C) | S_{lm} >[ U_{l,A}^1 - U_{Lmax,A}^1] < S_{lm} |]
 *          (\partial_C) \phi_q^1 
 *
 * Throws if pairlist is not symmetric or if the natom fields of the Basis and
 * ECPBasis objects are different.
 *
 * No screening beyond pair cache screening is used. A 2x reduction in work is
 * achieved if pairlist is symmetric. 
 *
 * NOTE: Due to restrictions in TeraChem, the routine will currently throw an
 * exception if the pairlist is not symmetric or if the basis and ecp basis do
 * not have the same number of atoms.
 *
 * Implementation: GPU only.
 * 
 * @param resources the list of CPU and/or GPU resources to use
 * @param pairlist the list of significant pairs describing basis1 and basis2
 * @param D an (nao1, nao2) Tensor with the total electronic density
 * @param ecp an ECPBasisSet object
 * @param threoe the threshold to discard integrals below
 * @param G [optional] an (natom, 3) Tensor to accumulate into. If not
 *  provided, this will be allocated.
 * @return an (natom, 3) Tensor with the gradient accumulated
 **/
static
std::shared_ptr<Tensor> ecpGrad(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<ECPBasis>& ecp,
    float threoe,
    const std::shared_ptr<Tensor>& V = 
          std::shared_ptr<Tensor>()
    );

// => Active-Space Integral Transformation Utility <= //

/**
 * Compute the fully transformed ERI's:
 *      
 *  (ij|kl) = C1_pi C2_qj (pq|rs) C3_rk C4_sl
 *
 * This algorithm uses (n1 * n2) J-matrix builds:
 *  
 *  (ij|kl) = C3_rk [J_rs (D_pq = C1_pi C2_qj)] C4_sl
 * 
 * RECOMMENDATION: the column dimension of C1_pi (ni) and C2_qj (nj) should
 * be minimized, e.g., these should be the active-space indices.
 *
 * If C1 == C2 (same pointer), a ~2x speedup will be achieved.
 *
 * Symmetrization is performed to reduce issues due to roundoff
 * If C1 == C2: (pq|rs) = (qp|rs) is enforced
 * If C3 == C2: (pq|rs) = (pq|sr) is enforced
 * If C1 == C3 and C2 == C4: (pq|rs) = (rs|pq) is enforced
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param ewald the interaction operator O(r_12)
 * @param pairlist12 the list of significant pairs describing basis1 and basis2
 * @param pairlist34 the list of significant pairs describing basis3 and basis4
 * @param C1 an (nao1,n1)-shape Tensor of orbital coefficients
 * @param C2 an (nao2,n2)-shape Tensor of orbital coefficients
 * @param C3 an (nao3,n3)-shape Tensor of orbital coefficients
 * @param C4 an (nao4,n4)-shape Tensor of orbital coefficients
 * @param thresp the value of Kest for which integral contributions are
 *  guaranteed to be computed in at least single precision.
 * @param thredp the value of Kest for which integral contributions are
 *  guaranteed to be computed in at least double precision.
 * @param I [optional] an (n1,n2,n3,n4) Tensor to accumulate into
 *  (overwritten). If not provided, this will be allocated.
 * @return an (n1,n2,n3,n4) Tensor with the MO-basis ERIs.
 **/  
static
std::shared_ptr<Tensor> eriJ(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist12,
    const std::shared_ptr<PairList>& pairlist34,
    const std::shared_ptr<Tensor>& C1,
    const std::shared_ptr<Tensor>& C2,
    const std::shared_ptr<Tensor>& C3,
    const std::shared_ptr<Tensor>& C4,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& I =
          std::shared_ptr<Tensor>()
    );

/**
 * Compute the contraction of the 
 *
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param ewald the interaction operator O(r_12)
 * @param pairlist12 the list of significant pairs describing basis1 and basis2
 **/
static 
std::shared_ptr<Tensor> eriGradJ(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& C,
    const std::shared_ptr<Tensor>& D2,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& G =
          std::shared_ptr<Tensor>()
    );

// ==> CPU-Based Routines <== //

static
std::shared_ptr<Tensor> overlapCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& S = std::shared_ptr<Tensor>()
    );

static
std::vector< std::shared_ptr<Tensor> > dipoleCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    double x0,
    double y0,
    double z0,
    const std::vector< std::shared_ptr<Tensor> >& X = 
          std::vector< std::shared_ptr<Tensor> >()
    );
    
static
std::vector< std::shared_ptr<Tensor> > quadrupoleCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    double x0,
    double y0,
    double z0,
    const std::vector< std::shared_ptr<Tensor> >& Q = 
          std::vector< std::shared_ptr<Tensor> >()
    );

static
std::vector< std::shared_ptr<Tensor> > nablaCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::vector< std::shared_ptr<Tensor> >& P = 
          std::vector< std::shared_ptr<Tensor> >()
    );

static
std::vector< std::shared_ptr<Tensor> > angularMomentumCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    double x0,
    double y0,
    double z0,
    const std::vector< std::shared_ptr<Tensor> >& L = 
          std::vector< std::shared_ptr<Tensor> >()
    );
    
static
std::shared_ptr<Tensor> kineticCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& T = std::shared_ptr<Tensor>()
    );

static 
std::shared_ptr<Tensor> potentialCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& xyzc,
    const std::shared_ptr<Tensor>& V = std::shared_ptr<Tensor>()
    );
    
static 
std::shared_ptr<Tensor> coulombCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist12,
    const std::shared_ptr<PairList>& pairlist34,
    const std::shared_ptr<Tensor>& D34,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& J12 = std::shared_ptr<Tensor>()
    );

static 
std::shared_ptr<Tensor> exchangeCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist12,
    const std::shared_ptr<PairList>& pairlist34,
    const std::shared_ptr<Tensor>& D24,
    bool D24symm,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& K13 = std::shared_ptr<Tensor>()
    );

static
std::vector<std::shared_ptr<Tensor> > overlapGradAdvCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& W,
    const std::vector<std::shared_ptr<Tensor> >& G12 =
          std::vector<std::shared_ptr<Tensor> >()
    );

static
std::vector<std::shared_ptr<Tensor> > kineticGradAdvCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::vector<std::shared_ptr<Tensor> >& G12 =
          std::vector<std::shared_ptr<Tensor> >()
    );

static
std::vector< std::vector< std::shared_ptr<Tensor> >> dipoleGradAdvCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    double x0,
    double y0,
    double z0,
    const std::shared_ptr<Tensor>& D,
    const std::vector<std::vector< std::shared_ptr<Tensor> >>& Dgrad12 =
          std::vector<std::vector< std::shared_ptr<Tensor> >>()
    );

static
std::shared_ptr<Tensor> espCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<Tensor>& xyz,
    const std::shared_ptr<Tensor>& w = std::shared_ptr<Tensor>()
    );

static
std::shared_ptr<Tensor> fieldCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<Tensor>& xyz,
    const std::shared_ptr<Tensor>& F = std::shared_ptr<Tensor>()
    );

static
std::shared_ptr<Tensor> coulombGradSymmCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist, // symm
    const std::shared_ptr<Tensor>& D,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& G =
          std::shared_ptr<Tensor>()
    );

static
std::vector<std::shared_ptr<Tensor> > coulombGradAdvCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist12, // symm
    const std::shared_ptr<PairList>& pairlist34, // symm
    const std::shared_ptr<Tensor>& D12,
    const std::shared_ptr<Tensor>& D34,
    float thresp,
    float thredp,
    const std::vector<std::shared_ptr<Tensor> >& G1234 =
          std::vector<std::shared_ptr<Tensor> >()
    );

static
std::shared_ptr<Tensor> exchangeGradSymmCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& G =
          std::shared_ptr<Tensor>()
    );

static
std::vector<std::shared_ptr<Tensor> > exchangeGradAdvCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist12,
    const std::shared_ptr<PairList>& pairlist34,
    const std::shared_ptr<Tensor>& D13,
    const std::shared_ptr<Tensor>& D24,
    bool D13symm,
    bool D24symm,
    bool Dsame,
    float thresp,
    float thredp,
    const std::vector<std::shared_ptr<Tensor> >& G1234 =
          std::vector<std::shared_ptr<Tensor> >()
    );

static 
std::shared_ptr<Tensor> potentialGradCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<Tensor>& xyzc,
    const std::shared_ptr<Tensor>& G = 
          std::shared_ptr<Tensor>()
    );
    
static
std::vector<std::shared_ptr<Tensor> > potentialGradAdvCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<Tensor>& xyzc,
    const std::vector<std::shared_ptr<Tensor> >& G12A = 
          std::vector<std::shared_ptr<Tensor> >()
    );

static
std::vector<std::shared_ptr<Tensor> > potentialGradAdv2CPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<Tensor>& xyzc,
    const std::vector<std::shared_ptr<Tensor> >& G12A = 
          std::vector<std::shared_ptr<Tensor> >()
    );
// ==> GPU-Based Routines <== //

// TODO: all Ewald-involving GPU routines

// ==> TeraChem-Based Routines <== //

static 
std::shared_ptr<Tensor> coulombTC(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist12,
    const std::shared_ptr<PairList>& pairlist34,
    const std::shared_ptr<Tensor>& D34,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& J12 = std::shared_ptr<Tensor>()
    );

static 
std::shared_ptr<Tensor> exchangeTC(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist12,
    const std::shared_ptr<PairList>& pairlist34,
    const std::shared_ptr<Tensor>& D24,
    bool D24symm,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& K13 = std::shared_ptr<Tensor>()
    );

static
std::shared_ptr<Tensor> coulombGradTC(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist, // symm
    const std::shared_ptr<Tensor>& D12,
    const std::shared_ptr<Tensor>& D34,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& G =
          std::shared_ptr<Tensor>()
    );

static
std::shared_ptr<Tensor> exchangeGradTC(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist, // symm
    const std::shared_ptr<Tensor>& D13,
    const std::shared_ptr<Tensor>& D24,
    bool D13symm,
    bool D24symm,
    bool Dsame,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& G =
          std::shared_ptr<Tensor>()
    );
    
static
std::shared_ptr<Tensor> ecpTC(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<ECPBasis>& ecp,
    float threoe,
    const std::shared_ptr<Tensor>& V = 
          std::shared_ptr<Tensor>()
    );

static 
std::shared_ptr<Tensor> ecpGradTC(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<ECPBasis>& ecp,
    float threoe,
    const std::shared_ptr<Tensor>& G = 
          std::shared_ptr<Tensor>()
    );

};

} // namespace lightspeed 

#endif 
