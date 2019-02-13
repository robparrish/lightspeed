#ifndef LIGHTSPEED_PURE_TRANSFORM_HPP
#define LIGHTSPEED_PURE_TRANSFORM_HPP

#include <memory>

namespace lightspeed {

class Basis;
class Tensor;

/**
 * Class PureTransform provides utilities to transform matrices between cart
 * and pure representations, while minimizing allocations via pointer aliasing.
 * This is used in many IntBox/GridBox routines to effectively hide
 * considerations of spherical harmonics from library designers (so that all
 * computational code can be written in cart representation), while avoiding
 * redundant allocations in the case that the basis sets involved do not
 * contain pure functions.
 *
 * Advanced users of client codes may use these routines to transform between
 * cart and pure representations, but should read the headers carefully.
 *
 * Note that "pure" throughout this class refers to the preferred
 * representation of the involved basis sets, and may be wholly pure, wholly
 * cart, or mixed. The case where the basis sets are wholly cart presents an
 * opportunity for optimization/reduced memory usage, as there is no difference
 * between pure and cart representations in this case.
 **/
class PureTransform {

public:
    
/**
 * Effective Operation:
 *  Returns a register Tensor of size (ncart1, ncart2)
 *  This is intended to assist in cases where the user wants a matrix in pure
 *  representation, but the Lightspeed coder wants to work in cart
 *  representation, without actually doing an allocation in the case that the
 *  involved basis sets are already in cart representation.
 *
 * Special Cases:
 *  - If basis1/basis2 are cart and Tpure is provided, returns Tpure
 *  - If basis1/basis2 are cart and Tpure is not provided, allocate/return 
 *  - If basis1/basis2 are not cart, allocate/return 
 *
 * Checks:
 *  - throws if Tpure is not (nao1, nao2) if Tpure is provided
 **/
static 
std::shared_ptr<Tensor> allocCart2(
    const std::shared_ptr<Basis>& basis1,
    const std::shared_ptr<Basis>& basis2,
    const std::shared_ptr<Tensor>& Tpure = std::shared_ptr<Tensor>());
    
/**
 * Effective Operation:
 *  Tpure += cartToPure(Tcart)
 *  Here, Tcart should have been acquired by a call to allocCart2 over Tpure,
 *  so that the accumulation is done correctly in the case that basis1/basis2
 *  are cart.
 *  
 * Special Cases:
 *  - If basis1/basis2 are cart: returns Tcart
 *  - If basis1/basis2 are not cart and Tpure given: transform and add into Tpure
 *  - If basis1/basis2 are not cart and not Tpure given: allocate, transform
 *    and add into Tpure
 *
 * Checks:
 *  - throws if Tcart is not (ncart1,ncart2)
 *  - throws if Tpure is not (nao1,nao2) if Tpure is provided
 **/
static
std::shared_ptr<Tensor> cartToPure2(
    const std::shared_ptr<Basis>& basis1,
    const std::shared_ptr<Basis>& basis2,
    const std::shared_ptr<Tensor>& Tcart,
    const std::shared_ptr<Tensor>& Tpure = std::shared_ptr<Tensor>());

/**
 * Effective Operation:
 *  Tcart = pureToCart(Tpure)
 *  
 * Special Cases:
 *  - basis1 and basis2 are fully cart: returns Tpure
 *
 * Checks:
 *  - throws if Tpure is not (nao1,nao2)
 **/
static
std::shared_ptr<Tensor> pureToCart2(
    const std::shared_ptr<Basis>& basis1,
    const std::shared_ptr<Basis>& basis2,
    const std::shared_ptr<Tensor>& Tpure);

/**
 * Effective Opertation:
 *  Ccart = pureToCart1(Cpure)
 *  This transforms the rows of Cpure to wholly Cartesian
 *  
 * Special Cases:
 *  - basis is fully cart: returns Cpure
 *
 * Checks:
 *  - throws if Cpure is not (nao1,norb) 
 **/
static 
std::shared_ptr<Tensor> pureToCart1(
    const std::shared_ptr<Basis>& basis,
    const std::shared_ptr<Tensor>& Cpure);

};

} // namespace lightspeed

#endif
