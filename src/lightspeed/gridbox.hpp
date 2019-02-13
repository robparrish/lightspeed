#ifndef LS_GRIDBOX_HPP
#define LS_GRIDBOX_HPP

#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <string>
#include <cmath>

/**
 * The following yak-shave provides a good hash function for
 * tuple<int,int,int>, for use in unordered_map and similar data structures.
 * You either can read this code, or you will never need to read it.
 **/ 
namespace {

size_t hash_combine2(size_t lhs, size_t rhs) {
  lhs^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
  return lhs;
}

}

namespace std {
template<>
    struct hash<std::tuple<int,int,int> >
    {
        size_t operator()(std::tuple<int,int,int> const& v) const
        {
            size_t seed = 0;    
            seed = hash_combine2(seed,std::get<0>(v));
            seed = hash_combine2(seed,std::get<1>(v));
            seed = hash_combine2(seed,std::get<2>(v));
            return seed;
        }
    };
}

namespace lightspeed {

class Tensor;
class Basis;
class PairList;
class ResourceList;
class HashedGrid;
    
/**
 * Class HashedGrid provides a simple way to partition a grid of nP points xyz
 * into subsets of spatially localized points. The approach is to divide space
 * into cubes (usually called "boxes") of side length R (a user-specified
 * parameter). Each box is assigned a key of three integers to use for indexing
 * (the list of translations from the origin). Box <0,0,0> contains the origin
 * as its lower-left vertex. 
 *
 * For a given point <x,y,z> in R3, the corresponding box index can be obtained
 * in O(1) by the following hash: key = <floor(x/R),floor(y/R),floor(z/R)>. 
 *
 * For a given box index <i,j,k>, the subset of xyz points in the HashedGrid
 * that are in the box in question (if any) can be acquired in O(1) time by
 * using an appropriate data structure. Such queries are handled by the "inds"
 * routine.
 *
 * There are several way to implement a HashedGrid. A particularly simple one
 * (presently used) is to build an
 * unordered_map<tuple<int,int,int>,std::vector<size_t>> mapping box keys to
 * the list of grid point indices P in each box.
 **/
class HashedGrid {

public:

HashedGrid(
    const std::shared_ptr<Tensor>& xyz,
    double R);

/// The (nP,3) tensor containing the grid points in this HashedGrid
std::shared_ptr<Tensor> xyz() const { return xyz_; }
/// The box spacing (distance between each box in x, y, or z)
double R() const { return R_; }
/// The number of boxes with >= 1 point in them in this HashedGrid
size_t nbox() const { return map_.size(); }
/// A handy string representation of the object
std::string string() const;

/**
 * Return the indices of points xyz that are in the box corresponding to index
 * key, in O(1) time.
 *
 * @param key the box key to query for points
 * @return a vector of point indices P in xyz that are within the box. This is
 *  an empty vector if no points are in the box.
 **/
const std::vector<size_t>& inds(
    const std::tuple<int,int,int>& key) const
    {
        std::unordered_map<std::tuple<int,int,int>, std::vector<size_t> >::const_iterator it = 
            map_.find(key);
        if (it == map_.end()) return map_null_;
        else return (*it).second;
    }

/**
 * Compute the hashkey (e.g., the box index) for a point <x,y,z> in R3 in O(1) time.
 * @param x the x coordinate
 * @param y the y coordinate
 * @param z the z coordinate
 * @return the desired hashkey indicating which box the point is in.
 **/
std::tuple<int,int,int> hashkey(
    double x,
    double y,
    double z) const { 
    return std::tuple<int,int,int>(
        floor(x * Rinv_),
        floor(y * Rinv_),
        floor(z * Rinv_));
    }

/**
 * Is any part of the sphere of radius Rc centered at <xA,yA,zA> overlapping
 * the HashedGrid box at key, with HashedGrid box size R?
 * @param key the HashedGrid box key
 * @param R the HashedGrid box spacing R
 * @param xA the x center of the sphere
 * @param yA the y center of the sphere
 * @param zA the z center of the sphere
 * @param Rc the radius of the sphere
 * @return true if there is overlap, false otherwise
 **/
static 
bool significant_box(
    const std::tuple<int,int,int>& key,
    double R,
    double xA,
    double yA,
    double zA,
    double Rc) {
    double C1x = std::get<0>(key) * R;
    double C2x = (std::get<0>(key) + 1) * R;
    double C1y = std::get<1>(key) * R;
    double C2y = (std::get<1>(key) + 1) * R;
    double C1z = std::get<2>(key) * R;
    double C2z = (std::get<2>(key) + 1) * R;
    double D2 = pow(Rc,2);
    D2 -= (xA < C1x) * pow(xA - C1x, 2);
    D2 -= (xA > C2x) * pow(xA - C2x, 2);
    D2 -= (yA < C1y) * pow(yA - C1y, 2);
    D2 -= (yA > C2y) * pow(yA - C2y, 2);
    D2 -= (zA < C1z) * pow(zA - C1z, 2);
    D2 -= (zA > C2z) * pow(zA - C2z, 2);
    return D2 >= 0.0;
    }

const std::unordered_map<std::tuple<int,int,int>, std::vector<size_t> >& map() const { return map_; }
    
private:

// The user-provided grid, a (nP,3) Tensor
std::shared_ptr<Tensor> xyz_;
// The user-provided box spacing parameter
double R_;
// The inverse of the user-provided box spacing parameter
double Rinv_;

// The unordered map representation of the HashedGrid, for O(1) lookup
std::unordered_map<std::tuple<int,int,int>, std::vector<size_t> > map_;
// An empty vector, to combine query and lookup into one "inds" subroutine
// while allowing return of const std::vector<size_t>&
std::vector<size_t> map_null_;

};
    
/**
 * GridBox is a simple library for computation of grid collocation properties
 * in Gaussian basis sets, including orbitals, densities, GGA-type properties
 * and gradients 
 * 
 * Features:
 *  - Grid structure agnostic (Becke, rectilinear, etc)
 *  - Spin specialization agnostic (R, U, etc)
 *  - Kernel depth agnostic (DFT, TD-DFT, etc)
 *  - LSDA/GGA-type grid density collocation, spectral potential collocation,
 *    and gradients
 *  - LSDA/GGA-type orbital collocation
 * 
 * As such, GridBox should prove useful in the following areas:
 *  - DFT
 *  - THC
 *  - Cubic grid properties (e.g., visualization)
 *  - FTC
 **/ 
class GridBox {

public:

// => LDA Routines <= //

/**
 * Compute the blurred Gaussian density field (LDA) on a grid:
 *
 *  rho_P += D_pq \phi_p^P \phi_q^P
 *
 * The computation is sieved so that only contributions with 
 * |rho_AP| >= thre are retained. 
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
    double thre,
    const std::shared_ptr<Tensor>& rho = std::shared_ptr<Tensor>()
    );

/**
 * Compute the blurred Gaussian potential (LDA) from a grid potential:
 *
 *  V_pq += \phi_p^P \phi_q^P v_P
 *
 * The computation is sieved so that only contributions with 
 * |v_AP| >= thre are retained. 
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
    double thre,
    const std::shared_ptr<Tensor>& V = std::shared_ptr<Tensor>()
    );

/**
 * Compute the gradient of a blurred Gaussian charge distribution (LDA)
 * interacting with a grid potential:
 *
 *  G_C += \partial_{C} D_{pq} \phi_p^P \phi_q^P v_P
 *
 * The computation is sieved so that only contributions with 
 * |E_AP| >= thre are retained. 
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
    double thre,
    const std::shared_ptr<Tensor>& G = std::shared_ptr<Tensor>()
    );
    
/**
 * Compute the gradient of a blurred Gaussian charge distribution (LDA)
 * interacting with a grid potential:
 *
 *  G1_C += D_{pq} (\partial_C \phi_p^P) \phi_q^P v_P
 *  G2_C += D_{pq} \phi_q^P (\partial_C \phi_q^P) v_P
 *
 * The computation is sieved so that only contributions with 
 * |E_AP| >= thre are retained. 
 *
 * This is the advanced version of the gradient routine, and returns the
 * separate gradient contributions from \phi_p and \phi_q.
 *
 * This routine throws if pairlist is symmetric
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
 * @param V [optional] a vector with an (natom1,3) and an (natom2,3) Tensor to
 *  accumulate the result into. If not provided, this will be allocated.
 * @return a vector with an (natom1,3) and an (natom2,3) Tensor with the result
 *  accumulated. 
 **/
static
std::shared_ptr<Tensor> ldaGradAdv(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<HashedGrid>& grid,
    const std::shared_ptr<Tensor>& v,
    double thre,
    const std::vector<std::shared_ptr<Tensor> >& G12 
        = std::vector<std::shared_ptr<Tensor> >()
    );

// => GGA Routines <= //

/**
 * Compute the blurred Gaussian density field (GGA) on a grid.
 *
 * Computes:     
 *
 *  rho_P += D_pq \phi_p^P \phi_q^P
 *  rho_P^x += \partial_{x_P} D_pq \phi_p^P \phi_q^P
 *  rho_P^y += \partial_{y_P} D_pq \phi_p^P \phi_q^P
 *  rho_P^z += \partial_{z_P} D_pq \phi_p^P \phi_q^P
 *
 * The computation is sieved so that only contributions with 
 * |rho_AP| >= thre are retained. 
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
 * @param rho [optional] an (nP,4) Tensor to accumulate the result into. If not
 *  provided, this will be allocated.
 * @return an (nP,4) Tensor with the result accumulated. Laid out as 
 *  (rho,rho^x,rho^y,rho^z) on each row.
 **/
static
std::shared_ptr<Tensor> ggaDensity(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<HashedGrid>& grid,
    double thre,
    const std::shared_ptr<Tensor>& rho = std::shared_ptr<Tensor>()
    );

/**
 * Compute the blurred Gaussian potential (GGA) from a grid potential.
 *
 * Computes the following contributions:     
 *
 *  V_pq += \phi_p^P \phi_q^P v_P
 *  V_pq += \partial_{x_P} \phi_p^P \phi_q^P v_P^x
 *  V_pq += \partial_{y_P} \phi_p^P \phi_q^P v_P^y
 *  V_pq += \partial_{z_P} \phi_p^P \phi_q^P v_P^z
 *
 * The computation is sieved so that only contributions with 
 * |v_AP| >= thre are retained. 
 *
 * A 2x reduction in work is obtained if pairlist is symmetric.
 *
 * Implementation: CPU only (Hermite).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param pairlist the PairList object defining basis1 and basis2
 * @param grid a HashedGrid to collocate on, collocation points P taken from
 *  grid->xyz() 
 * @param v an (nP,4) Tensor with the grid potential values. Laid out as 
 *  (v,v^x,v^y,v^z) on each row.
 * @param trans (nT,3) Tensor with translation vectors for the centers r_A. If
 *  you do not want translations, provide <0.0,0.0,0.0>.
 * @param alpha the blur exponent (-1.0 indicates no blurring)
 * @param thre cutoff value below which collocation values are clamped to zero.
 * @param V [optional] an (nao1,nao2) Tensor to accumulate the result into. If not
 *  provided, this will be allocated.
 * @return a (nao1,nao2) Tensor with the result accumulated. 
 **/
static
std::shared_ptr<Tensor> ggaPotential(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<HashedGrid>& grid,
    const std::shared_ptr<Tensor>& v,
    double thre,
    const std::shared_ptr<Tensor>& V = std::shared_ptr<Tensor>()
    );

/**
 * Compute the gradient of a blurred Gaussian charge distribution (GGA)
 * interacting with a grid potential.
 *
 * Computes the following contributions:     
 *
 *  G_C += \partial_{C} D_{pq} \phi_p^P \phi_q^P v_P
 *  G_C += \partial_{C} D_{pq} \partial_{x_P} \phi_p^P \phi_q^P v_P^x
 *  G_C += \partial_{C} D_{pq} \partial_{y_P} \phi_p^P \phi_q^P v_P^y
 *  G_C += \partial_{C} D_{pq} \partial_{z_P} \phi_p^P \phi_q^P v_P^z
 *
 * The computation is sieved so that only contributions with 
 * |E_AP| >= thre are retained. 
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
 * @param v an (nP,4) Tensor with the grid potential values. Laid out as 
 *  (v,v^x,v^y,v^z) on each row.
 * @param trans (nT,3) Tensor with translation vectors for the centers r_A. If
 *  you do not want translations, provide <0.0,0.0,0.0>.
 * @param alpha the blur exponent (-1.0 indicates no blurring)
 * @param thre cutoff value below which collocation values are clamped to zero.
 * @param V [optional] an (natom1,3) Tensor to accumulate the result into. If not
 *  provided, this will be allocated.
 * @return a (natom1,3) Tensor with the result accumulated. 
 **/
static
std::shared_ptr<Tensor> ggaGrad(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<HashedGrid>& grid,
    const std::shared_ptr<Tensor>& v,
    double thre,
    const std::shared_ptr<Tensor>& G = std::shared_ptr<Tensor>()
    );
    
/**
 * Compute the gradient of a blurred Gaussian charge distribution (GGA)
 * interacting with a grid potential.
 *
 * Computes the following contributions:     
 *
 *  G1_C += D_{pq} (\partial_C \phi_p^P) \phi_q^P v_P
 *  G1_C += D_{pq} \partial_{x_P} (\partial_C \phi_p^P) \phi_q^P v_P^x
 *  G1_C += D_{pq} \partial_{y_P} (\partial_C \phi_p^P) \phi_q^P v_P^y
 *  G1_C += D_{pq} \partial_{z_P} (\partial_C \phi_p^P) \phi_q^P v_P^z
 *
 *  G2_C += D_{pq} \phi_p^P (\partial_C \phi_q^P) v_P
 *  G2_C += D_{pq} \partial_{x_P} \phi_p^P (\partial_C \phi_q^P) v_P^x
 *  G2_C += D_{pq} \partial_{y_P} \phi_p^P (\partial_C \phi_q^P) v_P^y
 *  G2_C += D_{pq} \partial_{z_P} \phi_p^P (\partial_C \phi_q^P) v_P^z
 *
 * The computation is sieved so that only contributions with 
 * |E_AP| >= thre are retained. 
 *
 * This is the advanced version of the gradient routine, and returns the
 * separate gradient contributions from \phi_p and \phi_q.
 *
 * This routine throws if pairlist is symmetric
 *
 * Implementation: CPU only (Hermite).
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param pairlist the PairList object defining basis1 and basis2
 * @param D an (nao1,nao2) Tensor containing the OPDM  
 * @param grid a HashedGrid to collocate on, collocation points P taken from
 *  grid->xyz() 
 * @param v an (nP,4) Tensor with the grid potential values. Laid out as 
 *  (v,v^x,v^y,v^z) on each row.
 * @param trans (nT,3) Tensor with translation vectors for the centers r_A. If
 *  you do not want translations, provide <0.0,0.0,0.0>.
 * @param alpha the blur exponent (-1.0 indicates no blurring)
 * @param thre cutoff value below which collocation values are clamped to zero.
 * @param V [optional] a vector with an (natom1,3) and an (natom2,3) Tensor to
 *  accumulate the result into. If not provided, this will be allocated.
 * @return a vector with an (natom1,3) and an (natom2,3) Tensor with the result
 *  accumulated. 
 **/
static
std::shared_ptr<Tensor> ggaGradAdv(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<HashedGrid>& grid,
    const std::shared_ptr<Tensor>& v,
    double thre,
    const std::vector<std::shared_ptr<Tensor> >& G12 
        = std::vector<std::shared_ptr<Tensor> >()
    );
    
// => Meta Routines <= //

/**
 * Compute the Gaussian density field (Meta) on a grid.
 *
 * Computes:     
 *
 *  rho_P += D_pq \phi_p^P \phi_q^P]
 *  rho_P^x += \partial_{x_P} D_pq \phi_p^P \phi_q^P
 *  rho_P^y += \partial_{y_P} D_pq \phi_p^P \phi_q^P
 *  rho_P^z += \partial_{z_P} D_pq \phi_p^P \phi_q^P
 *      
 * and so forth for Hessians.
 *
 * The computation is sieved so that only contributions with 
 * |rho_AP| >= thre are retained. 
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
 * @param thre cutoff value below which collocation values are clamped to zero.
 * @param rho [optional] an (nP,10) Tensor to accumulate the result into. If not
 *  provided, this will be allocated.
 * @return an (nP,10) Tensor with the result accumulated. Laid out as 
 *  (0,x,y,z,xx,xy,xz,yy,yz,zz) on each row.
 **/
static
std::shared_ptr<Tensor> metaDensity(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<HashedGrid>& grid,
    double thre,
    const std::shared_ptr<Tensor>& rho = std::shared_ptr<Tensor>()
    );

/**
 * Compute the orbital collocation values:
 *
 * \psi_i^P += \sum_p C_pi \phi_p^P
 *
 * Implementation: CPU only.
 *
 * The significance criterion is |C| c_A exp(-a_A r_1P^2) < thre, where |C| is
 * the largest Cartesian orbital coefficient in the shell (across all orbitals)
 * 
 * @param resources the list of CPU and/or GPU resources to use.
 * @param basis the basis functions to build orbitals from.
 * @param xyz the grid coordinates to collocate to, an (nP,3) Tensor.
 * @param C the orbital coefficients, an (nao,norb) Tensor.
 * @param thre the orbital cutoff
 * @param psi [optional] an (nP,norb) Tensor to accumulate the orbital
 *  collocation into. If not provided, this will be allocated.
 * @return an (nP,norb) Tensor with the orbital collocation accumulated
 **/
static
std::shared_ptr<Tensor> orbitals(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Basis>& basis,
    const std::shared_ptr<Tensor>& xyz,
    const std::shared_ptr<Tensor>& C,
    double thre,
    const std::shared_ptr<Tensor>& psi = std::shared_ptr<Tensor>()
    );

/**
 * Compute the orbital collocation gradient contribution:
 *
 * G_A += \sum_P \partial_A (\psi_i^P) X_i^P
 *
 * where the orbitals are: 
 *
 * \psi_i^P = \sum_p C_pi \phi_p^P
 * 
 * Implementation: CPU only.
 *
 * The significance criterion is |C| |X| c_A exp(-a_A r_1P^2) < thre, where |C|
 * is the largest Cartesian orbital coefficient in the shell (across all
 * orbitals) and |X| is the largest collocation density maginitude element for
 * the point (across all orbitals)
 *
 * @param resources the list of CPU and/or GPU resources to use.
 * @param basis the basis functions to build orbitals from.
 * @param xyz the grid coordinates to collocate to, an (nP,3) Tensor.
 * @param C the orbital coefficients, an (nao,norb) Tensor.
 * @param xi the 
 * @param thre the orbital cutoff
 * @param GA [optional] an (nA,3) Tensor to accumulate the gradient into. If
 *  not provided, this will be allocated.
 * @return an (nA,3) Tensor with the gradient accumulated
 **/
static
std::shared_ptr<Tensor> orbitalsGrad(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Basis>& basis,
    const std::shared_ptr<Tensor>& xyz,
    const std::shared_ptr<Tensor>& C,
    const std::shared_ptr<Tensor>& xi,
    double thre,
    const std::shared_ptr<Tensor>& GA = std::shared_ptr<Tensor>()
    );
    
};

} // namespace lightspeed 

#endif
