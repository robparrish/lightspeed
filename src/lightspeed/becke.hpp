#ifndef LS_BECKE_HPP
#define LS_BECKE_HPP

#include <memory>
#include <stdexcept>
#include <vector>
#include <string>

namespace lightspeed {

class Tensor;
class ResourceList;
    
/**
 * Class LebedevGrid represents a Lebedev spherical quadrature grid, which exactly
 * integrates spherical harmonics up to order 2*L+1. The Lebedev grids have
 * octahedral symmetry, and are placed at rho=1. Not all of the Lebedev grids
 * are totally positive (some of the weights can be negative).
 *      
 * The spherical integral is:
 * \[
 * I_S =
 * \int_{\Omega}
 * \mathrm{d} \Omega
 * f(\Omega)
 * \approx w_P f(\Omega_P)
 * \]
 **/
class LebedevGrid {

public:

// => Factory Constructors <= //

/**
 * Construct a Lebedev grid with N points
 * @param N the number of points in the Lebedev grid
 * @return the constructed Lebedev grid
 *
 * Throws if N is not a Lebedev number.
 **/
static    
std::shared_ptr<LebedevGrid> build(size_t N);

/**
 * Return the vector of valid Lebedev grid sizes, in increasing order.
 **/
static 
std::vector<size_t> sizes();

// => Accessors <= //

// Return the number of points in this Lebedev grid
size_t size() const;
// A Lebedev grid exactly integrates all spherical harmonics of up to order 2*L+1
int L() const { return L_; }
// Are all the weights in this Lebedev grid positive?
bool is_positive() const;
// The grid points in Cartesian, a (size,4) Tensor with (x,y,z,w) on each row.
std::shared_ptr<Tensor> xyzw() const { return xyzw_; }
// The grid points in Spherical, a (size,4) Tensor with (rho,theta,phi,w) on each row.
std::shared_ptr<Tensor> rtpw() const;
// A handy string representation of the object
std::string string() const;

private:

LebedevGrid(
    int L,
    const std::shared_ptr<Tensor>& xyzw) :
    L_(L),
    xyzw_(xyzw)
    {}

int L_;
std::shared_ptr<Tensor> xyzw_;

static std::shared_ptr<LebedevGrid> build6();
static std::shared_ptr<LebedevGrid> build14();
static std::shared_ptr<LebedevGrid> build18(); // Technically not a Lebedev Grid, needed for SG-0
static std::shared_ptr<LebedevGrid> build26();
static std::shared_ptr<LebedevGrid> build38();
static std::shared_ptr<LebedevGrid> build50();
static std::shared_ptr<LebedevGrid> build74();
static std::shared_ptr<LebedevGrid> build86();
static std::shared_ptr<LebedevGrid> build110();
static std::shared_ptr<LebedevGrid> build146();
static std::shared_ptr<LebedevGrid> build170();
static std::shared_ptr<LebedevGrid> build194();
static std::shared_ptr<LebedevGrid> build230();
static std::shared_ptr<LebedevGrid> build266();
static std::shared_ptr<LebedevGrid> build302();
static std::shared_ptr<LebedevGrid> build350();
static std::shared_ptr<LebedevGrid> build434();
static std::shared_ptr<LebedevGrid> build590();
static std::shared_ptr<LebedevGrid> build770();
static std::shared_ptr<LebedevGrid> build974();
static std::shared_ptr<LebedevGrid> build1202();
static std::shared_ptr<LebedevGrid> build1454();
static std::shared_ptr<LebedevGrid> build1730();
static std::shared_ptr<LebedevGrid> build2030();
static std::shared_ptr<LebedevGrid> build2354();
static std::shared_ptr<LebedevGrid> build2702();
static std::shared_ptr<LebedevGrid> build3074();
static std::shared_ptr<LebedevGrid> build3470();
static std::shared_ptr<LebedevGrid> build3890();
static std::shared_ptr<LebedevGrid> build4334();
static std::shared_ptr<LebedevGrid> build4802();
static std::shared_ptr<LebedevGrid> build5294();
static std::shared_ptr<LebedevGrid> build5810();

};

/**
 * RadialGrid provide quadratures to integrate 
 * \[
 * I_R = 
 * \int_{0}^{\infty} 
 * \mathrm{d} r  
 * r^2 f(r)
 * \approx w_P f(r_P) 
 * \]
 * via various rules. 
 **/ 
class RadialGrid {

public:

// => Factory Constructors <= //

/**
 * Build a standard quadrature according to known rules
 * @param name: The rule name, one of:
 *   BECKE    - A.D. Becke, J. Chem. Phys., 88, 2547 (1988)
 *   HANDY    - C.W. Murray, N.C. Handy, G.J. Laming, Mol. Phys., 78, 997 (1993) (aka Euler-Maclaurin)
 *   AHLRICHS - O. Treutler, R.J. Ahlrichs, J. Chem. Phys., 102, 346 (1995) (aka Treutler, M4T2)
 *   KNOWLES  - M.E. Mura, P.J. Knowles, J. Chem. Phys., 104, 9848 (1996) (aka Mura)
 *   MULTIEXP - P.M.W. Gill, S.H. Chien, J. Comp. Chem., 24, 732 (2003) (aka Gill)
 *      Allowed MultiExp quadratures sizes are 1-26, 30, 35, 40, 45, 50 (throws otherwise)
 * @param size: The number of nodes
 * @param R: The radial scaling parameter in atomic units
 **/   
static std::shared_ptr<RadialGrid> build(
    const std::string& name,
    size_t size,
    double R);

/**
 * Build a standard quadrature according to known rules
 * @param name: The rule name, one of:
 *   BECKE    - A.D. Becke, J. Chem. Phys., 88, 2547 (1988)
 *   HANDY    - C.W. Murray, N.C. Handy, G.J. Laming, Mol. Phys., 78, 997 (1993) (aka Euler-Maclaurin)
 *   AHLRICHS - O. Treutler, R.J. Ahlrichs, J. Chem. Phys., 102, 346 (1995) (aka Treutler, M4T2)
 *   KNOWLES  - M.E. Mura, P.J. Knowles, J. Chem. Phys., 104, 9848 (1996) (aka Mura)
 *   MULTIEXP - P.M.W. Gill, S.H. Chien, J. Comp. Chem., 24, 732 (2003) (aka Gill)
 *      Allowed MultiExp quadratures sizes are 1-26, 30, 35, 40, 45, 50 (throws otherwise)
 * 
 * @param size: The number of nodes
 * @param N: The true atomic number
 *      The radial scaling parameter will be looked up in gridparam.h
 **/
static std::shared_ptr<RadialGrid> build_by_N(
    const std::string& name,
    size_t size,
    int N);

/**
 * Build an explicit RadialGrid with known nodes/weights/etc
 *  
 * @param name the name of the radial grid scheme
 * @param R the radial size parameter (e.g., the atomic radius)
 * @param rw (size, 2) Tensor with radial nodes/weights on each row. By
 *  convention, radial node increases with row.
 * @return the RadialGrid build from explicit rules
 **/
static std::shared_ptr<RadialGrid> build_explicit(
    const std::string& name,
    double R,
    const std::shared_ptr<Tensor>& rw) {
    return std::shared_ptr<RadialGrid>(new RadialGrid(
        name,
        R,  
        rw)); }

// => Accessors <= //

// The name of the Radial grid scheme (e.g., BECKE) 
std::string name() const { return name_; }
// The radial size parameter
double R() const { return R_; }
// Return the number of points in this Radial grid
size_t size() const;
// The grid points, a (size,2) Tensor with (rho,w) on each row. By convention, rho increases with row.
std::shared_ptr<Tensor> rw() const { return rw_; }
// A handy string representation of the object
std::string string() const;

private:

RadialGrid(
    const std::string& name,
    double R,
    const std::shared_ptr<Tensor> rw) :
    name_(name),
    R_(R),
    rw_(rw) 
    {}

std::string name_;
double R_;
std::shared_ptr<Tensor> rw_;
    
static std::shared_ptr<RadialGrid> build_becke(
    size_t size,
    double a);
static std::shared_ptr<RadialGrid> build_handy(
    size_t size,
    double a);
static std::shared_ptr<RadialGrid> build_ahlrichs(
    size_t size,
    double a);
static std::shared_ptr<RadialGrid> build_knowles(
    size_t size,
    double a);
static std::shared_ptr<RadialGrid> build_multiexp(
    size_t size,
    double a);

};

/**
 * Class AtomGrid provides a pruned quadrature to integrate R^3
 * around a given atomic center, with a given grid orientation.
 **/
class AtomGrid {
    
public:

/**
 * Verbatim constructor fills fields and indexing arrays below.
 **/
AtomGrid(
    int N,
    double x,
    double y,
    double z,
    const std::shared_ptr<Tensor>& orientation,
    const std::shared_ptr<RadialGrid>& radial,
    const std::vector<std::shared_ptr<LebedevGrid> >& spherical) :
    N_(N),
    x_(x),
    y_(y),
    z_(z),
    orientation_(orientation),
    radial_(radial),
    spherical_(spherical)
    {
        if (radial_->size() != spherical_.size()) throw std::runtime_error("AtomGrid: Radial and Lebedev sizes do not match.");
        size_ = 0;
        for (size_t P = 0; P < spherical_.size(); P++) {
            spherical_sizes_.push_back(spherical_[P]->size());
            spherical_starts_.push_back(size_);
            size_ += spherical_[P]->size();
        }
    }

       
/// The true atomic number of the element, e.g., H => 1
int N() const { return N_; }
/// The x center of the grid
double x() const { return x_; }
/// The y center of the grid
double y() const { return y_; }
/// The z center of the grid
double z() const { return z_; }
/** 
 * The 3 x 3 rotation matrix to standard molecular orientation
 * (possibly can be different for each atom)
 * By convention: xrotated_Au = xoriginal_At * O_tu
 **/
std::shared_ptr<Tensor> orientation() const { return orientation_; }
/// The radial quadrature
std::shared_ptr<RadialGrid> radial() const { return radial_; }
/// The spherical quadrature at each radial node
const std::vector<std::shared_ptr<LebedevGrid> >& spherical() const { return spherical_; }

/// Total number of points in the atomic quadrature
size_t size() const { return size_; }
/// Number of radial nodes 
size_t radial_size() const { return radial_->size(); }
/// Maximum number of spherical nodes
size_t max_spherical_size() const;
/// Is this grid pruned? E.g., are all spherical sizes not equal to max_spherical_size?
bool is_pruned() const;
/// Number of spherical nodes for each radial node
const std::vector<size_t>& spherical_sizes() const { return spherical_sizes_; }
/// Starting total index for each radial node (sum of spherical_sizes for previous nodes)
const std::vector<size_t>& spherical_starts() const { return spherical_starts_; }
/**
 * Total index in atomic quadrature, given a radial index and spherical index
 * @param radial_index
 * @param spherical_index
 * The complexity of this operation is O(1) in the radial grid size
 **/
size_t atomic_index(
    size_t radial_index, 
    size_t spherical_index) const;
/** 
 * Radial index for a given total atomic index
 * @param atomic_index
 * The complexity of this operation is O(N) in the radial grid size
 **/
size_t radial_index(
    size_t atomic_index) const;
/** 
 * Spherical index for a given total atomic index
 * @param atomic_index
 * The complexity of this operation is O(N) in the radial grid size
 **/
size_t spherical_index(
    size_t atomic_index) const;

/// A handy string representation of the object
std::string string() const;

/**
 * Return the Tensor of xyzw coordinates and weights of this AtomGrid in Cartesian. 
 * @return (size,4) Tensor with (x,y,z,w) on each row, computed on-the-fly.
 **/
std::shared_ptr<Tensor> xyzw() const;

private:

int N_;
double x_;
double y_;
double z_;
std::shared_ptr<Tensor> orientation_;
std::shared_ptr<RadialGrid> radial_;
std::vector<std::shared_ptr<LebedevGrid> > spherical_; 

size_t size_;
std::vector<size_t> spherical_sizes_;
std::vector<size_t> spherical_starts_;

};

/**
 * BeckeGrid provides a scheme to integrate R^3 around a molecule,
 * including atomic quadrature rules, orientation rules, and atomic
 * partitioning rules
 **/
class BeckeGrid {

public:

/**
 * Verbatim constructor, fills data fields below
 * Resources is used only to construct atomic weights, and is not stored with class
 **/ 
BeckeGrid(
    const std::shared_ptr<ResourceList>& resources,
    const std::string& name,
    const std::string& atomic_scheme,
    const std::vector<std::shared_ptr<AtomGrid> >& atomic
    );

/// The name of the molecular grid scheme, e.g., SG-0, SG-1, User
std::string name() const { return name_; }
/// The name of the atomic weighting scheme, one of: "BECKE", "FLAT", "AHLRICHS"
std::string atomic_scheme() const { return atomic_scheme_; }
/// The vector of AtomGrids (one-to-one with atoms in molecule)
const std::vector<std::shared_ptr<AtomGrid> >& atomic() const { return atomic_; }

/// The total number of nodes in the quadrature
size_t size() const { return size_; }
/// The total number of atoms in the quadrature (including ghosts)
size_t natom() const { return atomic_.size(); } 
/// The total number of nodes per atom
const std::vector<size_t>& atomic_sizes() const { return atomic_sizes_; }
/// The starting total index per atom (sum of previous atomic sizes)
const std::vector<size_t>& atomic_starts() const { return atomic_starts_; }
/// The radial size per atom
const std::vector<size_t>& radial_sizes() const { return radial_sizes_; }
/// The vector of spherical sizes per atom
const std::vector<std::vector<size_t> >& spherical_sizes() const { return spherical_sizes_; }
/**
 * Total index in molecular quadrature, given an atomic index, a radial index, and a spherical index
 * @param atomic_index
 * @param radial_index
 * @param spherical_index
 * The complexity of this operation is O(1) in the number of atoms
 **/
size_t total_index(
    size_t atomic_index,
    size_t radial_index, 
    size_t spherical_index) const;
/** 
 * Atomic index for a given total molecular index
 * @param total_index
 * The complexity of this operation is O(N) in the number of atoms
 * and then O(N) in the radial grid size
 **/
size_t atomic_index(
    size_t total_index) const; 
/** 
 * Radial index for a given total molecular index
 * @param total_index
 * The complexity of this operation is O(N) in the number of atoms
 * and then O(N) in the radial grid size
 **/
size_t radial_index(
    size_t total_index) const; 
/** 
 * Spherical index for a given total molecular index
 * @param total_index
 * The complexity of this operation is O(N) in the number of atoms
 * and then O(N) in the radial grid size
 **/
size_t spherical_index(
    size_t total_index) const; 

/// (size_,) vector with ind
const std::vector<int>& atomic_inds() const { return atomic_inds_; }

/// Maximum number of spherical nodes
size_t max_spherical_size() const;
/// Maximum number of radial nodes
size_t max_radial_size() const;
/// Maximum number of atomic nodes
size_t max_atomic_size() const;
/// Is this grid pruned? E.g., are all spherical sizes not equal to max_spherical_size?
bool is_pruned() const;
/// A handy string representation of this object
std::string string() const;

// => Grid Manipulation Routines <= //

// (size, 4) Tensor with x,y,z,w contents of grid including atomic weights (returns internal reference)
std::shared_ptr<Tensor> xyzw() const { return xyzw_; }
// (size, 3) Tensor with x,y,z contents of grid
std::shared_ptr<Tensor> xyz() const;
// (size, 4) Tensor with x,y,z,w contents of grid before atomic weights
std::shared_ptr<Tensor> xyzw_raw() const;

/**
 * Compute the atomic weight contribution to the  gradient for a potential field
 * on this quadrature grid
 *      
 *  G_{A} += \sum_{P} (\partial_{A} w_P) v_P 
 *
 * Implementation: CPU only.
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param v potential field (size,) Tensor
 * @param GA [optional] a (nA,3) Tensor accumulate the gradient contributions 
 *  into. If not provided, this will be allocated.
 * @return the gradient accumulated into GA
 **/
std::shared_ptr<Tensor> grad(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& v,
    const std::shared_ptr<Tensor>& GA = std::shared_ptr<Tensor>()
    );

private:

std::string name_;
std::string atomic_scheme_;
std::vector<std::shared_ptr<AtomGrid> > atomic_;

std::shared_ptr<Tensor> xyzw_;

size_t size_;
std::vector<size_t> atomic_starts_;
std::vector<size_t> atomic_sizes_;
std::vector<size_t> radial_sizes_;
std::vector<std::vector<size_t> > spherical_sizes_;

std::vector<int> atomic_inds_;

// (nA,nA) Tensor with a_ij size adjustment parameters
std::shared_ptr<Tensor> compute_a() const;
// (nA,nA) Tensor with inverse atomic distances
std::shared_ptr<Tensor> compute_rinv() const;
// (nP,4) Tensor with full x,y,z,w contents of grid including atomic weights
std::shared_ptr<Tensor> compute_xyzw(
    const std::shared_ptr<ResourceList>& resources) const;

// (nA,) Tensor with x,y,z positions of atoms
std::shared_ptr<Tensor> compute_xyz_atoms() const;

};

} // namespace lightspeed

#endif
