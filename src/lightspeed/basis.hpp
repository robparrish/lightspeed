#ifndef LS_BASIS_HPP
#define LS_BASIS_HPP

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace lightspeed {

class Tensor;

/**
 * Class Primitive represents a primitive Gaussian shell:
 *
 *  \phi_{p} (\vec r_1) = c_p 
 *  (x_1 - x_p)^l_p (x_1 - y_p)^m_p (z_1 - z_p)^n_p 
 *  \exp(-e_p * r_1p^2)
 *
 * The fields c,e,x,y,z below contain the real parameters c_p,e_p,x_p,y_p,z_p
 * in double precision, respectively.
 *
 * In many cases, there is a traditional "pre-normalized" way of reporting the
 * contraction coefficient, which we denote c0_p. E.g., this is the contraction
 * coefficient encountered in a Gaussian 94 or other basis set file from the
 * EMSL basis set exchange. We provide the field c0 to hold this value, as it
 * is not recoverable from the c and e fields alone.
 *
 * A Primitive shell contains a number of primitive functions according to its
 * total angular momentum:
 *      
 *  L_p = l_p + m_p + n_p
 *
 * In working code, we typically deal with the shells in "cartesian"
 * representation, in which each shell contains ncart = (L_p + 1) * (L_p + 2) /
 * 2 functions. In many basis sets, some or (usually) all of the shells are
 * defined in a "pure" representation of spherical harmonics (particularly real
 * solid hamonics). This representation removes contamination of lower L
 * components from the cartesian basis. For example, in a L_p = 2 "D" shell,
 * the linear combination x^2 + y^2 + z^2 actually lives in L = 0 rather than L
 * = 2, so it is removed in the pure spherical harmonic basis set. This reduces
 * the number of pure D functions to 5, compared to 6 with cartesians. In
 * general, there are npure = 2 * L_p + 1 spherical harmonic functions in a
 * given shell. A short table of the relevant sizes is:
 *
 *  Name L_p ncart npure
 *     S   0     1     1
 *     P   1     3     3
 *     D   2     6     5
 *     F   3    10     7
 *     G   4    15     9
 *
 * In cartesians, we use the CCA (lexical) ordering and normalization:
 *
 *  Name L_p l_p m_p n_p <p|p>
 *     S   0   0   0   0     1   
 *     P   1   1   0   0     1
 *             0   1   0     1
 *             0   0   1     1 
 *     D   2   2   0   0     1 
 *             1   1   0   1/3  
 *             1   0   1   1/3  
 *             0   2   0     1
 *             0   1   1   1/3
 *             0   0   2     1
 *     F   3   3   0   0     1 
 *             2   1   0   1/5  
 *             2   0   1   1/5  
 *             1   2   0   1/5
 *             1   1   1  1/15
 *             1   0   2   1/5
 *             0   3   0     1
 *             0   2   1   1/5
 *             0   1   2   1/5
 *             0   0   3     1
 *     G   4   4   0   0     1 
 *             3   1   0   1/7 
 *             3   0   1   1/7 
 *             2   2   0  3/35
 *             2   1   1  1/35
 *             2   0   2  3/35
 *             1   3   0   1/7
 *             1   2   1  1/35
 *             1   1   2  1/35
 *             1   0   3   1/7 
 *             0   4   0     1
 *             0   3   1   1/7
 *             0   2   2  3/35   
 *             0   1   3   1/7 
 *             0   0   4     1
 *      ...
 *
 * As there is only one contraction coefficient c_p across all functions within
 * a shell, some of the functions have different relative normalizations than
 * others, as indicated in the <p|p> field of the table above.
 *
 * In this code, individual shells are permitted to be either pure or cartesian
 * via the is_pure field, so a full Basis object may contain shells that are
 * both pure and cartesian (note though that all primitives with the same
 * shellIdx must have the same is_pure field). 
 *
 * When building PairList objects, we will often exploit the symmetry between
 * two identical Basis objects to discard the upper triangular PrimPairs. To do
 * so safely, we need to avoid double or half counting the diagonal. It is
 * therefore handy to know the absolute index of each Primitive in the Basis.
 * This is provided in the primIdx field.
 *
 * In user facing code (e.g., the finished overlap, potential, etc matrices),
 * the basis set appears in its desired spherical/cartesian representation. It
 * is therefore handy to know where each Primitive's 0-th function starts in
 * the absolute Basis index, including mixed spherical/cartesian
 * considerations. This is provided in the aoIdx field.
 *
 * In internal code, we usually transform the matrices wholly into Cartesian
 * representation, which generally will change the 0-th function start indices
 * for each Primitive. It is therefore handy to know where each Primitive's
 * 0-th function would start in a Basis using entirely cartesian shells. This
 * is provided in the cartIdx field.
 *
 * To reduce the memory requirements of various matrices (overlap, potential,
 * etc) as well as the linear algebra stages of electronic structure theory,
 * primitive shells are traditionally grouped together into "contracted"
 * shells. In this code, such contractions are indicated by common occurrences
 * of the shellIdx field. All primitives with the same shellIdx field must have
 * the same L and is_pure fields. Usually, all primitives with the same
 * shellIdx field should have the same x, y, z, aoIdx, cartIdx,and atomIdx
 * fields, but there are special cases where this does not have to be the case
 * (e.g., tessellated basis sets in periodic boundary conditions).
 *
 * Traditionally, several contracted shells are placed on each atom in a
 * molecule, and these shells move with that atom. It is therefore handy to
 * have an index to the atom for use in computing contributions to the
 * molecular gradient due to the motions of the basis functions. This is
 * provided in the atomIdx field.
 **/
class Primitive {

public:

/**
 * Verbatim Constructor: fills fields below (see accessor fields for details)
 **/
Primitive(
    double c,
    double e,
    double x,
    double y,
    double z,
    double c0,
    int L,
    bool is_pure,
    int aoIdx,
    int cartIdx,
    int primIdx,
    int shellIdx,
    int atomIdx
    ) :
    c_(c), 
    e_(e), 
    x_(x), 
    y_(y), 
    z_(z), 
    c0_(c0), 
    L_(L),
    is_pure_(is_pure),
    aoIdx_(aoIdx),
    cartIdx_(cartIdx),
    primIdx_(primIdx),
    shellIdx_(shellIdx),
    atomIdx_(atomIdx)
    {}

// => Accessors <= //

// The primitive contraction coefficient c_p
double c() const { return c_; }
// The primitive exponent e_p
double e() const { return e_; }
// The primitive x coordinate x_p
double x() const { return x_; }
// The primitive y coordinate y_p
double y() const { return y_; }
// The primitive z coordinate z_p
double z() const { return z_; }
// The pre-normalization contraction coefficient c0_p.
// (This is typically not used computationally, but is needed for Molden files
// and a few other places. Moreover, it is not recoverable from c_p, so we keep
// the value here.)
double c0() const { return c0_; }

// The total angular momentum (0 - S, 1 - P, 2 - D, 3 - F, 4 - G, etc)
int L() const { return L_; }
// Is this shell using pure spherical harmonics (true) or cartesians (false)
bool is_pure() const { return is_pure_; } 

// The true atomic orbital index (in spherical and/or cartesian representation) for the 0-th function in this shell
int aoIdx() const { return aoIdx_; }
// The atomic orbital index if basis is represented entirely in cartesians (>= aoIdx) for the 0-th function in this shell
int cartIdx() const { return cartIdx_; }
// The absolute index of the primitive within the basis set
int primIdx() const { return primIdx_; }
// The absolute index of the shell within the basis set
int shellIdx() const { return shellIdx_; }
// The absolute index of the atom within the molecule
int atomIdx() const { return atomIdx_; }

// Number of functions this shell covers in its desired spherical harmonic or cartesian representation
int nao() const { return (is_pure_ ? npure() : ncart()); }
// Number of functions this shell covers in a pure spherical harmonic representation (2L+1)
int npure() const { return 2 * L_ + 1; }
// Number of functions this shell covers in a cartesian representation (L+1) * (L+2) / 2
int ncart() const { return (L_ + 1) * (L_ + 2) / 2; }

// A handy string representation of the object
std::string string() const;

// => Equivalence (for Python) <= //

bool operator==(const Primitive& other) const {
    if (c_ != other.c_) return false;
    if (e_ != other.e_) return false;
    if (x_ != other.x_) return false;
    if (y_ != other.y_) return false;
    if (z_ != other.z_) return false;
    if (c0_ != other.c0_) return false;
    if (L_ != other.L_) return false;
    if (is_pure_ != other.is_pure_) return false;
    if (aoIdx_ != other.aoIdx_) return false;
    if (cartIdx_ != other.cartIdx_) return false;
    if (primIdx_ != other.primIdx_) return false;
    if (shellIdx_ != other.shellIdx_) return false;
    if (atomIdx_ != other.atomIdx_) return false;
    return true;
}

bool operator!=(const Primitive& other) const {
    return !((*this)==other);
}

private:

// => Fields <= //

double c_;
double e_;
double x_;
double y_;
double z_;
double c0_;

int L_;
bool is_pure_;

int aoIdx_;
int cartIdx_;
int primIdx_;
int shellIdx_;
int atomIdx_;
    
};

/**
 * Class Shell represents a contracted Gaussian shell. It contains the
 * essential data of a contracted group of Primitives with common Shell
 * indices. Note that typically, one perfers working with the uncontracted
 * Primitive representation of a Basis, but in some circumstances the Shell
 * representation is required.
 **/
class Shell {

public:

/**
 * Verbatim Constructor: fills fields below (see accessor fields for details)
 **/
Shell(
    const std::vector<double>& cs,
    const std::vector<double>& es,
    double x,
    double y,
    double z,
    const std::vector<double>& c0s,
    int L,
    bool is_pure,
    int aoIdx,
    int cartIdx,
    int primIdx,
    int shellIdx,
    int atomIdx
    ) :
    cs_(cs), 
    es_(es), 
    x_(x), 
    y_(y), 
    z_(z), 
    c0s_(c0s), 
    L_(L),
    is_pure_(is_pure),
    aoIdx_(aoIdx),
    cartIdx_(cartIdx),
    primIdx_(primIdx),
    shellIdx_(shellIdx),
    atomIdx_(atomIdx)
    {}

/**
 * Return a "zero shell" or "null shell", often used in density fitting. This
 * is a shell with a single s function with exponent of zero and constraction
 * coefficient of one, which serves as a proxy for not writing a Gaussian shell
 * in the given index. E.g., (A|B) = (A0|B0).
 **/
static
Shell zero() { return Shell({1.0}, {0.0}, 0.0, 0.0, 0.0, {1.0}, 0, false, 0, 0, 0, 0, 0); }

// => Accessors <= //

// The primitive contraction coefficients c_p
const std::vector<double>& cs() const { return cs_; }
// The primitive exponents e_p
const std::vector<double>& es() const { return es_; }
// The primitive x coordinate x_p
double x() const { return x_; }
// The primitive y coordinate y_p
double y() const { return y_; }
// The primitive z coordinate z_p
double z() const { return z_; }
// The pre-normalization contraction coefficient c0_p.
// (This is typically not used computationally, but is needed for Molden files
// and a few other places. Moreover, it is not recoverable from c_p, so we keep
// the value here.)
const std::vector<double>& c0s() const { return c0s_; }

// The total angular momentum (0 - S, 1 - P, 2 - D, 3 - F, 4 - G, etc)
int L() const { return L_; }
// Is this shell using pure spherical harmonics (true) or cartesians (false)
bool is_pure() const { return is_pure_; } 

// The true atomic orbital index (in spherical and/or cartesian representation) for the 0-th function in this shell
int aoIdx() const { return aoIdx_; }
// The atomic orbital index if basis is represented entirely in cartesians (>= aoIdx) for the 0-th function in this shell
int cartIdx() const { return cartIdx_; }
// The absolute index of the 0-th primitive in this shell within the basis set
int primIdx() const { return primIdx_; }
// The absolute index of the shell within the basis set
int shellIdx() const { return shellIdx_; }
// The absolute index of the atom within the molecule
int atomIdx() const { return atomIdx_; }

// Number of primitives in this shell
int nprim() const { return es_.size(); }
// Number of functions this shell covers in its desired spherical harmonic or cartesian representation
int nao() const { return (is_pure_ ? npure() : ncart()); }
// Number of functions this shell covers in a pure spherical harmonic representation (2L+1)
int npure() const { return 2 * L_ + 1; }
// Number of functions this shell covers in a cartesian representation (L+1) * (L+2) / 2
int ncart() const { return (L_ + 1) * (L_ + 2) / 2; }

// A handy string representation of the object
std::string string() const;

// => Equivalence (for Python) <= //

bool operator==(const Shell& other) const {
    if (cs_ != other.cs_) return false;
    if (es_ != other.es_) return false;
    if (x_ != other.x_) return false;
    if (y_ != other.y_) return false;
    if (z_ != other.z_) return false;
    if (c0s_ != other.c0s_) return false;
    if (L_ != other.L_) return false;
    if (is_pure_ != other.is_pure_) return false;
    if (aoIdx_ != other.aoIdx_) return false;
    if (cartIdx_ != other.cartIdx_) return false;
    if (primIdx_ != other.primIdx_) return false;
    if (shellIdx_ != other.shellIdx_) return false;
    if (atomIdx_ != other.atomIdx_) return false;
    return true;
}

bool operator!=(const Shell& other) const {
    return !((*this)==other);
}

private:

// => Fields <= //

std::vector<double> cs_;
std::vector<double> es_;
double x_;
double y_;
double z_;
std::vector<double> c0s_;

int L_;
bool is_pure_;

int aoIdx_;
int cartIdx_;
int primIdx_;
int shellIdx_;
int atomIdx_;
    
};
/**
 * Class Basis represents a basis set (a collection of Primitive objects
 * presumably associated with some molecular geometry).
 *
 * The core of a Basis object is simply a vector of Primitive objects that were
 * constructed according to some external procedure (such as putting
 * pre-specified Gaussians on each atom in a molecular geometry). The user
 * provides this data in the primitives field.
 *
 * Our convention is that the Basis be a collection of strictly dense
 * primitives. 
 *  
 * For a given collection of nuclei (nuclear charges and coordinates), there
 * are an extensive series of standard rules for which Primitives to put where
 * (e.g., the named basis sets STO-3G, 6-31G*, cc-pVDZ, etc). It is often handy
 * to keep track of that shorthand name so that we can echo it to the user at a
 * later time. This is provided for by the name field below. Note that this
 * field is not used in computational code, and the basis deliberately does not
 * care how it has been constructed.
 **/
class Basis {

public:

/**
 * Verbatim Constructor: fills fields below (see accessor fields for details)
 **/
Basis(
    const std::string& name,
    const std::vector<Primitive>& primitives) :
    name_(name),
    primitives_(primitives)
    {
        // Validity check
        for (size_t P = 0; P < primitives_.size(); P++) {
            const Primitive& prim = primitives_[P];
            // Primitive indices
            if (prim.primIdx() != P) throw std::runtime_error("Basis: primIdx field is not dense");
            if (P == 0) {
                // 0-th Primitive indices should be 0
                if (prim.aoIdx() != 0) throw std::runtime_error("Basis: aoIdx field is not dense");
                if (prim.cartIdx() != 0) throw std::runtime_error("Basis: cartIdx field is not dense");
                if (prim.shellIdx() != 0) throw std::runtime_error("Basis: shellIdx field is not dense");
                if (prim.atomIdx() != 0) throw std::runtime_error("Basis: atomIdx field is not dense");
            } else {
                // Check 1,2....-th Primitives against their preceeeding neighbors
                const Primitive& priml = primitives_[P-1];
                if (priml.shellIdx() == prim.shellIdx()) {
                    // Same-shell Primitives should have same aoIdx, cartIdx, atomIdx
                    if (priml.aoIdx() != prim.aoIdx()) throw std::runtime_error("Basis: aoIdx field is not dense");
                    if (priml.cartIdx() != prim.cartIdx()) throw std::runtime_error("Basis: cartIdx field is not dense");
                    if (priml.atomIdx() != prim.atomIdx()) throw std::runtime_error("Basis: atomIdx field is not dense");
                } else if (priml.shellIdx() + 1 == prim.shellIdx()) {
                    // Adjacent-shell Primitives should have adjacent aoIdx, cartIdx
                    if (priml.aoIdx() + priml.nao() != prim.aoIdx()) throw std::runtime_error("Basis: aoIdx field is not dense"); 
                    if (priml.cartIdx() + priml.ncart() != prim.cartIdx()) throw std::runtime_error("Basis: cartIdx field is not dense"); 
                    // and same or adjacent atomIdx 
                    if ((priml.atomIdx() != prim.atomIdx()) && (priml.atomIdx() + 1 != prim.atomIdx())) throw std::runtime_error("Basis: atomIdx field is not dense");
                } else {
                    // shellIdx should either be same or adjacent
                    throw std::runtime_error("Basis: shellIdx field is not dense");
                }
            }
        }    
        
        if (primitives_.size()) {
            const Primitive& prim = primitives_[primitives_.size() - 1];
            nao_ = prim.aoIdx() + prim.nao();
            ncart_ = prim.cartIdx() + prim.ncart();
            nshell_ = prim.shellIdx() + 1;
            natom_ = prim.atomIdx() + 1;
            nprim_ = primitives_.size();
        } else {
            nao_ = 0;
            ncart_ = 0;
            nshell_ = 0;
            natom_ = 0;
            nprim_ = 0;
        }
        
        // Memoize shells_ 
        shells_ = build_shells();
    }

// => Accessors <= //

// The name of the Basis object (meta - for user eyes)
std::string name() const { return name_; }
// The list of Primitive shells. No sorting or dense filling is required
const std::vector<Primitive>& primitives() const { return primitives_; }
// The total number of atomic orbitals
size_t nao() const { return nao_; }
// The total number of cartesian orbitals
size_t ncart() const { return ncart_; }
// The total number of primitives
size_t nprim() const { return nprim_; }
// The total number of shells
size_t nshell() const { return nshell_; }
// The total number of atoms
size_t natom() const { return natom_; }

// A handy string representation of the object
std::string string() const;

// => Extra Characteristics <= //

// The maximum L() for any Primitive in primitives()
int max_L() const;
// The maximum nao() for any Primitive in primitives()
int max_nao() const;
// The maximum npure() for any Primitive in primitives()
int max_npure() const;
// The maximum ncart() for any Primitive in primitives()
int max_ncart() const;
// Is any Primitive in primitives pure()?
bool has_pure() const;

// => Helper Methods <= //

/**
 * Determine the atomic coordinates present in this Basis
 * @return a (natom,3) Tensor with the atomic coordinates
 **/
std::shared_ptr<Tensor> xyz() const;

/**
 * Produce a new version of this Basis object with updated coordinates 
 * @param xyz (natom,3) Tensor with new atomic coordinates 
 * @return the new Basis at updated coordinates
 **/
std::shared_ptr<Basis> update_xyz(
    const std::shared_ptr<Tensor>& xyz) const;

/**
 * Extract a Basis which is constructed from a subset of the atoms in
 * the current basis.
 *
 * @param atom_range the indices of the atoms used to form the subset molecule
 * @return a Basis which is the desired subset of the current basis
 **/
std::shared_ptr<Basis> subset(
    const std::vector<size_t>& atom_range) const;

/**
 * Create a Basis which is constructed from the union of the atoms in the 
 * sub-Basis.
 *  
 * The natom field of the new basis is the sum of the natom fields of the
 * sub-bases. The atomIdx fields of the primitives of each subfragment are
 * offset set by the sum of the natom fields in the preceeding subbases
 * (similarly with other fields).
 *
 * @param bases the subbases
 * @return a Basis which is the desired union of the subbases
 **/
static std::shared_ptr<Basis> concatenate(
    const std::vector<std::shared_ptr<Basis> >& bases);

/**
 * Are Basis sets basis1 and basis2 computationally identical (same sizes and
 * same contents of Primitives)? The name flag is not checked.
 *
 * @param basis1 the first Basis in the equivalency check
 * @param basis2 the second Basis in the equivalency check
 * @return true if the Basis objects will give computationally identical
 * results, false otherwise.
 **/
static
bool equivalent(
    const std::shared_ptr<Basis>& basis1,
    const std::shared_ptr<Basis>& basis2) {
    if (basis1->nao() != basis2->nao()) return false;
    if (basis1->ncart() != basis2->ncart()) return false;
    if (basis1->nprim() != basis2->nprim()) return false;
    if (basis1->nshell() != basis2->nshell()) return false;
    if (basis1->natom() != basis2->natom()) return false;
    return (basis1->primitives() == basis2->primitives()); 
    }

/**
 * Generate the Shell representation of this Basis object
 * @return a vector of nshell Shell objects
 **/
const std::vector<Shell>& shells() const { return shells_; }

private:

// => Fields <= //

std::string name_;
std::vector<Primitive> primitives_;
size_t nao_;
size_t ncart_;
size_t nprim_;
size_t nshell_;
size_t natom_;

// Shell representation of this Basis (built by constructor)
std::vector<Shell> shells_;

protected:

std::vector<Shell> build_shells() const;

};


} // namespace lightspeed

#endif
