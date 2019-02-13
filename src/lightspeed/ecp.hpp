#ifndef LS_ECP_HPP
#define LS_ECP_HPP

#include <memory>
#include <string>
#include <vector>

namespace lightspeed {

class Tensor;

/**!
 * Class ECPShell is a data container for a simple ECP basis shell
 *
 * - Rob Parrish, 8 November, 2016
 **/
class ECPShell {

public:

/**
 * Verbatim Constructor: fills fields below (see accessor fields for details)
 **/
ECPShell(
    double x,
    double y,
    double z,
    int L,
    bool is_max_L,
    const std::vector<int>& ns,
    const std::vector<double>& cs,
    const std::vector<double>& es,
    size_t atomIdx,
    size_t shellIdx) : 
    x_(x),
    y_(y),
    z_(z),
    L_(L),
    is_max_L_(is_max_L),
    ns_(ns),
    cs_(cs),
    es_(es),
    atomIdx_(atomIdx),
    shellIdx_(shellIdx)
    {}

// => Accessors <= //

/// X position of shell (atomic center)
double x() const { return x_; }
/// Y position of shell (atomic center)
double y() const { return y_; }
/// Z position of shell (atomic center)
double z() const { return z_; }

/// Angular momentum of this shell
int L() const { return L_; }
/// Is this the maximum angular momentum on this atomic center?
bool is_max_L() const { return is_max_L_; }
/// Number of primitive Gaussians in this shell
size_t nprim() const { return es_.size(); }
/// The list of integral powers of contractions
const std::vector<int>& ns() const { return ns_; }
/// The list of primitive contraction coefficients
const std::vector<double>& cs() const { return cs_; }
/// The list of primitive Gaussian exponents
const std::vector<double>& es() const { return es_; }

/// Index of the atom this shell is centered on within all atoms in this basis set
size_t atomIdx() const { return atomIdx_; }
/// Index of this shell within its containing basis set
size_t shellIdx() const { return shellIdx_; }

// => Equivalence (Needed for Python) <= //

bool operator==(const ECPShell& other) const {
    return (
        x_ == other.x_ &&
        y_ == other.y_ &&
        z_ == other.z_ &&
        L_ == other.L_ &&
        is_max_L_ == other.is_max_L_ &&
        ns_ == other.ns_ &&
        cs_ == other.cs_ &&
        es_ == other.es_); }
bool operator!=(const ECPShell& other) const {
    return !((*this)==other);
}

private:
    
double x_;
double y_;
double z_;
bool is_max_L_;
int L_;
std::vector<int> ns_;
std::vector<double> cs_;
std::vector<double> es_;

size_t atomIdx_;
size_t shellIdx_;
};

/**!
 * Class ECPBasis is a simple wrapper around a list of atom-affiliated
 * ECPShells, plus some utility functions.
 *
 * - Rob Parrish, 8 November, 2016
 **/
class ECPBasis {

public:

// => Constructors <= //

/**!
 * Verbatim constructor, copies fields below.
 *
 * @param name the name of the Basis
 * @param shells the Gaussian shells in this Basis
 * @param nelecs the number of electrons per atom represented by ECP
 *
 * The natom() field below is always taken from the nelecs() vector. So one
 * need not worry about the holes in the indexing for atoms which do not have
 * ECPs, particularly for those at the end of the atom list.
 *
 * This checks the validity of/uses the ECPShell index fields
 **/
ECPBasis(
    const std::string& name,
    const std::vector<ECPShell>& shells,
    const std::vector<int>& nelecs);

/// Default constructor, no initialization
ECPBasis() {}

// => Accessors <= //

/// The basis set's name
std::string name() const { return name_; }
/// The complete list of shells
const std::vector<ECPShell>& shells() const { return shells_; }
/// The mapping from atom index and shell index within that atom to absolute shell index
const std::vector<std::vector<size_t>>& atoms_to_shell_inds() const { return atoms_to_shell_inds_; }
/// The number of electrons represented by each atom
const std::vector<int>& nelecs() const { return nelecs_; }

/// Total number of atoms in this basis set
size_t natom() const { return atoms_to_shell_inds_.size(); }
/// Total number of shells in this basis set
size_t nshell() const { return shells_.size(); }
/// Total number of primitives in this basis set
size_t nprim() const { return nprim_; }
/// Total number of electrons represented in this basis set
size_t nelec() const { return nelec_; }

/// Maximum angular momentum across all shells
int max_L() const;
/// Maximum number of primitives across all shells
size_t max_nprim() const;

/// Get a string representation of this basis set
std::string string() const;

// => Helper Methods <= //

/**
 * Determine the atomic coordinates present in this ECPBasis
 * @return a (natom,3) Tensor with the atomic coordinates
 * NOTE: If an atom has no ECPShell objects affiliated with it, the
 * coordinate of the atom is unknown, and will be returned as <0,0,0>.
 **/
std::shared_ptr<Tensor> xyz() const;

/**
 * Produce a new version of this Basis object with updated coordinates
 * @param xyz (natom,3) Tensor with new atomic coordinates
 * @return the new Basis at updated coordinates
 **/
std::shared_ptr<ECPBasis> update_xyz(
    const std::shared_ptr<Tensor>& xyz) const;

/**
 * Extract a basis set which is constructed from a subset of the atoms in
 * the current basis set.
 * 
 * @param atom_range the indices of the atoms used to form the subset basis
 * @return a ECPBasis which is the desired subset of the current basis 
 **/
std::shared_ptr<ECPBasis> subset(
    const std::vector<size_t>& atom_range) const;

/**
 * Create a basis set which is constructed from the union of the atoms in the 
 * sub-basis sets.
 *
 * @param bases the sub basis sets
 * @return a ECPBasis which is the desired union of the sub basis sets
 **/
static std::shared_ptr<ECPBasis> concatenate(
    const std::vector<std::shared_ptr<ECPBasis>>& bases);

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
    const std::shared_ptr<ECPBasis>& basis1,
    const std::shared_ptr<ECPBasis>& basis2) {
    if (basis1->natom() != basis2->natom()) return false;
    if (basis1->nshell() != basis2->nshell()) return false;
    if (basis1->nprim() != basis2->nprim()) return false;
    if (basis1->nelec() != basis2->nelec()) return false;
    if (basis1->nelecs() != basis2->nelecs()) return false;
    return (basis1->shells() == basis2->shells()); 
    }

protected:

std::string name_;
std::vector<ECPShell> shells_;
std::vector<std::vector<size_t>> atoms_to_shell_inds_;
std::vector<int> nelecs_;

size_t nelec_;
size_t nprim_;

};

} // namespace lightspeed

#endif
