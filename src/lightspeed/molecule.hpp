#ifndef LS_MOLECULE_HPP
#define LS_MOLECULE_HPP

#include <memory>
#include <cstddef>
#include <string>
#include <vector>

namespace lightspeed {

class Tensor;

// TODO: (minor) mildly expand documentation for this file

/**!
 * Class Atom is a data container for a simple atom
 *
 *  To specify the atom type, three fields are used (mostly for convenience)
 *  - label the user-specified label, e.g., He3 or Gh(He3) (used for printing)
 *  - symbol the capitalized atomic symbol, e.g., HE (or X for dummy)
 *  - N the integral atomic number (or 0 for dummy)
 *
 * To specify coordinates, the usual fields are:
 *  - x the x coordinate in au
 *  - y the y coordinate in au
 *  - z the z coordinate in au
 *
 * To specify the nuclear charge
 *  - Z the nuclear charge, e.g., the number of protons
 *
 * The index field is a back-reference to the atom's position in its parent
 * molecule, and is used in placing data pertaining to the molecule's atom in
 * various arrays.
 *
 * - Rob Parrish, 15 February, 2015
 **/
class Atom {

public:

// => Constructors <= //

/// Verbatim constructor, fills fields below
Atom(
    const std::string& label,
    const std::string& symbol,
    int N,
    double x,
    double y,
    double z,
    double Z,
    int atomIdx) :
    label_(label),
    symbol_(symbol),
    N_(N),
    x_(x),
    y_(y),
    z_(z),
    Z_(Z),
    atomIdx_(atomIdx)
    {}

// => Accessors <= //

/// User-specified atom label, e.g He4 or Gh(He), for printing
std::string label() const { return label_; }
/// Capitalized atomic symbol, e.g., HE (or X for dummy)
std::string symbol() const { return symbol_; }
/// True atomic number of element (or 0 for dummy)
int N() const { return N_; }
/// x coordinate in au
double x() const { return x_; }
/// y coordinate in au
double y() const { return y_; }
/// z coordinate in au
double z() const { return z_; }
/// Nuclear charge in au (might be not equal to N)
double Z() const { return Z_; }
/// Index of this atom within its containing molecule
int atomIdx() const { return atomIdx_; }

// => Methods <= //

/// Return the distance to Atom other in au
double distance(const Atom& other) const;

// => Equivalence (Needed for Python) <= //

bool operator==(const Atom& other) const {
    return (
        label_ == other.label_ &&
        symbol_ == other.symbol_ &&
        N_ == other.N_ &&
        x_ == other.x_ &&
        y_ == other.y_ &&
        z_ == other.z_ &&
        Z_ == other.Z_ &&
        atomIdx_ == other.atomIdx_);
}
bool operator!=(const Atom& other) const {
    return !((*this)==other);
}

private:
    
std::string label_;
std::string symbol_;
int N_;
double x_;
double y_;
double z_;
double Z_;
int atomIdx_;

};

/**!
 * Class Molecule is a simple wrapper around a vector of atoms, plus some utility
 * functions. The class is mostly immutable, to prevent clever bastards
 * from reorienting your molecule halfway through the computation!
 *
 * Our required convention is that the Molecule be a collection of strictly
 * dense atoms, e.g., atoms[A].atomIdx() == A.
 *
 * - Rob Parrish, 15 February, 2015
 **/
class Molecule {

public:

// => Constructors <= //

/**!
 * Verbatim constructor, copies fields below.
 *
 * Throws if atomIdx field in atoms do not match vector indices in atoms.
 **/
Molecule(
    const std::string& name,
    const std::vector<Atom>& atoms,
    double charge,
    double multiplicity
    );

// => Accessors <= //

/// The molecule's name (this is a convenience field, not used in Lightspeed)
std::string name() const { return name_; }
/// Number of atoms in this molecule
size_t natom() const { return atoms_.size(); }
/// The array of atoms which comprise this molecule
const std::vector<Atom>& atoms() const { return atoms_; }
/// The total molecule charge, as set by the user (this is a convenience field, not used in Lightspeed)
double charge() const { return charge_; }
/// The multiplicity, as set by the user (this is a convenience field, not used in Lightspeed)
double multiplicity() const { return multiplicity_; }

/// Set the name field
void set_name(const std::string& name) { name_ = name; }
/// Set the charge field
void set_charge(double charge) { charge_ = charge; }
/// Set the multiplicity field
void set_multiplicity(double multiplicity) { multiplicity_ = multiplicity; }

/// Get a string representation of the molecule
std::string string(bool print_coords = false) const;

// => Methods <= //

/// Total nuclear charge
double nuclear_charge() const;
/// Nuclear center-of-mass (3,)
std::shared_ptr<Tensor> nuclear_COM() const;
/// Nuclear inertial tensor (3,3), relative to nuclear center-of-mass
std::shared_ptr<Tensor> nuclear_I() const;

/// Return the nuclear repulsion energy in au for this molecule
double nuclear_repulsion_energy() const;
/// Return the first derivative of the nuclear repulsion energy in au for this molecule
std::shared_ptr<Tensor> nuclear_repulsion_grad() const;
/// Return the nuclear repulsion energy in au for this molecule
double nuclear_repulsion_energy_other(const std::shared_ptr<Molecule>& other) const;
// TODO: (minor) nuclear_repulsion_grad_other

// => Helper Methods <= //

/**
 * Determine the atomic coordinates present in this Molecule
 * @return a (natom,3) Tensor with the atomic coordinates
 **/
std::shared_ptr<Tensor> xyz() const;

/**
 * Determine the nuclear charges present in this Molecules
 * @retun a (natom,) Tensor with the nuclear charges
 **/
std::shared_ptr<Tensor> Z() const;

/**
 * Determine the atomic coordinates and nuclear charges present in this Molecule
 * @return a (natom,4) Tensor with the atomic coordinates and nuclear charges
 **/
std::shared_ptr<Tensor> xyzZ() const;

/**
 * Produce a new version of this Molecule object with updated coordinates
 * @param xyz (natom,3) Tensor with new atomic coordinates
 * @return the new Molecule at updated coordinates
 **/
std::shared_ptr<Molecule> update_xyz(
    const std::shared_ptr<Tensor>& xyz) const;

/**
 * Produce a new version of this Molecule object with updated charges
 * @param Z (natom,) Tensor with new nuclear charges
 * @return the new Molecule at updated charges
 **/
std::shared_ptr<Molecule> update_Z(
    const std::shared_ptr<Tensor>& Z) const;

/**
 * Extract a molecule which is constructed from a subset of the atoms in
 * the current molecule.
 *
 * The natom field of the new molecule is atom_range.size(). The atomIdx fields
 * of the atoms of the new molecule appear in dense increasing order.
 * 
 * @param atom_range the indices of the atoms used to form the subset molecule
 * @param charge the charge of the new molecule 
 * @param multiplicity the multiplicty of the new molecule
 * @return a Molecule which is the desired subset of the current molecule
 **/
std::shared_ptr<Molecule> subset(
    const std::vector<size_t>& atom_range,
    double charge,
    double multiplicty) const;

/**
 * Create a molecule which is constructed from the union of the atoms in the 
 * sub-molecules.
 *  
 * The natom field of the new molecule is the sum of the natom fields of the
 * sub-molecules. The atomIdx fields of the atoms of each subfragment are
 * offset set by the sum of the natom fields in the preceeding submolecules.
 *
 * @param mols the submolecules
 * @param charge the charge of the new molecule 
 * @param multiplicity the multiplicty of the new molecule
 * @return a Molecule which is the desired union of the submolecules
 **/
static std::shared_ptr<Molecule> concatenate(
    const std::vector<std::shared_ptr<Molecule> >& mols,
    double charge,
    double multiplicity);

/**
 * Are Molecules mol1 and mol2 computationally identical (same natom and same
 * contents of Atoms)? The name/charge/multiplicity flags are not checked.
 *
 * @param mol1 the first Molecule in the equivalency check
 * @param mol2 the second Molecule in the equivalency check
 * @return true if the Molecule objects will give computationally identical
 * results, false otherwise.
 **/
static
bool equivalent(
    std::shared_ptr<Molecule>& mol1,
    std::shared_ptr<Molecule>& mol2) {
    if (mol1->natom() != mol2->natom()) return false;
    return mol1->atoms() == mol2->atoms(); 
    }

private:

std::string name_;
std::vector<Atom> atoms_;
double charge_;
double multiplicity_;

};

} // namespace lightspeed

#endif
