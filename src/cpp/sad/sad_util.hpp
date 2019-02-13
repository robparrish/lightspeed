#ifndef LS_SAD_UTIL_HPP
#define LS_SAD_UTIL_HPP

#include <vector>
#include <stdexcept>
    
namespace lightspeed {

/**
 * Class SADAtom represents the L shells and frozen/active status for SAD
 * operations. This class is provided to de-mystify the process of SAD
 * occupation and MinAO basis set composition somewhat.
 *
 * E.g, the rules for O atom (atomic number N = 8) can be obtained by
 * considering.
 *
 *  const SADAtom& atomO = SADAtom::get(8);
 *
 * Inspecting the fields we see:
 *
 *  int ON = atomO.N(); // 8, obviously
 *  const std::vector<int>& OLs = atomO.Ls(); // { 0, 0, 1 }, meaning 1s, 2s, 2p
 *  const std::vector<bool>& Oacts = atomO.acts(); // { false, true, true }, meaning 1s is doubly occupied, and 2s/2p are fractionally occupied.
 **/
class SADAtom {

public:

// => Accessors <= //

/// The atomic number
int N() const { return N_; }
/// The vector of angular momentum quantum numbers of the atoms
const std::vector<int>& Ls() const { return Ls_; }
/// The vector of frozen core (false) or active/fractional (true) flags
const std::vector<bool>& acts() const { return acts_; }
/// The number of frozen core orbitals (sum of 2 * L + 1 for act == false)
int nfrzc() const;
/// The number of active/fractional orbitals (sum of 2 * L + 1 for act == true)
int nact() const;
/// The total number of SAD orbitals for this atom
int ntot() const { return nfrzc() + nact(); }

// A handy string representation of the object
std::string string() const;

// => Compute Methods <= //

/**
 * Compute the fractional occupation numbers for this atom, given a desired
 * total number of electron pairs npair. 
 *
 * Occupation Rules:
 *  if npair == 0.0 (a ghost atom), a vector of 0.0 is returned
 *  if nfrzc <= npair <= nfrzc + nact (the usual spot), nfrzc orbitals are
 *   frozen to occupation of 1.0, and nact orbitals are fractionally occupied
 *   with occupation of (npair - nfrzc) / nact
 *  if npair < nfrzc (deep cations), nfrzc orbitals are frozen to occupation of
 *   1.0.
 *  if npair > nfrac + nact (deep anions), nfrzc + nact are frozen to
 *   occupation of 1.0.
 * In the last two cases, the user is in such unknown territory that the
 * heuristic of SAD breaks down, and the user is usually better off manually
 * specifying occupations.
 *
 * @param npair the desired number of electron pairs to put in this atom
 * @return nocc a (ntot,) size vector with the orbital occupations in terms of
 * electron pairs.
 **/
std::vector<double> nocc(double npair) const;

// => Singleton Instance <= //

/**
 * Get a reference to the SADAtom rules for atom number N
 * @param N the atomic number
 * @return the reference to the SADAtom rules
 * throws if N is outside the supported range.
 **/
static
const SADAtom& get(int N);

static
std::string print_atoms();

private:

int N_;
std::vector<int> Ls_;
std::vector<bool> acts_;

static std::vector<SADAtom> atom_list__;
static std::vector<SADAtom> build_atom_list();

/**
 * Verbatim constructor, fills fields
 **/
SADAtom(
    int N,
    const std::vector<int>& Ls,
    const std::vector<bool>& acts) :
    N_(N),
    Ls_(Ls),
    acts_(acts)
    {
        if (Ls_.size() != acts_.size()) throw std::runtime_error("SADAtom: Ls/acts do not have same size");
    }

};

} // namespace lightspeed

#endif
