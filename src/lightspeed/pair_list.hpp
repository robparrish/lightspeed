#ifndef LS_PAIR_LIST_HPP
#define LS_PAIR_LIST_HPP

#include <lightspeed/basis.hpp>
#include <memory>
#include <vector>
#include <stdexcept>

namespace lightspeed {

/**
 * Class Pair represents a pair of primitive Gaussian basis functions.
 *
 * Simple object that holds pointers to a pair of Primitive objects as well as
 * the integral bound.
 * The integral bound is used for sieving when constructing a PairListL through
 * the PairList::build method
 * 
 * Note: ideally we would pass const Primitive& into this class, but this
 * causes nasty C++ problems regarding assigment operators attempting to
 * re-seat the reference. Therefore, we will construct Pair with const
 * Primitive*, but otherwise act as if we had const Primitive&. This is not
 * entirely satisfactory, but should not be the end of the world. -RMP
 **/
class Pair {

public:

/**
 * Verbatim Constructor: fills fields below (see accessor fields for details)
 **/
Pair(
    const Primitive* prim1,
    const Primitive* prim2,
    float bound):
    prim1_(prim1),
    prim2_(prim2),
    bound_(bound) {}

/**
 * Construct Pair with bound set as Schwarz bound
 * The integral bound of the pair is calculated as:
 * B_{pq} = \sqrt{(pq|pq)} = |c_p c_q| e^{-\frac{\alpha_p \alpha_q}{\alpha_p + \alpha_q} (r_p - r_q)^2} 2^{1/4} (\frac{\pi}{\alpha_p + \alpha_q})^{5/4}
 **/
static Pair build_schwarz(
    const Primitive& prim1,
    const Primitive& prim2);

// => Accessors <= //

// The integral bound B_{pq}
float bound() const { return bound_; }
// The first primitive in the Pair 
const Primitive& prim1() const { return *prim1_; }
// The second primitive in the Pair 
const Primitive& prim2() const { return *prim2_; }

// get a handy string representation of the object
std::string string() const;

// => Equivalence (for Python) <= //

bool operator==(const Pair& other) const {
    if (bound_ != other.bound_) return false;
    if (prim1_ != other.prim1_) return false;
    if (prim2_ != other.prim2_) return false;
    return true;
}

bool operator!=(const Pair& other) const {
    return !((*this)==other);
}

private:

// => Fields <= //

const Primitive* prim1_;
const Primitive* prim2_;
float bound_;

};

/**
 * Class PairListL represents a list of Pairs of the same angular momentum.
 *
 * The list must be sorted descending by bound, ascending by prim1.Idx, and 
 * ascending by prim2.Idx; this allows early exits of certain loops in 
 * integral computation.
 **/
class PairListL {

public:

/**
 * Verbatim Constructor: fills fields below (see accessor fields for details)
 **/
PairListL(
    bool is_symmetric,
    int L1,
    int L2,
    const std::vector<Pair>& pairs);

// => Accessors <= //

// Is the PairListL symmetric; if so only the lower triangular entries are 
// stored (prim1->primIdx >= prim2->primIdx).
bool is_symmetric() const { return is_symmetric_; }
// Angular momentum of the prim1 of each Pair
int L1() const { return L1_; }
// Angular momentum of the prim2 of each Pair
int L2() const { return L2_; }
// Vector of all primitive Pairs in the PairListL
const std::vector<Pair>& pairs() const { return pairs_; }
  
// get a handy string representation of the object
std::string string() const;
  
// => Equivalence (for Python) <= //

bool operator==(const PairListL& other) const {
    if (is_symmetric_ != other.is_symmetric_) return false;
    if (L1_ != other.L1_) return false;
    if (L2_ != other.L2_) return false;
    if (pairs_ != other.pairs_) return false;
    return true;
}

bool operator!=(const PairListL& other) const {
    return !((*this)==other);
}


private:

// => Fields <= //

bool is_symmetric_;
int L1_;
int L2_;
std::vector<Pair> pairs_;

};

/**
 * Class PairList represents a list of PairListL.
 **/
class PairList {

public:

/**
 * Verbatim Constructor: fills fields below (see accessor fields for details)
 **/
PairList(
    const std::shared_ptr<Basis>& basis1,
    const std::shared_ptr<Basis>& basis2,
    bool is_symmetric,
    float thre,
    const std::vector<PairListL>& pairlists):
    basis1_(basis1),
    basis2_(basis2),
    is_symmetric_(is_symmetric),
    thre_(thre),
    pairlists_(pairlists) 
    {
        if (is_symmetric && ((basis1->natom() != basis2->natom()) || basis1->nprim() != basis2->nprim())) {
            throw std::runtime_error("PairList: Basis sets are not symmetric");
        }
    }

/**
 * The build_schwarz method constructs the PairList from a pair of Basis 
 * objects using the schwarz bound for seiving.
 *
 * Pairs built with prim1 in basis1 and prim2 in basis2
 * 
 * All Pairs of the same L1, L2 combination go into a PairListL which is 
 * included.
 *
 * if is_symmetric==true, all PairListLs that are constructed will also have
 * is_symmetric==true; i.e. only lower triangular entries are stored.
 *
 * threpq is used for sieving during PairListL construction, only Pairs with
 * bound > threpq will be stored.
**/
static std::shared_ptr<PairList> build_schwarz(
    const std::shared_ptr<Basis>& basis1,
    const std::shared_ptr<Basis>& basis2,
    bool is_symmetric,
    float threpq);

// => Accessors <= //

// Pointer to basis1 used to construct the PairList
std::shared_ptr<Basis> basis1() const { return basis1_; }
// Pointer to basis2 used to construct the PairList
std::shared_ptr<Basis> basis2() const { return basis2_; }
// Is the PairList symmetric; if so only the lower triangular entries are 
// stored (prim1->primIdx >= prim2->primIdx) for all PairListLs contained.
bool is_symmetric() const { return is_symmetric_; }
// The threshold used for pair screening
float thre() const { return thre_; }
// Vector of all PairListLs contained in the PairList object.
const std::vector<PairListL>& pairlists() const { return pairlists_; }
  
// get a handy string representation of the object
std::string string() const;

private:

// => Fields <= //

std::shared_ptr<Basis> basis1_;
std::shared_ptr<Basis> basis2_;
bool is_symmetric_;
float thre_;
std::vector<PairListL> pairlists_;

};

/**
 * Class PairListUtil provides extra functions that are useful to computational
 * routines, but should not be exposed to the casual user.
 **/
class PairListUtil {

public:

/**
 * Comparator for sorting Pair objects.
 * criteria of sorting:
 * 1) bound descending
 * 2) primIdx of 1st primitive ascending
 * 3) primIdx of 2nd primitive ascending
 *
 * @param a the first Pair to make comparison 
 * @param b the second Pair to make comparison
 * @return true if a > b
 **/
static
bool compare_prim_pairs(
    const Pair& a,
    const Pair& b);

/**
 * Determine the maximum value of a Pair bound in pairlist
 *
 * @param pairlist the PairList to compute the bound for
 * @return the maximum value of Pair.bound() in the pairlist
 **/
static
double max_bound(
    const std::shared_ptr<PairList>& pairlist);
    
/**
 * Return a new PairList with only Pair bound elements above thre remaining.
 *
 * @param pairlist the original PairList
 * @param thre the new pair bound threshold
 * @return the newly truncated PairList
 **/
static 
std::shared_ptr<PairList> truncate_bra(
    const std::shared_ptr<PairList>& pairlist,
    float thre);

/**
 * Return a new PairList with new Pair bound elements assigned to the old Pair
 * bound elements multiplied by the maximum absolute D matrix entry for the
 * Cartesian pair (max taken over upper and lower triangular elements if
 * pairlist is symmetric), and then sieved according to the new bounds and
 * thre. I.e., produces a density-sieved PairList.
 * 
 * @param pairlist the original PairList
 * @param D the density matrix to sieve by, an (ncart1,ncart2) Tensor
 * @param thre the new pair bound threshold
 * @return the newly truncated PairList
 *
 * Throws if D is not (ncart1, ncart2) shape
 **/
static
std::shared_ptr<PairList> truncate_ket(
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    float thre);
    
};    

} // namespace lightspeed 

#endif
