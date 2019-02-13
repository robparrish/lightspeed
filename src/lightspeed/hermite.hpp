#ifndef LS_HERMITE_HPP
#define LS_HERMITE_HPP

#include <lightspeed/basis.hpp>
#include <lightspeed/pair_list.hpp>
#include <lightspeed/tensor.hpp>
#include <vector>
#include <algorithm>

namespace lightspeed {

/**
 * Class HermiteL builds a dense Hermite-Gaussian representation for a given
 * PairListL object. The HermiteL object constructs the geometric info on
 * centers and exponents of the Hermite Gaussians (stored in the field geom),
 * as well as a data buffer (stored in the read-write field data) sufficient to
 * hold the Hermite Gaussian representation of a density or potential matrix
 * for the PairListL object.
 *
 * The Hermite Gaussians are ordered in the lexical manner:
 *   ---------- 
 *   t tx ty tz 
 *   ----------    
 *   0  0  0  0
 *   ----------    
 *   1  1  0  0
 *   2  0  1  0
 *   3  0  0  1
 *   ----------
 *   4  2  0  0
 *   5  1  1  0
 *   6  1  0  1
 *   7  0  2  0
 *   8  0  1  1
 *   9  0  0  2
 *   ----------
 * and so forth.
 *
 * The number of Hermite function in a given L shell is:
 *   ----- 
 *   L   H
 *   -----
 *   0   1
 *   1   4
 *   2  10
 *   3  20 
 *   4  35
 *   5  56
 *   6  84
 *   7 120
 *   8 165
 *   -----
 * The general formula is H = (L+1)*(L+2)*(L+3)/6
 **/
class HermiteL {

public:

/**
 * Constructor: copies pointer/sizes from pairs, builds geom array, zeros data buffer.
 * @param pairs the PairListL pointer to build the HermiteL around
 **/
HermiteL(
    const PairListL* pairs
    ) : 
    pairs_(pairs),
    npair_(pairs->pairs().size()),
    L1_(pairs->L1()),
    L2_(pairs->L2()),
    L_(pairs->L1() + pairs->L2())
    {
        // Number of hermite Gaussians per shell
        H_ = (L_ + 1) * (L_ + 2) * (L_ + 3) / 6;

        // Allocate zero data buffer
        data_.resize(npair_*H(),0.0);
        // Build geometry info 
        geom_.resize(npair_*4);
        bounds_.resize(npair_);
        etas_.resize(npair_*2);
        inds_.resize(npair_*2);
        const std::vector<Pair>& prim_pairs = pairs->pairs();
        for (size_t ind = 0; ind < prim_pairs.size(); ind++) {
            const Pair& pair = prim_pairs[ind];
            const Primitive& prim1 = pair.prim1();
            const Primitive& prim2 = pair.prim2();
            double a = prim1.e() + prim2.e();
            double a_inv = 1.0 / a;
            double x = (prim1.e() * prim1.x() + prim2.e() * prim2.x()) * a_inv;
            double y = (prim1.e() * prim1.y() + prim2.e() * prim2.y()) * a_inv;
            double z = (prim1.e() * prim1.z() + prim2.e() * prim2.z()) * a_inv;
            geom_[4*ind + 0] = a;
            geom_[4*ind + 1] = x;
            geom_[4*ind + 2] = y;
            geom_[4*ind + 3] = z;
            bounds_[ind] = pair.bound();
            etas_[2*ind + 0] = prim1.e() * a_inv;
            etas_[2*ind + 1] = prim2.e() * a_inv;
            inds_[2*ind + 0] = prim1.atomIdx();
            inds_[2*ind + 1] = prim2.atomIdx();
        }
    }

// => Accessors <= //

// The PairListL object this HermiteL is built around
const PairListL* pairs() const { return pairs_; }
// The angular momentum of the first center in Cartesian representation
int L1() const { return L1_; }
// The angular momentum of the second center in Cartesian representation
int L2() const { return L2_; }
// The total angular momentum (L1 + L2)
int L() const { return L_; }
// The number of Hermite Gaussian functions per shell (L+1)*(L+2)*(L+3)/6
int H() const { return H_; }
// The total number of pairs in this HermiteL object
size_t npair() const { return npair_; }

// An npair x 4 array of the geometric info of all Hermite Gaussian shells.
// Each element holds (a,x,y,z), the exponent and geoemtric center of each
// Hermite Gaussian.
const std::vector<double>& geom() const { return geom_; }

// An npair x H array for use as a read-write data buffer
std::vector<double>& data() { return data_; }
const std::vector<double>& data() const { return data_; }

// An npair array of the bounds info of all Hermite Gaussian shells.
// Copied from the Pair bound fields
const std::vector<float>& bounds() const { return bounds_; }

// npair x 2 array of geometric center weights of all Hermite Gaussian shells
// each element holds (eta1,eta2) where e.g. 
// eta1 = e1/(e1+e2)
const std::vector<double>& etas() const { return etas_; }

// npair x 2 array of indices corresponding to the following:
// (atomIdx1, atomIdx2)
const std::vector<int>& inds() const { return inds_; }

// => Utility Methods <= //

// Zero the contents of the data field of this object
void zero() { std::fill(data_.begin(), data_.end(), 0.0); }

/**
 * Accumulate the Cartesian-to-Hermite transformation of Dcart into the data
 * field of this object
 *
 * @param Dcart the Cartesian representation of the density or potential
 *  matrix, an (ncart1,ncart2) Tensor.
 * @param phases apply (-1)^t phases to the Hermite expansion coefficients?
 *  Needed in Coulomb matrix operations.
 * @result this object's data() buffers will have the transformed
 *  representation of Dcart accumulated into them.
 *
 * This object is updated.
 **/
void cart_to_herm(
    const std::shared_ptr<Tensor>& Dcart,
    bool phases = false
    );

/**
 * Accumulate the Hermite-to-Cartesian transformation of the data field of this
 * object into Dcart.
 *
 * @param Dcart the Cartesian representation of the density or potential
 *  matrix, an (ncart1,ncart2) Tensor.
 * @param phases apply (-1)^t phases to the Hermite expansion coefficients?
 *  Needed in Coulomb matrix operations.
 * @result Dcart will be updated to have the transformed representation of this
 *  object's data() accumulated.
 *
 * Dcart is updated.
 **/
void herm_to_cart(
    const std::shared_ptr<Tensor>& Dcart,
    bool phases = false
    ) const;

/**
 * Accumulate the gradient contributions stemming from the E_pq^\vec t coupling
 * coefficients contracted against Dcart (pq indices) and the data field of
 * this object (\vec t indices).
 *
 * @param Dcart the Cartesian representation of the density or potential
 *  matrix, an (ncart1,ncart2) Tensor.
 * @param phases apply (-1)^t phases to the Hermite expansion coefficients?
 *  Needed in Coulomb matrix operations.
 * @result G1 and G2 will be updated to have the gradient contributions
 *  accumulated.
 *
 * G1 and G2 are updated.
 *
 * NOTE: if pairlist is not symmetric, G1 and G2 will separately contain the
 * gradient contributions from moving basis functions 1 and 2, respectively.
 * This is the classically expected behavior.  However, if pairlist is
 * symmetric, G1+G2 will contain the total gradient for moving basis functions
 * 1 and 2 - no guarantees are made as to the specific contents of G1 vs. G2 in
 * this case. The recommendation if pairlist is symmetric is to pass the same
 * tensor G in as both G1 and G2, and the total gradient contribution will be
 * accumulated.
 **/
void grad(
    const std::shared_ptr<Tensor>& Dcart,
    const std::shared_ptr<Tensor>& G1,
    const std::shared_ptr<Tensor>& G2,
    bool phases = false
    ) const;

private:

// => Fields <= //

const PairListL* pairs_;
int L_;
int L1_;
int L2_;
int H_;
size_t npair_;

std::vector<double> geom_;
std::vector<double> data_;
std::vector<float> bounds_;
std::vector<double> etas_;
std::vector<int> inds_;

};

/**
 * Class Hermite builds a dense Hermite-Gaussian representation for a given
 * PairList object. This is essentially a wrapper class to hold a list of
 * HermiteL objects corresponding to the PairListL object in the PairList.
 **/
class Hermite {
    
public:

/**
 * Constructor, builds a HermiteL for each PairListL in pairlist
 * @param pairlist the PairList object to build this Hermite around
 **/
Hermite(
    const std::shared_ptr<PairList>& pairlist) :
    pairlist_(pairlist)
    {
        const std::vector<PairListL>& pairlists = pairlist_->pairlists();  
        for (size_t ind = 0; ind < pairlists.size(); ind++) {
            hermites_.push_back(HermiteL(&pairlists[ind]));
        }
    }

// => Accessors <= //

// The PairList object this Hermite is build around
std::shared_ptr<PairList> pairlist() const { return pairlist_; }
// The HermiteL objects at the core of this Hermite
std::vector<HermiteL>& hermites() { return hermites_; }
const std::vector<HermiteL>& hermites() const { return hermites_; }

// => Utility Methods <= //

// Zero the contents of the data fields of this object's HermiteL objects.
void zero() {
        for (size_t ind = 0; ind < hermites_.size(); ind++) {
            hermites_[ind].zero();
        }
    }

/**
 * Accumulate the Cartesian-to-Hermite transformation of Dcart into the data
 * fields of this object's HermiteL objects.
 *
 * @param Dcart the Cartesian representation of the density or potential
 *  matrix, an (ncart1,ncart2) Tensor.
 * @param phases apply (-1)^t phases to the Hermite expansion coefficients?
 *  Needed in Coulomb matrix operations.
 * @result this object's hermites()[****]->data() buffers will have the transformed
 *  representation of Dcart accumulated into them.
 *
 * This object is updated.
 **/
void cart_to_herm(
    const std::shared_ptr<Tensor>& Dcart,
    bool phases = false
    ) {
        std::vector<size_t> dim;
        dim.push_back(pairlist_->basis1()->ncart());
        dim.push_back(pairlist_->basis2()->ncart());
        Dcart->shape_error(dim);
        for (size_t ind = 0; ind < hermites_.size(); ind++) {
            hermites_[ind].cart_to_herm(
                Dcart,
                phases);
        }
    }

/**
 * Accumulate the Hermite-to-Cartesian transformation of the data
 * fields of this object's HermiteL objects into Dcart.
 *
 * @param Dcart the Cartesian representation of the density or potential
 *  matrix, an (ncart1,ncart2) Tensor.
 * @param phases apply (-1)^t phases to the Hermite expansion coefficients?
 *  Needed in Coulomb matrix operations.
 * @result Dcart will be updated to have the transformed representation of this
 *  object's hermites()[****]->data() accumulated.
 *
 * Dcart is updated.
 **/
void herm_to_cart(
    const std::shared_ptr<Tensor>& Dcart,
    bool phases = false
    ) const {
        std::vector<size_t> dim;
        dim.push_back(pairlist_->basis1()->ncart());
        dim.push_back(pairlist_->basis2()->ncart());
        Dcart->shape_error(dim);
        for (size_t ind = 0; ind < hermites_.size(); ind++) {
            hermites_[ind].herm_to_cart(
                Dcart,
                phases);
        }
    }

/**
 * Accumulate the gradient contributions stemming from the E_pq^\vec t coupling
 * coefficients contracted against Dcart (pq indices) and the data fields of
 * this object's HermiteL objects (\vec t indices).
 *
 * @param Dcart the Cartesian representation of the density or potential
 *  matrix, an (ncart1,ncart2) Tensor.
 * @param phases apply (-1)^t phases to the Hermite expansion coefficients?
 *  Needed in Coulomb matrix operations.
 * @result G1 and G2 will be updated to have the gradient contributions
 *  accumulated.
 *
 * G1 and G2 are updated.
 *
 * NOTE: if pairlist is not symmetric, G1 and G2 will separately contain the
 * gradient contributions from moving basis functions 1 and 2, respectively.
 * This is the classically expected behavior.  However, if pairlist is
 * symmetric, G1+G2 will contain the total gradient for moving basis functions
 * 1 and 2 - no guarantees are made as to the specific contents of G1 vs. G2 in
 * this case. The recommendation if pairlist is symmetric is to pass the same
 * tensor G in as both G1 and G2, and the total gradient contribution will be
 * accumulated.
 **/
void grad(
    const std::shared_ptr<Tensor>& Dcart,
    const std::shared_ptr<Tensor>& G1,
    const std::shared_ptr<Tensor>& G2,
    bool phases = false
    ) const {
        std::vector<size_t> dim;
        dim.push_back(pairlist_->basis1()->ncart());
        dim.push_back(pairlist_->basis2()->ncart());
        Dcart->shape_error(dim);
        std::vector<size_t> dim1;
        dim1.push_back(pairlist_->basis1()->natom());
        dim1.push_back(3);
        G1->shape_error(dim1);
        std::vector<size_t> dim2;
        dim2.push_back(pairlist_->basis2()->natom());
        dim2.push_back(3);
        G2->shape_error(dim2);
        for (size_t ind = 0; ind < hermites_.size(); ind++) {
            hermites_[ind].grad(
                Dcart,
                G1,
                G2,
                phases);
        }
    }

private:

// => Fields <= //

std::shared_ptr<PairList> pairlist_;
std::vector<HermiteL> hermites_; 

};

/**
 * Class HermiteUtil provides static functions for a lot of random stuff one
 * often does with Hermite Gaussians.
 **/
class HermiteUtil {

public:

/**
 * Return the lookup table for Hermite Gaussian angular momentum quanta:
 *   ---------- 
 *   t tx ty tz 
 *   ----------    
 *   0  0  0  0
 *   ----------    
 *   1  1  0  0
 *   2  0  1  0
 *   3  0  0  1
 *   ----------
 *   4  2  0  0
 *   5  1  1  0
 *   6  1  0  1
 *   7  0  2  0
 *   8  0  1  1
 *   9  0  0  2
 *   ----------
 * and so forth.
 *
 * for each t, (tx,ty,tz) are packed as 10-bit fields (the right-most 30 bits).
 * 
 * If you happen to need more than L = 1023, you may be in the wrong line of
 * work.
 *
 * @param L the total angular momentum needed - 0: SS, 1: SP, 2: SD or PP, etc
 * @return the angular momentum quanta vectors as packed 32-bit integers
 **/
static
std::vector<int> angl(int L);

/**
 * Return a vector of the Gaussian prefactors K for each Hermite shell in herm
 * @param herm the HermiteL to look at
 * @return the magnitude of the HermiteL shells before any density matrix
 *  considerations are added (e.g., in the bra of the Coulomb matrix
 *  definition, the Gaussian prefactor)
 **/
static 
std::vector<double> dmax_bra(const HermiteL& herm);

/**
 * Return a vector of the maximum Hermite cofficients for each Hermite shell in
 * herm
 * @param herm the HermiteL to look at
 * @return the magnitude of the HermiteL shells including both density matrix
 *  and Gaussian prefactor considerations (max of data)
 **/
static 
std::vector<double> dmax_ket(const HermiteL& herm);
    
};

} // namespace lightspeed 

#endif
