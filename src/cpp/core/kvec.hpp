/**
 * \file kvec.hpp
 * \brief Implements KVecL, KBlockL and KBlock for computation of exchange matrix.
 **/
#ifndef LS_KVEC_HPP
#define LS_KVEC_HPP

#include <memory>
#include <lightspeed/pair_list.hpp>
#include <vector>
#include <stdexcept>

namespace lightspeed {

class PairList;

/**
 * Class KVecL holds a list of Pair sorted descending by the bound. All pairs
 * have the same 1st primitive, and different 2nd primitives with the same
 * angular momentum. The KVecL object is constructed from a pointer to the 1st
 * primitive and several vectors containing the geometric data, primitive/atom
 * indices of the 2nd primitive, and the bound of the pair.
 **/
class KVecL {

public:

/**
 * Verbatim Constructor of KVecL
 * @param prim1 Pointer to the 1st primitive
 * @param L2 Angular momentum of the 2nd primitives
 * @param nprim2 Number of pairs
 * @param geom2 Geometric data of the 2nd primitives
 * @param idx2 Primitive, Cartesian and atom indices of the 2nd primitives
 * @param bounds12 Bound of the pairs
 *
 * geom2 is of size (nprim2 * 5) and contains K12, e2, x2, y2, and z2.
 * idx2 is of size (nprim2 * 3) and contains primIdx2, cartIdx2 and atomIdx2.
 * bounds12 is of size (nprim2) and contains bound12.
 **/
KVecL(
    const Primitive* prim1,
    int L2,
    size_t nprim2,
    const std::vector<double>& geom2,
    const std::vector<int>& idx2,
    const std::vector<float>& bounds12) :
    prim1_(prim1),
    L2_(L2),
    nprim2_(nprim2),
    geom2_(geom2),
    idx2_(idx2),
    bounds12_(bounds12)
    {
        // check sizes
        if (geom2_.size() != nprim2_*5){
            throw std::runtime_error("KVecL: geom2 size is incorrect.");
        }
        if (idx2_.size() != nprim2_*3){
            throw std::runtime_error("KVecL: idx2 size is incorrect.");
        }
        if (bounds12_.size() != nprim2_){
            throw std::runtime_error("KVecL: bounds12 size is incorrect.");
        }
        // make sure that the pairs are sorted descending by bound
        for (size_t ind = 1; ind < bounds12_.size(); ind++){
            if (bounds12_[ind-1] < bounds12_[ind]){
                throw std::runtime_error("KVecL: bounds12 not sorted descending.");
            }
        }
    }

// => Accessors <= //

// First primitive
const Primitive& prim1() const { return *prim1_; }
// Angular momentum of the 2nd primitives
int L2() const { return L2_; }
// Number of pairs
size_t nprim2() const { return nprim2_; }
// Geometric data of the 2nd primitives
// Vector of size (nprim2 * 5), contains K12, e2, x2, y2, and z2.
const std::vector<double>& geom2() const { return geom2_; }
// Primitive and atom indices of the 2nd primitives
// Vector of size (nprim2 * 2), contains cartIdx2 and atomIdx2.
const std::vector<int>& idx2() const { return idx2_; }
// Bound of the pairs
// Vector of size (nprim2), contains bound12.
const std::vector<float>& bounds12() const { return bounds12_; }

private:

// Pointer to the 1st primitive
const Primitive* prim1_;
// Angular momentum of the 2nd primitive
int L2_;
// Number of pairs
size_t nprim2_;
// Vector of size (nprim2*5), contains K12, e2, x2, y2, z2 (data for 2nd primitives)
std::vector<double> geom2_;
// Vector of size (nprim2*2), cartesian and atom index
std::vector<int> idx2_;
// Vector of size (nprim2), should be sorted descending
std::vector<float> bounds12_;

};

/**
 * Class KBlockL is a wrapper class that holds a vector of KVecL obejcts of the
 * same angular momenta (L1 for the 1st primitives and L2 for the 2nd
 * primitives).
 **/
class KBlockL {

public:

/**
 * Verbatim Constructor of KBlockL
 * @param L1 Angular momentum of the 1st primitives
 * @param L2 Angular momentum of the 2nd primitives
 * @param kvecs Vector of the KVecL objects
 **/
KBlockL(
    int L1,
    int L2,
    const std::vector<KVecL>& kvecs) :
    L1_(L1),
    L2_(L2),
    kvecs_(kvecs)
    {
        // check consistency
        for (size_t ind = 0; ind < kvecs_.size(); ind++){
            const KVecL& kvec = kvecs_[ind];
            if (kvec.prim1().L() != L1){
                throw std::runtime_error("KBlockL: kvec.prim1.L() not equal to given L1.");
            }
            if (kvec.L2() != L2){
                throw std::runtime_error("KBlockL: kvec.L2() not equal to given L2.");
            }
        }
    }

// => Accessors <= //

// Angular mometum of the 1st primitives
int L1() const { return L1_; }
// Angular mometum of the 2nd primitives
int L2() const { return L2_; }
// Vector of KVecL objects
const std::vector<KVecL>& kvecs() const { return kvecs_; }

private:

// Angular mometum of the 1st primitives
int L1_;
// Angular mometum of the 2nd primitives
int L2_;
// Vector of KVecL objects
std::vector<KVecL> kvecs_;

};

/**
 * Class KBlock is a wrapper class that holds a vector of KBlockL objects.
 * KBlock also provides static functions to build the list of KBlockL objects
 * in the forward (build12) or the backward (build21) manner.
 *
 * In the forward approach, we build the (12| pair by looking for all the 2's
 * given 1. This may be used in the building of exchange K13 from density D24
 * and ERI's (12|34).
 *
 * In practice we employ the backward approach, where we build the (21| pairs
 * by looking for all the 1's given 2. This facilitates the pre-screening of
 * integrals in the calculation of exchange K13, since the access of D24 is
 * regular.
 **/
class KBlock {

public:

/**
 * Verbatim Constructor of KBlock
 * @param pairlist The pairlist object of basis set(s)
 * @param kblocks The vector of KBlockL objects
 **/
KBlock(
    const std::shared_ptr<PairList>& pairlist,
    const std::vector<KBlockL> kblocks) :
    pairlist_(pairlist),
    kblocks_(kblocks)
    {
    }

/**
 * Forward construction of KBlock
 * @param pairlist The pairlist of basis set(s)
 * @param thre Threshold for pre-screening
 * @return shared_ptr of KBlock object
 **/
static std::shared_ptr<KBlock> build12(
    const std::shared_ptr<PairList>& pairlist,
    float thre);

/**
 * Backward construction of KBlock
 * @param pairlist The pairlist of basis set(s)
 * @param thre Threshold for pre-screening
 * @return shared_ptr of KBlock object
 **/
static std::shared_ptr<KBlock> build21(
    const std::shared_ptr<PairList>& pairlist,
    float thre);

// => Accessors <= //

// Pairlist of basis set(s)
std::shared_ptr<PairList> pairlist() const { return pairlist_; }
// Vector of KBlockL obejcts
const std::vector<KBlockL>& kblocks() const { return kblocks_; }

private:

// Pairlist of basis set(s)
std::shared_ptr<PairList> pairlist_;
// Vector of KBlockL obejcts
std::vector<KBlockL> kblocks_;

/**
 * Symmetric construction of KBlock
 * @param pairlist The pairlist of basis set(s)
 * @param thre Threshold for pre-screening
 * @return shared_ptr of KBlock object
 **/
static std::shared_ptr<KBlock> build_symmetric(
    const std::shared_ptr<PairList>& pairlist,
    float thre);

/**
 * Build KBlockL object from pairs
 * @param L1 Angular momentum of the 1st primitives
 * @param L2 Angular momentum of the 2nd primitives
 * @param pairs Primitive pairs
 * @return KBlockL object of angular momenta (L1,L2)
 **/
static KBlockL build_kblockl(
    int L1,
    int L2,
    std::vector<Pair>& pairs);

};

} //namespace lightspeed

#endif
