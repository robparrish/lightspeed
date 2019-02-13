/**
 * \file kvec.cpp
 * \brief Implements KVecL, KBlockL and KBlock for computation of exchange matrix.
 **/
#include "kvec.hpp"
#include <set>
#include <cmath>
#include <tuple>
#include <algorithm>

namespace {

// criteria of sorting:
// 1) primIdx of 1st primitive ascending
// 2) bound descending
// 3) primIdx of 2nd primitive ascending
bool kblock12_comparator(const lightspeed::Pair& a, const lightspeed::Pair& b) {
    if (a.prim1().primIdx() == b.prim1().primIdx()) {
        if (a.bound() == b.bound()) {
            return (a.prim2().primIdx() < b.prim2().primIdx());
        }
        return (a.bound() > b.bound());
    }
    return (a.prim1().primIdx() < b.prim1().primIdx());
}

} // namespace

namespace lightspeed {

KBlockL KBlock::build_kblockl(
    int L1,
    int L2,
    std::vector<Pair>& pairs)
{
    // Sort pairs into predicate order
    std::sort(pairs.begin(),pairs.end(),kblock12_comparator);

    // Determine prim1 indices for each primIdx
    std::vector<size_t> starts;
    starts.push_back(0);
    for (size_t ind = 1; ind < pairs.size(); ind++) {
        if (pairs[ind].prim1().primIdx() != pairs[ind-1].prim1().primIdx()) {
            starts.push_back(ind);
        }
    }
    starts.push_back(pairs.size());

    // Form KVecL objects
    std::vector<KVecL> kvecls; 
    for (size_t block_ind = 0; block_ind < starts.size() - 1; block_ind++) {
        size_t start = starts[block_ind];
        size_t stop = starts[block_ind+1];
        size_t nprim2 = stop - start;
        if (nprim2 == 0) continue;
        const Primitive& prim1 = pairs[start].prim1();

        // check consistency
        if (prim1.L() != L1 || pairs[start].prim2().L() != L2) { 
            throw std::runtime_error("KBlock::build_kblockl: Inconsistent L1 or L2"); 
        }

        std::vector<double> geom2(5*nprim2);
        std::vector<int> idx2(3*nprim2);
        std::vector<float> bounds12(nprim2);
        for (size_t ind = 0; ind < nprim2; ind++) {
            const Primitive& prim2 = pairs[ind + start].prim2();
            geom2[5*ind + 0] = prim1.c() * prim2.c() * exp(
                - prim1.e() * prim2.e() / (prim1.e() + prim2.e()) * (
                pow(prim1.x() - prim2.x(),2) +
                pow(prim1.y() - prim2.y(),2) +
                pow(prim1.z() - prim2.z(),2)));
            geom2[5*ind + 1] = prim2.e();
            geom2[5*ind + 2] = prim2.x();
            geom2[5*ind + 3] = prim2.y();
            geom2[5*ind + 4] = prim2.z();
            idx2[3*ind + 0] = prim2.primIdx();
            idx2[3*ind + 1] = prim2.cartIdx();
            idx2[3*ind + 2] = prim2.atomIdx();
            bounds12[ind] = pairs[ind + start].bound();
        }

        kvecls.push_back(KVecL(
            &prim1,
            L2,
            nprim2,
            geom2,
            idx2,
            bounds12));
    }

    // Form KBlockL object
    return KBlockL(
        L1,
        L2,
        kvecls);
}

std::shared_ptr<KBlock> KBlock::build12(
    const std::shared_ptr<PairList>& pairlist,
    float thre)
{
    if (pairlist->is_symmetric()) { return KBlock::build_symmetric(pairlist, thre); }

    std::vector<KBlockL> kblockls;
    for (size_t lind = 0; lind < pairlist->pairlists().size(); lind++) {
        const PairListL& pairlistl = pairlist->pairlists()[lind];

        // Copy the pairs in (12| form
        std::vector<Pair> pairs;
        const std::vector<Pair>& pairsf = pairlistl.pairs();
        for (size_t ind = 0; ind < pairsf.size(); ind++) {
            const Pair& pair = pairsf[ind];
            if (pair.bound() >= thre) {
                pairs.push_back(pair);
            }
        }

        // Form KBlockL object
        kblockls.push_back(KBlock::build_kblockl(
            pairlistl.L1(),
            pairlistl.L2(),
            pairs));
    }

    return std::shared_ptr<KBlock>(new KBlock(pairlist, kblockls));
}
std::shared_ptr<KBlock> KBlock::build21(
    const std::shared_ptr<PairList>& pairlist,
    float thre)
{
    if (pairlist->is_symmetric()) { return KBlock::build_symmetric(pairlist, thre); }

    // Ordered to make (L2,L1) appear in order in KBlock
    std::vector<std::tuple<int, int, size_t> > orderl;
    for (size_t lind = 0; lind < pairlist->pairlists().size(); lind++) {
        const PairListL& pairlistl = pairlist->pairlists()[lind];
        orderl.push_back(std::tuple<int, int, size_t>(
            pairlistl.L2(),
            pairlistl.L1(),
            lind));
    }
    std::sort(orderl.begin(),orderl.end());

    std::vector<KBlockL> kblockls;
    for (size_t lind2 = 0; lind2 < pairlist->pairlists().size(); lind2++) {
        size_t lind = std::get<2>(orderl[lind2]);
        const PairListL& pairlistl = pairlist->pairlists()[lind];

        // Trick the pairs into 
        std::vector<Pair> pairs;
        const std::vector<Pair>& pairsf = pairlistl.pairs();
        for (int ind = 0; ind < pairsf.size(); ind++) {
            const Pair& pair = pairsf[ind];
            if (pair.bound() >= thre) {
                pairs.push_back(Pair(&pair.prim2(),&pair.prim1(),pair.bound()));
            }
        }
           
        // Form KBlockL object
        kblockls.push_back(KBlock::build_kblockl(
            pairlistl.L2(),
            pairlistl.L1(),
            pairs));
    }

    return std::shared_ptr<KBlock>(new KBlock(pairlist, kblockls));
}
std::shared_ptr<KBlock> KBlock::build_symmetric(
    const std::shared_ptr<PairList>& pairlist,
    float thre)
{
    if (!pairlist->is_symmetric()) {
        throw std::runtime_error("KBlock::build_symmetric: pairlist is not symmetric.");
    }

    // all possible combinations of (L1,L2) and (L2,L1)
    std::set<std::pair<int, int> > lcombos;
    for (size_t lind = 0; lind < pairlist->pairlists().size(); lind++) {
        const PairListL& pairlistl = pairlist->pairlists()[lind];
        int L1 = pairlistl.L1();
        int L2 = pairlistl.L2();
        lcombos.insert(std::pair<int,int>(L1,L2));
        lcombos.insert(std::pair<int,int>(L2,L1));
    }

    std::vector<KBlockL> kblockls;
    for (std::set<std::pair<int,int> >::const_iterator it = lcombos.begin();
        it != lcombos.end(); ++it) {

        std::pair<int, int> lcombo = *it;
        int L1 = lcombo.first;
        int L2 = lcombo.second;
        
        // Find possible pairs of (12| form
        // (L1,L2) task
        std::vector<Pair> pairs;
        for (size_t lind = 0; lind < pairlist->pairlists().size(); lind++) {
            const PairListL& pairlistl = pairlist->pairlists()[lind];
            if ((pairlistl.L1() != L1) || (pairlistl.L2() != L2)) { continue; }
            const std::vector<Pair>& pairsf = pairlistl.pairs();
            for (size_t ind = 0; ind < pairsf.size(); ind++) {
                const Pair& pair = pairsf[ind];
                if (pair.bound() >= thre) {
                    pairs.push_back(pair);
                }
            }
        }
        // (L2,L1) task
        for (size_t lind = 0; lind < pairlist->pairlists().size(); lind++) {
            const PairListL& pairlistl = pairlist->pairlists()[lind];
            if ((pairlistl.L2() != L1) || (pairlistl.L1() != L2)) { continue; }
            const std::vector<Pair>& pairsf = pairlistl.pairs();
            for (size_t ind = 0; ind < pairsf.size(); ind++) {
                const Pair& pair = pairsf[ind];
                // Do not duplicate diagonal pairs
                if (pair.prim1().primIdx() == pair.prim2().primIdx()) { continue; }
                if (pair.bound() >= thre) {
                    pairs.push_back(Pair(&pair.prim2(),&pair.prim1(),pair.bound()));
                }
            }
        }

        // Form KBlockL object
        kblockls.push_back(KBlock::build_kblockl(
            L1, 
            L2,
            pairs));
    }

    return std::shared_ptr<KBlock>(new KBlock(pairlist, kblockls));
}

} // namespace lightspeed
