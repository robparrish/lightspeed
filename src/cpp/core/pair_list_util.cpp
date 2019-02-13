#include <lightspeed/pair_list.hpp>
#include <lightspeed/tensor.hpp>
#include <algorithm>
#include <cmath>

namespace lightspeed {
    
// criteria of sorting:
// 1) bound descending
// 2) primIdx of 1st primitive ascending
// 3) primIdx of 2nd primitive ascending
bool PairListUtil::compare_prim_pairs(const Pair& a, const Pair& b) {
    if (a.bound() == b.bound()) {
        if (a.prim1().primIdx() == b.prim1().primIdx()) {
            return (a.prim2().primIdx() < b.prim2().primIdx());
        }
        return (a.prim1().primIdx() < b.prim1().primIdx());
    }
    return (a.bound() > b.bound());
}

double PairListUtil::max_bound(
    const std::shared_ptr<PairList>& pairlist)
{
    double val = 0.0;
    for (size_t lind = 0; lind < pairlist->pairlists().size(); lind++) {
        const PairListL& pairs = pairlist->pairlists()[lind];
        val = std::max(val,(pairs.pairs().size() ? pairs.pairs()[0].bound(): 0.0));
    }
    return val;
}
std::shared_ptr<PairList> PairListUtil::truncate_bra(
    const std::shared_ptr<PairList>& pairlist,
    float thre)
{
    const std::vector<PairListL>& pairlists = pairlist->pairlists();     
    std::vector<PairListL> pairlists2;
    for (size_t indL = 0; indL < pairlists.size(); indL++) {
        const PairListL& pairsL = pairlists[indL];
        const std::vector<Pair>& pairs = pairsL.pairs();
        std::vector<Pair> pairs2;
        for (size_t ind = 0; ind < pairs.size(); ind++) {
            if (pairs[ind].bound() < thre) break;
            pairs2.push_back(pairs[ind]);
        }
        pairlists2.push_back(PairListL(
            pairsL.is_symmetric(),
            pairsL.L1(),
            pairsL.L2(),
            pairs2));
    }
    
    return std::shared_ptr<PairList>(new PairList(
        pairlist->basis1(),
        pairlist->basis2(),
        pairlist->is_symmetric(),
        thre,
        pairlists2)); 
}
std::shared_ptr<PairList> PairListUtil::truncate_ket(
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    float thre)
{
    // Check that D is the right size
    std::vector<size_t> dim1;
    dim1.push_back(pairlist->basis1()->ncart());
    dim1.push_back(pairlist->basis2()->ncart());
    D->shape_error(dim1);

    const double* Dp = D->data().data(); 

    bool symm = pairlist->is_symmetric();
    size_t n2 = D->shape()[1];

    const std::vector<PairListL>& pairlists = pairlist->pairlists();     
    std::vector<PairListL> pairlists2;
    for (size_t indL = 0; indL < pairlists.size(); indL++) {
        const PairListL& pairsL = pairlists[indL];
        const std::vector<Pair>& pairs = pairsL.pairs();
        std::vector<Pair> pairs2;
        for (size_t ind = 0; ind < pairs.size(); ind++) {
            float B = pairs[ind].bound();    
            const Primitive& prim1 = pairs[ind].prim1();
            const Primitive& prim2 = pairs[ind].prim2();
            int c1 = prim1.ncart(); 
            int c2 = prim2.ncart(); 
            int a1 = prim1.cartIdx();
            int a2 = prim2.cartIdx();
            double val = 0.0;
            for (int p1 = 0; p1 < c1; p1++) {
                for (int p2 = 0; p2 < c2; p2++) {
                    val = std::max(val,fabs(Dp[(p1 + a1) * n2 + (p2 + a2)]));
                }
            }
            if (symm) {
                for (int p1 = 0; p1 < c1; p1++) {
                    for (int p2 = 0; p2 < c2; p2++) {
                        val = std::max(val,fabs(Dp[(p2 + a2) * n2 + (p1 + a1)]));
                    }
                }
            }
            float B2 = val * B;
            if (B2 > thre) {
                pairs2.push_back(Pair(&prim1,&prim2,B2));
            }
        }

        std::sort(pairs2.begin(),pairs2.end(),compare_prim_pairs);
        pairlists2.push_back(PairListL(
            pairsL.is_symmetric(),
            pairsL.L1(),
            pairsL.L2(),
            pairs2));
    }

    return std::shared_ptr<PairList>(new PairList(
        pairlist->basis1(),
        pairlist->basis2(),
        pairlist->is_symmetric(),
        thre,
        pairlists2)); 
}

} // namespace lightspeed 
