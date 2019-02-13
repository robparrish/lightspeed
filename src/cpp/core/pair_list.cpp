#include <lightspeed/pair_list.hpp>
#include <lightspeed/gridbox.hpp> // For HashedGrid: TODO: Move this to own file
#include <lightspeed/tensor.hpp>
#include "../util/string.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <cstdio>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace lightspeed {

Pair Pair::build_schwarz(
    const Primitive& prim1,
    const Primitive& prim2)
{
    double R2 = 0.0;
    R2 += pow(prim1.x()-prim2.x(),2);
    R2 += pow(prim1.y()-prim2.y(),2);
    R2 += pow(prim1.z()-prim2.z(),2);
    float bound = (float) (fabs(prim1.c()*prim2.c()) * 
        exp(-prim1.e() * prim2.e() / (prim1.e() + prim2.e()) * R2) * 
        pow(2.0,0.25) * pow(M_PI / (prim1.e() + prim2.e()),1.25));
    return Pair(&prim1, &prim2, bound);
}

std::shared_ptr<PairList> PairList::build_schwarz(
    const std::shared_ptr<Basis>& basis1,
    const std::shared_ptr<Basis>& basis2,
    bool is_symmetric,
    float threpq)
{
    if (is_symmetric && ((basis1->nprim() != basis2->nprim() || basis1->natom() != basis2->natom()))) {
        throw std::runtime_error("PairList::build_schwarz: basis sets are not symmetric");
    }

    const std::vector<Primitive>& prims1 = basis1->primitives();
    const std::vector<Primitive>& prims2 = basis2->primitives();

#ifdef _OPENMP
    int n_threads = omp_get_max_threads();
#else
    int n_threads = 1;
#endif
    // pairs is a 2D vector of Pair
    // 1st dimension is thread id
    // 2nd dimension is the index of Pair
    std::vector<std::vector<Pair> > pairs (n_threads);
    size_t nprimpair = prims1.size() * prims2.size();

//#define PAIRLIST_N2
#ifdef PAIRLIST_N2
    // Algorithm 1: Pairwise checking O(N^2)
    #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
    for (size_t ij = 0; ij < nprimpair; ij++) {
        size_t idx1 = ij / prims2.size();
        size_t idx2 = ij % prims2.size();
        const Primitive& prim1 = prims1[idx1];
        const Primitive& prim2 = prims2[idx2];
        if (is_symmetric && (prim2.primIdx() > prim1.primIdx())) {
            continue;
        }
        Pair pair = Pair::build_schwarz(prim1, prim2);
        if (pair.bound() > threpq) {
#ifdef _OPENMP
            int thread = omp_get_thread_num();
#else
            int thread = 0;
#endif
            pairs[thread].push_back(pair);
        }
    }
#else
    // Algorithm 2: Hash O(N)
    double Rmax = 0.0;
    for (size_t P = 0; P < prims1.size(); P++) {
        const Primitive& prim = prims1[P];
        double R2 = -2.0 / prim.e() * log(threpq / (prim.c() * prim.c() * (pow(2.0,0.25) * pow(M_PI / (prim.e() + prim.e()),1.25))));
        if (R2 < 0.0) continue; // Weird
        double R = sqrt(R2);
        Rmax = std::max(Rmax,R);
    }
    for (size_t P = 0; P < prims2.size(); P++) {
        const Primitive& prim = prims2[P];
        double R2 = -2.0 / prim.e() * log(threpq / (prim.c() * prim.c() * (pow(2.0,0.25) * pow(M_PI / (prim.e() + prim.e()),1.25))));
        if (R2 < 0.0) continue; // Weird
        double R = sqrt(R2);
        Rmax = std::max(Rmax,R);
    }

    // Hashed Grid
    std::vector<size_t> dim;
    dim.push_back(prims2.size());
    dim.push_back(3);
    std::shared_ptr<Tensor> xyz(new Tensor(dim));
    double* xyzp = xyz->data().data();
    for (size_t P = 0; P < prims2.size(); P++) {
        const Primitive& prim = prims2[P];
        xyzp[3*P+0] = prim.x();
        xyzp[3*P+1] = prim.y();
        xyzp[3*P+2] = prim.z();
    }
    const double Rval = 5.0; // Parameter!
    std::shared_ptr<HashedGrid> grid(new HashedGrid(xyz,Rval));
    const HashedGrid* gridp = grid.get();
    
    // Hash Lookup
    #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
    for (size_t P = 0; P < prims1.size(); P++) {
#ifdef _OPENMP
       int thread = omp_get_thread_num();
#else
       int thread = 0;
#endif
        const Primitive& prim1 = prims1[P];
        double xP = prim1.x();
        double yP = prim1.y();
        double zP = prim1.z();
        
        std::tuple<int,int,int> key = gridp->hashkey(xP,yP,zP);
        int nstep = (int) ceil(Rmax / Rval); 
        // Step into relevant boxes
        for (int nx = -nstep; nx <= nstep; nx++) {
        for (int ny = -nstep; ny <= nstep; ny++) {
        for (int nz = -nstep; nz <= nstep; nz++) {
            std::tuple<int,int,int> key2(
                std::get<0>(key) + nx,
                std::get<1>(key) + ny,
                std::get<2>(key) + nz);
            if (!HashedGrid::significant_box(key2, Rval, xP, yP, zP, Rmax)) continue;
            const std::vector<size_t>& inds = gridp->inds(key2); 
            for (size_t ind2 = 0; ind2 < inds.size(); ind2++) {
                size_t Q = inds[ind2];
                const Primitive& prim2 = prims2[Q];
                if (is_symmetric && (prim2.primIdx() > prim1.primIdx())) {
                    continue;
                }
                Pair pair = Pair::build_schwarz(prim1, prim2);
                if (pair.bound() > threpq) {
                    pairs[thread].push_back(pair);
                }
            }
        }}}
    }
#endif
    
    // pairsL is a 2D vector of Pair
    // 1st dimension is Ltot == L1*(max_l+1) + L2
    // 2nd dimension is the index of Pair of Ltot
    int max_l1 = basis1->max_L();
    int max_l2 = basis2->max_L();
    std::vector<std::vector<Pair> > pairsL ((max_l1+1)*(max_l2+1));
    for (int thread = 0; thread < n_threads; ++thread) {
        for (int i = 0; i < pairs[thread].size(); ++i) {
            int L1 = pairs[thread][i].prim1().L();
            int L2 = pairs[thread][i].prim2().L();
            pairsL[L1*(max_l2+1) + L2].push_back(pairs[thread][i]);
        }
    }

    // pairlists is a 1D vector of PairListL, to be used to build pairlist
    std::vector<PairListL> pairlists;
    for (int L1=0; L1 < max_l1+1; L1++){
        for (int L2=0; L2 < max_l2+1; L2++){
            int Ltot = L1*(max_l2+1) + L2;
            if (pairsL[Ltot].size() > 0) {
                std::sort(pairsL[Ltot].begin(), pairsL[Ltot].end(), PairListUtil::compare_prim_pairs);
                pairlists.push_back(PairListL(is_symmetric, L1, L2, pairsL[Ltot]));
            }
        }
    }

    std::shared_ptr<PairList> pairlist(new PairList(basis1, basis2, is_symmetric, threpq, pairlists));
    return pairlist;
}

PairListL::PairListL(
    bool is_symmetric,
    int L1,
    int L2,
    const std::vector<Pair>& pairs):
    is_symmetric_(is_symmetric),
    L1_(L1),
    L2_(L2),
    pairs_(pairs)
{
    // make sure that the pairs are sorted correctly
    for (size_t ind = 1; ind < pairs_.size(); ind++){
        if (!PairListUtil::compare_prim_pairs(pairs_[ind-1], pairs_[ind])){
            throw std::runtime_error("PairListL: pairs not sorted according to descending bounds, prim1().primIdx() ascending, prim2.().primIdx() ascending.");
        }
    }
}

std::string Pair::string() const 
{
    std::string s = "";
    s += "Pair:\n";
    s += sprintf2("  primIdx1 = %11d\n", prim1().primIdx());
    s += sprintf2("  primIdx2 = %11d\n", prim2().primIdx());
    s += sprintf2("  bound    = %11.3E\n", bound());
    return s;
}
std::string PairListL::string() const 
{
    std::string s = "";
    s += "PairListL:\n";
    s += sprintf2("  L1    = %11d\n", L1());
    s += sprintf2("  L2    = %11d\n", L2());
    s += sprintf2("  symm? = %11s\n", is_symmetric() ? "Yes" : "No"); 
    s += sprintf2("  npair = %11zu\n", pairs().size());
    return s;
}
std::string PairList::string() const 
{
    size_t npair = 0;
    for (int ang = 0; ang < pairlists().size(); ang++) {
        npair += pairlists()[ang].pairs().size();
    }

    std::string s = "";
    s += "PairList:\n";
    s += sprintf2("  symm? = %11s\n", is_symmetric() ? "Yes" : "No"); 
    s += sprintf2("  thre  = %11.3E\n", thre_);
    s += sprintf2("  npair = %11zu\n", npair);
    return s;
}
    
} // namespace lightspeed
