#include "casbox_util.hpp"
#include "bits.hpp"
#include <map>
#include <cstdio>

namespace lightspeed {

std::vector<CASSingle> CASBoxUtil::singles(
    int M, int N)
{
    // Generate the string space
    std::vector<uint64_t> strs = Bits::combinations(M, N);

    // Generate a reverse map to the string space
    // TODO: Formally should use unordered_map with C++11
    std::map<uint64_t, size_t> rev;
    for (size_t idxJ = 0; idxJ < strs.size(); idxJ++) {
        rev[strs[idxJ]] = idxJ;
    }
    std::vector<CASSingle> subs;
    for (size_t idxJ = 0; idxJ < strs.size(); idxJ++) {
        uint64_t strJ = strs[idxJ];
        for (int u = 0; u < M; u++) {
            if (!(strJ & (1ULL << u))) continue; // u|J> must be non-null
            uint64_t strJ2 = strJ ^ (1ULL << u); // N-1 electron string
            for (int t = 0; t < M; t++) {
                if (strJ2 & (1ULL << t)) continue; // t^+u|J> must be non-null
                uint64_t strI = strJ2 ^ (1ULL << t);
                size_t idxI = rev[strI]; // Index lookup
                int perm = Bits::popcount_range(strI,std::min(t,u)+1,std::max(t,u));
                int phase = 1 - 2 * (perm & 1);
                subs.push_back(CASSingle(
                    strI,
                    strJ,
                    idxI,   
                    idxJ,
                    t,
                    u,
                    phase));
            }
        }
    }

    //printf("CASDets: %zu\n", strs.size());
    //for (size_t K = 0; K < strs.size(); K++) {
    //    printf("%2zu: %zu\n", K, strs[K]);
    //}
    //printf("CASSingles: %zu\n", subs.size());
    //for (size_t S = 0; S < subs.size(); S++) {
    //    const CASSingle& sub = subs[S];
    //    printf("%2zu %2zu %2d %2d %2d\n",
    //        sub.idxI(),
    //        sub.idxJ(),
    //        sub.t(),
    //        sub.u(),
    //        sub.phase());
    //}

    return subs;
}

} // namespace lightspeed
