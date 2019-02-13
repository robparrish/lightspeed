#ifndef LS_BITS_HPP
#define LS_BITS_HPP

#include <vector>
#include <stdint.h>

namespace lightspeed {

/**
 * Class Bits is a static class providing bit fiddling tricks
 * or hardware implementations thereof
 **/
class Bits {

public:

// => Bit Fiddling Tricks <= //

/// returns the number of on bits in uint64_t x
static inline int popcount(uint64_t x) {
    x -= (x >> 1) & 0x5555555555555555; //put count of each 2 bits into those 2 bits
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333); //put count of each 4 bits into those 4 bits 
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f; //put count of each 8 bits into those 8 bits 
    return (x * 0x0101010101010101)>>56; //returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ... 
}

/// returns the number of on bits in uint64_t x in the bit range [start,stop)
static inline int popcount_range(uint64_t x, int start, int stop) {
    x -= (x >> stop) << stop;
    x >>= start;
    x -= (x >> 1) & 0x5555555555555555; //put count of each 2 bits into those 2 bits
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333); //put count of each 4 bits into those 4 bits 
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f; //put count of each 8 bits into those 8 bits 
    return (x * 0x0101010101010101)>>56; //returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ... 
}

/// returns the LSB in uint64_t b
static inline unsigned int ffs(uint64_t b) {
    const uint64_t magic = 0x022fdd63cc95386d; // the 4061955.
    const unsigned int magictable[64] =
    {
        0,  1,  2, 53,  3,  7, 54, 27,
        4, 38, 41,  8, 34, 55, 48, 28,
       62,  5, 39, 46, 44, 42, 22,  9,
       24, 35, 59, 56, 49, 18, 29, 11,
       63, 52,  6, 26, 37, 40, 33, 47,
       61, 45, 43, 21, 23, 58, 17, 10,
       51, 25, 36, 32, 60, 20, 57, 16,
       50, 31, 19, 15, 30, 14, 13, 12,
    };
    return magictable[((b&-b)*magic) >> 58];
}

/// returns the next lexographic permutation of uint64_t b
static inline uint64_t next_combination(uint64_t b) {
    uint64_t t = b | (b-1);
    return (t+1) | (((~t & -~t) - 1) >> ((Bits::ffs(b))+1));
    }

// returns all combinations of N things chosen K at a time
static std::vector<uint64_t> combinations(int N, int K) {
    uint64_t S = (1ULL<<K)-1; 
    std::vector<uint64_t> strings;
    while (S < (1ULL<<N)) {
        strings.push_back(S);
        S = Bits::next_combination(S);
    }
    return strings;
}

// returns (N choose K) = N! / (K! * (N-K)!)
static size_t ncombination(size_t N, size_t K) {
    if (K > N) {
        return 0;
    }
    size_t r = 1;
    for (size_t d = 1; d <= K; ++d) {
        r *= N--;
        r /= d;
    }
    return r;
}

// Return the ordered vector of set bits in string I
static std::vector<int> bits(uint64_t I) {
    std::vector<int> B2(Bits::popcount(I));
    int i = 0;
    while (I) {
        B2[i] = Bits::ffs(I);
        I ^= (1ULL<<B2[i]);
        i++;
    }
    return B2;
}

// Expand x to fill the places marked in m
static inline uint64_t expand(uint64_t x, uint64_t m)
{
    uint64_t y = 0;
    while (m) {
        int i = Bits::ffs(m);
        m ^= (1ULL << i);
        y |= (x & 1) << i;
        x >>= 1; 
    }
    return y;
}

static inline int combination_index(uint64_t I)
{
    // Will handle up to 8 bits in 16 places
    // TODO: Extend to more bits
    const unsigned int lookup[][9] = 
        {{    0,    1,    2,    3,    4,     5,     6,     7,     8},
         {    0,    1,    3,    6,   10,    15,    21,    28,    36},
         {    0,    1,    4,   10,   20,    35,    56,    84,   120},
         {    0,    1,    5,   15,   35,    70,   126,   210,   330},
         {    0,    1,    6,   21,   56,   126,   252,   462,   792},
         {    0,    1,    7,   28,   84,   210,   462,   924,  1716},
         {    0,    1,    8,   36,  120,   330,   792,  1716,  3432},
         {    0,    1,    9,   45,  165,   495,  1287,  3003,  6435}};

    int index = 0; int k = 0;
    while (I) {
        int l = Bits::ffs(I);
        I ^= (1ULL << l);
        index += lookup[k][l-k];
        k++;
    }
    
    return index;
}

};

} // namespace lightspeed
    
#endif
