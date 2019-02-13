#ifndef LS_CASBOX_UTIL_HPP
#define LS_CASBOX_UTIL_HPP

#include <vector>
#include <cstddef>
#include <stdint.h>
#include "bits.hpp"

namespace lightspeed {

/**
 * struct CASSingle represents a nonzero single substitution between string I
 * and string J, removing orbital u from J and then adding orbital t. u may be
 * the same orbital as t (so net-zero substitutions are possible).
 *
 * E_tu^IJ = <I | t^+ u | J >
 * 
 * I have added a lot of computationally extraneous stuff to make things
 * readable (so you can tell what the substitution is without referring to
 * external vectors enumerating the strings, etc). This should not be a
 * problem: the storage of all of the CASSingle objects is ~50 MB for
 * CAS(16,16). -RMP
 **/
class CASSingle {

public:

CASSingle(
    uint64_t strI,
    uint64_t strJ,
    size_t idxI,
    size_t idxJ,
    int t,
    int u,
    int phase) :
    strI_(strI),
    strJ_(strJ),
    idxI_(idxI),
    idxJ_(idxJ),
    t_(t),
    u_(u),
    phase_(phase)
    {}

uint64_t strI() const { return strI_; }
uint64_t strJ() const { return strJ_; }
size_t idxI() const { return idxI_; }
size_t idxJ() const { return idxJ_; }
int t() const { return t_; }
int u() const { return u_; }
int phase() const { return phase_; }
    
private:

uint64_t strI_; // string I
uint64_t strJ_; // string J
size_t idxI_; // lexical index of I
size_t idxJ_; // lexical index of J
int t_; // orbital added to string J
int u_; // orbital removed from string J
int phase_; // +1 or -1

};

class CASBoxUtil {

public:

// => Singles for Knowles-Handy Work <= //

/**
 * Return the set of CASSingle single substitutions for M orbitals with N
 * (alpha) electrons.
 **/
static
std::vector<CASSingle> singles(int M, int N);

// => Explicit Matrix Elements via Slater's Rules <= //

static inline double compute_H00(
    uint64_t Ra,
    uint64_t Rb,
    int M,
    const double* Hp,
    const double* Ip) {
    double Sval = 0.0;
    // One-electron contributions
    for (int p = 0; p < M; p++) {
        double Hval = Hp[p*M+p];
        Sval += ((Ra&(1ULL<<p))>>p) * Hval;
        Sval += ((Rb&(1ULL<<p))>>p) * Hval;
    }
    // Two-electron contributions
    for (int p = 0; p < M; p++) {
        for (int q = 0; q < M; q++) {
            double Ival = Ip[p*M*M*M + p*M*M + q*M + q] - Ip[p*M*M*M + q*M*M + p*M + q];
            double Jval = Ip[p*M*M*M + p*M*M + q*M + q];
            // alpha alpha
            Sval += 0.5 * ((Ra&(1ULL<<p))>>p) * ((Ra&(1ULL<<q))>>q) * Ival;
            // beta beta
            Sval += 0.5 * ((Rb&(1ULL<<p))>>p) * ((Rb&(1ULL<<q))>>q) * Ival;
            // alpha beta
            Sval += 0.5 * ((Ra&(1ULL<<p))>>p) * ((Rb&(1ULL<<q))>>q) * Jval;
            // beta alpha
            Sval += 0.5 * ((Rb&(1ULL<<p))>>p) * ((Ra&(1ULL<<q))>>q) * Jval;
        }
    }
    return Sval;
    }

static inline double compute_H01(
    uint64_t Ra,
    uint64_t Lb,
    uint64_t Rb,
    int M,
    const double* Hp,
    const double* Ip) {
    uint64_t Xb = Lb ^ Rb;
    int p = Bits::ffs(Xb);
    int q = Bits::ffs(Xb ^ (1ULL<<p));
    // One-electron contribution
    double Fval = Hp[p*M+q];
    // Two-electron contribution
    for (int j = 0; j < M; j++) {
        Fval += ((Rb&(1ULL<<j))>>j) * (Ip[p*M*M*M + q*M*M + j*M + j] - Ip[p*M*M*M + j*M*M + q*M + j]);
        Fval += ((Ra&(1ULL<<j))>>j) * Ip[p*M*M*M + q*M*M + j*M + j];
    }
    int par = 1 - 2*(Bits::popcount_range(Rb,p+1,q) & 1);
    return par*Fval;
    } 

static inline double compute_H02(
    uint64_t Lb,
    uint64_t Rb,
    int M,
    const double* Ip) {
    size_t Xb = Lb ^ Rb;
    size_t L3b = Xb & Lb;
    int pL = Bits::ffs(L3b);
    int qL = Bits::ffs(L3b ^ (1ULL<<pL));
    uint64_t R3b = Xb & Rb;
    int pR = Bits::ffs(R3b);
    int qR = Bits::ffs(R3b ^ (1ULL<<pR));
    double Ival = Ip[pL*M*M*M + pR*M*M + qL*M + qR] - Ip[pL*M*M*M + qR*M*M + qL*M + pR];
    int par = 1 - 2*((Bits::popcount_range(Lb,pL+1,qL) + Bits::popcount_range(Rb,pR+1,qR)) & 1);
    return par * Ival;
    }

static inline double compute_H11(
    uint64_t La,
    uint64_t Ra,
    uint64_t Lb,
    uint64_t Rb,
    int M,
    const double* Ip) {
    uint64_t Xa = La ^ Ra;
    uint64_t Xb = Lb ^ Rb;
    int pa = Bits::ffs(Xa);
    int qa = Bits::ffs(Xa ^ (1ULL << pa));
    int pb = Bits::ffs(Xb);
    int qb = Bits::ffs(Xb ^ (1ULL << pb));
    double Ival = Ip[pa*M*M*M + qa*M*M + pb*M + qb];
    int par = 1 - 2*((Bits::popcount_range(La,pa+1,qa) + Bits::popcount_range(Lb,pb+1,qb)) & 1);
    return par * Ival;
    }

};
    
} // namespace lightspeed

#endif 
