#include <lightspeed/casbox.hpp>
#include <lightspeed/tensor.hpp>
#include "../util/string.hpp"
#include "bits.hpp"
#include <set>
#include <stdexcept>
#include <cmath>

namespace lightspeed {

SeniorityBlock::SeniorityBlock(
    int M,
    int Na,
    int Nb,
    int Z
    ) : 
    M_(M),
    Na_(Na),
    Nb_(Nb),
    Z_(Z)
{
    // => Validity Checks <= //

    if (Na_ > M_) throw std::runtime_error("SeniorityBlock: Na must be <= M");
    if (Nb_ > M_) throw std::runtime_error("SeniorityBlock: Nb must be <= M");
    if (Na_ < Nb_) throw std::runtime_error("SeniorityBlock: Na must be >= Nb");
    if (Z_ > M_) throw std::runtime_error("SeniorityBlock: Z must be <= M");
    if (Z_ < H()) throw std::runtime_error("SeniorityBlock: Z must be >= H");
    if ((Z_ - H()) % 2 != 0) throw std::runtime_error("SeniorityBlock: Z-H must be even");
    
    // => DUAB Sizing <= //
    
    A_ = (Z_ - H()) / 2 + H(); 
    B_ = (Z_ - H()) / 2; 
    D_ = Na_ - A_; // Should also equal Nb_ - B_
    U_ = M_ - D_ - Z_;

    // => CSF Basis <= //
    
    compute_CSF_basis(1.0E-10); // Quick test of (16,8) is showing 5.E-13 deviations
}
std::vector<uint64_t> SeniorityBlock::unpaired_strings() const 
{
    return Bits::combinations(A() + B(), A());
}
std::vector<uint64_t> SeniorityBlock::paired_strings() const 
{
    return Bits::combinations(D() + U(), D());
}
std::vector<uint64_t> SeniorityBlock::interleave_strings() const 
{
    return Bits::combinations(M(), A() + B());
}
size_t SeniorityBlock::nunpaired() const 
{
    return Bits::ncombination(A() + B(), A());
}
size_t SeniorityBlock::npaired() const 
{
    return Bits::ncombination(D() + U(), D());
}
size_t SeniorityBlock::ninterleave() const 
{
    return Bits::ncombination(M(), Z());
}
std::shared_ptr<Tensor> SeniorityBlock::compute_S2() const
{
    std::vector<uint64_t> strings = unpaired_strings();    

    std::shared_ptr<Tensor> S2t(new Tensor({strings.size(), strings.size()}));
    double* S2p = S2t->data().data();

    double diag = 0.5 * (A() + B()) + pow(0.5 * (A() - B()), 2);

    for (size_t I1 = 0; I1 < strings.size(); I1++) {
        uint64_t S1 = strings[I1];
        for (size_t I2 = 0; I2 < strings.size(); I2++) {
            uint64_t S2 = strings[I2];
            int dist = Bits::popcount(S1 ^ S2);
            if (dist == 0) {
                // Diagonal
                (*S2p++) = diag;
            } else if (dist == 2) {
                // Off-diagonal
                uint64_t val = S1 ^ S2;
                int i1 = Bits::ffs(val);
                val ^= (1ULL << i1);
                int i2 = Bits::ffs(val);
                int delta = i2 - i1;
                // -1 if delta is odd, +1 if delta is even
                (*S2p++) = 1 - 2 * (delta % 2);
            } else {
                S2p++;
            }
        }
    }
    return S2t;
}
void SeniorityBlock::compute_CSF_basis(
    double thre)
{
    // Get S2 operator
    std::shared_ptr<Tensor> S2 = compute_S2();

    // Sizing
    size_t ndet = S2->shape()[0];

    // Diagonalize S2 operator
    std::shared_ptr<Tensor> V2(new Tensor(S2->shape()));
    std::shared_ptr<Tensor> s2(new Tensor({S2->shape()[0]}));
    Tensor::syev(S2, V2, s2);
    const double* V2p = V2->data().data();
    const double* s2p = s2->data().data();

    // Classify and partition S2 eigenvalues
    std::set<size_t> valid_inds;
    for (int Sind = H(); Sind <= Z(); Sind += 2) {
        double S = Sind / 2.0;
        double Sev = S * (S + 1.0);
    
        size_t nCSF = 0;
        for (size_t ind = 0; ind < ndet; ind++) {
            if (fabs(s2p[ind] - Sev) < thre) {
                valid_inds.insert(ind);
                nCSF++;
            }
        }

        CSF_basis_[Sind] = std::shared_ptr<Tensor>(new Tensor({ndet, nCSF}));
        double* V3p = CSF_basis_[Sind]->data().data();

        for (size_t ind = 0, ind3 = 0; ind < ndet; ind++) {
            if (fabs(s2p[ind] - Sev) < thre) {
                for (size_t ind2 = 0; ind2 < ndet; ind2++) {
                    V3p[ind2*nCSF + ind3] = V2p[ind2*ndet + ind];
                }
                ind3++;
            }
        }
    }

    // Check that all eigenvalues were classified
    if (valid_inds.size() != ndet) throw std::runtime_error("SeniorityBlock: did not find all S^2 eigenvalues expected");
}

} // namespace lightspeed
