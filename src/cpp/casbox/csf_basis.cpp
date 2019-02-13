#include <lightspeed/casbox.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/math.hpp>
#include "../util/string.hpp"
#include "bits.hpp"
#include <set>
#include <stdexcept>
#include <cmath>

namespace lightspeed {

std::vector<int> CSFBasis::seniority() const 
{
    std::vector<int> ret;
    for (auto it : seniority_blocks_) {
        ret.push_back(it.first);
    }
    return ret;
}
std::vector<std::shared_ptr<Tensor>> CSFBasis::det_to_CSF() const 
{
    std::vector<std::shared_ptr<Tensor>> ret;
    for (int Z : seniority()) {
        auto sen = seniority_blocks_.at(Z);
        if (sen->CSF_basis().count(S_)) ret.push_back(sen->CSF_basis().at(S_));
        else ret.push_back(std::shared_ptr<Tensor>(new Tensor({sen->nunpaired(), 0})));
    }
    return ret;
}
std::vector<size_t> CSFBasis::nCSF() const 
{
    std::vector<size_t> ret;
    for (int Z : seniority()) {
        auto sen = seniority_blocks_.at(Z);
        if (sen->CSF_basis().count(S_)) ret.push_back(sen->CSF_basis().at(S_)->shape()[1]);
        else ret.push_back(0);
    }
    return ret;
}
std::vector<size_t> CSFBasis::nunpaired() const 
{
    std::vector<size_t> ret;
    for (int Z : seniority()) {
        ret.push_back(seniority_blocks_.at(Z)->nunpaired());
    }
    return ret;
}
std::vector<size_t> CSFBasis::npaired() const 
{
    std::vector<size_t> ret;
    for (int Z : seniority()) {
        ret.push_back(seniority_blocks_.at(Z)->npaired());
    }
    return ret;
}
std::vector<size_t> CSFBasis::ninterleave() const 
{
    std::vector<size_t> ret;
    for (int Z : seniority()) {
        ret.push_back(seniority_blocks_.at(Z)->ninterleave());
    }
    return ret;
}
std::vector<size_t> CSFBasis::nblock() const 
{
    std::vector<size_t> ret;
    for (int Z : seniority()) {
        auto sen = seniority_blocks_.at(Z);
        ret.push_back(sen->ninterleave() * sen->npaired());
    }
    return ret;
}
std::vector<size_t> CSFBasis::sizes_CSF() const 
{
    auto nblock2 = nblock();
    auto nCSF2 = nCSF();
    std::vector<size_t> ret;
    for (size_t ind = 0; ind < nblock2.size(); ind++) {
        ret.push_back(nblock2[ind] * nCSF2[ind]);
    }
    return ret;
}
std::vector<size_t> CSFBasis::sizes_det() const 
{
    auto nblock2 = nblock();
    auto ndet2 = nunpaired();
    std::vector<size_t> ret;
    for (size_t ind = 0; ind < nblock2.size(); ind++) {
        ret.push_back(nblock2[ind] * ndet2[ind]);
    }
    return ret;
}
std::vector<size_t> CSFBasis::offsets_CSF() const
{
    auto sizes = sizes_CSF();
    std::vector<size_t> ret = {0};
    for (size_t ind = 0; ind < sizes.size() - 1; ind++) {
        ret.push_back(ret[ind] + sizes[ind]); 
    }
    return ret;
}
std::vector<size_t> CSFBasis::offsets_det() const
{
    auto sizes = sizes_det();
    std::vector<size_t> ret = {0};
    for (size_t ind = 0; ind < sizes.size() - 1; ind++) {
        ret.push_back(ret[ind] + sizes[ind]); 
    }
    return ret;
}
std::vector<std::vector<uint64_t>> CSFBasis::paired_strings() const
{
    std::vector<std::vector<uint64_t>> ret;
    for (int Z : seniority()) {
        ret.push_back(seniority_blocks_.at(Z)->paired_strings());
    }
    return ret;
}
std::vector<std::vector<uint64_t>> CSFBasis::unpaired_strings() const
{
    std::vector<std::vector<uint64_t>> ret;
    for (int Z : seniority()) {
        ret.push_back(seniority_blocks_.at(Z)->unpaired_strings());
    }
    return ret;
}
std::vector<std::vector<uint64_t>> CSFBasis::interleave_strings() const
{
    std::vector<std::vector<uint64_t>> ret;
    for (int Z : seniority()) {
        ret.push_back(seniority_blocks_.at(Z)->interleave_strings());
    }
    return ret;
}


size_t CSFBasis::total_nCSF() const
{
    std::vector<size_t> nCSF2 = nCSF();
    std::vector<size_t> nblock2 = nblock();
    size_t size = 0;
    for (size_t ind = 0; ind < nCSF2.size(); ind++) {
        size += nblock2[ind] * nCSF2[ind];
    }
    return size;
}
size_t CSFBasis::total_ndet() const
{
    std::vector<size_t> ndet2 = nunpaired();
    std::vector<size_t> nblock2 = nblock();
    size_t size = 0;
    for (size_t ind = 0; ind < ndet2.size(); ind++) {
        size += nblock2[ind] * ndet2[ind];
    }
    return size;
}

std::string CSFBasis::string() const 
{
    std::string s;
    s += "CSFBasis:\n";
    s += sprintf2("  S    = %11d\n", S());
    s += sprintf2("  nCSF = %11zu\n", total_nCSF());
    s += sprintf2("  ndet = %11zu\n", total_ndet());
    s += "\n";

    auto seniority2 = seniority();
    auto ninterleave2 = ninterleave();
    auto npaired2 = npaired();
    auto nunpaired2 = nunpaired();
    auto nCSF2 = nCSF();
    auto nblock2 = nblock();
    
    s += sprintf2("  %3s : %11s %11s %11s %11s %11s %11s %11s\n",
        "Z",
        "Ninterleave",
        "Npaired",
        "Nunpaired",
        "NCSF",
        "Sdet",
        "SCSF",
        "FLOPs"
        );
    size_t Sdet = 0;
    size_t SCSF = 0;
    size_t W = 0;
    for (size_t ind = 0; ind < seniority().size(); ind++) {
        s += sprintf2("  %3d : %11zu %11zu %11zu %11zu %11zu %11zu %11zu\n",
            seniority2[ind],
            ninterleave2[ind],
            npaired2[ind],
            nunpaired2[ind],
            nCSF2[ind],
            nblock2[ind] * nunpaired2[ind],
            nblock2[ind] * nCSF2[ind],
            nblock2[ind] * nunpaired2[ind] * nCSF2[ind]
            );
        Sdet += nblock2[ind] * nunpaired2[ind];
        SCSF += nblock2[ind] * nCSF2[ind];
        W    += nblock2[ind] * nunpaired2[ind] * nCSF2[ind];
    }
    s += sprintf2("  %3s : %11s %11s %11s %11s %11zu %11zu %11zu\n",
        "X",
        "", 
        "", 
        "", 
        "", 
        Sdet,
        SCSF,
        W);

    return s;
}

std::shared_ptr<Tensor> CSFBasis::transform_det_to_CSF(
    const std::shared_ptr<Tensor>& C) const
{
    return transform_det2_to_CSF(transform_det_to_det2(C));
}
std::shared_ptr<Tensor> CSFBasis::transform_CSF_to_det(
    const std::shared_ptr<Tensor>& C) const
{
    return transform_det2_to_det(transform_CSF_to_det2(C));
}

std::shared_ptr<Tensor> CSFBasis::transform_det2_to_CSF(
    const std::shared_ptr<Tensor>& C) const
{
    if (C->size() != total_ndet()) throw std::runtime_error("C is wrong size");
    double* Cp = C->data().data();

    std::shared_ptr<Tensor> D(new Tensor({total_nCSF()}));
    double* Dp = D->data().data(); 

    auto Ts = det_to_CSF();
    auto offsets_CSF2 = offsets_CSF();
    auto offsets_det2 = offsets_det();
    auto nblocks = nblock();
    auto ndets = nunpaired();
    auto nCSFs = nCSF();

    for (int Zind = 0; Zind < Ts.size(); Zind++) {
        if (nCSFs[Zind] == 0) continue;
        auto T = Ts[Zind];
        double* Tp = T->data().data();
        C_DGEMM('N', 'N', nblocks[Zind], nCSFs[Zind], ndets[Zind], 1.0, Cp + offsets_det2[Zind], ndets[Zind], Tp, nCSFs[Zind], 0.0, Dp + offsets_CSF2[Zind], nCSFs[Zind]); 
    } 

    return D;
}
std::shared_ptr<Tensor> CSFBasis::transform_CSF_to_det2(
    const std::shared_ptr<Tensor>& C) const
{
    if (C->size() != total_nCSF()) throw std::runtime_error("C is wrong size");
    double* Cp = C->data().data();

    std::shared_ptr<Tensor> D(new Tensor({total_ndet()}));
    double* Dp = D->data().data(); 

    auto Ts = det_to_CSF();
    auto offsets_CSF2 = offsets_CSF();
    auto offsets_det2 = offsets_det();
    auto nblocks = nblock();
    auto ndets = nunpaired();
    auto nCSFs = nCSF();

    for (int Zind = 0; Zind < Ts.size(); Zind++) {
        if (nCSFs[Zind] == 0) continue;
        auto T = Ts[Zind];
        double* Tp = T->data().data();
        C_DGEMM('N', 'T', nblocks[Zind], ndets[Zind], nCSFs[Zind], 1.0, Cp + offsets_CSF2[Zind], nCSFs[Zind], Tp, nCSFs[Zind], 0.0, Dp + offsets_det2[Zind], ndets[Zind]); 
    } 

    return D;
}

std::shared_ptr<Tensor> CSFBasis::transform_det_to_det2(
    const std::shared_ptr<Tensor>& C) const
{
    if (C->size() != total_ndet()) throw std::runtime_error("C is wrong size");
    const double* Cp = C->data().data();

    std::shared_ptr<Tensor> D(new Tensor({total_ndet()}));
    double* Dp = D->data().data(); 

    auto offsets = offsets_det();
    auto interleave_strings2 = interleave_strings();
    auto paired_strings2 = paired_strings();
    auto unpaired_strings2 = unpaired_strings();

    int M = seniority_blocks_.at(seniority()[0])->M(); // Gnarly
    uint64_t orbs = (1ULL << M) - 1; // Mask to flip holes/particles

    int Nb = seniority_blocks_.at(seniority()[0])->Nb(); // Gnarly
    size_t Sb = Bits::ncombination(M, Nb); // Gnarly

    for (int Zind = 0; Zind < offsets.size(); Zind++) {
        double* D2p = Dp + offsets[Zind];
        #pragma omp parallel for
        for (size_t IP = 0; IP < interleave_strings2[Zind].size() * paired_strings2[Zind].size(); IP++) {
            double* D3p = D2p + IP * unpaired_strings2[Zind].size();
            uint64_t I2 = interleave_strings2[Zind][IP / paired_strings2[Zind].size()];
            uint64_t P = paired_strings2[Zind][IP % paired_strings2[Zind].size()];
            uint64_t I = I2 ^ orbs;
            uint64_t P2 = Bits::expand(P, I);
        for (uint64_t U : unpaired_strings2[Zind]) {
            uint64_t U2 = Bits::expand(U, I2);
            uint64_t U3 = (~U2) & I2;
            uint64_t Ia = P2 | U2;
            uint64_t Ib = P2 | U3;
            size_t addra = Bits::combination_index(Ia);
            size_t addrb = Bits::combination_index(Ib);
            // printf("%2d: %11zu %11zu %11zu %11zu %11zu %11zu %11zu\n", Zind, I, P, U, Ia, Ib, addra, addrb);
            (*D3p++) = Cp[addra * Sb + addrb];
        }}
    }

    return D;
}
std::shared_ptr<Tensor> CSFBasis::transform_det2_to_det(
    const std::shared_ptr<Tensor>& C) const
{
    if (C->size() != total_ndet()) throw std::runtime_error("C is wrong size");
    const double* Cp = C->data().data();

    int M = seniority_blocks_.at(seniority()[0])->M();   // Gnarly
    int Na = seniority_blocks_.at(seniority()[0])->Na(); // Gnarly
    int Nb = seniority_blocks_.at(seniority()[0])->Nb(); // Gnarly
    size_t Sa = Bits::ncombination(M, Na); // Gnarly
    size_t Sb = Bits::ncombination(M, Nb); // Gnarly

    std::shared_ptr<Tensor> D(new Tensor({Sa, Sb}));
    double* Dp = D->data().data(); 

    auto offsets = offsets_det();
    auto interleave_strings2 = interleave_strings();
    auto paired_strings2 = paired_strings();
    auto unpaired_strings2 = unpaired_strings();

    uint64_t orbs = (1ULL << M) - 1; // Mask to flip holes/particles

    for (int Zind = 0; Zind < offsets.size(); Zind++) {
        const double* C2p = Cp + offsets[Zind];
        #pragma omp parallel for
        for (size_t IP = 0; IP < interleave_strings2[Zind].size() * paired_strings2[Zind].size(); IP++) {
            const double* C3p = C2p + IP * unpaired_strings2[Zind].size();
            uint64_t I2 = interleave_strings2[Zind][IP / paired_strings2[Zind].size()];
            uint64_t P = paired_strings2[Zind][IP % paired_strings2[Zind].size()];
            uint64_t I = I2 ^ orbs;
            uint64_t P2 = Bits::expand(P, I);
        for (uint64_t U : unpaired_strings2[Zind]) {
            uint64_t U2 = Bits::expand(U, I2);
            uint64_t U3 = (~U2) & I2;
            uint64_t Ia = P2 | U2;
            uint64_t Ib = P2 | U3;
            size_t addra = Bits::combination_index(Ia);
            size_t addrb = Bits::combination_index(Ib);
            // printf("%2d: %11zu %11zu %11zu %11zu %11zu %11zu %11zu\n", Zind, I, P, U, Ia, Ib, addra, addrb);
            Dp[addra * Sb + addrb] = (*C3p++);
        }}
    }

    return D;
}

} // namespace lightspeed
