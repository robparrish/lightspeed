#include <lightspeed/casbox.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/math.hpp>
#include <stdexcept>
#include "bits.hpp"
#include "casbox_util.hpp"

namespace lightspeed {

std::shared_ptr<Tensor> ExplicitCASBox::evecs(int S)
{
    if (!evecs_.count(S)) compute_block(S);
    return evecs_.at(S);
}
std::shared_ptr<Tensor> ExplicitCASBox::evals(int S)
{
    if (!evals_.count(S)) compute_block(S);
    return evals_.at(S);
}
std::shared_ptr<Tensor> ExplicitCASBox::evec(int S, size_t index)
{
    std::shared_ptr<Tensor> evecs2 = evecs(S);
    if (index >= evecs2->shape()[1]) throw std::runtime_error("ExplicitCASBox: index of state is too large.");
    std::shared_ptr<Tensor> evec2(new Tensor({evecs2->shape()[0]}));
    
    double* evec2p = evec2->data().data();
    const double* evecs2p = evecs2->data().data();
    for (size_t ind = 0; ind < evecs2->shape()[0]; ind++) {
        evec2p[ind] = evecs2p[ind * evecs2->shape()[1] + index];
    }
    return evec2;
}
double ExplicitCASBox::eval(int S, size_t index)
{
    std::shared_ptr<Tensor> evals2 = evals(S);
    if (index >= evals2->shape()[0]) throw std::runtime_error("ExplicitCASBox: index of state is too large.");
    return evals2->data()[index];
}

void ExplicitCASBox::compute_block(int S)
{
    std::shared_ptr<CSFBasis> csf = casbox_->CSF_basis(S);
    auto seniority = csf->seniority();
    auto offsets_CSF = csf->offsets_CSF();
    auto sizes_CSF = csf->sizes_CSF();

    // Construct H in CSF basis (by seniority block pair)
    size_t nCSF = csf->total_nCSF();
    std::shared_ptr<Tensor> H(new Tensor({nCSF, nCSF}));
    double* Hp = H->data().data();
    for (size_t ind1 = 0; ind1 < seniority.size(); ind1++) {
    for (size_t ind2 = 0; ind2 < seniority.size(); ind2++) {
        if (ind1 > ind2) continue; // Symmetry
        int Z1 = seniority[ind1];
        int Z2 = seniority[ind2];
        size_t size1 = sizes_CSF[ind1];
        size_t size2 = sizes_CSF[ind2];
        size_t offset1 = offsets_CSF[ind1];
        size_t offset2 = offsets_CSF[ind2];
        if (!size1) continue;
        if (!size2) continue;
        if (std::abs(Z1 - Z2) > 4) continue; // Cannot get coupling by Slater's rules
        auto sen1 = casbox_->seniority_block(Z1);
        auto sen2 = casbox_->seniority_block(Z2);
        std::shared_ptr<Tensor> Hblock = compute_Hblock(S, sen1, sen2);
        const double* H2p = Hblock->data().data();
        for (size_t i1 = 0; i1 < size1; i1++) {
        for (size_t i2 = 0; i2 < size2; i2++) { 
            Hp[(i1 + offset1) * nCSF + (i2 + offset2)] =
            Hp[(i2 + offset2) * nCSF + (i1 + offset1)] =
            H2p[i1 * size2 + i2];
        }}
    }}

    // Diagonalize H
    std::shared_ptr<Tensor> V(new Tensor({nCSF, nCSF}));
    std::shared_ptr<Tensor> h(new Tensor({nCSF}));
    Tensor::syev(H, V, h);
    
    // Store evecs/evals
    evecs_[S] = V;
    evals_[S] = h;
}
std::shared_ptr<Tensor> ExplicitCASBox::compute_Hblock(
    int S,
    const std::shared_ptr<SeniorityBlock>& sen1,
    const std::shared_ptr<SeniorityBlock>& sen2)
{
    // Potential integrals
    const double* I1p = casbox_->H()->data().data();
    const double* I2p = casbox_->I()->data().data();
    int M = casbox_->M();

    std::shared_ptr<Tensor> trans1 = sen1->CSF_basis().at(S);
    auto interleave1 = sen1->interleave_strings(); 
    auto paired1 = sen1->paired_strings(); 
    auto unpaired1 = sen1->unpaired_strings(); 
    size_t ninterleave1 = sen1->ninterleave();
    size_t npaired1 = sen1->npaired();
    size_t nunpaired1 = sen1->nunpaired();
    size_t ncsf1 = trans1->shape()[1];
    size_t total_ndet1 = ninterleave1 * npaired1 * nunpaired1;
    size_t total_ncsf1 = ninterleave1 * npaired1 * ncsf1;
    size_t nblock1 = ninterleave1 * npaired1;

    std::shared_ptr<Tensor> trans2 = sen2->CSF_basis().at(S);
    auto interleave2 = sen2->interleave_strings(); 
    auto paired2 = sen2->paired_strings(); 
    auto unpaired2 = sen2->unpaired_strings(); 
    size_t ninterleave2 = sen2->ninterleave();
    size_t npaired2 = sen2->npaired();
    size_t nunpaired2 = sen2->nunpaired();
    size_t ncsf2 = trans2->shape()[1];
    size_t total_ndet2 = ninterleave2 * npaired2 * nunpaired2;
    size_t total_ncsf2 = ninterleave2 * npaired2 * ncsf2;
    size_t nblock2 = ninterleave2 * npaired2;

    // Orbital space mask
    uint64_t orbs = (1ULL << sen1->M()) - 1;

    // Hamiltonian in det/det basis (seniority-ordered dets)
    std::shared_ptr<Tensor> H(new Tensor({total_ndet1, total_ndet2}));
    double* Hp = H->data().data();

    for (size_t indI1 = 0, index1 = 0; indI1 < ninterleave1; indI1++) {
        uint64_t I1 = interleave1[indI1];
    for (size_t indP1 = 0; indP1 < npaired1; indP1++) {
        uint64_t P1 = paired1[indP1];
    for (size_t indU1 = 0; indU1 < nunpaired1; indU1++, index1++) {
        uint64_t U1 = unpaired1[indU1];

        // Compute Ia1/Ib1
        uint64_t Id1 = I1 ^ orbs;
        uint64_t Pe1 = Bits::expand(P1, Id1);
        uint64_t Ue1 = Bits::expand(U1, I1);
        uint64_t Uf1 = (~Ue1) & I1;
        uint64_t Ia1 = Pe1 | Ue1;
        uint64_t Ib1 = Pe1 | Uf1;

    for (size_t indI2 = 0, index2 = 0; indI2 < ninterleave2; indI2++) {
        uint64_t I2 = interleave2[indI2];
    for (size_t indP2 = 0; indP2 < npaired2; indP2++) {
        uint64_t P2 = paired2[indP2];
    for (size_t indU2 = 0; indU2 < nunpaired2; indU2++, index2++) {
        uint64_t U2 = unpaired2[indU2];

        // Compute Ia2/Ib2
        uint64_t Id2 = I2 ^ orbs;
        uint64_t Pe2 = Bits::expand(P2, Id2);
        uint64_t Ue2 = Bits::expand(U2, I2);
        uint64_t Uf2 = (~Ue2) & I2;
        uint64_t Ia2 = Pe2 | Ue2;
        uint64_t Ib2 = Pe2 | Uf2;

        // Find Slater's rules intersections
        int popa = Bits::popcount(Ia1 ^ Ia2); 
        int popb = Bits::popcount(Ib1 ^ Ib2); 
        if (popa + popb > 4) continue; // Slater's rules preclude intersection

        // Slater's rules!
        double Hval = 0.0;
        if (popa == 0 && popb == 0) {
            Hval = CASBoxUtil::compute_H00(Ia1, Ib1, M, I1p, I2p);   
        } else if (popa == 0 && popb == 2) {
            Hval = CASBoxUtil::compute_H01(Ia1, Ib1, Ib2, M, I1p, I2p);
        } else if (popa == 2 && popb == 0) {
            Hval = CASBoxUtil::compute_H01(Ib1, Ia1, Ia2, M, I1p, I2p);
        } else if (popa == 0 && popb == 4) {
            Hval = CASBoxUtil::compute_H02(Ib1, Ib2, M, I2p);
        } else if (popa == 4 && popb == 0) {
            Hval = CASBoxUtil::compute_H02(Ia1, Ia2, M, I2p);
        } else if (popa == 2 && popb == 2) {
            Hval = CASBoxUtil::compute_H11(Ia1, Ia2, Ib1, Ib2, M, I2p);
        }
        Hp[index1 * total_ndet2 + index2] = Hval;
    }}}
    }}}

    std::shared_ptr<Tensor> H2(new Tensor({total_ndet1, total_ncsf2}));
    double* H2p = H2->data().data();

    double* T2p = trans2->data().data();

    C_DGEMM('N', 'N', total_ndet1 * nblock2, ncsf2, nunpaired2, 1.0, Hp, nunpaired2, T2p, ncsf2, 0.0, H2p, ncsf2);

    std::shared_ptr<Tensor> H3 = H2->transpose();
    double* H3p = H3->data().data();
    
    std::shared_ptr<Tensor> H4(new Tensor({total_ncsf2, total_ncsf1}));
    double* H4p = H4->data().data();

    double* T1p = trans1->data().data();

    C_DGEMM('N', 'N', total_ncsf2 * nblock1, ncsf1, nunpaired1, 1.0, H3p, nunpaired1, T1p, ncsf1, 0.0, H4p, ncsf1);

    return H4->transpose();
}

} // namespace lightspeed
