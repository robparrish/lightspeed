#include <lightspeed/casbox.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/resource_list.hpp>
#include <lightspeed/math.hpp>
#include "../util/string.hpp"
#include "bits.hpp"
#include "casbox_util.hpp"
#include <set>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace lightspeed {

namespace {

bool is_terachem_ready(const std::shared_ptr<ResourceList>& resources)
{
#ifdef HAVE_TERACHEM
    if (resources->ngpu() == 0) return false;
    return true;
#else
    return false;
#endif
}

}

CASBox::CASBox(
    int M,
    int Na,
    int Nb,
    const std::shared_ptr<Tensor>& H,
    const std::shared_ptr<Tensor>& I
    ) :
    M_(M),
    Na_(Na),
    Nb_(Nb),
    H_(H),
    I_(I)
{

    // => Validity Checks <= //

    if (M_ > 16) throw std::runtime_error("CASBox: M <= 16 for Bits class."); // TODO: Fix this
    if (Na_ > M_) throw std::runtime_error("CASBox: Na must be <= M");
    if (Nb_ > M_) throw std::runtime_error("CASBox: Nb must be <= M");
    if (Na_ < Nb_) throw std::runtime_error("CASBox: Na must be >= Nb");

    H->shape_error({(size_t)M_,(size_t)M_});
    I->shape_error({(size_t)M_,(size_t)M_,(size_t)M_,(size_t)M_});

    // => Modified Potential Integrals <= //
    
    K_ = H_->clone();
    double* Kp = K_->data().data();
    double* Ip = I_->data().data();
    for (int t = 0; t < M_; t++) {
    for (int u = 0; u < M_; u++) {
    for (int v = 0; v < M_; v++) {
        Kp[t*M_ + u] -= 0.5 * Ip[t*M_*M_*M_ + v*M_*M_ + v*M_ + u];
    }}}

    // => String Dimensions <= //
    
    Da_ = Bits::ncombination(M_,Na_);
    Db_ = Bits::ncombination(M_,Nb_);

    // => SeniorityBlock Setup <= //
    
    seniority_blocks_.clear();
    for (int Z = min_seniority(); Z <= max_seniority(); Z += 2) {
        seniority_blocks_[Z] = std::shared_ptr<SeniorityBlock>(new SeniorityBlock(
            M_,
            Na_,
            Nb_,
            Z));
    } 

    // => CSFBasis Setup <= //
    
    CSF_basis_.clear();
    for (int Z = min_seniority(); Z <= max_seniority(); Z += 2) {
        CSF_basis_[Z] = std::shared_ptr<CSFBasis>(new CSFBasis(
            Z,
            seniority_blocks_));
    }

    // => Evangelisti Preconditioner Setup <= //
    
    build_evangelisti();
}
std::vector<uint64_t> CASBox::stringsA() const
{
    return Bits::combinations(M_,Na_);
}
std::vector<uint64_t> CASBox::stringsB() const
{
    return Bits::combinations(M_,Nb_);
}
std::string CASBox::string() const
{
    std::string s;
    s += "CASBox:\n";
    s += sprintf2("  Number of Spatial Orbitals = %11d\n", M());
    s += sprintf2("  Number of Total Electrons  = %11d\n", Na() + Nb());
    s += sprintf2("  Number of Alpha Electrons  = %11d\n", Na());
    s += sprintf2("  Number of Beta  Electrons  = %11d\n", Nb());
    s += sprintf2("  Number of Alpha Strings    = %11zu\n", Da());
    s += sprintf2("  Number of Beta  Strings    = %11zu\n", Db());
    s += sprintf2("  Number of Total Strings    = %11zu\n", D());
    return s;
}

int CASBox::max_seniority() const
{
    int Nhigh = Na_ - Nb_;   // High spin electrons are always unpaired
    int Mfree = M_ - Nhigh;  // Leaving a set of free orbitals with an even number of pairable electrons 
    int Nhole = Nb_;         // The maximum number of docc
    int Npart = Mfree - Nb_; // The maximum number of uocc
    return 2 * std::min(Nhole, Npart) + Nhigh; // docc/uocc can combine to increase seniority by 2, plus high-spin
}
std::vector<int> CASBox::seniority() const
{
    std::vector<int> ret;
    for (int Z = min_seniority(); Z <= max_seniority(); Z += 2) {
        ret.push_back(Z);
    }
    return ret;
}
std::shared_ptr<SeniorityBlock> CASBox::seniority_block(int Z) const
{
    if (!seniority_blocks_.count(Z)) throw std::runtime_error("Key is not in seniority_blocks");
    return seniority_blocks_.at(Z);
}
std::shared_ptr<CSFBasis> CASBox::CSF_basis(int Z) const
{
    if (!CSF_basis_.count(Z)) throw std::runtime_error("Key is not in CSF_basis");
    return CSF_basis_.at(Z);
}
void CASBox::build_evangelisti()
{
    // Evangelisti Fock matrix
    std::vector<size_t> dim1;
    dim1.push_back(M_);
    Fpre_ = std::shared_ptr<Tensor>(new Tensor(dim1, "Fpre"));
    double* Hp = H_->data().data();
    double* Ip = I_->data().data();
    double* Fp = Fpre_->data().data();
    for (int p = 0; p < M_; p++) {
        Fp[p] = Hp[p*M_ + p];
        for (int j = 0; j < Na_; j++) {
            Fp[p] += 2. * Ip[p*M_*M_*M_ + p*M_*M_ + j*M_ + j] -
                     1. * Ip[p*M_*M_*M_ + j*M_*M_ + p*M_ + j];
        }
    }
    
    // Evangelisti reference determinant energy
    E0pre_ = 0.0;
    for (int i = 0; i < Nb_; i++) {
        E0pre_ += Hp[i * M_ + i] + Fp[i];
    }

    // Evangelisti preconditioner
    Hpre_.clear();
    for (int S = min_seniority(); S <= max_seniority(); S += 2) {
        Hpre_[S] = compute_Hpre(S);
    }
}
std::shared_ptr<Tensor> CASBox::compute_Hpre(int S) const
{
    // Get the CSF basis object
    std::shared_ptr<CSFBasis> csf = CSF_basis(S); 
    
    // Target
    std::shared_ptr<Tensor> H(new Tensor({csf->total_nCSF()}));
    double* Hp = H->data().data();

    // Seniority block info
    auto offsets = csf->offsets_CSF();
    auto interleave_strings = csf->interleave_strings();
    auto paired_strings = csf->paired_strings();
    auto unpaired_strings = csf->unpaired_strings();
    auto nCSF = csf->nCSF();

    // Reference alpha/beta string
    uint64_t ref = (1ULL << Nb_) - 1;
    // Mask to flip holes/particles
    uint64_t orbs = (1ULL << M_) - 1;

    // Evangelisti orbital eigenvalues
    const double* Fp = Fpre_->data().data();

    for (size_t Zind = 0; Zind < offsets.size(); Zind++) {
        double* H2p = Hp + offsets[Zind];
        // Index the seniority block
        for (uint64_t I2 : interleave_strings[Zind]) { // Interleaves of A/B orbitals
        for (uint64_t P : paired_strings[Zind]) {     // D strings (relative)
            uint64_t I = I2 ^ orbs;                   // Interleaves of D/U orbitals
            uint64_t P2 = Bits::expand(P, I);         // D strings (absolute)
            uint64_t Ia = P2 + I2; // Alpha string (absolute)
            uint64_t Ib = P2;      // Beta string (absolute)

            // Preconditioner value
            double val = E0pre_;
            for (int i = 0; i < Nb_; i++) {
                val -= (((1ULL << i) & (~Ia)) >> i) * Fp[i];
                val -= (((1ULL << i) & (~Ib)) >> i) * Fp[i];
            }
            for (int i = Nb_; i < M_; i++) {
                val += (((1ULL << i) & (Ia)) >> i) * Fp[i];
                val += (((1ULL << i) & (Ib)) >> i) * Fp[i];
            }

            // Assignment (same for all elements of spin-coupling block)
            for (int cind = 0; cind < nCSF[Zind]; cind++) {
               (*H2p++) = val;
            } 
        }}
    }

    return H;
}
std::shared_ptr<Tensor> CASBox::H_evangelisti(int Z) const
{
    if (!Hpre_.count(Z)) throw std::runtime_error("Key is not in Hpre");
    return Hpre_.at(Z);
}
std::shared_ptr<Tensor> CASBox::apply_evangelisti(
    int S, 
    const std::shared_ptr<Tensor>& C,
    double E) const
{
    std::shared_ptr<Tensor> H = H_evangelisti(S);
    C->shape_error(H->shape());
    std::shared_ptr<Tensor> V = C->clone();
    double* Vp = V->data().data();
    const double* Hp = H->data().data();
    size_t ndet = V->size();
    for (size_t ind = 0; ind < ndet; ind++) {
        double Hval = Hp[ind] - E;
        Hval = std::copysign(std::max(fabs(Hval), 1.0E-5), Hval); // To prevent singularities
        Vp[ind] /= - Hval;
    }
    return V; 
}
std::vector<std::shared_ptr<Tensor>> CASBox::guess_evangelisti(
    int S,
    size_t nguess) const
{
    std::shared_ptr<Tensor> H = H_evangelisti(S);
    const double* Hp = H->data().data();
    size_t ndet = H->size();
    if (nguess > ndet) throw std::runtime_error("Too many guesses requested");
    
    std::shared_ptr<CSFBasis> csf = CSF_basis(S); 
    auto sizes = csf->sizes_CSF();
    auto offsets = csf->offsets_CSF();
    auto seniority = csf->seniority();
    auto nblock = csf->nblock();
    auto nCSF = csf->nCSF();

    std::vector<std::tuple<double, size_t, size_t>> order;
    for (size_t Zind = 0; Zind < sizes.size(); Zind++) {
        const double* H2p = Hp + offsets[Zind];
        for (size_t ind = 0; ind < sizes[Zind]; ind++) {
            order.push_back(std::tuple<double, size_t, size_t>((*H2p++), Zind, ind));
        }
    }

    std::sort(order.begin(), order.end());

    std::vector<std::shared_ptr<Tensor>> ret;
    std::set<std::pair<size_t, size_t>> found_blocks;
    for (auto entry : order) {
        size_t Zind = std::get<1>(entry);
        size_t ind = std::get<2>(entry);
        size_t block = ind / nCSF[Zind];
        if (found_blocks.count(std::pair<size_t, size_t>(Zind, block))) continue;
        found_blocks.insert(std::pair<size_t, size_t>(Zind, block));
        for (int cind = 0; cind < nCSF[Zind]; cind++) {
            std::shared_ptr<Tensor> g(new Tensor({ndet}));
            double* gp = g->data().data();
            gp[offsets[Zind] + block * nCSF[Zind] + cind] = 1.0;
            ret.push_back(g);
        }
        if (ret.size() >= nguess) break; 
    }
    
    return ret;
}

std::shared_ptr<Tensor> CASBox::sigma_det(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& C) const
{

    C->shape_error({Da(),Db()});

    // => Setup <= //

    // Single-Substitution coupling coefficients
    std::vector<CASSingle> subsA = CASBoxUtil::singles(M_, Na_);
    std::vector<CASSingle> subsB = CASBoxUtil::singles(M_, Nb_);

    // C++03
    std::vector<size_t> dimD1;
    dimD1.push_back(M_);
    dimD1.push_back(M_);
    dimD1.push_back(Da_);
    dimD1.push_back(Db_);
     
    // Pre-transformed intermediate
    std::shared_ptr<Tensor> D(new Tensor({(size_t)M_, (size_t)M_, Da_, Db_}));
    
    // Post-transformed intermediate
    std::shared_ptr<Tensor> G(new Tensor({(size_t)M_, (size_t)M_, Da_, Db_}));

    // Target
    std::shared_ptr<Tensor> S(new Tensor({Da_, Db_}));

    // Pointers
    double* Cp = C->data().data();
    double* Sp = S->data().data();
    double* Dp = D->data().data();
    double* Gp = G->data().data();

    // => First-Half Transform <= //

    #pragma omp parallel for num_threads(resources->nthread())
    for (size_t subs_ind = 0; subs_ind < subsA.size(); subs_ind++) {
        const CASSingle& sub = subsA[subs_ind];
        size_t idxI = sub.idxI();
        size_t idxJ = sub.idxJ();
        int t = sub.t();
        int u = sub.u();
        int phase = sub.phase();
        double* D2p = Dp + t*M_*Da_*Db_ + u*Da_*Db_;
        double* C3p = Cp  + idxJ * Db_;
        double* D3p = D2p + idxI * Db_;
        for (size_t ind = 0; ind < Db_; ind++) {
            #pragma omp atomic
            (*D3p++) += phase * (*C3p++);
        }
    } 
    
    #pragma omp parallel for num_threads(resources->nthread())
    for (size_t subs_ind = 0; subs_ind < subsB.size(); subs_ind++) {
        const CASSingle& sub = subsB[subs_ind];
        size_t idxI = sub.idxI();
        size_t idxJ = sub.idxJ();
        int t = sub.t();
        int u = sub.u();
        int phase = sub.phase();
        double* D2p = Dp + t*M_*Da_*Db_ + u*Da_*Db_;
        double* C3p = Cp  + idxJ;
        double* D3p = D2p + idxI;
        for (size_t ind = 0; ind < Da_; ind++) {
            #pragma omp atomic
            (*D3p) += phase * (*C3p);
            D3p += Db_;
            C3p += Db_;
        }
    } 

    // => Potential Integral Contraction <= //

    std::vector<std::string> Iinds;
    Iinds.push_back("t");
    Iinds.push_back("u");
    Iinds.push_back("v");
    Iinds.push_back("w");

    std::vector<std::string> Dinds;
    Dinds.push_back("v");
    Dinds.push_back("w");
    Dinds.push_back("Ja");
    Dinds.push_back("Jb");

    std::vector<std::string> Ginds;
    Ginds.push_back("t");
    Ginds.push_back("u");
    Ginds.push_back("Ja");
    Ginds.push_back("Jb");

    std::vector<std::string> Kinds;
    Kinds.push_back("v");
    Kinds.push_back("w");

    std::vector<std::string> Sinds;
    Sinds.push_back("Ja");
    Sinds.push_back("Jb");

    // DGEMM by 2EIs    
    Tensor::einsum(
        Iinds,
        Dinds,
        Ginds,
        I_,
        D,
        G,
        0.5,
        0.0); 

    // DAXPY by 1EIs
    Tensor::einsum(
        Kinds,
        Dinds,
        Sinds,
        K_,
        D,
        S,
        1.0,
        0.0);

    // => Second-Half Transform <= //

    #pragma omp parallel for num_threads(resources->nthread())
    for (size_t subs_ind = 0; subs_ind < subsA.size(); subs_ind++) {
        const CASSingle& sub = subsA[subs_ind];
        size_t idxI = sub.idxI();
        size_t idxJ = sub.idxJ();
        int t = sub.t();
        int u = sub.u();
        int phase = sub.phase();
        double* G2p = Gp + t*M_*Da_*Db_ + u*Da_*Db_;
        double* S3p = Sp  + idxJ * Db_;
        double* G3p = G2p + idxI * Db_;
        for (size_t ind = 0; ind < Db_; ind++) {
          #pragma omp atomic
          (*S3p++) += phase * (*G3p++);
        }
    } 
    
    #pragma omp parallel for num_threads(resources->nthread())
    for (size_t subs_ind = 0; subs_ind < subsB.size(); subs_ind++) {
        const CASSingle& sub = subsB[subs_ind];
        size_t idxI = sub.idxI();
        size_t idxJ = sub.idxJ();
        int t = sub.t();
        int u = sub.u();
        int phase = sub.phase();
        double* G2p = Gp + t*M_*Da_*Db_ + u*Da_*Db_;
        double* S3p = Sp  + idxJ;
        double* G3p = G2p + idxI;
        for (size_t ind = 0; ind < Da_; ind++) {
          #pragma omp atomic
	  (*S3p) += phase * (*G3p);
            G3p += Db_;
            S3p += Db_;
        }
    } 

    //CIBox Testing
    //std::shared_ptr<Tensor> bsf = sigma_det_gpu(resources,C);
    //double *bsfp = bsf->data().data();

    //double accum = 0.0;
    //for (size_t i=0; i<Da_*Db_; ++i) {
    //accum += fabs(bsfp[i] - Sp[i]);
    //}
    //printf("The error in sigma is %18.14lf\n", accum);

    return S;
}
std::shared_ptr<Tensor> CASBox::opdm_det(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& A,
    const std::shared_ptr<Tensor>& B,
    bool total) const
{
    // C++03
    std::vector<size_t> dim2;
    dim2.push_back(Da()); 
    dim2.push_back(Db()); 
    A->shape_error(dim2);
    B->shape_error(dim2);
    // C++11
    //A->shape_error({Sa(),Sb()});
    //B->shape_error({Sa(),Sb()});

    // => Setup <= //

    // Target
    // C++03
    std::vector<size_t> dimN;
    dimN.push_back(M_);
    dimN.push_back(M_);
    std::shared_ptr<Tensor> D(new Tensor(dimN));
    
    // Pointers
    double* Ap = A->data().data();
    double* Bp = B->data().data();
    double* Dp = D->data().data();

    // Sign for beta contribution
    double signB = (total ? 1.0 : -1.0);
    
    // Single-Substitution coupling coefficients
    std::vector<CASSingle> subsA = CASBoxUtil::singles(M_, Na_);
    std::vector<CASSingle> subsB = CASBoxUtil::singles(M_, Nb_);

    // => OPDM Construction <= //

    #pragma omp parallel for num_threads(resources->nthread())
    for (size_t subs_ind = 0; subs_ind < subsA.size(); subs_ind++) {
        const CASSingle& sub = subsA[subs_ind];
        size_t idxI = sub.idxI();
        size_t idxJ = sub.idxJ();
        int t = sub.t();
        int u = sub.u();
        int phase = sub.phase();
        double* A3p = Ap + idxI * Db_;
        double* B3p = Bp + idxJ * Db_;
        double val = 0.0;
        for (size_t ind = 0; ind < Db_; ind++) {
            val += (*A3p++) * (*B3p++);
        }
        #pragma omp atomic
        Dp[t * M_ + u] += phase * val;
    } 
    
    #pragma omp parallel for num_threads(resources->nthread())
    for (size_t subs_ind = 0; subs_ind < subsB.size(); subs_ind++) {
        const CASSingle& sub = subsB[subs_ind];
        size_t idxI = sub.idxI();
        size_t idxJ = sub.idxJ();
        int t = sub.t();
        int u = sub.u();
        int phase = sub.phase();
        double* A3p = Ap + idxI;
        double* B3p = Bp + idxJ;
        double val = 0.0;
        for (size_t ind = 0; ind < Da_; ind++) {
            val += (*A3p) * (*B3p);
            A3p += Db_;
            B3p += Db_;
        }
        #pragma omp atomic
        Dp[t * M_ + u] += signB * phase * val;
    } 

    //CIBox Testing
    //std::shared_ptr<Tensor> bsf = opdm_det_gpu(resources,A,B);
    //double *bsfp = bsf->data().data();

    //double accum = 0.0;
    //for (size_t i=0; i<M_*M_; ++i) {
    //accum += fabs(bsfp[i] - Dp[i]);
    //}
    //printf("The error in opdm is %18.14lf\n", accum);
    
    return D; 
}
std::shared_ptr<Tensor> CASBox::tpdm_det(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& A,
    const std::shared_ptr<Tensor>& B,
    bool symmetrize) const
{
    // C++03
    std::vector<size_t> dim2;
    dim2.push_back(Da()); 
    dim2.push_back(Db()); 
    A->shape_error(dim2);
    B->shape_error(dim2);
    // C++11
    //A->shape_error({Sa(),Sb()});
    //B->shape_error({Sa(),Sb()});

    // => Setup <= //
      
    // C++03
    std::vector<size_t> dimD1;
    dimD1.push_back(M_);
    dimD1.push_back(M_);
    dimD1.push_back(Da_);
    dimD1.push_back(Db_);
     
    // Pre-transformed intermediate
    std::shared_ptr<Tensor> L(new Tensor(dimD1));
    
    // Post-transformed intermediate
    std::shared_ptr<Tensor> R(new Tensor(dimD1));

    // Target
    // C++03
    std::vector<size_t> dimN;
    dimN.push_back(M_);
    dimN.push_back(M_);
    dimN.push_back(M_);
    dimN.push_back(M_);
    std::shared_ptr<Tensor> D(new Tensor(dimN));

    // Pointers
    double* Ap = A->data().data();
    double* Bp = B->data().data();
    double* Lp = L->data().data();
    double* Rp = R->data().data();
    double* Dp = D->data().data();

    // Single-Substitution coupling coefficients
    std::vector<CASSingle> subsA = CASBoxUtil::singles(M_, Na_);
    std::vector<CASSingle> subsB = CASBoxUtil::singles(M_, Nb_);

    // => Bra Transform <= //

    #pragma omp parallel for num_threads(resources->nthread())
    for (size_t subs_ind = 0; subs_ind < subsA.size(); subs_ind++) {
        const CASSingle& sub = subsA[subs_ind];
        size_t idxI = sub.idxI();
        size_t idxJ = sub.idxJ();
        int t = sub.t();
        int u = sub.u();
        int phase = sub.phase();
        double* L2p = Lp + t*M_*Da_*Db_ + u*Da_*Db_;
        double* A3p = Ap  + idxJ * Db_;
        double* L3p = L2p + idxI * Db_;
        for (size_t ind = 0; ind < Db_; ind++) {
            #pragma omp atomic
            (*L3p++) += phase * (*A3p++);
        }
    } 

    #pragma omp parallel for num_threads(resources->nthread())
    for (size_t subs_ind = 0; subs_ind < subsB.size(); subs_ind++) {
        const CASSingle& sub = subsB[subs_ind];
        size_t idxI = sub.idxI();
        size_t idxJ = sub.idxJ();
        int t = sub.t();
        int u = sub.u();
        int phase = sub.phase();
        double* L2p = Lp + t*M_*Da_*Db_ + u*Da_*Db_;
        double* A3p = Ap  + idxJ;
        double* L3p = L2p + idxI;
        for (size_t ind = 0; ind < Da_; ind++) {
            #pragma omp atomic
            (*L3p) += phase * (*A3p);
            L3p += Db_;
            A3p += Db_;
        }
    } 

    // => Ket Transform <= //

    #pragma omp parallel for num_threads(resources->nthread())
    for (size_t subs_ind = 0; subs_ind < subsA.size(); subs_ind++) {
        const CASSingle& sub = subsA[subs_ind];
        size_t idxI = sub.idxI();
        size_t idxJ = sub.idxJ();
        int t = sub.t();
        int u = sub.u();
        int phase = sub.phase();
        double* R2p = Rp + t*M_*Da_*Db_ + u*Da_*Db_;
        double* B3p = Bp  + idxJ * Db_;
        double* R3p = R2p + idxI * Db_;
        for (size_t ind = 0; ind < Db_; ind++) {
            #pragma omp atomic
            (*R3p++) += phase * (*B3p++);
        }
    } 

    #pragma omp parallel for num_threads(resources->nthread())
    for (size_t subs_ind = 0; subs_ind < subsB.size(); subs_ind++) {
        const CASSingle& sub = subsB[subs_ind];
        size_t idxI = sub.idxI();
        size_t idxJ = sub.idxJ();
        int t = sub.t();
        int u = sub.u();
        int phase = sub.phase();
        double* R2p = Rp + t*M_*Da_*Db_ + u*Da_*Db_;
        double* B3p = Bp  + idxJ;
        double* R3p = R2p + idxI;
        for (size_t ind = 0; ind < Da_; ind++) {
            #pragma omp atomic
            (*R3p) += phase * (*B3p);
            R3p += Db_;
            B3p += Db_;
        }
    } 

    // => Contraction to TPDM <= //
    
    std::vector<std::string> Linds;
    Linds.push_back("t");
    Linds.push_back("u");
    Linds.push_back("Ka");
    Linds.push_back("Kb");

    std::vector<std::string> Rinds;
    Rinds.push_back("v");
    Rinds.push_back("w");
    Rinds.push_back("Ka");
    Rinds.push_back("Kb");

    std::vector<std::string> Dinds;
    Dinds.push_back("t");
    Dinds.push_back("u");
    Dinds.push_back("v");
    Dinds.push_back("w");

    Tensor::einsum(
        Linds,
        Rinds,
        Dinds,
        L,
        R,
        D,
        0.5,
        0.0); 

    // => OPDM Contribution <= //
    
    std::shared_ptr<Tensor> D1 = opdm_det(
        resources,
        A,
        B,
        true);
    double* D1p = D1->data().data();

    for (int t = 0; t < M_; t++) {
    for (int w = 0; w < M_; w++) {
    for (int u = 0; u < M_; u++) {
        Dp[t*M_*M_*M_ + u*M_*M_ + u*M_ + w] -= 0.5 * D1p[t*M_ + w];
    }}}

    // => Symmetrization <= //

    if (!symmetrize) return D;

    std::shared_ptr<Tensor> D2(new Tensor(dimN));
    double* D2p = D2->data().data();
    for (int t = 0; t < M_; t++) {
    for (int u = 0; u < M_; u++) {
    for (int v = 0; v < M_; v++) {
    for (int w = 0; w < M_; w++) {
        D2p[t*M_*M_*M_ + u*M_*M_ + v*M_ + w] = 0.25 * (
            Dp[t*M_*M_*M_ + u*M_*M_ + v*M_ + w] +
            Dp[t*M_*M_*M_ + u*M_*M_ + w*M_ + v] +
            Dp[u*M_*M_*M_ + t*M_*M_ + v*M_ + w] +
            Dp[u*M_*M_*M_ + t*M_*M_ + w*M_ + v]);
    }}}}

    //CIBox Testing
    //std::shared_ptr<Tensor> bsf = tpdm_det_gpu(resources,A,B,true);
    //double *bsfp = bsf->data().data();

    //double accum = 0.0;
    //for (size_t i=0; i<M_*M_*M_*M_; ++i) {
    //accum += fabs(bsfp[i] - D2p[i]);
    //}
    //printf("The error in tpdm is %18.14lf\n", accum);

    return D2;
    
}
std::shared_ptr<Tensor> CASBox::sigma(
    const std::shared_ptr<ResourceList>& resources,
    int S,
    const std::shared_ptr<Tensor>& C) const
{
    std::shared_ptr<Tensor> C2 = CSF_basis(S)->transform_CSF_to_det(C);
    std::shared_ptr<Tensor> HC2;

    if (is_terachem_ready(resources))
      HC2 = sigma_det_gpu(resources, C2);
    else
      HC2 = sigma_det(resources, C2);

    return CSF_basis(S)->transform_det_to_CSF(HC2);
}
std::shared_ptr<Tensor> CASBox::opdm(
    const std::shared_ptr<ResourceList>& resources,
    int S,
    const std::shared_ptr<Tensor>& A,
    const std::shared_ptr<Tensor>& B,
    bool total) const
{
    std::shared_ptr<Tensor> A2 = CSF_basis(S)->transform_CSF_to_det(A);
    std::shared_ptr<Tensor> B2 = (A == B ? A2 : CSF_basis(S)->transform_CSF_to_det(B));

    if (is_terachem_ready(resources))
      return opdm_det_gpu(resources, A2, B2);
    else
      return opdm_det(resources, A2, B2, total);
}
std::shared_ptr<Tensor> CASBox::tpdm(
    const std::shared_ptr<ResourceList>& resources,
    int S,
    const std::shared_ptr<Tensor>& A,
    const std::shared_ptr<Tensor>& B,
    bool symmetrize) const
{
    std::shared_ptr<Tensor> A2 = CSF_basis(S)->transform_CSF_to_det(A);
    std::shared_ptr<Tensor> B2 = (A == B ? A2 : CSF_basis(S)->transform_CSF_to_det(B));

    if (is_terachem_ready(resources))
      return tpdm_det_gpu(resources, A2, B2,symmetrize);
    else
      return tpdm_det(resources, A2, B2,symmetrize);
}

std::shared_ptr<Tensor> CASBox::dyson_orbital_a(
    const std::shared_ptr<CASBox>& casA,
    const std::shared_ptr<CASBox>& casB,
    int SA,
    int SB,
    const std::shared_ptr<Tensor>& A,
    const std::shared_ptr<Tensor>& B)
{
    if (casA->M() != casB->M()) throw std::runtime_error("CASBox: casA->M != casB->M");
    if (casA->Nb() != casB->Nb()) throw std::runtime_error("CASBox: casA->Nb != casB->Nb");
    if (casA->Na() != (casB->Na() + 1)) throw std::runtime_error("CASBox: casA->Na != casB->Na+1");

    std::shared_ptr<Tensor> C(new Tensor({(size_t)casA->M(), 1}));
    double* Cp = C->data().data();
    
    std::shared_ptr<Tensor> A2 = casA->CSF_basis(SA)->transform_CSF_to_det(A);
    std::shared_ptr<Tensor> B2 = casB->CSF_basis(SB)->transform_CSF_to_det(B);
    double* Ap = A2->data().data();
    double* Bp = B2->data().data();

    auto stringsB0 = casA->stringsB();
    auto stringsA0 = casA->stringsA();
    auto stringsAC = casB->stringsA();

    size_t ndetB0 = stringsB0.size();
    size_t ndetA0 = stringsA0.size();
    size_t ndetAC = stringsAC.size();

    for (size_t indA0 = 0; indA0 < stringsA0.size(); indA0++) {
        uint64_t stringA0 = stringsA0[indA0];
        for (size_t indAC = 0; indAC < stringsAC.size(); indAC++) {
            uint64_t stringAC = stringsAC[indAC];
            uint64_t diff = stringA0 ^ stringAC;
            if (Bits::popcount(diff) != 1) continue;
            int p = Bits::ffs(diff); 
            double phase = 1 - 2 * (Bits::popcount_range(stringA0, 0, p) & 1); 
            Cp[p] += phase * C_DDOT(ndetB0, Ap + indA0 * ndetB0, 1, Bp + indAC * ndetB0, 1);
        }
    }
    
    return C;
}
std::shared_ptr<Tensor> CASBox::dyson_orbital_b(
    const std::shared_ptr<CASBox>& casA,
    const std::shared_ptr<CASBox>& casB,
    int SA,
    int SB,
    const std::shared_ptr<Tensor>& A,
    const std::shared_ptr<Tensor>& B)
{
    if (casA->M() != casB->M()) throw std::runtime_error("CASBox: casA->M != casB->M");
    if (casA->Na() != casB->Na()) throw std::runtime_error("CASBox: casA->Na != casB->Na");
    if (casA->Nb() != (casB->Nb() + 1)) throw std::runtime_error("CASBox: casA->Nb != casB->Nb+1");

    std::shared_ptr<Tensor> C(new Tensor({(size_t)casA->M(), 1}));
    double* Cp = C->data().data();
    
    std::shared_ptr<Tensor> A2 = casA->CSF_basis(SA)->transform_CSF_to_det(A);
    std::shared_ptr<Tensor> B2 = casB->CSF_basis(SB)->transform_CSF_to_det(B);
    double* Ap = A2->data().data();
    double* Bp = B2->data().data();

    auto stringsA0 = casA->stringsA();
    auto stringsB0 = casA->stringsB();
    auto stringsBC = casB->stringsB();

    size_t ndetA0 = stringsA0.size();
    size_t ndetB0 = stringsB0.size();
    size_t ndetBC = stringsBC.size();

    for (size_t indB0 = 0; indB0 < stringsB0.size(); indB0++) {
        uint64_t stringB0 = stringsB0[indB0];
        for (size_t indBC = 0; indBC < stringsBC.size(); indBC++) {
            uint64_t stringBC = stringsBC[indBC];
            uint64_t diff = stringB0 ^ stringBC;
            if (Bits::popcount(diff) != 1) continue;
            int p = Bits::ffs(diff); 
            double phase = 1 - 2 * (Bits::popcount_range(stringB0, 0, p) & 1); 
            Cp[p] += phase * C_DDOT(ndetA0, Ap + indB0, ndetB0, Bp + indBC, ndetBC);
        }
    }
    
    return C;
}
    
std::string CASBox::amplitude_string(
    int S,
    const std::shared_ptr<Tensor>& C,
    double thre,
    size_t max_dets,
    int offset) const
{
    std::shared_ptr<Tensor> C2 = CSF_basis(S)->transform_CSF_to_det(C);
    double* C2p = C2->data().data();

    size_t nstrA = Da();
    size_t nstrB = Db();
    size_t ndet = D();
    std::vector<uint64_t> strsA = stringsA();
    std::vector<uint64_t> strsB = stringsB();

    std::vector<std::pair<double, size_t>> order;
    for (size_t ind = 0; ind < ndet; ind++) {
        order.push_back(std::pair<double, size_t>(-pow(C2p[ind], 2), ind));
    }
    std::sort(order.begin(), order.end());

    // Determine number of digits in orbital label
    int max_ind = ((int)(offset + M())) - 1;
    int ndigit = ceil(log10((double) max_ind));

    std::string val = ""; 
    double R2 = 1.0;
    for (size_t ind = 0; ind < std::min(max_dets, ndet); ind++) {
        if (R2 < thre) break;
        size_t index = order[ind].second;
        double coef = C2p[index];
        size_t indexA = index / nstrB;
        size_t indexB = index % nstrB;
        uint64_t strA = strsA[indexA];
        uint64_t strB = strsB[indexB];
        val += sprintf2("%18.14f ", coef);
        for (int p = 0; p < M(); p++) {
            bool occA = strA & (1ULL << p);
            bool occB = strB & (1ULL << p);
            if (!occA && !occB) continue;
            else if (occA && occB) val += sprintf2("X%-*d ", ndigit, p + offset);
            else if (occA && !occB) val += sprintf2("A%-*d ", ndigit, p + offset);
            else if (!occA && occB) val += sprintf2("B%-*d ", ndigit, p + offset);
            else throw std::runtime_error("Sanity check");
        }
        val += "\n";
        R2 -= pow(coef, 2);
    }
    
    return val;
}

std::shared_ptr<Tensor> CASBox::orbital_transformation_det(
    const std::shared_ptr<Tensor> D,
    const std::shared_ptr<Tensor> A) const
{
    // Size checks
    A->shape_error({Da(), Db()}); 
    D->shape_error({(size_t)M(), (size_t)M()});

    // Transformation coefficients (T = -L + U^-1 : D = LU)
    std::shared_ptr<Tensor> D2 = D->transpose();
    std::vector<int> ipiv(M());
    int info1 = C_DGETRF(M(),M(),D2->data().data(),M(),ipiv.data());
    if (info1) throw std::runtime_error("DGETRF failed");
    int info2 = C_DTRTRI('U', 'N', M(), D2->data().data(), M());
    if (info2) throw std::runtime_error("DTRTRI failed");
    std::shared_ptr<Tensor> V = D2->transpose();
    double* Vp = V->data().data();
    for (int p = 0; p < M(); p++) {
        for (int q = 0; q < M(); q++) {
            if (p == q) Vp[p * M() + q] -= 1.0; // Missing unit diagonal in -L
            if (p <= q) continue; 
            Vp[p * M() + q] *= -1.0; // Scale the strictly lower triangle by -1
        }
    } 

    // Get the effective order of orbital transformations (from LU pivot order)
    std::vector<int> ipiv2(M());
    std::iota(ipiv2.begin(),ipiv2.end(),0);
    for (int i = 0; i < M(); i++) {
        // Remember that ipiv is 1-based
        std::swap(ipiv2[i],ipiv2[ipiv[i]-1]);
    }

    // Transformation intermediates
    std::shared_ptr<Tensor> T1 = A->clone();
    std::shared_ptr<Tensor> T2 = A->clone();
    
    // Strings in CI vector
    std::vector<uint64_t> strsA = stringsA();
    std::vector<uint64_t> strsB = stringsB();

    // Permute CI vector to pivot order
    std::vector<std::tuple<uint64_t, size_t, int>> orderA;
    for (size_t ind = 0; ind < strsA.size(); ind++) {
        uint64_t strA = strsA[ind];
        uint64_t strA2 = 0;
        int phase = 0;
        while (strA) {
            int i = Bits::ffs(strA);
            strA ^= (1ULL) << i;
            int i2 = ipiv2[i];
            strA2 += (1ULL) << i2;
            phase += Bits::popcount_range(strA2, 0, i2);
        }
        phase = 1 - 2 * (phase & 1);
        orderA.push_back(std::tuple<uint64_t, size_t, int>(strA2, ind, phase));
    }
    std::sort(orderA.begin(), orderA.end());
    std::vector<std::tuple<uint64_t, size_t, int>> orderB;
    for (size_t ind = 0; ind < strsB.size(); ind++) {
        uint64_t strB = strsB[ind];
        uint64_t strB2 = 0;
        int phase = 0;
        while (strB) {
            int i = Bits::ffs(strB);
            strB ^= (1ULL) << i;
            int i2 = ipiv2[i];
            strB2 += (1ULL) << i2;
            phase += Bits::popcount_range(strB2, 0, i2);
        }
        phase = 1 - 2 * (phase & 1);
        orderB.push_back(std::tuple<uint64_t, size_t, int>(strB2, ind, phase));
    }
    std::sort(orderB.begin(), orderB.end());
    const double* T1f = T1->data().data();
    double* T2f = T2->data().data();
    for (size_t indA = 0; indA < Da_; indA++) {
        size_t indA2 = std::get<1>(orderA[indA]);
        int phaseA = std::get<2>(orderA[indA]);
    for (size_t indB = 0; indB < Db_; indB++) {
        size_t indB2 = std::get<1>(orderB[indB]);
        int phaseB = std::get<2>(orderB[indB]);
        T2f[indA2 * Db_ + indB2] = phaseA * phaseB * T1f[indA * Db_ + indB];
    }}
    std::swap(T1, T2);

    // Single-Substitution coupling coefficients
    std::vector<CASSingle> subsA = CASBoxUtil::singles(M_, Na_);
    std::vector<CASSingle> subsB = CASBoxUtil::singles(M_, Nb_);
    
    // Alpha orbital transformation
    for (int k = 0; k < M_; k++) {
        const double* T1p = T1->data().data();
        double* T2p = T2->data().data();
        T2->copy(T1);
        for (auto sub : subsA) {
            size_t idxI = sub.idxI();
            size_t idxJ = sub.idxJ();
            int t = sub.t();
            int u = sub.u();
            int phase = sub.phase();
            if (k != u) continue;
            double Vval = Vp[t * M_ + k];
            const double* T1d = T1p + idxJ * Db_;
            double* T2d = T2p + idxI * Db_;
            for (size_t ind = 0; ind < Db_; ind++) {
                *T2d += Vval * phase * *T1d; 
                T1d++;
                T2d++;
            }
        }
        std::swap(T1, T2);
    }

    // Beta orbital transformation
    for (int k = 0; k < M_; k++) {
        const double* T1p = T1->data().data();
        double* T2p = T2->data().data();
        T2->copy(T1);
        for (auto sub : subsB) {
            size_t idxI = sub.idxI();
            size_t idxJ = sub.idxJ();
            int t = sub.t();
            int u = sub.u();
            int phase = sub.phase();
            if (k != u) continue;
            double Vval = Vp[t * M_ + k];
            const double* T1d = T1p + idxJ;
            double* T2d = T2p + idxI;
            for (size_t ind = 0; ind < Da_; ind++) {
                *T2d += Vval * phase * *T1d; 
                T1d += Db_;
                T2d += Db_;
            }
        }
        std::swap(T1, T2);
    }
    
    return T1;
}
std::shared_ptr<Tensor> CASBox::metric_det(
    const std::shared_ptr<Tensor>& M) const
{
    // Orbital metric
    M->shape_error({(size_t)M_});
    const double* Mp = M->data().data();
    
    // Strings
    auto stringsA2 = stringsA(); 
    auto stringsB2 = stringsB(); 

    // String metrics
    std::shared_ptr<Tensor> Ma(new Tensor({Da()}));
    double* Map = Ma->data().data();
    for (size_t ind = 0; ind < stringsA2.size(); ind++) {
        uint64_t strA = stringsA2[ind]; 
        double prod = 1.0;
        while (strA) {
            int i = Bits::ffs(strA);
            strA ^= (1ULL << i);
            prod *= Mp[i];
        }
        Map[ind] = prod;
    }
    std::shared_ptr<Tensor> Mb(new Tensor({Db()}));
    double* Mbp = Mb->data().data();
    for (size_t ind = 0; ind < stringsB2.size(); ind++) {
        uint64_t strB = stringsB2[ind]; 
        double prod = 1.0;
        while (strB) {
            int i = Bits::ffs(strB);
            strB ^= (1ULL << i);
            prod *= Mp[i];
        }
        Mbp[ind] = prod;
    }

    // Det metric
    std::shared_ptr<Tensor> M2(new Tensor({Da_, Db_}));
    double* M2p = M2->data().data();
    for (size_t indA = 0; indA < Da_; indA++) {
    for (size_t indB = 0; indB < Db_; indB++) {
        (*M2p++) = Map[indA] * Mbp[indB];
    }}

    return M2;
}

} // namespace lightspeed
