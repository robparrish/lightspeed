#ifndef LS_CASBOX2_HPP
#define LS_CASBOX2_HPP

#include <memory>
#include <vector>
#include <map>
#include <cstddef>

namespace lightspeed {

class Tensor;
class ResourceList;

/**
 * Class SeniorityBlock describes a seniority sector in a CASCI wavefunction.
 * A seniority sector is the set of all determinants with common seniority
 * number Z (the number of unpaired alpha/beta electrons). Typically, there
 * are many such determinants which can be sub-grouped into "blocks" of
 * "spin-coupled determinants." A spin coupling block consists of all dets
 * with common seniority number Z *and* common open-shell orbital indices
 * (unpaired strings). There are many spin-coupled blocks within a seniority
 * section, with the different blocks differing in the ordering of the
 * closed-shell doubly occupied/empty orbitals (paired strings) and the
 * ordering of the merger between closed- and open-shell orbitals (interleave
 * strings).
 *
 * Each spin-coupling block can be transformed to the configuration state
 * function (CSF) basis by a linear transformation of the involved open-shell
 * indices. This transformation diagonalizes the S^2 operator within the
 * spin-coupling block, and is the same for all spin-coupling blocks within a
 * seniority sector. 
 *
 * SeniorityBlock encapsulates the canonical ordering of determinants in a
 * seniority sector, and the transformation matrices from the spin-coupling
 * blocks of dets to the CSF basis for given spin quantum numbers S.
 **/
class SeniorityBlock {

public:

/**
 * SeniorityBlock Constructor
 *
 * @param M number of spatial orbitals
 * @param Na total number of alpha electrons
 * @param Nb total number of beta electrons
 * @param Z seniority number (number of unpaired electrons)
 **/
SeniorityBlock(
    int M,
    int Na,
    int Nb,
    int Z
    );

// Number of spatial orbitals in this SeniorityBlock
int M() const { return M_; }
// Number of alpha electrons in this SeniorityBlock
int Na() const { return Na_; }
// Number of beta electrons in this SeniorityBlock
int Nb() const { return Nb_; }
// Number of unpaired electrons in this SeniorityBlock (seniority number)
int Z() const { return Z_; }
// Number of excess high-spin electrons in this SeniorityBlock (Na - Nb)
int H() const { return Na_ - Nb_; }

// Number of doubly occupied orbitals
int D() const { return D_; }
// Number of unoccupied orbitals
int U() const { return U_; }
// Number of alpha singly-occupied orbitals
int A() const { return A_; }
// Number of beta singly-occupied orbitals
int B() const { return B_; }

/**
 * The interleave strings describe how to place the A/B open-shell
 * orbitals into the full orbital space. Their popcount is always Z. The
 * interleave strings for the D/U closed-shell orbitals can be obtained by 
 * I2 = ((1ULL << M) - 1) ^ I, where ((1ULL << M) - 1) is a mask with 1 for
 * every orbital in the active space.
 **/
std::vector<uint64_t> interleave_strings() const;
/**
 * The paired strings describe how to place the D doubly-occupied orbitals into
 * the closed-shell orbitals. Their popcount is always D. The
 * paired strings for the U doubly-unoccupied orbitals can be obtained by 
 * I2 = ((1ULL << (M-Z)) - 1) ^ I, where ((1ULL << (M-Z)) - 1) is a mask with 1 for
 * every orbital in the closed-shell orbitals.
 **/
std::vector<uint64_t> paired_strings() const;
/**
 * The unpaired strings describe how to place the A alpha orbitals into
 * the open-shell orbitals. Their popcount is always A. The
 * unpaired strings for the B open-shell orbitals can be obtained by 
 * I2 = ((1ULL << (Z)) - 1) ^ I, where ((1ULL << (Z)) - 1) is a mask with 1 for
 * every orbital in the open-shell orbitals.
 **/
std::vector<uint64_t> unpaired_strings() const;

// The number of interleave strings, O(1) operation
size_t ninterleave() const;
// The number of paired strings, O(1) operation
size_t npaired() const;
// The number of unpaired strings, O(1) operation
size_t nunpaired() const;

// Map from spin index S to det-to-CSF transformation matrix
const std::map<int, std::shared_ptr<Tensor>>& CSF_basis() const { return CSF_basis_; }

// => Computational Helpers <= //

/**
 * Compute the S^2 operator for all seniority blocks in this SeniorityBlock object
 *
 * @return S^2, (V, V) Tensor, where V is the number of unpaired strings in
 *   this SeniorityBlock.
 **/
std::shared_ptr<Tensor> compute_S2() const;

private:

int M_;
int Na_;
int Nb_;
int Z_;

int D_;
int U_;
int A_;
int B_;

/**
 * Compute the CSF basis for this SeniorityBlock
 *
 * @param thre absolute threshold for eigenvalue to be classified as a given S(S+1) eigenvalue
 * @result the CSF basis is built
 **/
void compute_CSF_basis(double thre);

// map of <S, Tensor of size (V, C)> where V is the number of unpaired
// strings and C is the number of CSFs corresponding to S
std::map<int, std::shared_ptr<Tensor>> CSF_basis_;
    
};

/**
 * Class CSFBasis describes the layout of a vector of a given S^2 symmetry in
 * the CSF basis and its transformation to the det basis.
 *
 * A vector in the CSF basis for S^2 symmetry index S will have the following
 * nested block structure, stored slowest to fastest (C-order):
 *  - Seniority sector index (Z) [e.g., 0, 2, 4, 6 for CAS(6,6)] 
 *  - Interleave index [ways to merge the D/U and A/B strings]
 *  - Paired index [ways to arrange D/U orbitals]
 *  - CSF index [linear combinations of A/B strings predicated on previous inds]
 *
 * The corresponding ordering in the det basis is non-lexical, but is useful as
 * an intermediate representation. The ordering is:
 *  - Seniority sector index (Z) [e.g., 0, 2, 4, 6 for CAS(6,6)] 
 *  - Interleave index [ways to merge the D/U and A/B strings]
 *  - Paired index [ways to arrange D/U orbitals]
 *  - Unpaired index [ways to arrange A/B orbitals]
 *
 * All indices are rectilinear *except* for seniority sector. Therefore, all
 * quantities are reported with seniority sector labels.
 *
 * All seniority sectors are present in a given CSFBasis, even if some dets do
 * not contribute to the CSFs for the S index. This means that the
 * lexical-to-seniority reordering of the det basis is the same (in both size
 * and operation count) for all S indices.
 **/ 
class CSFBasis {

public:

/**
 * Verbatim constructor, makes a copy of the seniority_blocks map.
 **/
CSFBasis(
    int S,
    const std::map<int, std::shared_ptr<SeniorityBlock>>& seniority_blocks) :
    S_(S), seniority_blocks_(seniority_blocks) {}

// Spin index (0 - singlet, 1 - doublet, 2 - triplet, etc)
int S() const { return S_; }

// A handy string representation of the object (std::string not det strings!)
std::string string() const;

// Total number of CSF basis functions for this value of S
size_t total_nCSF() const;
// Total number of determinants (the same for all S)
size_t total_ndet() const;

// => Basis Layout Data <= //

// The Z (seniority) indices for this S (same for all S)
std::vector<int> seniority() const;
// The (nunpaired, nCSF) transformation matrices for each SeniorityBlock sector (may have zero dims)
std::vector<std::shared_ptr<Tensor>> det_to_CSF() const;
// The number of CSFs generated from each coupling set with the proper S value
std::vector<size_t> nCSF() const;
// The number of unpaired dets in each coupling set (ways to arrange A/B orbitals)
std::vector<size_t> nunpaired() const;
// The number of paired dets (ways to arrange the D/U orbitals)
std::vector<size_t> npaired() const;
// The number of interleave configurations (ways to merge A/B with D/U)
std::vector<size_t> ninterleave() const;
// ninterleave * npaired - the number of coupling sets for each SeniorityBlock sector
std::vector<size_t> nblock() const;

// Offsets into each Z sector in the CSF basis
std::vector<size_t> offsets_CSF() const;
// Sizes of each Z sector in the CSF basis
std::vector<size_t> sizes_CSF() const;
// Offsets into each Z sector in the det basis
std::vector<size_t> offsets_det() const;
// Sizes of each Z sector in the det basis
std::vector<size_t> sizes_det() const;

// Unpaired strings (A/B orderings)
std::vector<std::vector<uint64_t>> unpaired_strings() const;
// Paired strings (D/U orderings)
std::vector<std::vector<uint64_t>> paired_strings() const;
// Interleave strings (D/U // A/B merge patterns)
std::vector<std::vector<uint64_t>> interleave_strings() const;

// => CSF/DET Basis Transformations <= //

std::shared_ptr<Tensor> transform_det_to_CSF(
    const std::shared_ptr<Tensor>& C) const;

std::shared_ptr<Tensor> transform_CSF_to_det(
    const std::shared_ptr<Tensor>& C) const;

std::shared_ptr<Tensor> transform_det_to_det2(
    const std::shared_ptr<Tensor>& C) const;

std::shared_ptr<Tensor> transform_det2_to_det(
    const std::shared_ptr<Tensor>& C) const;

std::shared_ptr<Tensor> transform_det2_to_CSF(
    const std::shared_ptr<Tensor>& C) const;

std::shared_ptr<Tensor> transform_CSF_to_det2(
    const std::shared_ptr<Tensor>& C) const;

private:

// Spin index (0 - singlet, 1 - doublet, 2 - triplet, etc)
int S_;
// Copy of map of SeniorityBlock objects
std::map<int, std::shared_ptr<SeniorityBlock>> seniority_blocks_;

};

/**
 * Class CASBox provides encapsulation of and computations involving a complete
 * active space configuration interaction (CASCI) problem. CASBox stores
 * vectors in the CSF basis, but expands vectors to the det basis for
 * computational operations, including the sigma vector algorithm of Knowles
 * and Handy. 
 *
 * Algorithms to compute guesses, Hamiltonian-vector products (sigma vectors),
 * Evangelisti preconditioner-vector products, OPDMs, and TPDMs (including
 * transition density variants) are included. These are the necessary tools to
 * solve a wide variety of CASCI/CASSCF problems, using a variety of
 * convergence algorithms.
 *
 * Unless otherwise noted, all CASCI vectors in this class are in the CSF
 * basis. Methods marked _det work in the det basis.
 **/
class CASBox {

public:
    
/**
 * CASBox Constructor
 *
 * @param M number of spatial orbitals 
 * @param Na number of alpha electrons
 * @param Nb number of beta electrons
 * @param H (M,M)-sized Tensor with external potential integrals (t|f|u)
 * @param I (M,M,M,M)-sized Tensor with (tu|vw) spatial ERIs
 *
 * It is assumed that the orbitals are ordered energetically increasing (if
 * not, the guess and preconditioner will not make much sense).
 **/
CASBox(
    int M,
    int Na,
    int Nb,
    const std::shared_ptr<Tensor>& H,
    const std::shared_ptr<Tensor>& I
    );

// => Accessors <= //

// Number of spatial orbitals in this CAS
int M() const { return M_; }
// Number of alpha electrons in this CAS
int Na() const { return Na_; }
// Number of beta electrons in this CAS
int Nb() const { return Nb_; }
// Number of alpha strings in this CAS (M choose Na)
size_t Da() const { return Da_; }
// Number of beta strings in this CAS (M choose Nb)
size_t Db() const { return Db_; }
// Total number of electrons in this CAS
int N() const { return Na_ + Nb_; }
// Total number of determinants in this CAS
size_t D() const { return Da_ * Db_; }

// External potential integrals (u|f|u), (M,M) Tensor
std::shared_ptr<Tensor> H() const { return H_; } 
// Spatial ERIs (tu|vw), (M,M,M,M) Tensor
std::shared_ptr<Tensor> I() const { return I_; } 
  
// A handy string representation of the object (std::string not det strings!)
std::string string() const;

// => Det Basis Occupation Strings <= //

/**
 * Compute the valid alpha strings in the CAS space (in binary).
 * @return strings the lexically-sorted list of alpha strings, size Sa
 **/
std::vector<uint64_t> stringsA() const;
    
/**
 * Compute the valid beta strings in the CAS space (in binary).
 * @return strings the lexically-sorted list of beta strings, size Sb
 **/
std::vector<uint64_t> stringsB() const;

// => Seniority Blocks <= //

// The minimum possible seniority number (electrons maximally paired)
int min_seniority() const { return Na_ - Nb_; }
// The maximum possible seniority number (electrons maximally unpaired)
int max_seniority() const;
/**
 * Return the valid seniority (Z) and spin (S) indices for this CASBox object.
 * These are [min_Z, min_Z+2,...,max_Z], and are the same for Z and S.
 **/
std::vector<int> seniority() const;

/**
 * Get the SeniorityBlock corresponding to seniority number Z
 *
 * @param Z the seniority number
 * @return the SeniorityBlock corresponding to Z
 **/
std::shared_ptr<SeniorityBlock> seniority_block(int Z) const;

/**
 * Get the CSFBasis corresponding to spin index S
 *
 * @param S the spin index, must be in [min_Z, max_Z] counting by 2
 * @return the CSFBasis corresponding to S
 **/
std::shared_ptr<CSFBasis> CSF_basis(int S) const;

// => Evangelisti Preconditioner <= //

/**
 * Return the energy of the Evangelisti reference determinant (the closed shell
 * det with the lowest Nb orbitals doubly occupied).
 * @return E0 the energy of the reference state
 **/ 
double E0_evangelisti() const { return E0pre_; }

/**
 * Return the RHF orbital energies for the Evangelisti reference determinant. 
 * @return F a (M,) Tensor with the orbital energies
 **/ 
std::shared_ptr<Tensor> F_evangelisti() const { return Fpre_; }

/**
 * Get the diagonal Evangelisti preconditionor for spin index S
 *
 * @param S the spin index
 * @return the Evangelisti Hamiltonian corresponding to S
 **/
std::shared_ptr<Tensor> H_evangelisti(int S) const;

/**
 * Return a new Tensor with the diagonal Evangelisti preconditioner
 * applied.
 *
 * @param S the spin index
 * @param C the Tensor to apply the preconditioner to
 * @param Eval the Ritz value or other estimated eigenvalue of the Tensor
 * @return L = - C / (D - eval) where D is the Koopman's energy of each
 * det. The denominator value is clamped to an absolute value of 1.0E-5 to
 * prevent trivial singularities.
 **/
std::shared_ptr<Tensor> apply_evangelisti(
    int S,
    const std::shared_ptr<Tensor>& C,
    double Eval) const;

/**
 * Return a number of orthonormal guess vectors diagonalizing the Evangelisti
 * preconditioner.
 *
 * @param S the spin index S
 * @param nguess the desired number of guess vectors
 * @return a vector of (nSCF,) Tensor guesses. >= nguess vectors will be
 * returned, with more than the requested number being provided to keep
 * spin-coupling blocks complete.
 **/
std::vector<std::shared_ptr<Tensor>> guess_evangelisti(
    int S,
    size_t nguess) const;

// => Det Basis Operations <= //

/**
 * Compute the application of the Hamiltonian on a trial vector in the det basis:
 *  S_I = H_IJ C_J
 *   
 * @param resources the ResourceList to use for this rate-limiting operation
 * @param C the trial vector
 * @return S the resultant sigma vector
 **/
std::shared_ptr<Tensor> sigma_det(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& C) const;

/**
 * Compute the active space total or spin-polarization OPDM.
 *  
 * The total (spin-free) OPDM is:
 *  DT_tu = <A|t_a^+ u_a|B> + <A|t_b^+ u_b|B>
 * The spin-polarization OPDM is:
 *  DS_tu = <A|t_a^+ u_a|B> - <A|t_b^+ u_b|B>
 * From these two objects, the alpha/beta OPDMs can be formed as:
 *  DA_tu = 0.5 * (DT_tu + DS_tu)
 *  DB_tu = 0.5 * (DT_tu - DS_tu)
 *
 * @param resources the ResourceList to use for this rate-limiting operation
 * @param A the bra state Tensor
 * @param B the ket state Tensor
 * @param total if true, return the total OPDM, if false, return the
 *  spin-polarization OPDM. 
 * @return D the (transition) OPDM, an (M,M) Tensor
 **/ 
std::shared_ptr<Tensor> opdm_det(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& A,
    const std::shared_ptr<Tensor>& B,
    bool total) const;

/**
 * Compute the active space spin-free TPDM.
 *
 * @param resources the ResourceList to use for this rate-limiting operation
 * @param A the bra state Tensor
 * @param B the ket state Tensor
 * @param symmetrize symmetrize the TPDM (true) or not (false)
 * @return D the (transition) TPDM, an (M,M,M,M) Tensor
 **/ 
std::shared_ptr<Tensor> tpdm_det(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& A,
    const std::shared_ptr<Tensor>& B,
    bool symmetrize) const;

// => TeraChem GPU CI Operations <= //

/**
 * Compute the application of the Hamiltonian on a trial vector in the det basis using GPU resources:
 *  S_I = H_IJ C_J
 *   
 * @param resources the ResourceList to use for this rate-limiting operation
 * @param C the trial vector
 * @return S the resultant sigma vector
 **/
std::shared_ptr<Tensor> sigma_det_gpu(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& C) const;

/**
 * Compute the active space total OPDM.
 *  
 * The total (spin-free) OPDM is:
 *  DT_tu = <A|t_a^+ u_a|B> + <A|t_b^+ u_b|B>
 *
 * @param resources the ResourceList to use for this rate-limiting operation
 * @param A the bra state Tensor
 * @param B the ket state Tensor
 * @return D the (transition) OPDM, an (M,M) Tensor
 **/ 
std::shared_ptr<Tensor> opdm_det_gpu(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& A,
    const std::shared_ptr<Tensor>& B) const;

/**
 * Compute the active space spin-free TPDM.
 *
 * @param resources the ResourceList to use for this rate-limiting operation
 * @param A the bra state Tensor
 * @param B the ket state Tensor
 * @param symmetrize symmetrize the TPDM (true) or not (false)
 * @return D the (transition) TPDM, an (M,M,M,M) Tensor
 **/ 
std::shared_ptr<Tensor> tpdm_det_gpu(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& A,
    const std::shared_ptr<Tensor>& B,
    bool symmetrize) const;

// => CSF Basis Operations <= //
    
/**
 * Compute the application of the Hamiltonian on a trial vector:
 *  S_I = H_IJ C_J
 *   
 * @param resources the ResourceList to use for this rate-limiting operation
 * @param S the spin index
 * @param C the trial vector
 * @return Sigma the resultant sigma vector
 **/
std::shared_ptr<Tensor> sigma(
    const std::shared_ptr<ResourceList>& resources,
    int S,
    const std::shared_ptr<Tensor>& C) const;

/**
 * Compute the active space total or spin-polarization OPDM.
 *  
 * The total (spin-free) OPDM is:
 *  DT_tu = <A|t_a^+ u_a|B> + <A|t_b^+ u_b|B>
 * The spin-polarization OPDM is:
 *  DS_tu = <A|t_a^+ u_a|B> - <A|t_b^+ u_b|B>
 * From these two objects, the alpha/beta OPDMs can be formed as:
 *  DA_tu = 0.5 * (DT_tu + DS_tu)
 *  DB_tu = 0.5 * (DT_tu - DS_tu)
 *
 * @param resources the ResourceList to use for this rate-limiting operation
 * @param S the spin index
 * @param A the bra state Tensor
 * @param B the ket state Tensor
 * @param total if true, return the total OPDM, if false, return the
 *  spin-polarization OPDM. 
 * @return D the (transition) OPDM, an (M,M) Tensor
 **/ 
std::shared_ptr<Tensor> opdm(
    const std::shared_ptr<ResourceList>& resources,
    int S,
    const std::shared_ptr<Tensor>& A,
    const std::shared_ptr<Tensor>& B,
    bool total) const;

/**
 * Compute the active space spin-free TPDM.
 *
 * @param resources the ResourceList to use for this rate-limiting operation
 * @param S the spin index
 * @param A the bra state Tensor
 * @param B the ket state Tensor
 * @param symmetrize symmetrize the TPDM (true) or not (false)
 * @return D the (transition) TPDM, an (M,M,M,M) Tensor
 **/ 
std::shared_ptr<Tensor> tpdm(
    const std::shared_ptr<ResourceList>& resources,
    int S,
    const std::shared_ptr<Tensor>& A,
    const std::shared_ptr<Tensor>& B,
    bool symmetrize) const;

// => Orbital Transformation Utility <= //

/**
 * Transform a det-basis CI vector A from its current orbitals to 
 *
 * @param D (M,M) Tensor with old orbitals in the rows and new orbitals in the
*    columns
 * @param A det-basis CI vector to transform
 * @return A' the transformed CI vector
 **/
std::shared_ptr<Tensor> orbital_transformation_det(
    const std::shared_ptr<Tensor> D,
    const std::shared_ptr<Tensor> A) const;

// => Utility Methods <= //

/**
 * Return a human-readable summary string of the det composition of
 * CSF-basis CAS wavefunction C.
 *
 * @param S the spin index
 * @param C the wavefunction (CSF-basis)
 * @param thre threshold on r^2 to stop printing at
 * @param max_dets max number of dets to stop printing at
 * @param offset the offset of the lowest orbital in the active space
 * @return the amplitude summary string
 **/
std::string amplitude_string(
    int S,
    const std::shared_ptr<Tensor>& C,
    double thre,
    size_t max_dets,
    int offset) const;

/**
 * Compute the Dyson orbital coefficients:
 *
 * c_p = <A|a_p^+|B>
 *
 * for N-electron state A and N-1-electron state B differing by an alpha electron
 * (e.g., A: doublet anion -> B: neutral singlet). Both involved CASBox
 * systems must have identical M and Nb, and casA->Na must be 1 greater than
 * casB->Na.
 *
 * @param casA the CASBox for the N-electron state A
 * @param casB the CASBox for the N-1-electron state B
 * @param SA the spin index for state A
 * @param SB the spin index for state B
 * @param A the bra state Tensor
 * @param B the ket state Tensor
 * @return c (M, 1) shape Tensor with Dyson orbital coefficients
 **/
static 
std::shared_ptr<Tensor> dyson_orbital_a(
    const std::shared_ptr<CASBox>& casA,
    const std::shared_ptr<CASBox>& casB,
    int SA,
    int SB,
    const std::shared_ptr<Tensor>& A,
    const std::shared_ptr<Tensor>& B);

/**
 * Compute the Dyson orbital coefficients:
 *
 * c_p = <A|a_p^+|B>
 *
 * for N-electron state A and N-1-electron state B differing by a beta electron
 * (e.g., A: neutral singlet -> B: doublet cation). Both involved CASBox
 * systems must have identical M and Na, and casA->Nb must be 1 greater than
 * casB->Nb.
 *
 * @param casA the CASBox for the N-electron state A
 * @param casB the CASBox for the N-1-electron state B
 * @param SA the spin index for state A
 * @param SB the spin index for state B
 * @param A the bra state Tensor
 * @param B the ket state Tensor
 * @return c (M, 1) shape Tensor with Dyson orbital coefficients
 **/
static 
std::shared_ptr<Tensor> dyson_orbital_b(
    const std::shared_ptr<CASBox>& casA,
    const std::shared_ptr<CASBox>& casB,
    int SA,
    int SB,
    const std::shared_ptr<Tensor>& A,
    const std::shared_ptr<Tensor>& B);

/** 
 * Compute the metric tensor in the determinant basis
 *
 * M_{Ia Ib} = M_{Ia} M_{Ib}
 * M_{Ia} = \prod_{i \in I_a} M_{i}
 *
 * @param M (M,) Tensor, active-space metric integrals
 * @returns M2 (Da, Db) Tensor, det-space metric integrals
 **/
std::shared_ptr<Tensor> metric_det(
    const std::shared_ptr<Tensor>& M) const;

private:

// Input Sizing
int Na_;
int Nb_;
int M_;
// Derived Sizing
size_t Da_;
size_t Db_;
      
// Input Potential Integral Tensors
std::shared_ptr<Tensor> H_;
std::shared_ptr<Tensor> I_;
// Derived Potential Integral Tensors
std::shared_ptr<Tensor> K_;

std::map<int, std::shared_ptr<SeniorityBlock> > seniority_blocks_;
std::map<int, std::shared_ptr<CSFBasis> > CSF_basis_;

// => Evangelisti Preconditioner <= //
    
// Sets up the evangelisti preconditioner fields below
void build_evangelisti();

// Evangelisti det energy
double E0pre_; 
// Evangelisti orbital energies
std::shared_ptr<Tensor> Fpre_; 
// Evangelisti preconditioner
std::map<int, std::shared_ptr<Tensor> > Hpre_;

// Helper to build a given element of the Evangelisti preconditioner for a given S
std::shared_ptr<Tensor> compute_Hpre(int S) const;

};

class ExplicitCASBox {

public:

ExplicitCASBox(
    const std::shared_ptr<CASBox>& casbox) :
    casbox_(casbox) {}

// The CASBox object this ExplicitCASBox is built around
std::shared_ptr<CASBox> casbox() const { return casbox_; }

std::shared_ptr<Tensor> evecs(int S);
std::shared_ptr<Tensor> evals(int S);

std::shared_ptr<Tensor> evec(int S, size_t index);
double eval(int S, size_t index);

private: 

std::shared_ptr<CASBox> casbox_;

// Eigenvectors in CSF basis
std::map<int, std::shared_ptr<Tensor> > evecs_;
// Eigenvalues 
std::map<int, std::shared_ptr<Tensor> > evals_;

void compute_block(int S);

std::shared_ptr<Tensor> compute_Hblock(
    int S,
    const std::shared_ptr<SeniorityBlock>& sen1,
    const std::shared_ptr<SeniorityBlock>& sen2);

};


} // namespace lightspeed

#endif
