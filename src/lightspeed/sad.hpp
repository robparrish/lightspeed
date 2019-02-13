#ifndef LS_SAD_HPP
#define LS_SAD_HPP

#include <memory>
#include <vector>

namespace lightspeed {

class ResourceList;
class Tensor;
class Basis;
class Molecule;
    
/**
 * Class SAD handles considerations for a superpostion of atomic densities
 * (SAD) guess for the OPDM. This implementation of SAD relies on the existence
 * of "MinAO" basis sets such as cc-pvdz-minao whose fully contracted core and
 * valence orbitals outline the spherically-averaged orbitals of the free atoms
 * (either Hartree-Fock or natural orbitals, depending on the basis set). By
 * fractionally-occupying the MinAO orbitals and then projecting into a desired
 * basis, the SAD guess can be obtained without solving an electronic structure
 * problem on-the-fly for each atom.
 *
 * The operations in this class are all computationally inexpensive, but
 * involve a rather tedious amount of code due to having to lay out the
 * periodic table and decide which orbitals are core (fully occupied) and
 * valence (fractionally occupied).
 *
 * We provide several layers of flexibility for the user regarding the
 * fractional occupations of the MinAOs. At the most-flexible level, the
 * individual orbital occupations can be manipulated, which can be useful in
 * converging the SCF for metal-containing systems with ligands. At an
 * intermediate level, the orbital occupations can be taken to be spherically
 * symmetric with the valence electrons fractionally occupied to provide for
 * atomic charges. At the lowest level, the orbital occupations can be taken to
 * be spherically symmetric with the valence electrons fractionally occupied to
 * provide for neutral atoms. The latter is usually wholly sufficient to
 * converge the SCF (even in the case of systems with net total charge), but
 * the others are provided for trickier cases.
 **/
class SAD {

public:

/**
 * Compute the superposition of atomic density matrices (SAD) C matrix
 * 
 * @param resources ResourceList to use in computing SAD C matrix
 * @param nocc1 (basis1->nfunction(),) Tensor with fractional occupations of
 *  orbitals of basis1
 * @param basis1 the MinAO Basis (e.g., cc-pvdz-minao) 
 * @param basis2 the target Basis (e.g., 6-31gs)
 * @param necp for each atom, how many orbitals to skip in basis1 due to ECP
 *  treatment?
 * @return Csad (basis2->nfunction(),basis1->nfunction()) Tensor with SAD
 *  orbitals in basis2
 **/
static
std::shared_ptr<Tensor> sad_orbitals(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& nocc1,
    const std::shared_ptr<Basis>& basis1,
    const std::shared_ptr<Basis>& basis2,
    const std::vector<int>& necp);

/**
 * Compute the SAD fractional occupations for Molecule mol with atomic charges
 * charges. 
 *
 * @param mol the Molecule to determine SAD fractional occupations for.
 * @param charges to the electronic charge of each atom (in electron pairs).
 **/
static
std::shared_ptr<Tensor> sad_nocc(
    const std::shared_ptr<Molecule>& mol,
    const std::shared_ptr<Tensor>& charges);

/**
 * Compute the SAD fractional occupations for Molecule mol with atomic charges
 * charges taken to balance the Z fields of the atoms
 *
 * @param mol the Molecule to determine SAD fractional occupations for.
 **/
static
std::shared_ptr<Tensor> sad_nocc_neutral(
    const std::shared_ptr<Molecule>& mol);

/**
 * Returns a descriptive string explaning the available atoms and pattern of
 * frozen/active shells for SAD. Essentially explains which shells are
 * considered frozen core and active valence across the periodic table.
 **/
static
std::string sad_nocc_atoms();

/**
 * Compute an orthonormal set of orbitals in basis2 by least squares projection
 * and Lowdin orthogonalization. Note that this forms three PairList objects,
 * so is not perhaps the most efficient approach for molecule -> molecule
 * projections. But it is fine for atom -> atom projections.
 *
 * @param resources ResourceList to use in computing SAD C matrix
 * @param C1 (nbf1, nmo) Tensor of orbitals in basis1
 * @param basis1 reference Basis
 * @param basis2 target Basis
 * @param threpq threshold for pq pairs in overlap integrals
 * @param threpow threshold for condition number in inverse square root
 * @return C2 (nbf2, nmo) Tensor of orthonormalized least-squares orbitals in basis2
 **/ 
static
std::shared_ptr<Tensor> project_orbitals(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& C1,
    const std::shared_ptr<Basis>& basis1,
    const std::shared_ptr<Basis>& basis2,
    double threpq = 1.0E-14,
    double threpow = 1.0E-12);


};

} // namespace lightspeed 

#endif
