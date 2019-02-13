#ifndef LS_MOLDEN_HPP
#define LS_MOLDEN_HPP

#include <cstddef>
#include <memory>

namespace lightspeed {

class Molecule;
class Basis;
class Tensor;

class Molden {

public:

/**
 * Write a MOLDEN file for orbital visualization purposes
 *
 * @param filename the complete filename of the molden file (opened with "w")
 * @param basis the Basis object for the 
 * @param C (nao, norb) Tensor of orbital coefficients
 * @param eps (norb,) Tensor of orbital energies
 * @param occ (norb,) Tensor of orbital occupation numbers
 * @param alpha alpha (true) or beta (false) orbitals?
 **/
static void save_molden_file(
    const std::string& filename,
    const std::shared_ptr<Molecule>& molecule,
    const std::shared_ptr<Basis>& basis,
    const std::shared_ptr<Tensor>& C,
    const std::shared_ptr<Tensor>& eps,
    const std::shared_ptr<Tensor>& occ,
    bool alpha);

};    

} // namespace lightspeed

#endif
