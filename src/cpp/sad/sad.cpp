#include <lightspeed/sad.hpp>
#include <lightspeed/molecule.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/basis.hpp>
#include <lightspeed/pair_list.hpp>
#include <lightspeed/intbox.hpp>
#include "sad_util.hpp"
#include <cmath>

namespace lightspeed { 

std::string SAD::sad_nocc_atoms()
{
    return SADAtom::print_atoms();
}

std::shared_ptr<Tensor> SAD::sad_nocc(
    const std::shared_ptr<Molecule>& mol,
    const std::shared_ptr<Tensor>& charges)
{
    std::vector<size_t> dim;
    dim.push_back(mol->atoms().size());
    charges->shape_error(dim);

    const double* Qp = charges->data().data();

    std::vector<double> nocc;    

    for (size_t A2 = 0; A2 < mol->atoms().size(); A2++) {
        const Atom& atom = mol->atoms()[A2];
        double Q = Qp[A2];
        const SADAtom& atom2 = SADAtom::get(atom.N());
        std::vector<double> nocc2 = atom2.nocc(Q);
        nocc.insert(nocc.end(), nocc2.begin(), nocc2.end());
    } 

    std::vector<size_t> dim2;
    dim2.push_back(nocc.size());
    std::shared_ptr<Tensor> T(new Tensor(dim2));
    std::vector<double>& Td = T->data();
    for (size_t ind = 0; ind < nocc.size(); ind++) {
        Td[ind] = nocc[ind];
    }    
    
    return T;
}
std::shared_ptr<Tensor> SAD::sad_nocc_neutral(
    const std::shared_ptr<Molecule>& mol)
{
    std::vector<size_t> dim;
    dim.push_back(mol->atoms().size());
    std::shared_ptr<Tensor> Q(new Tensor(dim));
    double* Qp = Q->data().data();
    
    for (size_t A2 = 0; A2 < mol->atoms().size(); A2++) {
        const Atom& atom = mol->atoms()[A2];
        Qp[A2] = 0.5 * atom.Z(); // number of electron pairs is Z / 2
    }

    return SAD::sad_nocc(mol,Q);
}

std::shared_ptr<Tensor> SAD::sad_orbitals(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& nocc1,
    const std::shared_ptr<Basis>& basis1,
    const std::shared_ptr<Basis>& basis2,
    const std::vector<int>& necp)
{
    // Validity checks
    if (basis1->natom() != basis2->natom()) throw std::runtime_error("SAD::sad_orbitals: basis1->natom() != basis2->natom()");
    if (necp.size() != basis1->natom()) throw std::runtime_error("SAD::sad_orbitals: basis1->natom() != necp.size()");
    nocc1->shape_error({basis1->nao()});

    const double* nocc1p = nocc1->data().data();
    for (size_t i = 0; i < basis1->nao(); i++) {
        if (nocc1p[i] < 0.0) throw std::runtime_error("SAD::sad_orbitals: nocc1 < 0.0");
    }

    // Total number of SAD orbitals
    int nsad = basis1->nao();
    for (int n : necp) nsad -= n;
    // Target
    std::shared_ptr<Tensor> Cf(new Tensor({basis2->nao(), (size_t)nsad}));
    double* Cfp = Cf->data().data();
    
    // => Atom Loop <= //
    
    size_t aoOff1 = 0;
    size_t aoOff2 = 0;
    size_t sadOff = 0;
    for (size_t A = 0; A < basis1->natom(); A++) {
        // Atomic basis sets
        std::vector<size_t> Alist;
        Alist.push_back(A);
        std::shared_ptr<Basis> atom1 = basis1->subset(Alist);
        std::shared_ptr<Basis> atom2 = basis2->subset(Alist);
        // MinAO C Matrix (Identity for now, in non-ECP columns)
        int nsad2 = atom1->nao() - necp[A];
        std::shared_ptr<Tensor> C1(new Tensor({atom1->nao(), (size_t)nsad2}));
        double* C1p = C1->data().data();
        for (int i = 0; i < nsad2; i++) {
            C1p[(i + necp[A]) * nsad2 + i] = 1.0;
        }
        // Basis2 C Matrix
        std::shared_ptr<Tensor> C2 = SAD::project_orbitals(
            resources,
            C1,
            atom1,
            atom2);
        const double* C2p = C2->data().data();
        // C Matrix placement    
        for (size_t p = 0; p < atom2->nao(); p++) {
            for (size_t i = 0; i < nsad2; i++) {
                Cfp[(p + aoOff2) * nsad + (i + sadOff)] = 
                    C2p[p * nsad2 + i] * sqrt(nocc1p[i + aoOff1 + necp[A]]);
            }
        }
        // Offset subblock 
        sadOff += nsad2;
        aoOff1 += atom1->nao();;
        aoOff2 += atom2->nao();
    }
    
    return Cf;
}

std::shared_ptr<Tensor> SAD::project_orbitals(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& C1,
    const std::shared_ptr<Basis>& basis1,
    const std::shared_ptr<Basis>& basis2,
    double threpq,
    double threpow)
{
    // Validity checks
    C1->ndim_error(2);
    size_t nmo = C1->shape()[1];
    size_t nbf1 = basis1->nao();
    size_t nbf2 = basis2->nao();
    std::vector<size_t> dim1;
    dim1.push_back(nbf1);
    dim1.push_back(nmo);
    C1->shape_error(dim1);

    // Overlap Integrals
    std::shared_ptr<PairList> pairlist12 = PairList::build_schwarz(basis1,basis2,false,threpq); 
    std::shared_ptr<PairList> pairlist22 = PairList::build_schwarz(basis2,basis2,true,threpq); 
    std::shared_ptr<Tensor> S12 = IntBox::overlap(
        resources,
        pairlist12);
    std::shared_ptr<Tensor> S22 = IntBox::overlap(
        resources,
        pairlist22);

    // Projection
    std::shared_ptr<Tensor> S22inv = Tensor::power(S22,-1.0,threpow);
    std::shared_ptr<Tensor> A = Tensor::chain({S12,C1},{true,false});
    std::shared_ptr<Tensor> M = Tensor::chain({A,S22inv,A},{true,false,false});
    std::shared_ptr<Tensor> Minv12 = Tensor::power(M,-1.0/2.0,threpow);
    std::shared_ptr<Tensor> C2 = Tensor::chain({S22inv,A,Minv12},{false,false,false}); 

    return C2;
}

} // namespace lightspeed
