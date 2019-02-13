#include <lightspeed/molecule.hpp>
#include <lightspeed/tensor.hpp>
#include "../util/string.hpp"
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <map>

namespace lightspeed {

double Atom::distance(const Atom& other) const 
{
    double dx = x_ - other.x_;
    double dy = y_ - other.y_;
    double dz = z_ - other.z_;
    return sqrt(dx*dx + dy*dy + dz*dz);
}

Molecule::Molecule(
    const std::string& name,
    const std::vector<Atom>& atoms,
    double charge,
    double multiplicity) :
    name_(name),
    atoms_(atoms),
    charge_(charge),
    multiplicity_(multiplicity)
{
    // Check that atom indices are setup correctly
    for (size_t A = 0; A < atoms_.size(); A++) {
        if (atoms[A].atomIdx() != A) throw std::runtime_error("Molecule: atom index field is not correct");
    }
}
std::string Molecule::string(bool print_coords) const
{
    std::string str = "";
    str += sprintf2("Molecule: %s\n", name_.c_str());
    str += sprintf2("  Natom        = %11zu\n", natom());
    str += sprintf2("  Charge       = %11.3f\n", charge());
    str += sprintf2("  Multiplicity = %11.3f\n", multiplicity());
    if (print_coords) {
        str += sprintf2("\n");
        str += sprintf2("  %-12s %18s %18s %18s\n", 
            "Atom", "X", "Y", "Z");
        for (size_t A = 0; A < atoms_.size(); A++) {
            const Atom& atom = atoms_[A];
            str += sprintf2("  %-12s %18.12f %18.12f %18.12f\n", 
                atom.label().c_str(),
                atom.x(),
                atom.y(),
                atom.z());
        }   
    }
    return str;
}
double Molecule::nuclear_charge() const 
{
    double Z = 0.0;
    for (size_t A = 0; A < atoms_.size(); A++) {
        Z += atoms_[A].Z();
    }
    return Z;
}
std::shared_ptr<Tensor> Molecule::nuclear_COM() const 
{
    // C++03
    std::vector<size_t> three;
    three.push_back(3);
    std::shared_ptr<Tensor> X(new Tensor(three,"Nuclear COM"));
    // C++11
    //std::shared_ptr<Tensor> X(new Tensor({3},"Nuclear COM"));
    double* Xp = X->data().data();
    for (size_t Aind = 0; Aind < atoms_.size(); Aind++) {
        const Atom& atom = atoms_[Aind];
        Xp[0] += atom.x() * atom.Z(); 
        Xp[1] += atom.y() * atom.Z(); 
        Xp[2] += atom.z() * atom.Z(); 
    }
    double Ztot = nuclear_charge();
    X->scale(1.0/Ztot);
    return X;
}
std::shared_ptr<Tensor> Molecule::nuclear_I() const
{
    std::shared_ptr<Tensor> X = nuclear_COM();
    double* Xp = X->data().data();
    // C++03
    std::vector<size_t> three2;
    three2.push_back(3);
    three2.push_back(3);
    std::shared_ptr<Tensor> I(new Tensor(three2,"Nuclear I"));
    // C++11
    //std::shared_ptr<Tensor> I(new Tensor({3,3},"Nuclear I"));
    double* Ip = I->data().data();
    for (size_t Aind = 0; Aind < atoms_.size(); Aind++) {
        const Atom& atom = atoms_[Aind];
        Ip[0*3 + 0] += (atom.x() - Xp[0]) * (atom.x() - Xp[0]) * atom.Z(); 
        Ip[0*3 + 1] += (atom.x() - Xp[0]) * (atom.y() - Xp[1]) * atom.Z(); 
        Ip[0*3 + 2] += (atom.x() - Xp[0]) * (atom.z() - Xp[2]) * atom.Z(); 
        Ip[1*3 + 0] += (atom.y() - Xp[1]) * (atom.x() - Xp[0]) * atom.Z(); 
        Ip[1*3 + 1] += (atom.y() - Xp[1]) * (atom.y() - Xp[1]) * atom.Z(); 
        Ip[1*3 + 2] += (atom.y() - Xp[1]) * (atom.z() - Xp[2]) * atom.Z(); 
        Ip[2*3 + 0] += (atom.z() - Xp[2]) * (atom.x() - Xp[0]) * atom.Z(); 
        Ip[2*3 + 1] += (atom.z() - Xp[2]) * (atom.y() - Xp[1]) * atom.Z(); 
        Ip[2*3 + 2] += (atom.z() - Xp[2]) * (atom.z() - Xp[2]) * atom.Z(); 
    }
    double Ztot = nuclear_charge();
    I->scale(1.0/Ztot);
    return I;
}

double Molecule::nuclear_repulsion_energy() const
{
    double E = 0.0;
    for (size_t A = 0; A < atoms_.size(); A++) {
        for (size_t B = A + 1; B < atoms_.size(); B++) {
            double ZAB = atoms_[A].Z() * atoms_[B].Z();
            double rAB = atoms_[A].distance(atoms_[B]);
            E += ZAB / rAB;
        }
    }
    return E;
}
double Molecule::nuclear_repulsion_energy_other(
    const std::shared_ptr<Molecule>& other) const
{
    const std::vector<Atom>& atomsB = other->atoms(); 

    double E = 0.0;
    for (size_t A = 0; A < atoms_.size(); A++) {
        for (size_t B = 0; B < atomsB.size(); B++) {
            double ZA = atoms_[A].Z();
            double ZB = atomsB[B].Z();
            double ZAB = ZA * ZB;
            if (ZAB != 0.0) {
                double rAB = atoms_[A].distance(atomsB[B]);
                E += ZAB / rAB;
            }
        }
    }
    return E;
}
std::shared_ptr<Tensor> Molecule::nuclear_repulsion_grad() const 
{
    // C++03
    std::vector<size_t> dim;
    dim.push_back(natom());
    dim.push_back(3);
    std::shared_ptr<Tensor> G(new Tensor(dim,"G"));
    // C++11
    //std::shared_ptr<Tensor> G(new Tensor({natom(), 3},"G"));
    double* Gp = G->data().data();
    
    for (size_t A = 0; A < atoms_.size(); A++) {
        for (size_t B = 0; B < atoms_.size(); B++) {
            if (A == B) continue;
            int A2 = atoms_[A].atomIdx();
            double ZAB = atoms_[A].Z() * atoms_[B].Z();
            double rAB = atoms_[A].distance(atoms_[B]);
            double rABm3 = pow(rAB,-3.0);
            Gp[A2*3 + 0] -= (atoms_[A].x() - atoms_[B].x()) * ZAB * rABm3;
            Gp[A2*3 + 1] -= (atoms_[A].y() - atoms_[B].y()) * ZAB * rABm3;
            Gp[A2*3 + 2] -= (atoms_[A].z() - atoms_[B].z()) * ZAB * rABm3;
        }
    }
             
    return G;
}
std::shared_ptr<Tensor> Molecule::xyz() const
{
    // C++03
    std::vector<size_t> dim;
    dim.push_back(natom());
    dim.push_back(3);
    std::shared_ptr<Tensor> xyz(new Tensor(dim,"xyz"));
    // C++11
    //std::shared_ptr<Tensor> xyz(new Tensor({natom(), 3}));
    double* xyzp = xyz->data().data();
    for (size_t A2 = 0; A2 < atoms_.size(); A2++) {
        const Atom& atom = atoms_[A2];
        int A = atom.atomIdx();
        xyzp[3*A+0] = atom.x();
        xyzp[3*A+1] = atom.y();
        xyzp[3*A+2] = atom.z();
    }
    return xyz;
}
std::shared_ptr<Tensor> Molecule::Z() const
{
    // C++03
    std::vector<size_t> dim;
    dim.push_back(natom());
    std::shared_ptr<Tensor> Z(new Tensor(dim,"Z"));
    // C++11
    //std::shared_ptr<Tensor> Z(new Tensor({natom()}));
    double* Zp = Z->data().data();
    for (size_t A2 = 0; A2 < atoms_.size(); A2++) {
        const Atom& atom = atoms_[A2];
        int A = atom.atomIdx();
        Zp[A] = atom.Z();
    }
    return Z;
}
std::shared_ptr<Tensor> Molecule::xyzZ() const
{
    // C++03
    std::vector<size_t> dim;
    dim.push_back(natom());
    dim.push_back(4);
    std::shared_ptr<Tensor> xyzZ(new Tensor(dim,"xyzZ"));
    // C++11
    //std::shared_ptr<Tensor> xyzZ(new Tensor({natom(), 4}));
    double* xyzZp = xyzZ->data().data();
    for (size_t A2 = 0; A2 < atoms_.size(); A2++) {
        const Atom& atom = atoms_[A2];
        int A = atom.atomIdx();
        xyzZp[4*A+0] = atom.x();
        xyzZp[4*A+1] = atom.y();
        xyzZp[4*A+2] = atom.z();
        xyzZp[4*A+3] = atom.Z();
    }
    return xyzZ;
}
std::shared_ptr<Molecule> Molecule::update_xyz(
    const std::shared_ptr<Tensor>& xyz) const
{
    // C++03
    std::vector<size_t> dim;
    dim.push_back(natom());
    dim.push_back(3);
    xyz->shape_error(dim);
    // C++11
    //xyz->shape_error({natom(),3});
    const double* xyzp = xyz->data().data();
    std::vector<Atom> atoms;
    for (size_t A2 = 0; A2 < atoms_.size(); A2++) {
        const Atom& atom = atoms_[A2];
        int A = atom.atomIdx();
        atoms.push_back(Atom(
            atom.label(),
            atom.symbol(),
            atom.N(),
            xyzp[3*A+0], 
            xyzp[3*A+1], 
            xyzp[3*A+2], 
            atom.Z(),
            atom.atomIdx()));
    }
    return std::shared_ptr<Molecule>(new Molecule(name_,atoms,charge_,multiplicity_));
}
std::shared_ptr<Molecule> Molecule::update_Z(
    const std::shared_ptr<Tensor>& Z) const
{
    // C++03
    std::vector<size_t> dim;
    dim.push_back(natom());
    Z->shape_error(dim);
    // C++11
    //Z->shape_error({natom()});
    const double* Zp = Z->data().data();
    std::vector<Atom> atoms;
    for (size_t A2 = 0; A2 < atoms_.size(); A2++) {
        const Atom& atom = atoms_[A2];
        int A = atom.atomIdx();
        atoms.push_back(Atom(
            atom.label(),
            atom.symbol(),
            atom.N(),
            atom.x(),
            atom.y(),
            atom.z(),
            Zp[A],
            atom.atomIdx()));
    }
    return std::shared_ptr<Molecule>(new Molecule(name_,atoms,charge_,multiplicity_));
}
std::shared_ptr<Molecule> Molecule::subset(
    const std::vector<size_t>& atom_range,
    double charge,
    double multiplicity) const
{
    std::vector<Atom> subset_atoms;
    size_t atomIdx = 0;
    for (size_t A2 = 0; A2 < atom_range.size(); A2++) {
        size_t A = atom_range[A2];
        if (A >= natom()) throw std::runtime_error("Molecule:subset: Invalid index");
        const Atom& atom = atoms_[A];
        subset_atoms.push_back(Atom(
            atom.label(),
            atom.symbol(),
            atom.N(),
            atom.x(),
            atom.y(),
            atom.z(),
            atom.Z(),
            atomIdx++));
    } 
    return std::shared_ptr<Molecule>(new Molecule(name_,subset_atoms,charge,multiplicity));
}
std::shared_ptr<Molecule> Molecule::concatenate(
    const std::vector<std::shared_ptr<Molecule> >& mols,
    double charge,
    double multiplicity)
{
    std::string name = (mols.size() ? mols[0]->name() : "");
    std::vector<Atom> atoms;
    size_t offset = 0;
    for (size_t mol_ind = 0; mol_ind < mols.size(); mol_ind++) {
        std::shared_ptr<Molecule> mol = mols[mol_ind];
        for (size_t atom_ind = 0; atom_ind < mol->atoms().size(); atom_ind++) {
            const Atom& atom = mol->atoms()[atom_ind];
            atoms.push_back(Atom(
                atom.label(),
                atom.symbol(),
                atom.N(),
                atom.x(),
                atom.y(),
                atom.z(),
                atom.Z(),
                atom.atomIdx() + offset));
        }
        offset += mol->natom();
    }
    return std::shared_ptr<Molecule>(new Molecule(name,atoms,charge,multiplicity));
}

} // namespace lightspeed
