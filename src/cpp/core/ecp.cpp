#include <lightspeed/ecp.hpp>
#include <lightspeed/tensor.hpp>
#include "../util/string.hpp"

namespace lightspeed {

ECPBasis::ECPBasis(
    const std::string& name,
    const std::vector<ECPShell>& shells,
    const std::vector<int>& nelecs) :
    name_(name),
    shells_(shells),
    nelecs_(nelecs)
{
    // Check that the atom indices make sense
    for (size_t A = 1; A < shells_.size(); A++) {
        if (shells_[A-1].atomIdx() > shells_[A].atomIdx()) {
            throw std::runtime_error("ECPBasis: atomic indices are not strictly increasing");
        }
    }

    // Check that the shell, function and cartesian indices make sense
    size_t nshell2 = 0;
    for (auto sh : shells_) {
        if (sh.shellIdx() != nshell2) {
            throw std::runtime_error("ECPBasis: shell index is not correct");
        }
        nshell2++;
    }
    
    // Figure out how many atoms there are 
    size_t natom2 = 0;
    for (auto sh : shells) {
        natom2 = std::max(natom2, sh.atomIdx());
    }
    natom2++;

    if (natom2 > nelecs_.size()) { 
        throw std::runtime_error("BasisSet: shell atom index is greater than natom");
    }

    // Populate the atoms_to_shell_inds_ field
    atoms_to_shell_inds_.resize(nelecs_.size());
    for (size_t P = 0; P < shells_.size(); P++) {
        atoms_to_shell_inds_[shells_[P].atomIdx()].push_back(P);
    }

    nelec_ = 0;
    for (auto s : nelecs_) {
        nelec_ += s;
    }

    nprim_ = 0;
    for (auto s : shells_) {
        nprim_ += s.nprim();
    }
}
std::shared_ptr<ECPBasis> ECPBasis::subset(
    const std::vector<size_t>& atom_range) const
{
    std::vector<ECPShell> shells;
    std::vector<int> nelecs;
    size_t nshell2 = 0;
    for (size_t A2 = 0; A2 < atom_range.size(); A2++) {
        size_t A = atom_range[A2];
        nelecs.push_back(nelecs_[A]);
        if (A >= natom()) { throw std::runtime_error("BasisSet:subset: incorrect atom index"); }
    
        const std::vector<size_t>& shell_inds = atoms_to_shell_inds()[A];
        for (size_t P2 = 0; P2 < shell_inds.size(); P2++) {
            size_t P = shell_inds[P2];
            const ECPShell& sh = shells_[P]; 
            shells.push_back(ECPShell(
                sh.x(),
                sh.y(),
                sh.z(),
                sh.L(),
                sh.is_max_L(),
                sh.ns(),
                sh.cs(),
                sh.es(),
                A2, 
                nshell2));
            nshell2++;
        }
    }
    return std::shared_ptr<ECPBasis>(new ECPBasis(name(), shells, nelecs));
}
std::shared_ptr<ECPBasis> ECPBasis::concatenate(
    const std::vector<std::shared_ptr<ECPBasis>>& bases) 
{
    std::string name = (bases.size() ? bases[0]->name() : "");
    std::vector<ECPShell> shells;
    std::vector<int> nelecs;
    size_t natom2 = 0;
    size_t nshell2 = 0;
    for (auto bas : bases) {
        nelecs.insert(nelecs.end(),bas->nelecs().begin(),bas->nelecs().end());
        for (auto shell_inds : bas->atoms_to_shell_inds()) {
            for (size_t P : shell_inds) {
                const ECPShell& sh = bas->shells()[P]; 
                shells.push_back(ECPShell(
                    sh.x(),
                    sh.y(),
                    sh.z(),
                    sh.L(),
                    sh.is_max_L(),
                    sh.ns(),
                    sh.cs(),
                    sh.es(),
                    sh.atomIdx() + natom2, 
                    nshell2));
                nshell2++;
            } 
        }
        natom2 += bas->natom();
    }
    return std::shared_ptr<ECPBasis>(new ECPBasis(name, shells, nelecs));
}
int ECPBasis::max_L() const 
{
    int val = 0;
    for (size_t ind = 0; ind < shells_.size(); ind++) {
        val = std::max(val,shells_[ind].L());
    }
    return val;
}
size_t ECPBasis::max_nprim() const 
{
    size_t val = 0L;
    for (size_t ind = 0; ind < shells_.size(); ind++) {
        val = std::max(val,shells_[ind].nprim());
    }
    return val;
}
std::string ECPBasis::string() const
{
    std::string str = "";
    str += sprintf2("ECPBasis: %s\n", name_.c_str());
    str += sprintf2("  Natom  = %5zu\n", natom());
    str += sprintf2("  Nelec  = %5zu\n", nelec());
    str += sprintf2("  Nshell = %5zu\n", nshell());
    str += sprintf2("  Max AM = %5d\n", max_L());
    return str;
}
std::shared_ptr<Tensor> ECPBasis::xyz() const 
{
    std::shared_ptr<Tensor> xyz(new Tensor({natom(),3},"xyz"));
    double* xyzp = xyz->data().data();
    for (auto sh : shells_) {
        xyzp[3*sh.atomIdx() + 0] = sh.x();
        xyzp[3*sh.atomIdx() + 1] = sh.y();
        xyzp[3*sh.atomIdx() + 2] = sh.z();
    }
    return xyz;
}
std::shared_ptr<ECPBasis> ECPBasis::update_xyz(
    const std::shared_ptr<Tensor>& xyz) const
{
    xyz->shape_error({natom(),3});
    double* xyzp = xyz->data().data();
    std::vector<ECPShell> shells;
    for (auto sh : shells_) {
        shells.push_back(ECPShell(
            xyzp[3*sh.atomIdx() + 0],
            xyzp[3*sh.atomIdx() + 1],
            xyzp[3*sh.atomIdx() + 2],
            sh.L(),
            sh.is_max_L(),
            sh.ns(),
            sh.cs(),
            sh.es(),
            sh.atomIdx(), 
            sh.shellIdx()));
    }
    return std::shared_ptr<ECPBasis>(new ECPBasis(name_,shells,nelecs_));
}

} // namespace lightspeed
