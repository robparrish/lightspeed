#include <lightspeed/basis.hpp>
#include <lightspeed/tensor.hpp>
#include "../util/string.hpp"
#include <algorithm>
#include <cstdio>
#include <map>

namespace {

bool compare_atoms(const lightspeed::Primitive& prim1, const lightspeed::Primitive& prim2)
{
    return prim1.atomIdx() < prim2.atomIdx();
}

}

namespace lightspeed {

std::string Primitive::string() const
{
    std::string s = "";
    s += "Primitive:\n";
    s += sprintf2("  c        = %14.6E\n", c());
    s += sprintf2("  e        = %14.6E\n", e());
    s += sprintf2("  x        = %14.6E\n", x());
    s += sprintf2("  y        = %14.6E\n", y());
    s += sprintf2("  z        = %14.6E\n", z());
    s += sprintf2("  c0       = %14.6E\n", c0());
    s += sprintf2("  L        = %14d\n", L());
    s += sprintf2("  pure?    = %14s\n", is_pure() ? "Yes" : "No");
    s += sprintf2("  aoIdx    = %14d\n", aoIdx());
    s += sprintf2("  cartIdx  = %14d\n", cartIdx());
    s += sprintf2("  primIdx  = %14d\n", primIdx());
    s += sprintf2("  shellIdx = %14d\n", shellIdx());
    s += sprintf2("  atomIdx  = %14d\n", atomIdx());
    s += sprintf2("  nao      = %14d\n", nao());
    s += sprintf2("  npure    = %14d\n", npure());
    s += sprintf2("  ncart    = %14d\n", ncart());
    return s;
}

std::string Shell::string() const
{
    std::string s = "";
    s += "Shell:\n";
    s += "  cs       = ";
    for (double c : cs_) s += sprintf2("%14.6E ", c);
    s += "\n";
    s += "  es       = ";
    for (double e : es_) s += sprintf2("%14.6E ", e);
    s += "\n";
    s += sprintf2("  x        = %14.6E\n", x());
    s += sprintf2("  y        = %14.6E\n", y());
    s += sprintf2("  z        = %14.6E\n", z());
    s += "  c0s      = ";
    for (double c0 : c0s_) s += sprintf2("%14.6E ", c0);
    s += "\n";
    s += sprintf2("  L        = %14d\n", L());
    s += sprintf2("  pure?    = %14s\n", is_pure() ? "Yes" : "No");
    s += sprintf2("  aoIdx    = %14d\n", aoIdx());
    s += sprintf2("  cartIdx  = %14d\n", cartIdx());
    s += sprintf2("  primIdx  = %14d\n", primIdx());
    s += sprintf2("  shellIdx = %14d\n", shellIdx());
    s += sprintf2("  atomIdx  = %14d\n", atomIdx());
    s += sprintf2("  nprim    = %14d\n", nprim());
    s += sprintf2("  nao      = %14d\n", nao());
    s += sprintf2("  npure    = %14d\n", npure());
    s += sprintf2("  ncart    = %14d\n", ncart());
    return s;
}

std::string Basis::string() const
{
    std::string s = "";
    s += sprintf2("Basis: %s\n", name_.c_str());
    s += sprintf2("  nao     = %5zu\n", nao());
    s += sprintf2("  ncart   = %5zu\n", ncart());
    s += sprintf2("  nprim   = %5zu\n", nprim());
    s += sprintf2("  nshell  = %5zu\n", nshell());
    s += sprintf2("  natom   = %5zu\n", natom());
    s += sprintf2("  pure?   = %5s\n", has_pure() ? "Yes" : "No");
    s += sprintf2("  max L   = %5d\n", max_L());
    return s;
}

int Basis::max_L() const 
{
    int max_val = 0;
    for (size_t ind = 0; ind < primitives_.size(); ind++) {
        max_val = std::max(max_val, primitives_[ind].L());
    }
    return max_val;
}
int Basis::max_nao() const 
{
    int max_val = 0;
    for (size_t ind = 0; ind < primitives_.size(); ind++) {
        max_val = std::max(max_val, primitives_[ind].nao());
    }
    return max_val;
}
int Basis::max_npure() const 
{
    int max_val = 0;
    for (size_t ind = 0; ind < primitives_.size(); ind++) {
        max_val = std::max(max_val, primitives_[ind].npure());
    }
    return max_val;
}
int Basis::max_ncart() const 
{
    int max_val = 0;
    for (size_t ind = 0; ind < primitives_.size(); ind++) {
        max_val = std::max(max_val, primitives_[ind].ncart());
    }
    return max_val;
}
bool Basis::has_pure() const
{
    for (size_t ind = 0; ind < primitives_.size(); ind++) {
        if (primitives_[ind].is_pure()) return true;
    }
    return false;
}

std::shared_ptr<Tensor> Basis::xyz() const
{
    // C++03
    std::vector<size_t> dim;
    dim.push_back(natom());
    dim.push_back(3);
    std::shared_ptr<Tensor> xyz(new Tensor(dim,"xyz"));
    // C++11
    //std::shared_ptr<Tensor> xyz(new Tensor({natom(), 3}));
    double* xyzp = xyz->data().data();
    for (size_t P2 = 0; P2 < primitives_.size(); P2++) {
        const Primitive& prim = primitives_[P2];
        int A = prim.atomIdx();
        xyzp[3*A+0] = prim.x();
        xyzp[3*A+1] = prim.y();
        xyzp[3*A+2] = prim.z();
    }
    return xyz;
}

std::shared_ptr<Basis> Basis::update_xyz(
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
    std::vector<Primitive> prims;
    for (size_t P2 = 0; P2 < primitives_.size(); P2++) {
        const Primitive& prim = primitives_[P2];
        int A = prim.atomIdx();
        prims.push_back(Primitive(
            prim.c(),
            prim.e(),
            xyzp[3*A+0],
            xyzp[3*A+1],
            xyzp[3*A+2],
            prim.c0(),
            prim.L(),
            prim.is_pure(),
            prim.aoIdx(),
            prim.cartIdx(),
            prim.primIdx(),
            prim.shellIdx(),
            prim.atomIdx()));
            
    }
    return std::shared_ptr<Basis>(new Basis(name_,prims));
}

std::shared_ptr<Basis> Basis::subset(
    const std::vector<size_t>& atom_range) const
{
    std::vector<Primitive> prims;
    size_t aoOff = 0;
    size_t cartOff = 0;
    size_t primOff = 0;
    size_t shellOff = 0;
    for (size_t A2 = 0; A2 < atom_range.size(); A2++) {
        size_t A = atom_range[A2];
        if (A >= natom()) throw std::runtime_error("Basis::subset: Invalid atom index");

        // Find the range of primitives for this atom O(log(N))
        
        Primitive dummy(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            false,
            0,
            0,
            0,
            0,
            A);
    
        std::pair<std::vector<Primitive>::const_iterator, std::vector<Primitive>::const_iterator> bounds = 
            std::equal_range(primitives_.begin(),primitives_.end(), dummy, compare_atoms);
        size_t Pstart = std::distance(primitives_.begin(), bounds.first);
        size_t Pstop = std::distance(primitives_.begin(), bounds.second);

        const Primitive& prim_start = primitives_[Pstart];
        size_t aoStart = prim_start.aoIdx();
        size_t cartStart = prim_start.cartIdx();
        size_t primStart = prim_start.primIdx();
        size_t shellStart = prim_start.shellIdx();
        
        for (size_t P2 = Pstart; P2 < Pstop; P2++) {
            const Primitive& prim = primitives_[P2];
            prims.push_back(Primitive(
                prim.c(),
                prim.e(),
                prim.x(),
                prim.y(),
                prim.z(),
                prim.c0(),
                prim.L(),
                prim.is_pure(),
                prim.aoIdx()-aoStart+aoOff,
                prim.cartIdx()-cartStart+cartOff,
                prim.primIdx()-primStart+primOff,
                prim.shellIdx()-shellStart+shellOff,
                A2)); 
        }
        
        const Primitive& prim_stop = prims[prims.size() - 1];
        aoOff = prim_stop.aoIdx() + prim_stop.nao();
        cartOff = prim_stop.cartIdx() + prim_stop.ncart();
        primOff = prim_stop.primIdx() + 1;
        shellOff = prim_stop.shellIdx() + 1;
    }
    
    return std::shared_ptr<Basis>(new Basis(name_,prims));
}
std::shared_ptr<Basis> Basis::concatenate(
    const std::vector<std::shared_ptr<Basis> >& mols)
{
    std::string name = (mols.size() ? mols[0]->name() : "");
    std::vector<Primitive> prims;
    size_t aoOff = 0;
    size_t cartOff = 0;
    size_t primOff = 0;
    size_t shellOff = 0;
    size_t atomOff = 0;
    for (size_t mol_ind = 0; mol_ind < mols.size(); mol_ind++) {
        std::shared_ptr<Basis> mol = mols[mol_ind];
        for (size_t P2 = 0; P2 < mol->primitives().size(); P2++) {
            const Primitive& prim = mol->primitives()[P2];
            prims.push_back(Primitive(
                prim.c(),
                prim.e(),
                prim.x(),
                prim.y(),
                prim.z(),
                prim.c0(),
                prim.L(),
                prim.is_pure(),
                prim.aoIdx()+aoOff,
                prim.cartIdx()+cartOff,
                prim.primIdx()+primOff,
                prim.shellIdx()+shellOff,
                prim.atomIdx()+atomOff));
        } 
        aoOff += mol->nao();
        cartOff += mol->ncart();
        primOff += mol->nprim();
        shellOff += mol->nshell();
        atomOff += mol->natom();
    }
    return std::shared_ptr<Basis>(new Basis(name,prims));
}
std::vector<Shell> Basis::build_shells() const 
{
    if (!primitives_.size()) return {};
    std::vector<size_t> blocks;
    blocks.push_back(0);
    for (size_t P = 0; P < primitives_.size(); P++) {
        if (primitives_[P].shellIdx() != primitives_[blocks[blocks.size() - 1]].shellIdx()) {
            blocks.push_back(P);
        }
    }
    blocks.push_back(primitives_.size());

    std::vector<Shell> vals;
    for (size_t block_ind = 0; block_ind < blocks.size() - 1; block_ind++) {
        size_t start = blocks[block_ind];
        size_t stop = blocks[block_ind+1];
        size_t nprim = stop - start;
        std::vector<double> cs(nprim);
        std::vector<double> es(nprim);
        std::vector<double> c0s(nprim);
        for (size_t P = 0; P < nprim; P++) {
            const Primitive& prim = primitives_[P + start];
            cs[P] = prim.c();
            es[P] = prim.e();
            c0s[P] = prim.c0();
        }
        const Primitive& prim = primitives_[start];
        vals.push_back(Shell(
            cs,
            es,
            prim.x(),
            prim.y(),
            prim.z(),
            c0s,
            prim.L(),
            prim.is_pure(),
            prim.aoIdx(),
            prim.cartIdx(),
            prim.primIdx(),
            prim.shellIdx(),
            prim.atomIdx()));
    }
    return vals;
}

} // namespace lightspeed
