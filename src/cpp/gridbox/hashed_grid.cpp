#include <lightspeed/gridbox.hpp>
#include <lightspeed/tensor.hpp>
#include "../util/string.hpp"
#include <stdexcept>
#include <cstdio>

namespace lightspeed {

HashedGrid::HashedGrid(
    const std::shared_ptr<Tensor>& xyz,
    double R) :
    xyz_(xyz),
    R_(R),
    Rinv_(1.0 / R)
{
    xyz->ndim_error(2);
    std::vector<size_t> dim;
    dim.push_back(xyz->shape()[0]); 
    dim.push_back(3);
    xyz->shape_error(dim);
    if (R <= 0.0) throw std::runtime_error("HashedGrid: R <= 0.0");

    // => Hash Map of Grid <= //

    size_t nP = xyz->shape()[0];    
    const double* xyzp = xyz->data().data();
    for (size_t P = 0; P < nP; P++) {
        std::tuple<int,int,int> key = hashkey(
            xyzp[3*P+0],
            xyzp[3*P+1],
            xyzp[3*P+2]);
        map_[key].push_back(P);
    } 
}
std::string HashedGrid::string() const
{
    std::string s = "";
    s += "HashedGrid:\n";
    s += sprintf2("  R    = %11.3E\n", R_);
    s += sprintf2("  nbox = %11zu\n", nbox());
    return s;
}

} // namespace lightspeed
