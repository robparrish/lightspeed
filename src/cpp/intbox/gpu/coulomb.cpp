#include <lightspeed/intbox.hpp>
#include <lightspeed/basis.hpp>
#include <lightspeed/pair_list.hpp>
#include <lightspeed/pure_transform.hpp>
#include <lightspeed/resource_list.hpp>
#include <lightspeed/ewald.hpp>
#include <lightspeed/rys.hpp>
#include <lightspeed/hermite.hpp>
#include <cmath>
#include <cstring>

namespace lightspeed {
        
std::shared_ptr<Tensor> IntBox::coulombGPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist12,
    const std::shared_ptr<PairList>& pairlist34,
    const std::shared_ptr<Tensor>& D34,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& J12
    )
{
    // Cartesian representation of D34
    std::shared_ptr<Tensor> D34T = PureTransform::pureToCart2(
        pairlist34->basis1(),
        pairlist34->basis2(),
        D34);

    // Cartesian representation of J12
    std::shared_ptr<Tensor> J12T = PureTransform::allocCart2(
        pairlist12->basis1(),
        pairlist12->basis2(),
        J12);

    // Find the max element in the bra
    float max12 = PairListUtil::max_bound(pairlist12);

    // Truncate the ket (including density)
    std::shared_ptr<PairList> pairlist34T = PairListUtil::truncate_ket(
        pairlist34,
        D34T,
        thresp / max12);
        
    // Find the max element in the ket (including density)
    float max34 = PairListUtil::max_bound(pairlist34T);

    // Truncate the bra
    std::shared_ptr<PairList> pairlist12T = PairListUtil::truncate_bra(
        pairlist12,
        thresp / max34);

    // Hermite representations
    std::shared_ptr<Hermite> herms12(new Hermite(pairlist12T));
    std::shared_ptr<Hermite> herms34(new Hermite(pairlist34T));

    // Cart-to-Hermite transform of D34
    herms34->cart_to_herm(D34T, true);

    // Hermite-to-cart transform of J12
    herms12->herm_to_cart(J12T, false);
    
    // Cartesian-to-pure transform of J12
    std::shared_ptr<Tensor> J12U = PureTransform::cartToPure2(
        pairlist12->basis1(),
        pairlist12->basis2(),
        J12T,
        J12);
    
    return J12U;
}

} // namespace lightspeed
