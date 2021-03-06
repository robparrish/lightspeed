#include "tcintbox/tcintbox.hpp"
#include "tcintbox/tc2ls.hpp"
#include <lightspeed/tensor.hpp>
#include <lightspeed/intbox.hpp>
#include <lightspeed/basis.hpp>
#include <lightspeed/ecp.hpp>
#include <lightspeed/pair_list.hpp>
#include <lightspeed/pure_transform.hpp>
#include <lightspeed/resource_list.hpp>
#include <stdexcept>

namespace lightspeed {
        
std::shared_ptr<Tensor> IntBox::ecpGradTC(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<ECPBasis>& ecp,
    float threoe,
    const std::shared_ptr<Tensor>& G
    )
{
    if (!pairlist->is_symmetric()) throw std::runtime_error("IntBox::ecpGradTC: pairlist is not symmetric");
    if (ecp->natom() != pairlist->basis1()->natom()) throw std::runtime_error("IntBox::ecpGradTC: ecp and basis natom do not match.");

    std::shared_ptr<Basis> basis = pairlist->basis1();

    std::shared_ptr<Tensor> DT = TCTransform::LStoTC(
        D,
        basis);

    std::shared_ptr<Tensor> G2 = G;
    if (!G2) {
        G2 = std::shared_ptr<Tensor>(new Tensor({basis->natom(), 3}, "G (ECP)"));
    }
    
    // Repack points
    std::shared_ptr<Tensor> xyzc(new Tensor({ecp->natom(),4}));
    double* xyzcp = xyzc->data().data();
    // for (const ECPShell& sh : ecp->shells()) { // DOES NOT WORK
    // TC requires a dense set of positions -> not available in ECPBasis.
    // Therefore, we use the info from basis instead
    for (const Shell& sh : basis->shells()) {
        size_t A = sh.atomIdx();
        xyzcp[A*4 + 0] = sh.x();
        xyzcp[A*4 + 1] = sh.y();
        xyzcp[A*4 + 2] = sh.z();
        xyzcp[A*4 + 3] = 0.0; // Should not be needed
    }

    std::shared_ptr<TCIntBox> tc = TCIntBox::instance(resources);

    tc->set_basis(
        pairlist->basis1(),
        pairlist->thre());
    tc->set_ecp_basis(
        ecp);

    tc->computeECPGrad(
        threoe,
        1.0, // TODO: WTF
        xyzc->data().data(),
        DT->data().data(),
        G2->data().data(),
        G2->data().data());
    
    tc->clear_basis();
    tc->clear_ecp_basis();
    
    return G2;
}

} // namespace lightspeed
