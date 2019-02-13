#include "tcintbox/tcintbox.hpp"
#include "tcintbox/tc2ls.hpp"
#include <lightspeed/tensor.hpp>
#include <lightspeed/intbox.hpp>
#include <lightspeed/basis.hpp>
#include <lightspeed/pair_list.hpp>
#include <lightspeed/pure_transform.hpp>
#include <lightspeed/resource_list.hpp>
#include <lightspeed/ewald.hpp>
#include <stdexcept>

namespace lightspeed {

std::shared_ptr<Tensor> IntBox::exchangeTC(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist12,
    const std::shared_ptr<PairList>& pairlist34,
    const std::shared_ptr<Tensor>& D24,
    bool D24symm,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& K13
    )
{
    if (pairlist12 != pairlist34) throw std::runtime_error("IntBox::exchangeTC: pairlist12 != pairlist34");
    if (!pairlist12->is_symmetric()) throw std::runtime_error("IntBox::exchangeTC: pairlist12 is not symmetric");
    
    std::shared_ptr<Basis> basis = pairlist12->basis1();

    std::shared_ptr<Tensor> DT = TCTransform::LStoTC(
        D24,
        basis);
    std::shared_ptr<Tensor> KT(new Tensor({basis->ncart(),basis->ncart()}));

    auto ewald2 = TCTransform::ewaldTC(ewald);

    std::shared_ptr<TCIntBox> tc = TCIntBox::instance(resources);
    tc->set_basis(
        pairlist12->basis1(),
        pairlist12->thre());
    for (auto task : ewald2) {
        if (D24symm) {
            tc->computeKFockSym(
                thresp,
                thredp,
                std::get<0>(task),
                std::get<1>(task),
                std::get<2>(task),
                DT->data().data(),
                KT->data().data());
        } else {
            tc->computeKFockGen(
                thresp,
                thredp,
                std::get<0>(task),
                std::get<1>(task),
                std::get<2>(task),
                DT->data().data(),
                KT->data().data());
        }
    }
    tc->clear_basis();
    
    std::shared_ptr<Tensor> KU = TCTransform::TCtoLS(
        KT,
        basis);

    if (K13) { 
        K13->axpby(KU,1.0,1.0);
        return K13;
    } else {
        return KU; 
    }
}

} // namespace lightspeed
