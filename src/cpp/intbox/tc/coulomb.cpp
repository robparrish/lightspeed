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
        
std::shared_ptr<Tensor> IntBox::coulombTC(
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
    if (pairlist12 != pairlist34) throw std::runtime_error("IntBox::coulombTC: pairlist12 != pairlist34");
    if (!pairlist12->is_symmetric()) throw std::runtime_error("IntBox::coulombTC: pairlist12 is not symmetric");

    std::shared_ptr<Basis> basis = pairlist12->basis1();

    std::shared_ptr<Tensor> DT = TCTransform::LStoTC(
        D34,
        basis);
    std::shared_ptr<Tensor> JT(new Tensor({basis->ncart(),basis->ncart()}));

    auto ewald2 = TCTransform::ewaldTC(ewald);

    std::shared_ptr<TCIntBox> tc = TCIntBox::instance(resources);
    tc->set_basis(
        pairlist12->basis1(),
        pairlist12->thre());
    for (auto task : ewald2) {
        tc->computeJFockGen(
            thresp,
            thredp,
            std::get<0>(task),
            std::get<1>(task),
            std::get<2>(task),
            DT->data().data(),
            JT->data().data());
    }
    tc->clear_basis();
    
    std::shared_ptr<Tensor> JU = TCTransform::TCtoLS(
        JT,
        basis);

    if (J12) { 
        J12->axpby(JU,1.0,1.0);
        return J12;
    } else {
        return JU; 
    }
}

} // namespace lightspeed
