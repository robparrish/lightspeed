#include "tcintbox/tcintbox.hpp"
#include "tcintbox/tc2ls.hpp"
#include <lightspeed/intbox.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/basis.hpp>
#include <lightspeed/resource_list.hpp>
#include <lightspeed/pair_list.hpp>
#include <lightspeed/ewald.hpp>
#include <stdexcept>

namespace lightspeed { 

std::shared_ptr<Tensor> IntBox::exchangeGradTC(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D13,
    const std::shared_ptr<Tensor>& D24,
    bool D13symm,
    bool D24symm,
    bool Dsame,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& G
    )
{
    if (!pairlist->is_symmetric()) throw std::runtime_error("IntBox::coulombGradTC: pairlist is not symmetric");

    std::shared_ptr<Basis> basis = pairlist->basis1();

    std::shared_ptr<Tensor> D24T = TCTransform::LStoTC(
        D24,
        basis);
    std::shared_ptr<Tensor> D13T = (D13 == D24 ? D24T : TCTransform::LStoTC(
        D13,
        basis));
    std::shared_ptr<Tensor> GT(new Tensor({basis->natom(),3}));

    auto ewald2 = TCTransform::ewaldTC(ewald);

    std::shared_ptr<TCIntBox> tc = TCIntBox::instance(resources);
    tc->set_basis(
        pairlist->basis1(),
        pairlist->thre());
    for (auto task : ewald2) {
        if (Dsame && D13symm) {
            tc->computeKGradSym(
                thresp,
                thredp,
                std::get<0>(task),
                std::get<1>(task),
                std::get<2>(task),
                D13T->data().data(),
                GT->data().data());
        } else if (Dsame) {
            tc->computeKGradGen(
                thresp,
                thredp,
                std::get<0>(task),
                std::get<1>(task),
                std::get<2>(task),
                D13T->data().data(),
                GT->data().data());
        } else {
            tc->computeKGradGenGen(
                thresp,
                thredp,
                std::get<0>(task),
                std::get<1>(task),
                std::get<2>(task),
                D13T->data().data(),
                D24T->data().data(),
                GT->data().data());
        }
    }
    tc->clear_basis();
    GT->scale(-4.0); // To account for TC scale

    if (G) { 
        G->axpby(GT,1.0,1.0);
        return G;
    } else {
        return GT; 
    }
}
    
} // namespace lightspeed
