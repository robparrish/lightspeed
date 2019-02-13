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

std::shared_ptr<Tensor> IntBox::coulombGradTC(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist, // symm
    const std::shared_ptr<Tensor>& D12,
    const std::shared_ptr<Tensor>& D34,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& G
    )
{
    if (!pairlist->is_symmetric()) throw std::runtime_error("IntBox::coulombGradTC: pairlist is not symmetric");

    std::shared_ptr<Basis> basis = pairlist->basis1();
    std::shared_ptr<Tensor> GT(new Tensor({basis->natom(),3}));

    if (D12 == D34) {
        std::shared_ptr<Tensor> D34T = TCTransform::LStoTC(
            D34,
            basis);

        auto ewald2 = TCTransform::ewaldTC(ewald);

        std::shared_ptr<TCIntBox> tc = TCIntBox::instance(resources);
        tc->set_basis(
            pairlist->basis1(),
            pairlist->thre());
        for (auto task : ewald2) {
            tc->computeJGradGen(
                thresp,
                thredp,
                std::get<0>(task),
                std::get<1>(task),
                std::get<2>(task),
                D34T->data().data(),
                D34T->data().data(),
                GT->data().data());
        }
        tc->clear_basis();
        GT->scale(2.0); // To account for TC scale
    } else {
        // Hack: TODO: IntBox is incorrect when given two different D matrices
        std::shared_ptr<Tensor> DAT = TCTransform::LStoTC(
            D12,
            basis);
        std::shared_ptr<Tensor> DBT = TCTransform::LStoTC(
            D34,
            basis);
        double* DATp = DAT->data().data();
        double* DBTp = DBT->data().data();
        size_t npair = DAT->size();
        for (size_t pq = 0; pq < npair; pq++) {
            double X = DATp[pq];
            double Y = DBTp[pq];
            double A = 0.5 * (X + Y);
            double B = 0.5 * (X - Y);
            DATp[pq] = A;
            DBTp[pq] = B;
        }
        std::shared_ptr<Tensor> GT2(new Tensor({basis->natom(),3}));

        auto ewald2 = TCTransform::ewaldTC(ewald);

        std::shared_ptr<TCIntBox> tc = TCIntBox::instance(resources);
        tc->set_basis(
            pairlist->basis1(),
            pairlist->thre());
        for (auto task : ewald2) {
            tc->computeJGradGen(
                thresp,
                thredp,
                std::get<0>(task),
                std::get<1>(task),
                std::get<2>(task),
                DAT->data().data(),
                DAT->data().data(),
                GT->data().data());
            tc->computeJGradGen(
                thresp,
                thredp,
                std::get<0>(task),
                std::get<1>(task),
                std::get<2>(task),
                DBT->data().data(),
                DBT->data().data(),
                GT2->data().data());
        }
        tc->clear_basis();
        GT->axpby(GT2,-1.0,1.0);
        GT->scale(2.0); // To account for TC scale
    }


    if (G) { 
        G->axpby(GT,1.0,1.0);
        return G;
    } else {
        return GT; 
    }
}

} // namespace lightspeed
