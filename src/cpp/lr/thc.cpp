#include <lightspeed/lr.hpp>
#include <lightspeed/resource_list.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/becke.hpp>
#include <lightspeed/gridbox.hpp>
#include <lightspeed/math.hpp>

namespace lightspeed {

std::shared_ptr<Tensor> THC::mo_thc_X(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Basis>& basis,
    const std::shared_ptr<BeckeGrid>& grid,
    const std::shared_ptr<Tensor>& C,
    double thre_grid,
    double weight_pow
    )
{
    std::shared_ptr<Tensor> XPa = GridBox::orbitals(
        resources,
        basis,
        grid->xyz(),
        C,
        thre_grid);  
    double* XPap = XPa->data().data();

    std::shared_ptr<Tensor> xyzw = grid->xyzw();
    const double* xyzwp = xyzw->data().data();

    size_t ngrid = XPa->shape()[0];
    size_t nmo = XPa->shape()[1];

    for (size_t P = 0; P < ngrid; P++) {
        double w = xyzwp[4*P+3];
        if (w < 0.0) throw std::runtime_error("THC::mo_thc_X: negative weight");
        double wpow = pow(w, weight_pow);
        for (size_t a = 0; a < nmo; a++) {
            (*XPap++) *= wpow;
        }
    }
    
    return XPa;
}
std::vector<std::shared_ptr<Tensor>> THC::mo_thc_V(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<Basis>& auxiliary,
    const std::shared_ptr<PairList>& pairlist,
    const std::vector<std::shared_ptr<Tensor>>& CAs,
    const std::vector<std::shared_ptr<Tensor>>& CBs,
    const std::vector<std::shared_ptr<Tensor>>& XAs,
    const std::vector<std::shared_ptr<Tensor>>& XBs,
    double thre_df_cond,
    double thre_thc_cond
    )
{
    // DF integrals including metric
    std::vector<std::shared_ptr<Tensor>> Bs = DF::mo_df(
        resources,
        ewald,
        auxiliary,
        pairlist,
        CAs,
        CBs,
        thre_df_cond);

    if (XAs.size() != CAs.size()) throw std::runtime_error("THC::mo_thc_V: XAs.size() != CAs.size()");
    if (XBs.size() != CBs.size()) throw std::runtime_error("THC::mo_thc_V: XBs.size() != CBs.size()");

    std::vector<std::shared_ptr<Tensor>> Vs;
    for (size_t ind = 0; ind < Bs.size(); ind++) {
        std::shared_ptr<Tensor> B = Bs[ind];
        std::shared_ptr<Tensor> XA = XAs[ind];
        std::shared_ptr<Tensor> XB = XBs[ind];

        XA->ndim_error(2);
        XB->ndim_error(2);
        XA->shape_error({XB->shape()[0],XA->shape()[1]});
        XA->shape_error({XA->shape()[0],B->shape()[1]});
        XB->shape_error({XB->shape()[0],B->shape()[2]});
        
        // THC metric
        std::shared_ptr<Tensor> SA = Tensor::chain({XA,XA},{false,true});
        std::shared_ptr<Tensor> SB = Tensor::chain({XB,XB},{false,true});
        std::shared_ptr<Tensor> S = Tensor::einsum({"P","Q"}, {"P", "Q"}, {"P", "Q"}, SA, SB);
        SA.reset();
        SB.reset();
        std::shared_ptr<Tensor> Sinv = Tensor::power(S, -1.0, thre_thc_cond);
        S.reset();

        // E matrix
        size_t nP = XA->shape()[0];
        size_t na = XA->shape()[1];
        size_t nb = XB->shape()[1];
        size_t nA = B->shape()[0];
        
        double* Bp = B->data().data();
        double* XAp = XA->data().data();
        double* XBp = XB->data().data();

        std::shared_ptr<Tensor> T1(new Tensor({nP,na}));
        double* T1p = T1->data().data();

        std::shared_ptr<Tensor> E(new Tensor({nA,nP}));
        double* Ep = E->data().data();

        for (size_t A = 0; A < nA; A++) {
            C_DGEMM('N','T',nP,na,nb,1.0,XBp,nb,Bp+A*na*nb,nb,0.0,T1p,na);
            for (size_t P = 0; P < nP; P++) {
                double val = 0.0;
                for (size_t a = 0; a < na; a++) val += XAp[P*na+a] * T1p[P*na+a];
                Ep[A*nP+P] = val;
            }
        }

        // V matrix
        std::shared_ptr<Tensor> V = Tensor::chain({E,Sinv},{false, false});
        Vs.push_back(V);
    }

    return Vs;
}

} // namespace lightspeed
