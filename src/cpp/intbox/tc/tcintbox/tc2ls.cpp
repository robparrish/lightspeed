#include "tc2ls.hpp"
#include <lightspeed/tensor.hpp>
#include <lightspeed/basis.hpp>
#include <lightspeed/am.hpp>
#include <lightspeed/ewald.hpp>
#include "../../../core/pure_transform_util.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lightspeed {

std::vector<std::tuple<double,double,double>> TCTransform::ewaldTC(
    const std::shared_ptr<Ewald>& ewald)
{
    std::vector<std::tuple<double,double,double>> ewald2;
    if (ewald->is_coulomb()) {
        ewald2.push_back(std::tuple<double,double,double>(ewald->scales()[0],0.0,0.0));
    } else if (ewald->is_lr()) {
        ewald2.push_back(std::tuple<double,double,double>(0.0,ewald->lr_scale(),ewald->lr_omega()));
    } else if (ewald->is_sr()) {
        ewald2.push_back(std::tuple<double,double,double>(ewald->sr_scale(),-ewald->sr_scale(),ewald->sr_omega()));
    } else {
        for (size_t ind = 0; ind < ewald->scales().size(); ind++) {
            double scale = ewald->scales()[ind];
            double omega = ewald->omegas()[ind];
            double scalfr = (omega == -1.0 ? scale : 0.0);
            double scallr = (omega != -1.0 ? scale : 0.0);
            double omega2 = (omega == -1.0 ? 0.0 : omega);
            ewald2.push_back(std::tuple<double,double,double>(scalfr,scallr,omega2));
        }
    }
    return ewald2;
}

std::shared_ptr<Tensor> TCTransform::LStoTC(
    const std::shared_ptr<Tensor>& T1,
    const std::shared_ptr<Basis>& primary)
{
    std::shared_ptr<Tensor> T2(new Tensor({primary->ncart(), primary->ncart()}));

    double* T1p = T1->data().data();
    double* T2p = T2->data().data();

    // => Scratch Arrays <= //

    int nt = 1;
    #ifdef _OPENMP
        nt = omp_get_max_threads();
    #endif

    std::vector<std::vector<double>> scratch1(nt);
    std::vector<std::vector<double>> scratch2(nt);
    for (int t = 0; t < nt; t++) {
        scratch1[t].resize(primary->max_ncart()*primary->max_ncart());
        scratch2[t].resize(primary->max_ncart()*primary->max_ncart());
    }

    // => Master Loop <= //
    
    std::vector<AngularMomentum> am_info = AngularMomentum::build(primary->max_L());
    size_t nPQ = primary->nshell() * primary->nshell();
    int nbf   = primary->nao();
    int ncart = primary->ncart();
    #pragma omp parallel for schedule(dynamic)
    for (size_t PQ = 0; PQ < nPQ; PQ++) {
        size_t P = PQ / primary->nshell();
        size_t Q = PQ % primary->nshell();
        const Shell& Pshell = primary->shells()[P];
        const Shell& Qshell = primary->shells()[Q];
        int  nP = Pshell.nao(); 
        int  nQ = Qshell.nao(); 
        int  oP = Pshell.aoIdx();
        int  oQ = Qshell.aoIdx();
        int  cP = Pshell.ncart(); 
        int  cQ = Qshell.ncart(); 
        int  aP = Pshell.cartIdx();
        int  aQ = Qshell.cartIdx();
        int  lP = Pshell.L();
        int  lQ = Qshell.L();
        bool sP = Pshell.is_pure();
        bool sQ = Qshell.is_pure();
        int t = 0;
        #ifdef _OPENMP
            t = omp_get_thread_num();
        #endif
        double* S1p = scratch1[t].data();
        double* S2p = scratch2[t].data();

        for (int p = 0; p < nP; p++) {
            for (int q = 0; q < nQ; q++) {
                S1p[p*nQ + q] = T1p[(p + oP)*nbf + (q + oQ)];
            }
        }

        PureTransformUtil::pureToCart2(am_info,lP,lQ,sP,sQ,S1p,S2p);
        TCTransform::LStoTC(lP,lQ,S1p,S2p);

        for (int p = 0; p < cP; p++) {
            for (int q = 0; q < cQ; q++) {
                T2p[(p + aP)*ncart + (q + aQ)] = S1p[p*cQ + q];
            }
        }

    }

    return T2;
}
std::shared_ptr<Tensor> TCTransform::TCtoLS(
    const std::shared_ptr<Tensor>& T2,
    const std::shared_ptr<Basis>& primary)
{
    std::shared_ptr<Tensor> T1(new Tensor({primary->nao(), primary->nao()}));

    double* T1p = T1->data().data();
    double* T2p = T2->data().data();

    // => Scratch Arrays <= //

    int nt = 1;
    #ifdef _OPENMP
        nt = omp_get_max_threads();
    #endif

    std::vector<std::vector<double>> scratch1(nt);
    std::vector<std::vector<double>> scratch2(nt);
    for (int t = 0; t < nt; t++) {
        scratch1[t].resize(primary->max_ncart()*primary->max_ncart());
        scratch2[t].resize(primary->max_ncart()*primary->max_ncart());
    }

    // => Master Loop <= //
    
    std::vector<AngularMomentum> am_info = AngularMomentum::build(primary->max_L());
    size_t nPQ = primary->nshell() * primary->nshell();
    int nbf   = primary->nao();
    int ncart = primary->ncart();
    #pragma omp parallel for schedule(dynamic)
    for (size_t PQ = 0; PQ < nPQ; PQ++) {
        size_t P = PQ / primary->nshell();
        size_t Q = PQ % primary->nshell();
        const Shell& Pshell = primary->shells()[P];
        const Shell& Qshell = primary->shells()[Q];
        int  nP = Pshell.nao(); 
        int  nQ = Qshell.nao(); 
        int  oP = Pshell.aoIdx();
        int  oQ = Qshell.aoIdx();
        int  cP = Pshell.ncart(); 
        int  cQ = Qshell.ncart(); 
        int  aP = Pshell.cartIdx();
        int  aQ = Qshell.cartIdx();
        int  lP = Pshell.L();
        int  lQ = Qshell.L();
        bool sP = Pshell.is_pure();
        bool sQ = Qshell.is_pure();
        int t = 0;
        #ifdef _OPENMP
            t = omp_get_thread_num();
        #endif
        double* S1p = scratch1[t].data();
        double* S2p = scratch2[t].data();

        for (int p = 0; p < cP; p++) {
            for (int q = 0; q < cQ; q++) {
                S1p[p*cQ + q] = T2p[(p + aP)*ncart + (q + aQ)];
            }
        }

        TCTransform::TCtoLS(lP,lQ,S1p,S2p);
        PureTransformUtil::cartToPure2(am_info,lP,lQ,sP,sQ,S1p,S2p);

        for (int p = 0; p < nP; p++) {
            for (int q = 0; q < nQ; q++) {
                T1p[(p + oP)*nbf + (q + oQ)] = S1p[p*nQ + q];
            }
        }
    
    }

    return T1;
}

} // namespace lightspeed
