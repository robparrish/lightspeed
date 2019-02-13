#include <lightspeed/lr.hpp>
#include <lightspeed/resource_list.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/ewald.hpp>
#include <lightspeed/basis.hpp>
#include <lightspeed/pair_list.hpp>
#include <lightspeed/math.hpp>
#include "mints/potential4c.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace lightspeed {

std::shared_ptr<Tensor> DF::metric(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<Basis>& auxiliary
    )
{
    std::shared_ptr<Tensor> J(new Tensor({auxiliary->nao(), auxiliary->nao()}));
    double* Jp = J->data().data();

    int nthread = resources->nthread();
    std::vector<std::shared_ptr<PotentialInt4C>> ints;
    for (int thread = 0; thread < nthread; thread++) {
        ints.push_back(std::shared_ptr<PotentialInt4C>(new PotentialInt4C(
            ewald->scales(),
            ewald->omegas(),
            auxiliary->max_L(),
            0,
            auxiliary->max_L(),
            0)));
    }

    size_t naux = auxiliary->nao();
    const std::vector<Shell>& shells = auxiliary->shells();
    Shell zero = Shell::zero();

    #pragma omp parallel for num_threads(nthread) schedule(dynamic)
    for (size_t PQ = 0; PQ < shells.size() * shells.size(); PQ++) {
        size_t P = PQ / shells.size();
        size_t Q = PQ % shells.size();
        if (P > Q) continue;
        const Shell& shellP = shells[P];
        const Shell& shellQ = shells[Q];
        int oP = shellP.aoIdx();
        int oQ = shellQ.aoIdx();
        int nP = shellP.nao();
        int nQ = shellQ.nao();
#ifdef _OPENMP
        int thread = omp_get_thread_num(); 
#else
        int thread = 0;
#endif
        ints[thread]->compute_shell0(shellP, zero, shellQ, zero); 
        const double* data = ints[thread]->data().data();
        for (int p = 0; p < nP; p++) {
            for (int q = 0; q < nQ; q++) {
                Jp[(p + oP) * naux + (q + oQ)] = 
                Jp[(q + oQ) * naux + (p + oP)] = 
                data[p*nQ + q];
            }
        } 
    }

    return J;
}

std::shared_ptr<Tensor> DF::ao_df(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<Basis>& auxiliary,
    const std::shared_ptr<PairList>& pairlist,
    double thre_df_cond
    )
{
    // Inverse square root of metric
    std::shared_ptr<Tensor> J = DF::metric(resources, ewald, auxiliary);
    std::shared_ptr<Tensor> Jm12 = Tensor::power(J, -1.0/2.0, thre_df_cond);
    J.reset();      

    // AO (A|pq) integrals
    bool symm = pairlist->is_symmetric();
    std::shared_ptr<Basis> basisP = pairlist->basis1();   
    std::shared_ptr<Basis> basisQ = pairlist->basis2();   
 
    size_t naoA = auxiliary->nao();
    size_t naoP = basisP->nao();
    size_t naoQ = basisQ->nao();

    std::shared_ptr<Tensor> Apq(new Tensor({naoA, naoP, naoQ}));
    double* Apqp = Apq->data().data();

    int nthread = resources->nthread();
    std::vector<std::shared_ptr<PotentialInt4C>> ints;
    for (int thread = 0; thread < nthread; thread++) {
        ints.push_back(std::shared_ptr<PotentialInt4C>(new PotentialInt4C(
            ewald->scales(),
            ewald->omegas(),
            auxiliary->max_L(),
            0,
            basisP->max_L(),
            basisQ->max_L())));
    }

    const std::vector<Shell>& shellsA = auxiliary->shells();
    const std::vector<Shell>& shellsP = basisP->shells();
    const std::vector<Shell>& shellsQ = basisQ->shells();
    Shell zero = Shell::zero();

    // A small hack - use shell inds in primitive pairs to tell shell pairs
    // TODO: Move this up to pairlist (shell_pairs method)
    std::set<std::pair<int,int>> shell_pairs2;
    for (auto pairlistl : pairlist->pairlists()) {
        for (const Pair& pair : pairlistl.pairs()) {
            shell_pairs2.insert(std::pair<int,int>(pair.prim1().shellIdx(), pair.prim2().shellIdx()));
        } 
    }
    std::vector<std::pair<int, int>> shell_pairs;
    for (auto pair : shell_pairs2) {
        shell_pairs.push_back(pair);
    }
    // printf(std::is_sorted(shell_pairs.begin(), shell_pairs.end()) ? "Sorted\n" : "Not Sorted\n");
    
    #pragma omp parallel for num_threads(nthread) schedule(dynamic)
    for (size_t APQtask = 0; APQtask < auxiliary->nshell() * shell_pairs.size(); APQtask++) {
        int A         = APQtask / shell_pairs.size();
        size_t PQtask = APQtask % shell_pairs.size();
        int P = shell_pairs[PQtask].first;
        int Q = shell_pairs[PQtask].second;
        const Shell& shellA = shellsA[A];
        const Shell& shellP = shellsP[P];
        const Shell& shellQ = shellsQ[Q];
        int oA = shellA.aoIdx();
        int oP = shellP.aoIdx();
        int oQ = shellQ.aoIdx();
        int nA = shellA.nao();
        int nP = shellP.nao();
        int nQ = shellQ.nao();
#ifdef _OPENMP
        int thread = omp_get_thread_num(); 
#else
        int thread = 0;
#endif
        ints[thread]->compute_shell0(shellA, zero, shellP, shellQ); 
        const double* data = ints[thread]->data().data();
        for (int a = 0; a < nA; a++) {
            for (int p = 0; p < nP; p++) {
                for (int q = 0; q < nQ; q++) {
                    double val = (*data++);
                    Apqp[(a + oA) * naoP * naoQ + (p + oP) * naoQ + (q + oQ)] = val;
                    if (symm)
                        Apqp[(a + oA) * naoP * naoQ + (q + oQ) * naoQ + (p + oP)] = val;
                }
            } 
        }
    }

    // Apply fitting metric
    std::shared_ptr<Tensor> Qpq = Tensor::einsum(
        {"P", "A"},
        {"A", "p", "q"},
        {"P", "p", "q"},
        Jm12,
        Apq);

    return Qpq;
}
    
std::vector<std::shared_ptr<Tensor>> DF::mo_df(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<Basis>& auxiliary,
    const std::shared_ptr<PairList>& pairlist,
    const std::vector<std::shared_ptr<Tensor>>& CAs,
    const std::vector<std::shared_ptr<Tensor>>& CBs,
    double thre_df_cond
    )
{
    if (CAs.size() != CBs.size()) throw std::runtime_error("DF::mo_df: CAs.size() != CBs.size()");
    for (auto CA : CAs) CA->ndim_error(2);
    for (auto CB : CBs) CB->ndim_error(2);
    for (auto CA : CAs) if (CA->shape()[0] != pairlist->basis1()->nao()) throw std::runtime_error("DF::mo_df: CA->shape[0] != pairlist->basis1()->nao()");
    for (auto CB : CBs) if (CB->shape()[0] != pairlist->basis1()->nao()) throw std::runtime_error("DF::mo_df: CB->shape[0] != pairlist->basis1()->nao()");

    // TODO: This could be sped up by the usual tricks
    std::shared_ptr<Tensor> Apq = DF::ao_df(resources, ewald, auxiliary, pairlist, thre_df_cond);
    
    std::vector<std::shared_ptr<Tensor>> Aias;
    for (int T = 0; T < CAs.size(); T++) {
        std::shared_ptr<Tensor> Api = Tensor::einsum(
            {"A", "p", "q"},
            {"q", "i"},
            {"A", "p", "i"},
            Apq,
            CAs[T]);
        std::shared_ptr<Tensor> Aia = Tensor::einsum(    
            {"A", "p", "i"},
            {"p", "a"},
            {"A", "i", "a"},
            Api,
            CBs[T]);
        Aias.push_back(Aia);
    }
    return Aias;
}

} // namespace lightspeed
