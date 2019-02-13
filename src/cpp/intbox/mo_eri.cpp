#include <lightspeed/intbox.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/basis.hpp>
#include <lightspeed/pair_list.hpp>
#include <cstring>
#include <cmath>

namespace lightspeed { 

std::shared_ptr<Tensor> IntBox::eriJ(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist12,
    const std::shared_ptr<PairList>& pairlist34,
    const std::shared_ptr<Tensor>& C1,
    const std::shared_ptr<Tensor>& C2,
    const std::shared_ptr<Tensor>& C3,
    const std::shared_ptr<Tensor>& C4,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& I
    )
{
    // Ndim errors
    C1->ndim_error(2);
    C2->ndim_error(2);
    C3->ndim_error(2);
    C4->ndim_error(2);

    // Sizing
    size_t nao1 = pairlist12->basis1()->nao();
    size_t nao2 = pairlist12->basis2()->nao();
    size_t nao3 = pairlist34->basis1()->nao();
    size_t nao4 = pairlist34->basis2()->nao();
    size_t n1 = C1->shape()[1];
    size_t n2 = C2->shape()[1];
    size_t n3 = C3->shape()[1];
    size_t n4 = C4->shape()[1];

    // Shape errors 
    std::vector<size_t> dim1;
    dim1.push_back(nao1);
    dim1.push_back(n1);
    C1->shape_error(dim1);
    std::vector<size_t> dim2;
    dim2.push_back(nao2);
    dim2.push_back(n2);
    C2->shape_error(dim2);
    std::vector<size_t> dim3;
    dim3.push_back(nao3);
    dim3.push_back(n3);
    C3->shape_error(dim3);
    std::vector<size_t> dim4;
    dim4.push_back(nao4);
    dim4.push_back(n4);
    C4->shape_error(dim4);

    // Target
    std::vector<size_t> dim1234;
    dim1234.push_back(n1);
    dim1234.push_back(n2);
    dim1234.push_back(n3);
    dim1234.push_back(n4);
    std::shared_ptr<Tensor> I2 = I;
    if (!I) {
        I2 = std::shared_ptr<Tensor>(new Tensor(dim1234));
    }
    I2->shape_error(dim1234);
    double* Ip = I2->data().data();
    
    // Symmetric in 1 <-> 2 or not?
    bool symm = (C1 == C2); 

    // Loop over ij pairs
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            if (symm && i < j) continue;
            // Build effective density matrix
            std::vector<size_t> dim12;
            dim12.push_back(nao1);
            dim12.push_back(nao2);
            std::shared_ptr<Tensor> D(new Tensor(dim12));
            double* Dp = D->data().data();
            const double* C1p = C1->data().data();
            const double* C2p = C2->data().data();
            //C_DGER(nbf,nbf,1.0,C1p+i,n1,C2p+j,n2,Dp,nao2);
            for (int p = 0; p < nao1; p++) {
                for (int q = 0; q < nao2; q++) {
                    Dp[p * nao2 + q] = C1p[p * n1 + i] * C2p[q * n2 + j];
                }
            }
            // Build J matrix
            std::shared_ptr<Tensor> J = IntBox::coulomb(
                resources,
                ewald,
                pairlist34,
                pairlist12,
                D,
                thresp,
                thredp);
            // Build transformed integrals
            std::vector<std::shared_ptr<Tensor> > Mtask;
            Mtask.push_back(C3);
            Mtask.push_back(J);
            Mtask.push_back(C4);
            std::vector<bool> Ttask;
            Ttask.push_back(true);
            Ttask.push_back(false);
            Ttask.push_back(false);
            std::shared_ptr<Tensor> L = Tensor::chain(Mtask, Ttask);
            const double* Lp = L->data().data();
            // Assign transformed integrals
            ::memcpy(Ip + i*(n2*n3*n4) + j*(n3*n4), Lp, n3*n4*sizeof(double));
            if (symm && i != j) {
                ::memcpy(Ip + j*(n2*n3*n4) + i*(n3*n4), Lp, n3*n4*sizeof(double));
            } 
        }
    }

    // Symmetrize
    if (C1 == C2) {
        for (size_t rs = 0; rs < n3 * n4; rs++) {
            for (size_t p = 0; p < n1; p++) {
                for (size_t q = 0; q < p; q++) {
                    Ip[p*n2*n3*n4 + q*n3*n4 + rs] = 
                    Ip[q*n2*n3*n4 + p*n3*n4 + rs] = 0.5 * (
                    Ip[p*n2*n3*n4 + q*n3*n4 + rs] +
                    Ip[q*n2*n3*n4 + p*n3*n4 + rs]);
                }
            }
        }
    }
    if (C3 == C4) {
        for (size_t pq = 0; pq < n1 * n2; pq++) {
            double* I2p = Ip + pq*n3*n4;
            for (size_t r = 0; r < n3; r++) {
                for (size_t s = 0; s < r; s++) {
                    I2p[r*n4 + s] = 
                    I2p[s*n4 + r] = 0.5 * (
                    I2p[r*n4 + s] +
                    I2p[s*n4 + r]);
                }
            }
        }
    }
    if (C1 == C3 && C2 == C4) {
        for (size_t pq = 0; pq < n1 * n2; pq++) {
            for (size_t rs = 0; rs < pq; rs ++) {
                Ip[pq*n3*n4 + rs] =
                Ip[rs*n3*n4 + pq] = 0.5 * (
                Ip[pq*n3*n4 + rs] +
                Ip[rs*n3*n4 + pq]);
            }
        }
    }

    return I2;
}

std::shared_ptr<Tensor> IntBox::eriGradJ(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& C,
    const std::shared_ptr<Tensor>& D2,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& G)
{
    if (!pairlist->is_symmetric()) throw std::runtime_error("IntBox::eriGradJ: pairlist must be symmetric");
    C->ndim_error(2);
    D2->ndim_error(4);
    size_t nao = pairlist->basis1()->nao();
    size_t norb = C->shape()[1];
    std::vector<size_t> dimC; 
    dimC.push_back(nao);
    dimC.push_back(norb);
    C->shape_error(dimC);
    std::vector<size_t> dimD2;
    dimD2.push_back(norb);
    dimD2.push_back(norb);
    dimD2.push_back(norb);
    dimD2.push_back(norb);
    D2->shape_error(dimD2);
    std::vector<size_t> dimG;
    dimG.push_back(pairlist->basis1()->natom());
    dimG.push_back(3);
    std::shared_ptr<Tensor> G2 = G;
    if (!G) {
        G2 = std::shared_ptr<Tensor>(new Tensor(dimG));
    }
    G2->shape_error(dimG);    

    size_t norb2 = norb * norb;
    std::vector<size_t> dimD4;
    dimD4.push_back(norb2);
    dimD4.push_back(norb2);
    std::vector<size_t> dimd4;
    dimd4.push_back(norb2);
    std::shared_ptr<Tensor> D4(new Tensor(dimD4));
    double* D4p = D4->data().data(); 
    const double* D2p = D2->data().data();
    ::memcpy(D4p,D2p,norb2*norb2*sizeof(double));
    std::shared_ptr<Tensor> U4(new Tensor(dimD4));
    std::shared_ptr<Tensor> d4(new Tensor(dimd4));
    Tensor::syev(D4,U4,d4);

    std::vector<size_t> dimU2;
    dimU2.push_back(norb);
    dimU2.push_back(norb);
    std::shared_ptr<Tensor> U2(new Tensor(dimU2));
    double* U2p = U2->data().data();

    const double* U4p = U4->data().data();
    const double* d4p = d4->data().data();

    std::vector<std::shared_ptr<Tensor> > tensors;
    tensors.push_back(C);
    tensors.push_back(U2);
    tensors.push_back(C);
    std::vector<bool> trans;
    trans.push_back(false);
    trans.push_back(false);
    trans.push_back(true);

    for (size_t i = 0; i < norb2; i++) {
        double dval = d4p[i];
        for (size_t t = 0; t < norb; t++) {
            for (size_t u = 0; u < norb; u++) {
                U2p[t*norb + u] = U4p[(t*norb + u)*norb2 + i];
            }
        }
        U2->scale(sqrt(fabs(dval)));
        std::shared_ptr<Tensor> D = Tensor::chain(tensors,trans);
        std::shared_ptr<Tensor> G3 = IntBox::coulombGrad(
            resources,
            ewald,
            pairlist,
            D,
            D,
            thresp,
            thredp);
        G2->axpby(G3,(0.0 < dval) - (dval < 0.0),1.0);
    }

    return G2;
}

} // namespace lightspeed

