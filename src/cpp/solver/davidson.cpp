#include <lightspeed/solver.hpp>
#include <lightspeed/tensor.hpp>
#include "../util/string.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <cstdio>

namespace lightspeed {

Davidson::Davidson(
    size_t nstate,     
    size_t nmax,      
    double convergence,
    double canonical_tolerance,
    const std::vector<std::shared_ptr<Storage> >& moms):
    nstate_(nstate),
    nmax_(nmax),
    convergence_(convergence),
    canonical_tolerance_(canonical_tolerance)
{
    if(nmax_ < nstate_)
        throw std::runtime_error("Davidson:: nmax >> nstate for good performance.");
    
    std::vector<size_t> dimMax;
    dimMax.push_back(nmax_);
    dimMax.push_back(nmax_);
    
    A_ = std::shared_ptr<Tensor>(new Tensor(dimMax,"A"));
    S_ = std::shared_ptr<Tensor>(new Tensor(dimMax,"S"));

    if (moms.size() > 0){
        if (moms.size() != nstate){
            throw std::runtime_error("Davidson:: Maximum overlap requested, but number of guess vectors does not match nstate.");
        }
        moms_.insert(  moms_.end(),  moms.begin(),  moms.end() );
    }

}

std::string Davidson::string() const 
{
    std::string str = "";
    str += sprintf2("Davidson:\n");
    str += sprintf2("  Nstate        = %11zu\n", nstate());
    str += sprintf2("  Nmax          = %11zu\n", nmax());
    str += sprintf2("  Convergence   = %11.3E\n", convergence());
    if (moms_.size() > 0) {
    str += sprintf2("  Maximum overlap method requested.\n");
    }
    return str;
}
bool Davidson::is_converged() const
{
    return max_rnorm() <= convergence();
}
double Davidson::max_rnorm() const 
{
    return *std::max_element(rnorms_.begin(),rnorms_.end());
}
void Davidson::add_vectors(
    const std::vector<std::shared_ptr<Storage> >& b,
    const std::vector<std::shared_ptr<Storage> >& Ab,
    const std::vector<std::shared_ptr<Storage> >& Sb)
{
    // Throw if b, Ab and Sb does not have the same size
    if(b.size() != Ab.size() || b.size() != Sb.size())
        throw std::runtime_error("Davidson:: Sizes of b, Ab and Sb are not compatible");

    // Throw if ntot exceeds  the maximum subspace size
    size_t nold =  Abs_.size();
    size_t nnew = Ab.size();
    size_t ntot = nold + nnew;
    if (ntot > nmax()) {
        throw std::runtime_error("Davidson::add_vectors: number of new vectors exceeds subspace nmax");
    }

    // Copying data to private containers
    bs_.insert(  bs_.end(),  b.begin(),  b.end() );
    Abs_.insert( Abs_.end(), Ab.begin(), Ab.end() );
    Sbs_.insert( Sbs_.end(), Sb.begin(), Sb.end() );
    
    // Determine the diagonal elements of S to select vectors for numerical accuracy
    double* Sp = S_->data().data();
    double S_diag[ntot];
    for (int i = 0; i < nold; i++) {
        S_diag[i] = Sp[i * nmax() + i];
    }
    for (int i = nold; i < ntot; i++) {
        S_diag[i] = Storage::dot(bs_[i],bs_[i]);
    }

    // Computing the sub-space overlap matrix S_{IJ}=<b_I|b_J>
    for (int i = 0; i < nold; i++) {
        for (int j = nold; j < ntot; j++) {
            Sp[i * nmax() + j] =  
            Sp[j * nmax() + i] =  
            (S_diag[j] > S_diag[i] ? 
            Storage::dot(bs_ [i],Sbs_[j]) :
            Storage::dot(Sbs_[i],bs_ [j]));
        }
    }
    for (int i = nold; i < ntot; i++) {
        for (int j = nold; j < ntot; j++) {
            if (i > j) continue;
            Sp[i * nmax() + j] =  
            Sp[j * nmax() + i] =  
            (S_diag[j] > S_diag[i] ? 
            Storage::dot(bs_ [i],Sbs_[j]) :
            Storage::dot(Sbs_[i],bs_ [j]));
        }
    }
    
    // Computing the sub-space "stiffness matrix" A = <\sigma_I|b_J> 
    double* Ap = A_->data().data();
    for (int i = 0; i < nold; i++) {
        for (int j = nold; j < ntot; j++) {
            Ap[i * nmax() + j] =  
            Ap[j * nmax() + i] =  
            (Sp[j * nmax() + j] > Sp[i * nmax() + i] ? 
            Storage::dot(bs_ [i],Abs_[j]) :
            Storage::dot(Abs_[i],bs_ [j]));
        }
    }
    for (int i = nold; i < ntot; i++) {
        for (int j = nold; j < ntot; j++) {
            if (i > j) continue;
            Ap[i * nmax() + j] =  
            Ap[j * nmax() + i] =  
            (Sp[j * nmax() + j] > Sp[i * nmax() + i] ? 
            Storage::dot(bs_ [i],Abs_[j]) :
            Storage::dot(Abs_[i],bs_ [j]));
        }
    }

    // Working copies of A/S
    std::vector<size_t> dim1;
    dim1.push_back(ntot);
    dim1.push_back(ntot);
    std::shared_ptr<Tensor> A2(new Tensor(dim1,"A2")); 
    std::shared_ptr<Tensor> S2(new Tensor(dim1,"S2")); 
    double* A2p = A2->data().data();
    double* S2p = S2->data().data();
    for (int i = 0; i < ntot; i++) {
        for (int j = 0; j < ntot; j++) {
            A2p[i*ntot + j] = Ap[i*nmax() + j];
            S2p[i*ntot + j] = Sp[i*nmax() + j];
        }   
    }
        
    // Solving the sub-space Generalized Symmetric Definite Eigenproblem:
    // A_IJ C_JK = S_IJ C_JK E_K : C_IK S_IJ C_JL = I_KL
    std::vector<size_t> dim0;
    dim0.push_back(ntot);
    std::shared_ptr<Tensor> C2(new Tensor(dim1,"C2")); 
    std::shared_ptr<Tensor> a2(new Tensor(dim0,"a2")); 
    // Using canonical orthogonalization in place of Cholesky decomposition
    Tensor::generalized_syev2(A2,S2,C2,a2,canonical_tolerance_); 
    double* C2p = C2->data().data();
    double* a2p = a2->data().data();
            
    // Generate maximum-overlap metric (p)
    std::shared_ptr<Tensor> C2i;
    std::shared_ptr<Tensor> a2i;
    if (moms_.size() > 0){
        C2i = C2->clone();
        a2i = a2->clone();
        double* C2ip = C2i->data().data();
        double* a2ip = a2i->data().data();
        std::vector<int> save_index;
        for (int i = 0; i < moms_.size(); i++) {
            std::vector<std::pair<double,int> > order;
            for (int j = 0; j < ntot; j++) {
                std::shared_ptr<Storage> x = Storage::zeros_like(bs_[0]);
                for (int k = 0; k < ntot; k++) {
                    Storage::axpby(bs_[k],x,C2p[k*ntot+j],1.0);
                }
                double p = pow(Storage::dot(moms_[i],x),2);
                order.push_back(std::pair<double,int>(p,j));
            }
            std::sort(order.rbegin(), order.rend());
            save_index.push_back(order[0].second);
            printf("Maximum-overlap metric for state: %d\n", i);
            for (int j = 0; j < ntot; j++) {
                printf("%f %d\n", order[j].first, order[j].second);
            }
            printf("\n");
        }
        for (int i = 0; i < save_index.size(); i++) {
            printf("Interchanging orbital %d with %d\n",i, save_index[i]);
            for (int j = 0; j < ntot; j++) {
                C2ip[j*ntot+i] = C2p[j*ntot+save_index[i]];
                C2ip[j*ntot+save_index[i]] = C2p[j*ntot+i];
                a2ip[i] = a2p[save_index[i]];
                a2ip[save_index[i]] = a2p[i];
            }
        }
        C2p = C2i->data().data();
        a2p = a2i->data().data();
    }

    // Stash the eigenvalues
    evals_.clear();
    for (int i = 0; i < std::min(nstate_, ntot); i++) {
        evals_.push_back(a2p[i]);
    }
    
    // Computing eigenvectors as X_I = \sum_J C2_{JI}b_J
    evecs_.clear();
    for (int i = 0; i < std::min(nstate_, ntot); i++) {
        std::shared_ptr<Storage> x = Storage::zeros_like(bs_[0]);
        double norm = 0.0;
        for (int j = 0; j < ntot; j++) {
            norm += pow(C2p[j*ntot+i],2);
        }
        if(norm==0.0) throw std::runtime_error("Davidson::add_vectors: Eigen-structure is defficient.");
        for (int j = 0; j < ntot; j++) {
            Storage::axpby(bs_[j],x,C2p[j*ntot+i],1.0);
        }
        evecs_.push_back(x);
    } 

    // Compute residuals/residual norms
    // r_I = \sum_J C2_{JI}*Ab_J -E_I*x_I
    rs_.clear();
    rnorms_.clear();
    for (int i = 0; i < std::min(nstate_, ntot); i++) {
        std::shared_ptr<Storage> r = Storage::zeros_like(bs_[0]);
        Storage::axpby(evecs_[i],r,-evals_[i],0.0);
        for (int j = 0; j < ntot; j++) {
            Storage::axpby(Abs_[j],r,C2p[j*ntot+i],1.0);
        }
        rnorms_.push_back(sqrt(Storage::dot(r,r)));
        rs_.push_back(r);
    } 

    // Update the subspace to be preconditioned externally at will
    gs_.clear();
    hs_.clear();
    for (int i = 0; i < std::min(nstate_, ntot); i++) {
        if (rnorms_[i] <= convergence()) continue;
        gs_.push_back(rs_[i]); 
        hs_.push_back(evals_[i]);
    }
}
void Davidson::add_preconditioned(
    const std::vector<std::shared_ptr<Storage> >& ds)
{
    // Decide if subspace collapse or proceeding as normal
    if (bs_.size() + ds.size() > nmax()) {
        A_->zero();
        S_->zero();
        bs_.clear();
        Abs_.clear(); 
        Sbs_.clear(); 
        cs_ = evecs_;
        return;
    } else {
        cs_ = ds;
    }
}

} // namespace lightspeed
