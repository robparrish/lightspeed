#include <lightspeed/tensor.hpp>
#include <lightspeed/math.hpp>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <limits>
#include <cstdio>

namespace lightspeed { 

std::shared_ptr<Tensor> Tensor::potrf(
    const std::shared_ptr<Tensor>& A,
    bool lower)
{
    A->square_error();
    size_t n = A->shape()[0];
    std::shared_ptr<Tensor> A2 = A->clone();
    
    double* A2p = A2->data().data();
    
    char trans = (lower ? 'U' : 'L');

    int info = C_DPOTRF(trans, n, A2p, n);

    if (info) throw std::runtime_error("C_DPOTRF failed");
    
    if (lower) {
        for (size_t i = 0; i < n; i++) {
            for (size_t j = i+1; j < n; j++) {
                A2p[i * n + j] = 0.0;
            }
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < i; j++) {
                A2p[i * n + j] = 0.0;
            }
        }
    }
    
    return A2;
}
std::shared_ptr<Tensor> Tensor::trtri(
    const std::shared_ptr<Tensor>& A,
    bool lower)
{
    A->square_error();
    size_t n = A->shape()[0];
    std::shared_ptr<Tensor> A2 = A->clone();
    
    double* A2p = A2->data().data();

    char trans = (lower ? 'U' : 'L');

    int info = C_DTRTRI(trans, 'N',  n, A2p, n);

    if (info) throw std::runtime_error("C_DTRTRI failed");
    
    if (lower) {
        for (size_t i = 0; i < n; i++) {
            for (size_t j = i+1; j < n; j++) {
                A2p[i * n + j] = 0.0;
            }
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < i; j++) {
                A2p[i * n + j] = 0.0;
            }
        }
    }
    
    return A2;
}
std::shared_ptr<Tensor> Tensor::gesv(
    const std::shared_ptr<Tensor>& A,
    std::shared_ptr<Tensor>& f)
{
    A->square_error();
    
    size_t n = A->shape()[0];
    size_t m = f->shape()[0];

    std::vector<size_t> mndim;
    mndim.push_back(m);
    mndim.push_back(n);
    f->shape_error(mndim);

    std::shared_ptr<Tensor> A2 = A->clone();
    std::shared_ptr<Tensor> f2 = f->clone();

    std::vector<int> ipiv(n);

    int info = C_DGESV(n,m,A2->data().data(),n,ipiv.data(),f2->data().data(),n);
    if (info) throw std::runtime_error("C_DGESV failed.");
    return f2;
}
void Tensor::syev(
    const std::shared_ptr<Tensor>& A,
    std::shared_ptr<Tensor>& U,
    std::shared_ptr<Tensor>& a,
    bool ascending,
    bool syevd)
{
    A->square_error();

    size_t n = A->shape()[0];

    std::vector<size_t> ndim;
    ndim.push_back(n);

    a->shape_error(ndim);

    U->shape_error(A->shape());

    std::shared_ptr<Tensor> A2 = A->clone();
    std::shared_ptr<Tensor> a2 = a->clone();
    
    if (syevd) {
        int info;
        double nwork;
        int niwork;
        info = C_DSYEVD('V','U',n,A2->data().data(),n,a2->data().data(),&nwork,-1,&niwork,-1);  
        if ((size_t) nwork >= std::numeric_limits<int>::max()) {
            throw std::runtime_error("nwork is too big.");
        }
        std::vector<double> work((size_t)nwork);
        std::vector<int> iwork(niwork);
        info = C_DSYEVD('V','U',n,A2->data().data(),n,a2->data().data(),work.data(),(size_t)nwork,iwork.data(),niwork);  
        if (info != 0) {
            throw std::runtime_error("DSYEVD failed to converge.");
        }
    } else {
        int info;
        double nwork;
        info = C_DSYEV('V','U',n,A2->data().data(),n,a2->data().data(),&nwork,-1);
        std::vector<double> work((size_t)nwork);
        info = C_DSYEV('V','U',n,A2->data().data(),n,a2->data().data(),work.data(),(size_t)nwork);
        if (info != 0) {
            throw std::runtime_error("DSYEV failed to converge.");
        }
    }
 
    double* ap = a->data().data();
    double* Up = U->data().data();
    double* U2p = A2->data().data();
    double* a2p = a2->data().data();
    if (ascending) {
        for (size_t i = 0; i < n; i++) {
            ap[i] = a2p[i];
            for (size_t j = 0; j < n; j++) {
                Up[j * n + i] = U2p[i * n + j];
            }
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            ap[i] = a2p[n - i - 1];
            for (size_t j = 0; j < n; j++) {
                Up[j * n + i] = U2p[(n - i - 1) * n + j];
            }
        }
    } 
}
void Tensor::generalized_syev(
    const std::shared_ptr<Tensor>& A,
    const std::shared_ptr<Tensor>& S,
    std::shared_ptr<Tensor>& U,
    std::shared_ptr<Tensor>& a,
    bool ascending,
    bool syevd)
{
    A->square_error();
    S->shape_error(A->shape());
    size_t n = A->shape()[0];
    std::vector<size_t> ndim;
    ndim.push_back(n);
    a->shape_error(ndim);
    U->shape_error(A->shape());

    // Balancing
    std::shared_ptr<Tensor> A2 = A->clone();
    std::shared_ptr<Tensor> S2 = S->clone();
    double* A2p = A2->data().data();  
    double* S2p = S2->data().data();  
    std::shared_ptr<Tensor> Sb(new Tensor(ndim));
    double* Sbp = Sb->data().data();
    for (int i = 0; i < n; i++) {
        if (S2p[i*n+i] <= 0.0) throw std::runtime_error("Zero or negative diagonal in S");
        Sbp[i] = 1. / sqrt(S2p[i*n+i]);
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A2p[i*n + j] *= Sbp[i] * Sbp[j];
            S2p[i*n + j] *= Sbp[i] * Sbp[j];
        }
    }
    
    // Generalized eigenproblem
    std::shared_ptr<Tensor> X = Tensor::cholesky_orthogonalize(S2);
    std::shared_ptr<Tensor> A3(new Tensor(A->shape()));
    std::vector<std::shared_ptr<Tensor> > chain1;
    chain1.push_back(X);
    chain1.push_back(A2);
    chain1.push_back(X);
    std::vector<bool> trans1;
    trans1.push_back(true);
    trans1.push_back(false);
    trans1.push_back(true);
    Tensor::chain(chain1,trans1,A3);
    //Tensor::chain({X,A2,X},{true,false,false},A3); C++11
    std::shared_ptr<Tensor> U3(new Tensor(A->shape()));
    Tensor::syev(A3,U3,a,ascending,syevd); 

    //Transforming the eignevectors back to the original basis U = X U3 
    std::vector<std::shared_ptr<Tensor> > chain2;
    chain2.push_back(X);
    chain2.push_back(U3);
    std::vector<bool> trans2;
    trans2.push_back(false);
    trans2.push_back(false);
    Tensor::chain(chain2,trans2,U);
    //Tensor::chain({X,U3},{false,false},U); C++11

    // Balancing
    double* Up = U->data().data();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Up[i*n+j] *= Sbp[i];
        }
    } 
}
void Tensor::generalized_syev2(
    const std::shared_ptr<Tensor>& A,
    const std::shared_ptr<Tensor>& S,
    std::shared_ptr<Tensor>& U,
    std::shared_ptr<Tensor>& a,
    bool ascending,
    bool syevd,
    double tolerance)
{
    A->square_error();
    S->shape_error(A->shape());
    size_t n = A->shape()[0];
    std::vector<size_t> ndim;
    ndim.push_back(n);
    a->shape_error(ndim);
    U->shape_error(A->shape());

    // Balancing
    std::shared_ptr<Tensor> A2 = A->clone();
    std::shared_ptr<Tensor> S2 = S->clone();
    double* A2p = A2->data().data();  
    double* S2p = S2->data().data();  
    std::shared_ptr<Tensor> Sb(new Tensor(ndim));
    double* Sbp = Sb->data().data();
    for (int i = 0; i < n; i++) {
        if (S2p[i*n+i] <= 0.0) throw std::runtime_error("Zero or negative diagonal in S");
        Sbp[i] = 1. / sqrt(S2p[i*n+i]);
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A2p[i*n + j] *= Sbp[i] * Sbp[j];
            S2p[i*n + j] *= Sbp[i] * Sbp[j];
        }
    }

    // Generalized eigenproblem
    std::shared_ptr<Tensor> X = Tensor::canonical_orthogonalize(S2,tolerance);
    size_t m = X->shape()[1];

    // Performing the similarity transformation A3 = X' A2 X
    std::vector<std::shared_ptr<Tensor> > chain1;
    chain1.push_back(X);
    chain1.push_back(A2);
    chain1.push_back(X);
    std::vector<bool> trans1;
    trans1.push_back(true);
    trans1.push_back(false);
    trans1.push_back(false);
    std::shared_ptr<Tensor> A3 = Tensor::chain(chain1,trans1);

    // Solving the eigenproblem for A3
    std::shared_ptr<Tensor> U3(new Tensor(A3->shape()));
    std::vector<size_t> dim3;
    dim3.push_back(m);
    std::shared_ptr<Tensor> a3(new Tensor(dim3));
    Tensor::syev(A3,U3,a3,ascending,syevd); 

    //Transforming the eignevectors back to the original basis U = X U3 
    std::vector<std::shared_ptr<Tensor> > chain2;
    chain2.push_back(X);
    chain2.push_back(U3);
    std::vector<bool> trans2;
    trans2.push_back(false);
    trans2.push_back(false);
    std::shared_ptr<Tensor> U2 = Tensor::chain(chain2,trans2);

    // Copying  U2 and a3 into U/a, leaving last n-m columns as zeros
    U->zero();
    double *U2p = U2->data().data();
    double* Up = U->data().data();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            Up[i*n+j] = U2p[i*m+j];
        }
    }
    a->zero();
    double *a3p = a3->data().data();
    double* ap = a->data().data();    
    for (int j = 0; j < m; j++) {
        ap[j] = a3p[j];
    }

    // Balancing
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Up[i*n+j] *= Sbp[i];
        }
    } 
}
void Tensor::gesvd(
    const std::shared_ptr<Tensor>& A,
    std::shared_ptr<Tensor>& U,
    std::shared_ptr<Tensor>& s,
    std::shared_ptr<Tensor>& V,
    bool full_matrices,
    bool gesdd)
{
    size_t m = A->shape()[0]; 
    size_t n = A->shape()[1]; 
    size_t k = std::min(m,n);

    if (full_matrices) {
        std::vector<size_t> dim1;
        dim1.push_back(m);
        dim1.push_back(m);
        U->shape_error(dim1);
        std::vector<size_t> dim2;
        dim2.push_back(k);
        s->shape_error(dim2);
        std::vector<size_t> dim3;
        dim3.push_back(n);
        dim3.push_back(n);
        V->shape_error(dim3);
    } else {
        std::vector<size_t> dim1;
        dim1.push_back(m);
        dim1.push_back(k);
        U->shape_error(dim1);
        std::vector<size_t> dim2;
        dim2.push_back(k);
        s->shape_error(dim2);
        std::vector<size_t> dim3;
        dim3.push_back(k);
        dim3.push_back(n);
        V->shape_error(dim3);
    }
#if 0 //C++11
    if (full_matrices) {
        U->shape_error({m,m});
        s->shape_error({k});     
        V->shape_error({n,n});
    } else {
        U->shape_error({m,k});
        s->shape_error({k});     
        V->shape_error({k,n});
    }
#endif

    std::shared_ptr<Tensor> A2 = A->clone();
    size_t ldaU;
    size_t ldaV;
    char jobu;
    char jobv;
    if (full_matrices) {
        ldaU = m;
        ldaV = n;
        jobu = 'A';
        jobv = 'A';
    } else {
        ldaU = k;
        ldaV = n;
        jobu = 'S';
        jobv = 'S';
    }

    if (gesdd) {
        int info;
        double nwork;
        std::vector<int> iwork(8*k);
        info = C_DGESDD(
            jobv,
            n,
            m,
            A2->data().data(),
            n,
            s->data().data(),
            V->data().data(),
            ldaV,
            U->data().data(),
            ldaU,
            &nwork,
            -1,
            iwork.data()
            );

        if ((size_t) nwork >= std::numeric_limits<int>::max()) {
            throw std::runtime_error("nwork is too big.");
        }
        
        std::vector<double> work((size_t)nwork);
        info = C_DGESDD(
            jobv,
            n,
            m,
            A2->data().data(),
            n,
            s->data().data(),
            V->data().data(),
            ldaV,
            U->data().data(),
            ldaU,
            work.data(),
            (size_t)nwork,
            iwork.data()
            );

        if (info) throw std::runtime_error("DGESDD did not converge.");
    } else {
        int info;
        double nwork;
        info = C_DGESVD(
            jobv,
            jobu,
            n,
            m,
            A2->data().data(),
            n,
            s->data().data(),
            V->data().data(),
            ldaV,
            U->data().data(),
            ldaU,
            &nwork,
            -1);
        
        std::vector<double> work((size_t)nwork);
        info = C_DGESVD(
            jobv,
            jobu,
            n,
            m,
            A2->data().data(),
            n,
            s->data().data(),
            V->data().data(),
            ldaV,
            U->data().data(),
            ldaU,
            work.data(),
            (size_t)nwork
            );

        if (info) throw std::runtime_error("DGESVD did not converge.");
    }
}
std::shared_ptr<Tensor> Tensor::power(
    const std::shared_ptr<Tensor>& S,
    double power,
    double condition,
    bool throwNaN)
{
    if (power == 0.0) {
        std::shared_ptr<Tensor> Spow = S->clone();
        Spow->identity();
        return Spow;
    }
    if (power == 1.0) {
        return S->clone();
    }

    std::shared_ptr<Tensor> U(new Tensor(S->shape()));
    std::vector<size_t> dim1;
    dim1.push_back(S->shape()[0]);
    std::shared_ptr<Tensor> s(new Tensor(dim1));
    //std::shared_ptr<Tensor> s(new Tensor({S->shape()[0]})); //C++11
    Tensor::syev(S,U,s,false);

    size_t n = S->shape()[0];
    double* sp = s->data().data();

    double smax = 0.0;
    for (size_t i = 0; i < n; i++) {
        smax = std::max(smax,fabs(sp[i]));
    }

    bool foundNaN = false;
    for (size_t i = 0; i < n; i++) {
        if (fabs(sp[i]) >= condition * smax) {
            sp[i] = pow(sp[i], power);
            if (!std::isfinite(sp[i])) {
                sp[i] = 0.0;
                foundNaN = true;
            }
        } else {
            sp[i] = 0.0;
        }
    }
    if (throwNaN && foundNaN) {
        throw std::runtime_error("Tensor::power - non-finite diagonal encountered.");
    }

    std::shared_ptr<Tensor> V = U->clone();
    double* Vp = V->data().data();
    for (size_t i = 0; i < n; i++) {
        C_DSCAL(n,sp[i],&Vp[i],n);
    }
     
    std::shared_ptr<Tensor> Spow(new Tensor(S->shape()));
    std::vector<std::shared_ptr<Tensor> > chain1;
    chain1.push_back(V);
    chain1.push_back(U);
    std::vector<bool> trans1;
    trans1.push_back(false);
    trans1.push_back(true);
    Tensor::chain(chain1,trans1, Spow);
    //Tensor::chain({V,U},{false,true}, Spow); //C++11
    Spow->symmetrize();
    return Spow;
}
std::shared_ptr<Tensor> Tensor::lowdin_orthogonalize(
    const std::shared_ptr<Tensor>& S)
{
    return Tensor::power(S,-1.0/2.0,1.0E-12);
}
std::shared_ptr<Tensor> Tensor::cholesky_orthogonalize(
    const std::shared_ptr<Tensor>& S)
{
    return Tensor::trtri(Tensor::potrf(S,false),false);
}
std::shared_ptr<Tensor> Tensor::canonical_orthogonalize(
    const std::shared_ptr<Tensor>& S,
    double condition)
{
    std::shared_ptr<Tensor> U(new Tensor(S->shape()));
    std::vector<size_t> dim1;
    dim1.push_back(S->shape()[0]);
    std::shared_ptr<Tensor> s(new Tensor(dim1));
    //std::shared_ptr<Tensor> s(new Tensor({S->shape()[0]})); //C++11
    Tensor::syev(S,U,s,false);

    size_t n = S->shape()[0];
    double* sp = s->data().data();
    if (sp[0] < 0.0) throw std::runtime_error("S is not positive"); 
    size_t m = 0;
    for (size_t i = 0; i < n; i++) {
        if (sp[i] / sp[0] >= condition) {
            sp[i] = pow(sp[i], -1.0/2.0);
            m++;
        } else {
            break;
        }
    }
    
    std::vector<size_t> dim2;
    dim2.push_back(n);
    dim2.push_back(m);
    std::shared_ptr<Tensor> X(new Tensor(dim2)); 
    //std::shared_ptr<Tensor> X(new Tensor({n,m})); //C++11
    double* Xp = X->data().data();
    double* Up = U->data().data(); 
    
    for (size_t i = 0; i < m; i++) {
        C_DAXPY(n,sp[i],&Up[i],n,&Xp[i],m);
    }

    return X; 
}

double Tensor::invert_lu()
{
    // Validity check
    square_error();

    // Sizing/pointers
    size_t n = shape_[0];
    double* Ap = data_.data();

    // LU decomposition
    std::vector<int> ipiv(n);
    int info1 = C_DGETRF(n, n, Ap, n, ipiv.data());
    if (info1 > 0) return 0.0; // Determinant is exactly 0

    // Compute the determinant
    double det = 1.0;
    for (size_t i = 0; i < n; i++) det *= Ap[i * n + i];
    // Permutation parity (remember that FORTRAN is 1-based)
    for (size_t i = 0; i < n; i++) {
        if (ipiv[i] != i+1) det *= -1.0;
    }

    // DGETRI workspace query
    double dwork;
    int info2 = C_DGETRI(n, Ap, n, ipiv.data(), &dwork, -1);
    if (info2) throw std::runtime_error("Bad DGETRI workspace query");

    // DGETRI
    std::vector<double> work((size_t) dwork);
    int info3 = C_DGETRI(n, Ap, n, ipiv.data(), work.data(), work.size());
    if (info3) throw std::runtime_error("DGETRI failed");
    
    return det;
}

} // namespace lightspeed
