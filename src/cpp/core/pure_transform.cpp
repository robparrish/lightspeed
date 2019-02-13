#include <lightspeed/pure_transform.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/basis.hpp>
#include <lightspeed/am.hpp>
#include "pure_transform_util.hpp"

namespace lightspeed {

std::shared_ptr<Tensor> PureTransform::allocCart2(
    const std::shared_ptr<Basis>& basis1,
    const std::shared_ptr<Basis>& basis2,
    const std::shared_ptr<Tensor>& Tpure)
{
    // => Checks <= //
    
    // throw if Tpure is not (nao1, nao2) if Tpure is provided
    if (Tpure) {
        // C++03
        std::vector<size_t> dim;
        dim.push_back(basis1->nao());
        dim.push_back(basis2->nao());
        Tpure->shape_error(dim);
        // C++11
        //Tpure->shape_error({basis1->nao(),basis2->nao()});
    }

    // => Early Exit Possibilities <= //

    // If basis1/basis2 are cart and Tpure is provided, return Tpure
    if (Tpure && !basis1->has_pure() && !basis2->has_pure()) {
        return Tpure;
    }

    // => Allocation <= //
  
    // If basis1/basis2 are cart and Tpure is not provided, allocate/return 
    // If basis1/basis2 are not cart, allocate/return 
    // C++03
    std::vector<size_t> dim;
    dim.push_back(basis1->ncart());
    dim.push_back(basis2->ncart());
    return std::shared_ptr<Tensor>(new Tensor(dim));
    // C++11
    //return std::shared_ptr<Tensor>(new Tensor({basis1->ncart(),basis2->ncart()}));
}
std::shared_ptr<Tensor> PureTransform::cartToPure2(
    const std::shared_ptr<Basis>& basis1,
    const std::shared_ptr<Basis>& basis2,
    const std::shared_ptr<Tensor>& Tcart,
    const std::shared_ptr<Tensor>& Tpure)
{
    // => Checks <= //

    // throw if Tpure is not (nao1, nao2) if Tpure is provided
    if (Tpure) {
        // C++03
        std::vector<size_t> dim;
        dim.push_back(basis1->nao());
        dim.push_back(basis2->nao());
        Tpure->shape_error(dim);
        // C++11
        //Tpure->shape_error({basis1->nao(),basis2->nao()});
    }
    // throw if Tcart is not (ncart1, ncart2) if Tcart is provided
    if (true) {
        // C++03
        std::vector<size_t> dim;
        dim.push_back(basis1->ncart());
        dim.push_back(basis2->ncart());
        Tcart->shape_error(dim);
        // C++11
        //Tcart->shape_error({basis1->ncart(),basis2->ncart()});
    }

    // => Early Exit Possibilities <= //

    // If basis1/basis2 are cart: returns Tcart
    if (!basis1->has_pure() && !basis2->has_pure()) {
        return Tcart;
    }
      
    // => Allocation <= //
  
    std::shared_ptr<Tensor> Tpure2 = Tpure;
    if (!Tpure) {
        // C++03
        std::vector<size_t> dim;
        dim.push_back(basis1->nao());
        dim.push_back(basis2->nao());
        Tpure2 = std::shared_ptr<Tensor>(new Tensor(dim));
        // C++11
        //Tpure2 = std::shared_ptr<Tensor>(new Tensor({basis1->nao(),basis2->nao()}));
    }

    // => Spherical Transform <= //

    // Transform Tcart to Tpure2 (accumulate)

    // Targets
    double* TCp = Tcart->data().data();
    double* TPp = Tpure2->data().data();

    const std::vector<Shell>& shells1 = basis1->shells();
    const std::vector<Shell>& shells2 = basis2->shells();

    int nao1 = basis1->nao(); 
    int ncart1 = basis1->ncart(); 
    int nao2 = basis2->nao(); 
    int ncart2 = basis2->ncart(); 

    std::vector<AngularMomentum> am_info = AngularMomentum::build(
        std::max(basis1->max_L(), basis2->max_L()));

    #pragma omp parallel for schedule(dynamic)
    for (size_t ind12 = 0; ind12 < shells1.size() * shells2.size(); ind12++) {
        size_t ind1 = ind12 / shells2.size();
        size_t ind2 = ind12 % shells2.size();
        const Shell& shell1 = shells1[ind1];
        const Shell& shell2 = shells2[ind2];
        int  n1 = shell1.nao(); 
        int  n2 = shell2.nao(); 
        int  o1 = shell1.aoIdx();
        int  o2 = shell2.aoIdx();
        int  c1 = shell1.ncart(); 
        int  c2 = shell2.ncart(); 
        int  a1 = shell1.cartIdx();
        int  a2 = shell2.cartIdx();
        int  l1 = shell1.L();
        int  l2 = shell2.L();
        bool s1 = shell1.is_pure();
        bool s2 = shell2.is_pure();
                
        // Scratch registers
        double S1p[c1*c2];
        double S2p[c1*c2];

        for (int p = 0; p < c1; p++) {
            for (int q = 0; q < c2; q++) {
                S1p[p*c2 + q] = TCp[(p + a1)*ncart2 + (q + a2)];
            }
        }

        PureTransformUtil::cartToPure2(am_info,l1,l2,s1,s2,S1p,S2p);

        for (int p = 0; p < n1; p++) {
            for (int q = 0; q < n2; q++) {
                TPp[(p + o1)*nao2 + (q + o2)] += S1p[p*n2 + q];
            }
        }

        
    } 

    return Tpure2;
}
std::shared_ptr<Tensor> PureTransform::pureToCart2(
    const std::shared_ptr<Basis>& basis1,
    const std::shared_ptr<Basis>& basis2,
    const std::shared_ptr<Tensor>& Tpure)
{
    // => Checks <= //

    // throw if Tpure is not (nao1, nao2)
    // C++03
    std::vector<size_t> dim;
    dim.push_back(basis1->nao());
    dim.push_back(basis2->nao());
    Tpure->shape_error(dim);
    // C++11
    //Tpure->shape_error({basis1->nao(),basis2->nao()});

    // => Early Exit Possibilities <= //

    // If basis1/basis2 are cart: returns Tpure
    if (!basis1->has_pure() && !basis2->has_pure()) {
        return Tpure;
    }
      
    // => Allocation <= //
  
    // C++03
    std::vector<size_t> dim2;
    dim2.push_back(basis1->ncart());
    dim2.push_back(basis2->ncart());
    std::shared_ptr<Tensor> Tcart(new Tensor(dim2));
    // C++11
    //std::shared_ptr<Tensor> Tcart(new Tensor({basis1->ncart(),basis2->ncart()}));

    // => Spherical Transform <= //

    // Transform Tcart to Tpure2 (accumulate)

    // Targets
    double* TCp = Tcart->data().data();
    double* TPp = Tpure->data().data();

    const std::vector<Shell>& shells1 = basis1->shells();
    const std::vector<Shell>& shells2 = basis2->shells();

    int nao1 = basis1->nao(); 
    int ncart1 = basis1->ncart(); 
    int nao2 = basis2->nao(); 
    int ncart2 = basis2->ncart(); 

    std::vector<AngularMomentum> am_info = AngularMomentum::build(
        std::max(basis1->max_L(), basis2->max_L()));

    #pragma omp parallel for schedule(dynamic)
    for (size_t ind12 = 0; ind12 < shells1.size() * shells2.size(); ind12++) {
        size_t ind1 = ind12 / shells2.size();
        size_t ind2 = ind12 % shells2.size();
        const Shell& shell1 = shells1[ind1];
        const Shell& shell2 = shells2[ind2];
        int  n1 = shell1.nao(); 
        int  n2 = shell2.nao(); 
        int  o1 = shell1.aoIdx();
        int  o2 = shell2.aoIdx();
        int  c1 = shell1.ncart(); 
        int  c2 = shell2.ncart(); 
        int  a1 = shell1.cartIdx();
        int  a2 = shell2.cartIdx();
        int  l1 = shell1.L();
        int  l2 = shell2.L();
        bool s1 = shell1.is_pure();
        bool s2 = shell2.is_pure();
                
        // Scratch registers
        double S1p[c1*c2];
        double S2p[c1*c2];

        for (int p = 0; p < n1; p++) {
            for (int q = 0; q < n2; q++) {
                S1p[p*n2 + q] = TPp[(p + o1)*nao2 + (q + o2)];
            }
        }

        PureTransformUtil::pureToCart2(am_info,l1,l2,s1,s2,S1p,S2p);

        for (int p = 0; p < c1; p++) {
            for (int q = 0; q < c2; q++) {
                TCp[(p + a1)*ncart2 + (q + a2)] = S1p[p*c2 + q];
            }
        }
        
    } 

    return Tcart;
}
std::shared_ptr<Tensor> PureTransform::pureToCart1(
    const std::shared_ptr<Basis>& basis,
    const std::shared_ptr<Tensor>& Cpure)
{
    // => Checks <= //

    // throw if Tpure is not (nao1, norb)
    // C++03
    Cpure->ndim_error(2);
    std::vector<size_t> dim;
    dim.push_back(basis->nao());
    dim.push_back(Cpure->shape()[1]);
    Cpure->shape_error(dim);

    // => Early Exit Possibilities <= //

    // If basis is cart: returns Cpure
    if (!basis->has_pure()) {
        return Cpure;
    }
      
    // => Allocation <= //
  
    // C++03
    std::vector<size_t> dim2;
    dim2.push_back(basis->ncart());
    dim2.push_back(Cpure->shape()[1]);
    std::shared_ptr<Tensor> Ccart(new Tensor(dim2));

    // => Spherical Transform <= //

    // Transform Ccart to Cpure2 (accumulate)

    // Targets
    double* CCp = Ccart->data().data();
    double* CPp = Cpure->data().data();

    // Sizes
    size_t norb = Cpure->shape()[1];

    const std::vector<Shell>& shells = basis->shells();
    std::vector<AngularMomentum> am_info = AngularMomentum::build(basis->max_L());

    #pragma omp parallel for
    for (size_t sind = 0; sind < shells.size(); sind++) {
        const Shell& shell = shells[sind];
        int  n = shell.nao(); 
        int  o = shell.aoIdx();
        int  c = shell.ncart(); 
        int  a = shell.cartIdx();
        int  l = shell.L();
        bool s = shell.is_pure();
        double* CC2p = CCp + a * norb; 
        const double* CP2p = CPp + o * norb; 
        PureTransformUtil::pureToCart1(am_info,l,s,CC2p,CP2p,norb);
    }

    return Ccart;
}

} // namespace lightspeed 
