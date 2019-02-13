#include <lightspeed/gridbox.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/resource_list.hpp>
#include <lightspeed/basis.hpp>
#include <lightspeed/pure_transform.hpp>
#include <cmath>

namespace lightspeed {

std::shared_ptr<Tensor> GridBox::orbitals(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Basis>& basis,
    const std::shared_ptr<Tensor>& xyz,
    const std::shared_ptr<Tensor>& C,
    double thre,
    const std::shared_ptr<Tensor>& psi)
{
    // The significance criterion is
    // |C| c_A exp(-a_A r_1P^2) < thre, where |C| is the largest Cartesian
    // orbital coefficient in the shell (across all orbitals)
     
    // => Safety Checks <= //

    // Check the xyz is the correct size
    xyz->ndim_error(2);
    std::vector<size_t> dim3;
    dim3.push_back(xyz->shape()[0]);
    dim3.push_back(3);
    xyz->shape_error(dim3);

    // => Setup <= //

    // Cartesian-basis orbital coefficients
    std::shared_ptr<Tensor> C2 = PureTransform::pureToCart1(
        basis,  
        C);    

    // Target
    std::shared_ptr<Tensor> psi2 = psi;
    if (!psi2) {
        std::vector<size_t> dim2;
        dim2.push_back(xyz->shape()[0]);
        dim2.push_back(C2->shape()[1]);
        psi2 = std::shared_ptr<Tensor>(new Tensor(dim2));
    }
    std::vector<size_t> dim2;
    dim2.push_back(xyz->shape()[0]);
    dim2.push_back(C2->shape()[1]);
    psi2->shape_error(dim2);

    // Pointers
    const double* C2p = C2->data().data();
    const double* xyzp = xyz->data().data();
    double* psip = psi2->data().data(); 
    const std::vector<Primitive>& prims = basis->primitives();
        
    // Sizes
    size_t nP = xyz->shape()[0];
    size_t norb = C2->shape()[1];

    // Max absolute C for each primitive shell
    std::vector<double> Cmax(prims.size());
    for (size_t A = 0; A < prims.size(); A++) {
        const Primitive& prim = prims[A];
        int oA = prim.cartIdx();
        int nA = prim.ncart(); 
        double val = 0.0;
        const double* C3p = C2p + oA * norb;
        for (int ind = 0; ind < nA * norb; ind++) {
            val = std::max(val,fabs(*C3p++));
        }
        Cmax[A] = val;
    }

    #pragma omp parallel for schedule(dynamic) num_threads(resources->nthread())
    for (size_t P = 0; P < nP; P++) {
        double xP = xyzp[3*P + 0];
        double yP = xyzp[3*P + 1];
        double zP = xyzp[3*P + 2];
        for (size_t A = 0; A < prims.size(); A++) {
            const Primitive& prim = prims[A];
            int L = prim.L();
            double cA = prim.c();
            double eA = prim.e();
            double xA = prim.x();
            double yA = prim.y();
            double zA = prim.z();
            int oA = prim.cartIdx();
            int nA = prim.ncart(); 
            double xPA = xP - xA;
            double yPA = yP - yA;
            double zPA = zP - zA;
            double R2 = pow(xPA,2) + pow(yPA,2) + pow(zPA,2); 
            double K = cA * exp(-eA * R2);
            double Cval = Cmax[A];
            if (Cval * K < thre) continue; // Significance check
            double chi[nA];
            for (int i = 0, index = 0; i <= L; i++) {
                int l = L - i;
            for (int j = 0; j <= i; j++, index++) {
                int m = i - j;
                int n = j;
                chi[index] = K * pow(xPA,l) * pow(yPA,m) * pow(zPA,n);
            }}
            for (int index = 0; index < nA; index++) {
                double chival = chi[index];
                double* psi3p = psip + P * norb;
                const double* C3p = C2p + (index + oA) * norb;
                for (int i = 0; i < norb; i++) {
                    #pragma omp atomic
                    (*psi3p++) += chival * (*C3p++);
                }
            }
        }
    }

    return psi2;
}
std::shared_ptr<Tensor> GridBox::orbitalsGrad(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Basis>& basis,
    const std::shared_ptr<Tensor>& xyz,
    const std::shared_ptr<Tensor>& C,
    const std::shared_ptr<Tensor>& xi,
    double thre,
    const std::shared_ptr<Tensor>& G1)
{
    // The significance criterion is
    // |C|| X| c_A exp(-a_A r_1P^2) < thre, where |C| is the largest Cartesian
    // orbital coefficient in the shell (across all orbitals) and |X| is the
    // largest collocation density maginitude element for the point (across all
    // orbitals)
     
    // => Safety Checks <= //

    // Check the xyz is the correct size
    xyz->ndim_error(2);
    std::vector<size_t> dim3;
    dim3.push_back(xyz->shape()[0]);
    dim3.push_back(3);
    xyz->shape_error(dim3);

    // => Setup <= //

    // Cartesian-basis orbital coefficients
    std::shared_ptr<Tensor> C2 = PureTransform::pureToCart1(
        basis,  
        C);    

    // Check that xi is the correct size
    std::vector<size_t> dim2;
    dim2.push_back(xyz->shape()[0]);
    dim2.push_back(C2->shape()[1]);
    xi->shape_error(dim2);

    // Target
    std::shared_ptr<Tensor> G2 = G1;
    if (!G2) {
        std::vector<size_t> dimG;
        dimG.push_back(basis->natom());
        dimG.push_back(3);
        G2 = std::shared_ptr<Tensor>(new Tensor(dimG));
    }
    std::vector<size_t> dimG;
    dimG.push_back(basis->natom());
    dimG.push_back(3);
    G2->shape_error(dimG);

    // Pointers
    const double* C2p = C2->data().data();
    const double* xyzp = xyz->data().data();
    const double* xip = xi->data().data(); 
    double* G2p = G2->data().data();
    const std::vector<Primitive>& prims = basis->primitives();
        
    // Sizes
    size_t nP = xyz->shape()[0];
    size_t norb = C2->shape()[1];

    // Max absolute C for each primitive shell
    std::vector<double> Cmax(prims.size());
    for (size_t A = 0; A < prims.size(); A++) {
        const Primitive& prim = prims[A];
        int oA = prim.cartIdx();
        int nA = prim.ncart(); 
        double val = 0.0;
        const double* C3p = C2p + oA * norb;
        for (int ind = 0; ind < nA * norb; ind++) {
            val = std::max(val,fabs(*C3p++));
        }
        Cmax[A] = val;
    }

    // Max absolute xi for each point
    std::vector<double> Xmax(nP);
    for (size_t P = 0; P < nP; P++) {
        double val = 0.0;
        const double* xi2p = xip + P * norb;
        for (int i = 0; i < norb; i++) {
            val = std::max(val,fabs(*xi2p++));
        }
        Xmax[P] = val;
    }

    #pragma omp parallel for schedule(dynamic) num_threads(resources->nthread())
    for (size_t P = 0; P < nP; P++) {
        double xP = xyzp[3*P + 0];
        double yP = xyzp[3*P + 1];
        double zP = xyzp[3*P + 2];
        double Xval = Xmax[P];
        for (size_t A = 0; A < prims.size(); A++) {
            const Primitive& prim = prims[A];
            int L = prim.L();
            double cA = prim.c();
            double eA = prim.e();
            double xA = prim.x();
            double yA = prim.y();
            double zA = prim.z();
            int oA = prim.cartIdx();
            int nA = prim.ncart(); 
            int AA = prim.atomIdx();
            double xPA = xP - xA;
            double yPA = yP - yA;
            double zPA = zP - zA;
            double R2 = pow(xPA,2) + pow(yPA,2) + pow(zPA,2); 
            double K = cA * exp(-eA * R2);
            double Cval = Cmax[A];
            if (Xval * Cval * K < thre) continue; // Significance check
            double chix[nA];
            double chiy[nA];
            double chiz[nA];
            for (int i = 0, index = 0; i <= L; i++) {
                int l = L - i;
            for (int j = 0; j <= i; j++, index++) {
                int m = i - j;
                int n = j;
                chix[index] = K * (-2.0 * eA * pow(xPA,l+1) + l * pow(xPA,(l == 0 ? 0 : l-1))) * pow(yPA,m) * pow(zPA,n);
                chiy[index] = K * (-2.0 * eA * pow(yPA,m+1) + m * pow(yPA,(m == 0 ? 0 : m-1))) * pow(xPA,l) * pow(zPA,n);
                chiz[index] = K * (-2.0 * eA * pow(zPA,n+1) + n * pow(zPA,(n == 0 ? 0 : n-1))) * pow(xPA,l) * pow(yPA,m);
            }}
            double Gx = 0.0;
            double Gy = 0.0;
            double Gz = 0.0;
            for (int index = 0; index < nA; index++) {
                double chixval = chix[index];
                double chiyval = chiy[index];
                double chizval = chiz[index];
                const double* xi3p = xip + P * norb;
                const double* C3p = C2p + (index + oA) * norb;
                for (int i = 0; i < norb; i++) {
                    Gx += chixval * (*C3p) * (*xi3p); 
                    Gy += chiyval * (*C3p) * (*xi3p); 
                    Gz += chizval * (*C3p) * (*xi3p); 
                    C3p++;
                    xi3p++;
                }
            }
            #pragma omp atomic
            G2p[3*A + 0] += Gx;
            #pragma omp atomic
            G2p[3*A + 1] += Gy;
            #pragma omp atomic
            G2p[3*A + 2] += Gz;
        }
    }

    return G2;
}

} // namespace lightspeed

