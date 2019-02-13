#include <lightspeed/dftbox.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/resource_list.hpp>
#include <lightspeed/becke.hpp>
#include <lightspeed/pair_list.hpp>
#include <lightspeed/gridbox.hpp>
#include <cstdio>

namespace lightspeed {
    
std::vector<std::shared_ptr<Tensor> > DFTBox::rksPotential(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Functional>& functional,
    const std::shared_ptr<BeckeGrid>& becke,
    const std::shared_ptr<HashedGrid>& hash,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    double thre,
    const std::vector<std::shared_ptr<Tensor> >& V)
{
    std::shared_ptr<Tensor> xyzw = becke->xyzw();
    const double* xyzwp = xyzw->data().data();

    size_t npoint = becke->size();

    if (!(V.size() == 0 || V.size() == 2)) throw std::runtime_error("DFTBox::rksPotential: V must be size 0 or size 2");
    std::shared_ptr<Tensor> V3;
    if (V.size() == 2) V3 = V[1];

    // => Density Characteristics <= //

    std::shared_ptr<Tensor> rho;
    std::shared_ptr<Tensor> rho2;
    if (functional->type() == 0) {
        // Density collocation
        rho2 = GridBox::ldaDensity(
            resources,
            pairlist,
            D,
            hash,
            thre);
        // Density Packing
        std::vector<size_t> dim;
        dim.push_back(npoint);
        dim.push_back(2);
        rho = std::shared_ptr<Tensor>(new Tensor(dim));
        double* rhop = rho->data().data();
        const double* rho2p = rho2->data().data();
        for (size_t P = 0; P < npoint; P++) {
            double rhoa = rho2p[1*P + 0];
            rhop[2*P + 0] = rhoa;
            rhop[2*P + 1] = rhoa;
        }
    } else if (functional->type() == 1) {
        // Density collocation
        rho2 = GridBox::ggaDensity(
            resources,
            pairlist,
            D,
            hash,
            thre);
        // Density Packing
        std::vector<size_t> dim;
        dim.push_back(npoint);
        dim.push_back(5);
        rho = std::shared_ptr<Tensor>(new Tensor(dim));
        double* rhop = rho->data().data();
        const double* rho2p = rho2->data().data();
        for (size_t P = 0; P < npoint; P++) {
            double rhoa = rho2p[4*P + 0];
            double sigmaaa = 
                rho2p[4*P + 1] * rho2p[4*P + 1] +
                rho2p[4*P + 2] * rho2p[4*P + 2] +
                rho2p[4*P + 3] * rho2p[4*P + 3];
            rhop[5*P + 0] = rhoa;
            rhop[5*P + 1] = rhoa;
            rhop[5*P + 2] = sigmaaa;
            rhop[5*P + 3] = sigmaaa;
            rhop[5*P + 4] = sigmaaa;
        }
    } else {
        throw std::runtime_error("DFTBox::rksPotential: Functional type not implemented");
    }   

    // => Local XC Functional Evaluation <= //
    
    std::shared_ptr<Tensor> exc = functional->compute(rho,1);

    // => Potential Characteristics <= //

    const double* excp = exc->data().data();
    const double* rho2p = rho2->data().data();
    std::shared_ptr<Tensor> V2;
    double Q = 0.0;
    double Exc = 0.0;
    if (functional->type() == 0) {
        // Potential Packing
        std::vector<size_t> dim;
        dim.push_back(npoint);
        std::shared_ptr<Tensor> v(new Tensor(dim));
        double* vp = v->data().data();
        for (size_t P = 0; P < npoint; P++) {
            double wval = xyzwp[4*P + 3]; // quadrature weight
            vp[P] = wval * excp[3*P + 1];
            Exc += wval * excp[3*P + 0];
            Q += wval * rho2p[1*P + 0];
        }
        // Potential collocation
        V2 = GridBox::ldaPotential(
            resources,
            pairlist,
            hash,
            v,
            thre,
            V3);
    } else if (functional->type() == 1) {
        // Potential Packing
        std::vector<size_t> dim;
        dim.push_back(npoint);
        dim.push_back(4);
        std::shared_ptr<Tensor> v(new Tensor(dim));
        double* vp = v->data().data();
        for (size_t P = 0; P < npoint; P++) {
            double wval = xyzwp[4*P + 3]; // quadrature weight
            vp[4*P + 0] = wval * (excp[6*P + 1]);
            vp[4*P + 1] = wval * (2.0 * excp[6*P + 3] * rho2p[4*P + 1] + excp[6*P + 4] * rho2p[4*P + 1]);
            vp[4*P + 2] = wval * (2.0 * excp[6*P + 3] * rho2p[4*P + 2] + excp[6*P + 4] * rho2p[4*P + 2]);
            vp[4*P + 3] = wval * (2.0 * excp[6*P + 3] * rho2p[4*P + 3] + excp[6*P + 4] * rho2p[4*P + 3]);
            Exc += wval * excp[6*P + 0];
            Q += wval * rho2p[4*P + 0];
        }
        // Potential collocation
        V2 = GridBox::ggaPotential(
            resources,
            pairlist,
            hash,
            v,
            thre,
            V3);
    } else {
        throw std::runtime_error("DFTBox::rksPotential: Functional type not implemented");
    }   

    if (V.size()) {
        std::vector<size_t> dimc;
        dimc.push_back(2);
        V[0]->shape_error(dimc);
        double* Cp = V[0]->data().data(); 
        Cp[0] = Exc;
        Cp[1] = Q;
        return V;
    } else {
        std::vector<size_t> dimc;
        dimc.push_back(2);
        std::vector<std::shared_ptr<Tensor> > V4;
        V4.push_back(std::shared_ptr<Tensor>(new Tensor(dimc)));
        V4.push_back(V2);
        double* Cp = V4[0]->data().data(); 
        Cp[0] = Exc;
        Cp[1] = Q;
        return V4;
    }
}
std::shared_ptr<Tensor> DFTBox::rksGrad(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Functional>& functional,
    const std::shared_ptr<BeckeGrid>& becke,
    const std::shared_ptr<HashedGrid>& hash,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    double thre,
    const std::shared_ptr<Tensor>& G)
{
    std::vector<size_t> dimg;
    dimg.push_back(becke->natom());
    dimg.push_back(3);
    std::shared_ptr<Tensor> G2;
    if (!G) {
        G2 = std::shared_ptr<Tensor>(new Tensor(dimg));
    }
    G2->shape_error(dimg);

    std::shared_ptr<Tensor> xyzw = becke->xyzw();
    const double* xyzwp = xyzw->data().data();

    size_t npoint = becke->size();
    std::vector<size_t> dimt;
    dimt.push_back(1);
    dimt.push_back(3);
    std::shared_ptr<Tensor> trans(new Tensor(dimt));

    // => Density Characteristics <= //

    std::shared_ptr<Tensor> rho;
    std::shared_ptr<Tensor> rho2;
    if (functional->type() == 0) {
        // Density collocation
        rho2 = GridBox::ggaDensity(
            resources,
            pairlist,
            D,
            hash,
            thre);
        // Density Packing
        std::vector<size_t> dim;
        dim.push_back(npoint);
        dim.push_back(2);
        rho = std::shared_ptr<Tensor>(new Tensor(dim));
        double* rhop = rho->data().data();
        const double* rho2p = rho2->data().data();
        for (size_t P = 0; P < npoint; P++) {
            double rhoa = rho2p[4*P + 0];
            rhop[2*P + 0] = rhoa;
            rhop[2*P + 1] = rhoa;
        }
    } else if (functional->type() == 1) {
        // Density collocation
        rho2 = GridBox::metaDensity(
            resources,
            pairlist,
            D,
            hash,
            thre);
        // Density Packing
        std::vector<size_t> dim;
        dim.push_back(npoint);
        dim.push_back(5);
        rho = std::shared_ptr<Tensor>(new Tensor(dim));
        double* rhop = rho->data().data();
        const double* rho2p = rho2->data().data();
        for (size_t P = 0; P < npoint; P++) {
            double rhoa = rho2p[10*P + 0];
            double sigmaaa = 
                rho2p[10*P + 1] * rho2p[10*P + 1] +
                rho2p[10*P + 2] * rho2p[10*P + 2] +
                rho2p[10*P + 3] * rho2p[10*P + 3];
            rhop[5*P + 0] = rhoa;
            rhop[5*P + 1] = rhoa;
            rhop[5*P + 2] = sigmaaa;
            rhop[5*P + 3] = sigmaaa;
            rhop[5*P + 4] = sigmaaa;
        }
    } else {
        throw std::runtime_error("DFTBox::rksGrad: Functional type not implemented");
    }   

    // => Local XC Functional Evaluation <= //
    
    std::shared_ptr<Tensor> exc = functional->compute(rho,1);
    const double* excp = exc->data().data();

    // => Potential Characteristics <= //

    const double* rho2p = rho2->data().data();
    std::vector<size_t> dime;
    dime.push_back(npoint);
    std::shared_ptr<Tensor> e(new Tensor(dime));
    double* ep = e->data().data();
    if (functional->type() == 0) {
        // Potential Packing
        std::vector<size_t> dim;
        dim.push_back(npoint);
        std::shared_ptr<Tensor> v(new Tensor(dim));
        double* vp = v->data().data();
        for (size_t P = 0; P < npoint; P++) {
            double wval = xyzwp[4*P + 3]; // quadrature weight
            vp[1*P + 0] = 2.0 * wval * excp[3*P + 1];
            ep[P] = excp[3*P + 0];
        }
        // Potential grad
        GridBox::ldaGrad(
            resources,
            pairlist,
            D,
            hash,
            v,
            thre,
            G2);
    } else if (functional->type() == 1) {
        // Potential Packing
        std::vector<size_t> dim;
        dim.push_back(npoint);
        dim.push_back(4);
        std::shared_ptr<Tensor> v(new Tensor(dim));
        double* vp = v->data().data();
        for (size_t P = 0; P < npoint; P++) {
            double wval = xyzwp[4*P + 3]; // quadrature weight
            vp[4*P + 0] = 2.0 * wval * excp[6*P + 1];
            vp[4*P + 1] = 2.0 * wval * (2.0 * excp[6*P + 3] * rho2p[10*P + 1] + 1.0 * excp[6*P + 4] * rho2p[10*P + 1]);
            vp[4*P + 2] = 2.0 * wval * (2.0 * excp[6*P + 3] * rho2p[10*P + 2] + 1.0 * excp[6*P + 4] * rho2p[10*P + 2]);
            vp[4*P + 3] = 2.0 * wval * (2.0 * excp[6*P + 3] * rho2p[10*P + 3] + 1.0 * excp[6*P + 4] * rho2p[10*P + 3]);
            ep[P] = excp[6*P + 0];
        }
        // Potential grad
        GridBox::ggaGrad(
            resources,
            pairlist,
            D,
            hash,
            v,
            thre,
            G2);
    } else {
        throw std::runtime_error("DFTBox::rksGrad: Functional type not implemented");
    }   

    // => Point Motion <= //

    const std::vector<int>& atom_inds = becke->atomic_inds();
    if (functional->type() == 0) {
        double* G2p = G2->data().data();
        for (size_t P = 0; P < npoint; P++) {
            double wval = xyzwp[4*P + 3]; // quadrature weight
            int A = atom_inds[P];
            G2p[3*A + 0] += 2.0 * wval * excp[3*P+1] * rho2p[4*P + 1];
            G2p[3*A + 1] += 2.0 * wval * excp[3*P+1] * rho2p[4*P + 2];
            G2p[3*A + 2] += 2.0 * wval * excp[3*P+1] * rho2p[4*P + 3];
        }
    } else if (functional->type() == 1) {
        // Density collocation
        double* G2p = G2->data().data();
        for (size_t P = 0; P < npoint; P++) {
            double wval = xyzwp[4*P + 3]; // quadrature weight
            int A = atom_inds[P];
            G2p[3*A + 0] += 2.0 * wval * excp[6*P+1] * rho2p[10*P + 1];
            G2p[3*A + 1] += 2.0 * wval * excp[6*P+1] * rho2p[10*P + 2];
            G2p[3*A + 2] += 2.0 * wval * excp[6*P+1] * rho2p[10*P + 3];
            G2p[3*A + 0] += 2.0 * wval * (2.0 * excp[6*P + 3] * rho2p[10*P + 1] + 1.0 * excp[6*P + 4] * rho2p[10*P + 1]) * rho2p[10*P + 4];
            G2p[3*A + 1] += 2.0 * wval * (2.0 * excp[6*P + 3] * rho2p[10*P + 1] + 1.0 * excp[6*P + 4] * rho2p[10*P + 1]) * rho2p[10*P + 5];
            G2p[3*A + 2] += 2.0 * wval * (2.0 * excp[6*P + 3] * rho2p[10*P + 1] + 1.0 * excp[6*P + 4] * rho2p[10*P + 1]) * rho2p[10*P + 6];
            G2p[3*A + 0] += 2.0 * wval * (2.0 * excp[6*P + 3] * rho2p[10*P + 2] + 1.0 * excp[6*P + 4] * rho2p[10*P + 2]) * rho2p[10*P + 5];
            G2p[3*A + 1] += 2.0 * wval * (2.0 * excp[6*P + 3] * rho2p[10*P + 2] + 1.0 * excp[6*P + 4] * rho2p[10*P + 2]) * rho2p[10*P + 7];
            G2p[3*A + 2] += 2.0 * wval * (2.0 * excp[6*P + 3] * rho2p[10*P + 2] + 1.0 * excp[6*P + 4] * rho2p[10*P + 2]) * rho2p[10*P + 8];
            G2p[3*A + 0] += 2.0 * wval * (2.0 * excp[6*P + 3] * rho2p[10*P + 3] + 1.0 * excp[6*P + 4] * rho2p[10*P + 3]) * rho2p[10*P + 6];
            G2p[3*A + 1] += 2.0 * wval * (2.0 * excp[6*P + 3] * rho2p[10*P + 3] + 1.0 * excp[6*P + 4] * rho2p[10*P + 3]) * rho2p[10*P + 8];
            G2p[3*A + 2] += 2.0 * wval * (2.0 * excp[6*P + 3] * rho2p[10*P + 3] + 1.0 * excp[6*P + 4] * rho2p[10*P + 3]) * rho2p[10*P + 9];
        }
    } else {
        throw std::runtime_error("DFTBox::rksGrad: Functional type not implemented");
    }
    
    // => Becke Weight Gradients <= //

    becke->grad(
        resources,
        e,
        G2);

    return G2;
}

} // namespace lightspeed
