#include <lightspeed/intbox.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/resource_list.hpp>
#include <lightspeed/pair_list.hpp>
#include <lightspeed/ecp.hpp>
#include <lightspeed/ewald.hpp>

namespace lightspeed { 

namespace {

bool is_terachem_ready(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist12,
    const std::shared_ptr<PairList>& pairlist34)
{
#ifdef HAVE_TERACHEM
    if (resources->ngpu() == 0) return false;
    if (pairlist12 != pairlist34) return false;
    if (!pairlist12->is_symmetric()) return false;
    if (pairlist12->basis1()->max_L() > 2) return false;
    return true;
#else
    return false;
#endif
}

}

std::shared_ptr<Tensor> IntBox::overlap(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& S
    )
{
    return IntBox::overlapCPU(
        resources,
        pairlist,
        S
        );
}

std::vector< std::shared_ptr<Tensor> > IntBox::dipole(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    double x0, 
    double y0, 
    double z0,
    const std::vector< std::shared_ptr<Tensor> >& Dipole
    )
{
    return IntBox::dipoleCPU(
        resources,
        pairlist,
        x0, 
        y0, 
        z0,
        Dipole
        );
}

std::vector< std::shared_ptr<Tensor>> IntBox::dipoleGrad(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    double x0, 
    double y0, 
    double z0,
    const std::shared_ptr<Tensor>& D,
    const std::vector< std::shared_ptr<Tensor> >& Dgrad
    )
{
    if (!pairlist->is_symmetric()) throw std::runtime_error("IntBox::dipoleGrad: pairlist must be symmetric");

    std::vector<size_t> dim2;
    dim2.push_back(pairlist->basis1()->natom());
    dim2.push_back(3);

    std::vector < std::shared_ptr<Tensor> > Dgrad2 = Dgrad;
    if (Dgrad.size() == 0) {
        for (int d = 0; d < 3; d++) {
            Dgrad2.push_back(std::shared_ptr<Tensor>(new Tensor(dim2)));
        }        
    }
    else if (Dgrad.size() != 3) {
        std::runtime_error("IntBox::dipoleGrad: Dgrad should be of size 0 or 2");
    }
    // Validate dimensions
    for (int d = 0; d < 3; d++) {
        Dgrad2[d]->shape_error(dim2);
    }

    std::vector<std::vector <std::shared_ptr<Tensor>> > Dgradtemp;
    for (int d = 0; d < 3; d++) {
        Dgradtemp.push_back(std::vector< std::shared_ptr<Tensor>>());
        Dgradtemp[d].push_back(Dgrad2[d]);
        Dgradtemp[d].push_back(Dgrad2[d]);
    
    }

    IntBox::dipoleGradAdv(
        resources,
        pairlist,
        x0,
        y0,
        z0,
        D,
        Dgradtemp);
    
    return Dgrad2;
}

std::vector< std::vector<std::shared_ptr<Tensor> >> IntBox::dipoleGradAdv(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    double x0, 
    double y0, 
    double z0,
    const std::shared_ptr<Tensor>& D,
    const std::vector< std::vector<std::shared_ptr<Tensor>> >& Dgrad
    )
{
    return IntBox::dipoleGradAdvCPU(
        resources,
        pairlist,
        x0,
        y0,
        z0,
        D,
        Dgrad
        );
}
std::vector< std::shared_ptr<Tensor> > IntBox::quadrupole(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    double x0, 
    double y0, 
    double z0,
    const std::vector< std::shared_ptr<Tensor> >& Q
    )
{
    return IntBox::quadrupoleCPU(
        resources,
        pairlist,
        x0, 
        y0, 
        z0,
        Q
        );
}

std::vector< std::shared_ptr<Tensor> > IntBox::nabla(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::vector< std::shared_ptr<Tensor> >& P
    )
{
    return IntBox::nablaCPU(
        resources,
        pairlist,
        P
        );
}

std::vector< std::shared_ptr<Tensor> > IntBox::angularMomentum(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    double x0, 
    double y0, 
    double z0,
    const std::vector< std::shared_ptr<Tensor> >& L
    )
{
    return IntBox::angularMomentumCPU(
        resources,
        pairlist,
        x0, 
        y0, 
        z0,
        L
        );
}

std::shared_ptr<Tensor> IntBox::kinetic(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& T
    )
{
    return IntBox::kineticCPU(
        resources,
        pairlist,
        T
        );
}

std::shared_ptr<Tensor> IntBox::potential(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& xyzc,
    const std::shared_ptr<Tensor>& V
    )
{
    return IntBox::potentialCPU(
        resources,
        ewald,
        pairlist,
        xyzc,
        V
        );
}

std::shared_ptr<Tensor> IntBox::coulomb(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist12,
    const std::shared_ptr<PairList>& pairlist34,
    const std::shared_ptr<Tensor>& D34,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& J12
    )
{
    if (is_terachem_ready(resources,pairlist12,pairlist34)) {
        //printf("coulombTC\n");
        return IntBox::coulombTC(
            resources,
            ewald,
            pairlist12,
            pairlist34,
            D34,
            thresp,
            thredp,
            J12
            );
    } else {
        //printf("coulombCPU\n");
        return IntBox::coulombCPU(
            resources,
            ewald,
            pairlist12,
            pairlist34,
            D34,
            thresp,
            thredp,
            J12
            );
    }
}

std::shared_ptr<Tensor> IntBox::esp(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<Tensor>& xyz,
    const std::shared_ptr<Tensor>& w
    )
{
    return IntBox::espCPU(
        resources,
        ewald,
        pairlist,
        D,
        xyz,
        w
        );
}


std::shared_ptr<Tensor> IntBox::field(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<Tensor>& xyz,
    const std::shared_ptr<Tensor>& F
    )
{
    return IntBox::fieldCPU(
        resources,
        ewald,
        pairlist,
        D,
        xyz,
        F
        );
}

std::shared_ptr<Tensor> IntBox::exchange(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist12,
    const std::shared_ptr<PairList>& pairlist34,
    const std::shared_ptr<Tensor>& D24,
    bool D24symm,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& K13
    )
{
    if (is_terachem_ready(resources,pairlist12,pairlist34)) {
        //printf("exchangeTC\n");
        return IntBox::exchangeTC(
            resources,
            ewald,
            pairlist12,
            pairlist34,
            D24,
            D24symm,
            thresp,
            thredp,
            K13
            );
    } else {
        //printf("exchangeCPU\n");
        return IntBox::exchangeCPU(
            resources,
            ewald,
            pairlist12,
            pairlist34,
            D24,
            D24symm,
            thresp,
            thredp,
            K13
            );
    }
}

std::shared_ptr<Tensor> IntBox::overlapGrad(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& W,
    const std::shared_ptr<Tensor>& G
    )
{
    if (!pairlist->is_symmetric()) throw std::runtime_error("IntBox::overlapGrad: pairlist must be symmetric");

    std::vector<size_t> dim2;
    dim2.push_back(pairlist->basis1()->natom());
    dim2.push_back(3);
    std::shared_ptr<Tensor> G2 = G;
    if (!G) {
        G2 = std::shared_ptr<Tensor>(new Tensor(dim2));
    }
    G2->shape_error(dim2);

    std::vector<std::shared_ptr<Tensor> > Gtemp;
    Gtemp.push_back(G2);
    Gtemp.push_back(G2);
    
    IntBox::overlapGradAdv(
        resources,
        pairlist,
        W,
        Gtemp);
    
    return G2;
}

std::vector<std::shared_ptr<Tensor> > IntBox::overlapGradAdv(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& W,
    const std::vector<std::shared_ptr<Tensor> >& G12
    )
{
    return IntBox::overlapGradAdvCPU(
        resources,
        pairlist,
        W,
        G12
        );
}

std::shared_ptr<Tensor> IntBox::kineticGrad(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& W,
    const std::shared_ptr<Tensor>& G
    )
{
    if (!pairlist->is_symmetric()) throw std::runtime_error("IntBox::kineticGrad: pairlist must be symmetric");

    std::vector<size_t> dim2;
    dim2.push_back(pairlist->basis1()->natom());
    dim2.push_back(3);
    std::shared_ptr<Tensor> G2 = G;
    if (!G) {
        G2 = std::shared_ptr<Tensor>(new Tensor(dim2));
    }
    G2->shape_error(dim2);

    std::vector<std::shared_ptr<Tensor> > Gtemp;
    Gtemp.push_back(G2);
    Gtemp.push_back(G2);
    
    IntBox::kineticGradAdv(
        resources,
        pairlist,
        W,
        Gtemp);
    
    return G2;
}

std::vector<std::shared_ptr<Tensor> > IntBox::kineticGradAdv(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::vector<std::shared_ptr<Tensor> >& G12
    )
{
    return IntBox::kineticGradAdvCPU(
        resources,
        pairlist,
        D,
        G12
        );
}

std::shared_ptr<Tensor> IntBox::coulombGrad(
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
    if (is_terachem_ready(resources,pairlist,pairlist)) {
        //printf("coulombGradTC\n");
        return IntBox::coulombGradTC(
            resources,
            ewald,
            pairlist,
            D12,
            D34,
            thresp,
            thredp,
            G
            );
    }
    //printf("coulombGradCPU\n");

    // symmetric J gradient
    if (D12 == D34) {
        return IntBox::coulombGradSymmCPU(
            resources,
            ewald,
            pairlist,
            D12,
            thresp,
            thredp,
            G
            );
    }

    // non-symmetric J gradient
    else {
        std::vector<size_t> dim2;
        dim2.push_back(pairlist->basis1()->natom());
        dim2.push_back(3);
        std::shared_ptr<Tensor> G2 = G;
        if (!G) {
            G2 = std::shared_ptr<Tensor>(new Tensor(dim2));
        }
        G2->shape_error(dim2);

        std::vector<std::shared_ptr<Tensor> > Gtemp =
            IntBox::coulombGradAdv(
                resources,
                ewald,
                pairlist,
                pairlist,
                D12,
                D34,
                thresp,
                thredp
                );

        for (int a = 0; a < Gtemp.size(); a++) {
            for (int i = 0; i < G2->size(); i++) {
                G2->data()[i] += Gtemp[a]->data()[i];
            }
        }
        return G2;
    }
}

std::vector<std::shared_ptr<Tensor> > IntBox::coulombGradAdv(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist12, // symm
    const std::shared_ptr<PairList>& pairlist34, // symm
    const std::shared_ptr<Tensor>& D12,
    const std::shared_ptr<Tensor>& D34,
    float thresp,
    float thredp,
    const std::vector<std::shared_ptr<Tensor> >& G1234
    )
{
    return IntBox::coulombGradAdvCPU(
        resources,
        ewald,
        pairlist12,
        pairlist34,
        D12,
        D34,
        thresp,
        thredp,
        G1234
        );
}

std::shared_ptr<Tensor> IntBox::exchangeGrad(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D13,
    const std::shared_ptr<Tensor>& D24,
    bool D13symm,
    bool D24symm,
    bool Dsame,
    float thresp,
    float thredp,
    const std::shared_ptr<Tensor>& G
    )
{
    // check pairlist symmetry
    if (!pairlist->is_symmetric()) {
        throw std::runtime_error("IntBox::exchangeGrad: pairlist must be symmetric");
    }

    if (is_terachem_ready(resources,pairlist,pairlist)) {
        //printf("exchangeGradTC\n");
        return IntBox::exchangeGradTC(
            resources,
            ewald,
            pairlist,
            D13,
            D24,
            D13symm,
            D24symm,
            Dsame,
            thresp,
            thredp,
            G
            );
    }
    //printf("exchangeGradCPU\n");

    // fully symmetric K gradient
    if (D13symm && D24symm && Dsame) {
        return IntBox::exchangeGradSymmCPU(
            resources,
            ewald,
            pairlist,
            D13,
            thresp,
            thredp,
            G
            );
    }
    
    // partially symmetric or non-symmetric K gradient
    else {
        std::vector<size_t> dim2;
        dim2.push_back(pairlist->basis1()->natom());
        dim2.push_back(3);
        std::shared_ptr<Tensor> G2 = G;
        if (!G) {
            G2 = std::shared_ptr<Tensor>(new Tensor(dim2));
        }
        G2->shape_error(dim2);

        std::vector<std::shared_ptr<Tensor> > Gtemp =
            IntBox::exchangeGradAdv(
                resources,
                ewald,
                pairlist,
                pairlist,
                D13,
                D24,
                D13symm,
                D24symm,
                Dsame,
                thresp,
                thredp
                );

        for (int a = 0; a < Gtemp.size(); a++) {
            for (int i = 0; i < G2->size(); i++) {
                G2->data()[i] += Gtemp[a]->data()[i];
            }
        }
        return G2;
    }
}

std::vector<std::shared_ptr<Tensor> > IntBox::exchangeGradAdv(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist12,
    const std::shared_ptr<PairList>& pairlist34,
    const std::shared_ptr<Tensor>& D13,
    const std::shared_ptr<Tensor>& D24,
    bool D13symm,
    bool D24symm,
    bool Dsame,
    float thresp,
    float thredp,
    const std::vector<std::shared_ptr<Tensor> >& G1234
    )
{
    return IntBox::exchangeGradAdvCPU(
        resources,
        ewald,
        pairlist12,
        pairlist34,
        D13,
        D24,
        D13symm,
        D24symm,
        Dsame,
        thresp,
        thredp,
        G1234
        );
}

std::shared_ptr<Tensor> IntBox::potentialGrad(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<Tensor>& xyzc,
    const std::shared_ptr<Tensor>& G
    )
{
    return IntBox::potentialGradCPU(
        resources,
        ewald,
        pairlist,
        D,
        xyzc,
        G);
}

std::vector<std::shared_ptr<Tensor> > IntBox::potentialGradAdv(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<Tensor>& xyzc,
    const std::vector<std::shared_ptr<Tensor> >& G12A 
    )
{
    return IntBox::potentialGradAdvCPU(
        resources,
        ewald,
        pairlist,
        D,
        xyzc,
        G12A 
        );
}

std::vector<std::shared_ptr<Tensor> > IntBox::potentialGradAdv2(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<Tensor>& xyzc,
    const std::vector<std::shared_ptr<Tensor> >& G12A 
    )
{
    return IntBox::potentialGradAdv2CPU(
        resources,
        ewald,
        pairlist,
        D,
        xyzc,
        G12A 
        );
}

std::shared_ptr<Tensor> IntBox::ecp(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<ECPBasis>& ecp,
    float threoe,
    const std::shared_ptr<Tensor>& V
    )
{
    if (is_terachem_ready(resources, pairlist, pairlist)) {
        return IntBox::ecpTC(
            resources,
            pairlist,
            ecp,
            threoe,
            V
            );
    } else {
        throw std::runtime_error("IntBox::ecp: ecpCPU is not implemented.");
    }
}

std::shared_ptr<Tensor> IntBox::ecpGrad(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<ECPBasis>& ecp,
    float threoe,
    const std::shared_ptr<Tensor>& G
    )
{
    if (is_terachem_ready(resources, pairlist, pairlist)) {
        return IntBox::ecpGradTC(
            resources,
            pairlist,
            D,
            ecp,
            threoe,
            G
            );
    } else {
        throw std::runtime_error("IntBox::ecpGrad: ecpGradCPU is not implemented.");
    }
}

} // namespace lightspeed
