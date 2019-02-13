#include "tcintbox.hpp"
#include <lightspeed/resource_list.hpp>
#include <lightspeed/gpu_context.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/basis.hpp>
#include <lightspeed/ecp.hpp>
#include "../../../util/string.hpp"
#include <stdexcept>
#include <cstring>
#include <cmath>

#ifdef HAVE_TERACHEM
#include <intbox/intbox.h> // TeraChem's intbox.h 
#endif

namespace lightspeed {

std::shared_ptr<TCIntBox> TCIntBox::instance_ = std::shared_ptr<TCIntBox>();

std::shared_ptr<TCIntBox> TCIntBox::instance(
    const std::shared_ptr<ResourceList>& resources)
{
    if (!instance_) {
        instance_ = std::shared_ptr<TCIntBox>(new TCIntBox(resources));
    }
    return instance_;
}

#ifdef HAVE_TERACHEM

TCIntBox::TCIntBox(
    const std::shared_ptr<ResourceList>& resources) : 
    resources_(resources)
{
    if (resources_->ngpu() < 1) {
        throw std::runtime_error("No GPUs in this ResourceList object");
    }

    std::vector<int> dev_ids;
    for (auto gpu : resources_->gpus()) {
        dev_ids.push_back(gpu->id());
    }

    IBoxStartup(
        resources_->ngpu(),
        const_cast<int*>(dev_ids.data()),
        //1ULL*1024ULL*1024ULL*1024ULL, // 1 GB pinned memory per GPU
        2ULL*1024ULL*1024ULL*1024ULL, // 2 GB pinned memory per GPU
        512 // 512 byte alignment
        );
}
TCIntBox::~TCIntBox()
{
    //IBoxShutdown(); // TODO: This line fails for some strange reason. It is only being called once
}
std::string TCIntBox::string() const
{
    std::string str = "";

    str += sprintf2( "TCIntBox:\n");
    str += sprintf2( "  TCIntBox is Activated\n");
    str += sprintf2( "\n");
    
    return str;
}
void TCIntBox::set_basis(
    const std::shared_ptr<Basis>& basis,
    float threpair)
{
    if (basis->max_L() > 2) {
        throw std::runtime_error("TCIntBox: Can only use up to D functions");
    }

    // Make sure we can check out the TCIntBox session
    if (basis_working_) {
        throw std::runtime_error("TCIntBox: Only one active TCIntBox session at a time.");
    }
    basis_working_ = true;

    // Have we seen this Basis/threpair problem last time? (avoids forming pairlist in TeraChem's IntBox)
    bool reinitialize = false;
    if (!last_basis_) {
        reinitialize = true;
    } else {
        if (threpair != last_threpair_ || basis != last_basis_) {
            reinitialize = true;
        }
    }
    
    // If TCIntBox is already aware of the problem, leave now
    if (!reinitialize) return;

    // Clear existing problem if it exists
    if (last_basis_) {
        IBoxClear();
    }

    // Set up the new problem
    for (auto shell : basis->shells()) {
        IBoxInitShell(
            shell.L(),
            shell.atomIdx(),
            shell.nprim(),
            shell.es().data(),
            shell.cs().data()); 
    }

    std::shared_ptr<Tensor> XYZ = basis->xyz();
    const double* XYZp = XYZ->data().data();
    IBoxUpdateCoors(threpair,XYZp);

    // Cache the current problem
    last_threpair_ = threpair;
    last_basis_ = basis;
}
void TCIntBox::clear_basis()
{
    basis_working_ = false;
}
void TCIntBox::set_ecp_basis(
    const std::shared_ptr<ECPBasis>& ecp_basis)
{
    if (ecp_basis->max_L() > 3) {
        throw std::runtime_error("IntBox: Can only use up to F ECPs");
    }

    // Make sure we can check out the IntBox session
    if (ecp_basis_working_) {
        throw std::runtime_error("IntBox: Only one active IntBox session at a time.");
    }
    ecp_basis_working_ = true;

    // Have we seen this problem last time?
    bool reinitialize = false;
    if (!last_ecp_basis_) {
        reinitialize = true;
    } else {
        if (ecp_basis != last_ecp_basis_) {
            reinitialize = true;
        }
    }
    
    // If IntBox is already aware of the problem, leave now
    if (!reinitialize) return;

    // Clear existing problem if it exists
    if (last_ecp_basis_) {
        //IntBoxECPBasisClear(); // TODO: There is a definite memory leak in intbox due to this
    }

    // Allocate memory for ECP Basis
    std::vector<int> LprimCnt(5); // ECP_ANGL_TYPES (or segfault)
    std::vector<int> primCnt(5);
    for (const ECPShell& sh : ecp_basis->shells()) {
        if (sh.is_max_L()) {
            LprimCnt[sh.L()] += sh.nprim();
        } else {
            primCnt[sh.L()] += sh.nprim();
        }
    }
    IBoxStartupECP(
        LprimCnt.data(),
        primCnt.data());

    // Place the ECP Basis
    for (const ECPShell& sh : ecp_basis->shells()) {
        for (size_t K = 0; K < sh.nprim(); K++) {
            IBoxInitECP(
                sh.is_max_L(),
                sh.L(),
                sh.atomIdx(),
                sh.ns()[K],
                sh.es()[K],
                sh.cs()[K]);
        }
    }

    // Cache the current problem
    last_ecp_basis_ = ecp_basis;
}
void TCIntBox::clear_ecp_basis()
{
    ecp_basis_working_ = false;
}
void TCIntBox::computeOverlap(
    double* S)
{
    IBoxOverlap(
    S);
}
void TCIntBox::computeOverlapGrad(
    const double* W,
    double* grad)
{
    IBoxOverlapGrad(
    W,
    grad);
}
void TCIntBox::computeAntiSymOverlapGrad(
    const double* W,
    double* grad)
{
    IBoxAntiSymOverlapGrad(
    W,
    grad);
}
void TCIntBox::compute1eVCore(
    double scalfr, 
    double scallr, 
    double omega, 
    float thre,  
    int nPtZ,     
    const double* xyzc, 
    double* hout)
{
    IBox1eVCore(
    scalfr, 
    scallr, 
    omega, 
    thre,  
    nPtZ,     
    xyzc, 
    hout);
}
void TCIntBox::compute1eVGrad(
    double scalfr, 
    double scallr, 
    double omega, 
    float thre,
    int nPtZ,         
    const double* xyzc,   
    const double* P, 
    double* qmgrad,  
    double* ptgrad)
{
    IBox1eVGrad(
    scalfr, 
    scallr, 
    omega, 
    thre,
    nPtZ,         
    xyzc,   
    P, 
    qmgrad,  
    ptgrad);
}
void TCIntBox::compute1eKCore(
    double* hout)
{
    IBox1eKCore(
    hout);
}
void TCIntBox::compute1eKGrad(
    const double* P,
    double* grad)
{
    IBox1eKGrad(
    P,
    grad);
}
void TCIntBox::computeJFockGen(
    float thresp, 
    float thredp,
    double scalfr, 
    double scallr, 
    double omega,
    const double* P, 
    double* fock)
{
    IBoxJFockGen(
    thresp, 
    thredp,
    scalfr, 
    scallr, 
    omega,
    P, 
    fock);
}
void TCIntBox::computeJGradGen(
    float thresp, 
    float thredp,
    double scalfr, 
    double scallr, 
    double omega,
    const double* X,
    const double* Y, 
    double* grad)
{
    IBoxJGradGen(
    thresp, 
    thredp,
    scalfr, 
    scallr, 
    omega,
    X,
    Y, 
    grad);
}
void TCIntBox::computeKFockGen(
    float thresp, 
    float thredp,
    double scalfr, 
    double scallr, 
    double omega,
    const double* P,
    double* fock)
{
    IBoxKFockGen(
    thresp, 
    thredp,
    scalfr, 
    scallr, 
    omega,
    P,
    fock);
}
void TCIntBox::computeKFockSym(
    float thresp, 
    float thredp,
    double scalfr, 
    double scallr, 
    double omega,
    const double* P,
    double* fock)
{
    IBoxKFockSym(
    thresp, 
    thredp,
    scalfr, 
    scallr, 
    omega,
    P,
    fock);
}
void TCIntBox::computeKGradGenGen(
    float thresp, 
    float thredp,
    double scalfr, 
    double scallr, 
    double omega,
    const double* X, 
    const double* Y,
    double* grad)
{
    IBoxKGradGenGen(
    thresp, 
    thredp,
    scalfr, 
    scallr, 
    omega,
    X, 
    Y,
    grad);
}
void TCIntBox::computeKGradGen(
    float thresp, 
    float thredp,
    double scalfr, 
    double scallr, 
    double omega,
    const double* X, 
    double* grad)
{
    IBoxKGradGen(
    thresp, 
    thredp,
    scalfr, 
    scallr, 
    omega,
    X, 
    grad);
}
void TCIntBox::computeKGradSym(
    float thresp, 
    float thredp,
    double scalfr, 
    double scallr, 
    double omega,
    const double* P, 
    double* grad)
{
    IBoxKGradSym(
    thresp, 
    thredp,
    scalfr, 
    scallr, 
    omega,
    P, 
    grad);
}
void TCIntBox::computeECPCore(
    const float thre,
    double scal,
    const double* xyzc,
    double* H
)
{
    IBoxECPCore(
        thre,
        scal,
        xyzc,
        H
    );
}

void TCIntBox::computeECPGrad(
    const float thre,
    double scal,
    const double* xyzc,
    const double* P,
    double* qmgrad,
    double* ptgrad
)
{
    IBoxECPGrad(
        thre,
        scal,
        xyzc,
        P,
        qmgrad,
        ptgrad
    );
}

#else

TCIntBox::TCIntBox(
    const std::shared_ptr<ResourceList>& resources) :
    resources_(resources)
{
}
TCIntBox::~TCIntBox()
{
}
std::string TCIntBox::string() const
{
    std::string str = "";

    str += sprintf2( "TCIntBox:\n");
    str += sprintf2( "  TCIntBox is Deactivated\n");
    str += sprintf2( "\n");
    
    return str;
}
void TCIntBox::set_basis(
    const std::shared_ptr<Basis>& basis,
    float threpair)
{
    throw std::runtime_error("TCIntBox: Lightspeed was not compiled with TCIntBox.");
}
void TCIntBox::clear_basis()
{
    throw std::runtime_error("TCIntBox: Lightspeed was not compiled with TCIntBox.");
}
void TCIntBox::set_ecp_basis(
    const std::shared_ptr<ECPBasis>& basis)
{
    throw std::runtime_error("TCIntBox: Lightspeed was not compiled with TCIntBox.");
}
void TCIntBox::clear_ecp_basis()
{
    throw std::runtime_error("TCIntBox: Lightspeed was not compiled with TCIntBox.");
}
void TCIntBox::computeOverlap(
    double* S)
{
    throw std::runtime_error("TCIntBox: Lightspeed was not compiled with TCIntBox.");
}
void TCIntBox::computeOverlapGrad(
    const double* W,
    double* grad)
{
    throw std::runtime_error("TCIntBox: Lightspeed was not compiled with TCIntBox.");
}
void TCIntBox::computeAntiSymOverlapGrad(
    const double* W,
    double* grad)
{
    throw std::runtime_error("TCIntBox: Lightspeed was not compiled with TCIntBox.");
}
void TCIntBox::compute1eVCore(
    double scalfr, 
    double scallr, 
    double omega, 
    float thre,  
    int nPtZ,     
    const double* xyzc, 
    double* hout)
{
    throw std::runtime_error("TCIntBox: Lightspeed was not compiled with TCIntBox.");
}
void TCIntBox::compute1eVGrad(
    double scalfr, 
    double scallr, 
    double omega, 
    float thre,
    int nPtZ,         
    const double* xyzc,   
    const double* P, 
    double* qmgrad,  
    double* ptgrad)
{
    throw std::runtime_error("TCIntBox: Lightspeed was not compiled with TCIntBox.");
}
void TCIntBox::compute1eKCore(
    double* hout)
{
    throw std::runtime_error("TCIntBox: Lightspeed was not compiled with TCIntBox.");
}
void TCIntBox::compute1eKGrad(
    const double* P,
    double* grad)
{
    throw std::runtime_error("TCIntBox: Lightspeed was not compiled with TCIntBox.");
}
void TCIntBox::computeJFockGen(
    float thresp, 
    float thredp,
    double scalfr, 
    double scallr, 
    double omega,
    const double* P, 
    double* fock)
{
    throw std::runtime_error("TCIntBox: Lightspeed was not compiled with TCIntBox.");
}
void TCIntBox::computeJGradGen(
    float thresp, 
    float thredp,
    double scalfr, 
    double scallr, 
    double omega,
    const double* X,
    const double* Y, 
    double* grad)
{
    throw std::runtime_error("TCIntBox: Lightspeed was not compiled with TCIntBox.");
}
void TCIntBox::computeKFockGen(
    float thresp, 
    float thredp,
    double scalfr, 
    double scallr, 
    double omega,
    const double* P,
    double* fock)
{
    throw std::runtime_error("TCIntBox: Lightspeed was not compiled with TCIntBox.");
}
void TCIntBox::computeKFockSym(
    float thresp, 
    float thredp,
    double scalfr, 
    double scallr, 
    double omega,
    const double* P,
    double* fock)
{
    throw std::runtime_error("TCIntBox: Lightspeed was not compiled with TCIntBox.");
}
void TCIntBox::computeKGradGenGen(
    float thresp, 
    float thredp,
    double scalfr, 
    double scallr, 
    double omega,
    const double* X, 
    const double* Y,
    double* grad)
{
    throw std::runtime_error("TCIntBox: Lightspeed was not compiled with TCIntBox.");
}
void TCIntBox::computeKGradGen(
    float thresp, 
    float thredp,
    double scalfr, 
    double scallr, 
    double omega,
    const double* X, 
    double* grad)
{
    throw std::runtime_error("TCIntBox: Lightspeed was not compiled with TCIntBox.");
}
void TCIntBox::computeKGradSym(
    float thresp, 
    float thredp,
    double scalfr, 
    double scallr, 
    double omega,
    const double* P, 
    double* grad)
{
    throw std::runtime_error("TCIntBox: Lightspeed was not compiled with TCIntBox.");
}
void TCIntBox::computeECPCore(
    const float thre,
    double scal,
    const double* xyzc,
    double* H
)
{
    throw std::runtime_error("TCIntBox: Lightspeed was not compiled with TCIntBox.");

}
void TCIntBox::computeECPGrad(
    const float thre,
    double scal,
    const double* xyzc,
    const double* P,
    double* qmgrad,
    double* ptgrad)
{
    throw std::runtime_error("TCIntBox: Lightspeed was not compiled with TCIntBox.");
}
#endif

} // namespace lightspeed
