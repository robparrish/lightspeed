#ifndef LS_TCINTBOX_HPP
#define LS_TCINTBOX_HPP

#include <cstddef>
#include <vector>
#include <cstdio>
#include <memory>

namespace lightspeed {

class ResourceList;
class Basis;
class ECPBasis;
class Tensor;

class TCIntBox {

public:

~TCIntBox();
std::string string() const;

std::shared_ptr<ResourceList> resources() const { return resources_; }

protected:

TCIntBox(
    const std::shared_ptr<ResourceList>& resources);

std::shared_ptr<ResourceList> resources_;

public:

void set_basis(
    const std::shared_ptr<Basis>& basis,
    float threpair);
void clear_basis();
bool basis_working() const { return basis_working_; }

void set_ecp_basis(
    const std::shared_ptr<ECPBasis>& ecp_basis);
void clear_ecp_basis();
bool ecp_basis_working() const { return ecp_basis_working_; }

protected:
    
bool basis_working_ = false;
// Last basis given to TCIntBox
std::shared_ptr<Basis> last_basis_;
float last_threpair_ = 0.0;
bool ecp_basis_working_ = false;
// Last ecp_basis given to TCIntBox
std::shared_ptr<ECPBasis> last_ecp_basis_;

public:

static std::shared_ptr<TCIntBox> instance(
    const std::shared_ptr<ResourceList>& resources);

static void clear_instance() { instance_ = nullptr; }

protected:

static std::shared_ptr<TCIntBox> instance_;

public:
      
// > Direct IntBox Wrappers (See IntBox's intbox.h) < // 

void computeOverlap(
    double* S);

void computeOverlapGrad(
    const double* W,
    double* grad);

void computeAntiSymOverlapGrad(
    const double* W,
    double* grad);

void compute1eVCore(
    double scalfr, 
    double scallr, 
    double omega, 
    float thre,  
    int nPtZ,     
    const double* xyzc, 
    double* hout);

void compute1eVGrad(
    double scalfr, 
    double scallr, 
    double omega, 
    float thre,
    int nPtZ,         
    const double* xyzc,   
    const double* P, 
    double* qmgrad,  
    double* ptgrad);

void compute1eKCore(
    double* hout);

void compute1eKGrad(
    const double* P,
    double* grad);

void computeJFockGen(
    float thresp, 
    float thredp,
    double scalfr, 
    double scallr, 
    double omega,
    const double* P, 
    double* fock);

void computeJGradGen(
    float thresp, 
    float thredp,
    double scalfr, 
    double scallr, 
    double omega,
    const double* X,
    const double* Y, 
    double* grad);

void computeKFockGen(
    float thresp, 
    float thredp,
    double scalfr, 
    double scallr, 
    double omega,
    const double* P,
    double* fock);

void computeKFockSym(
    float thresp, 
    float thredp,
    double scalfr, 
    double scallr, 
    double omega,
    const double* P,
    double* fock);

void computeKGradGenGen(
    float thresp, 
    float thredp,
    double scalfr, 
    double scallr, 
    double omega,
    const double* X, 
    const double* Y,
    double* grad);

void computeKGradGen(
    float thresp, 
    float thredp,
    double scalfr, 
    double scallr, 
    double omega,
    const double* X, 
    double* grad);

void computeKGradSym(
    float thresp, 
    float thredp,
    double scalfr, 
    double scallr, 
    double omega,
    const double* P, 
    double* grad);

void computeECPCore(const float thredp,
    const double scal,
    const double* xyzc,
    double* H);

void computeECPGrad(const float thredp,
    const double scal,
    const double* xyzc,
    const double* P,
    double* qmgrad,
    double* ptgrad);

};

} // namespace lightspeed

#endif

