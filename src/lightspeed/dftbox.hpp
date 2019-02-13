#ifndef LS_DFT_BOX_HPP
#define LS_DFT_BOX_HPP

#include <memory>
#include <string>
#include <vector>

namespace lightspeed { 

// Ignore this: I hate PIMPL, but will use it here to cleanly fork between
// LibXC and our own custom functional definitions without making users crazy
// by knowing about such details. -RMP
class FunctionalImpl;

class Tensor;
class ResourceList;
class PairList;
class BeckeGrid;
class HashedGrid;

/**
 * Class Functional provides opaque access to common generalized Kohn-Sham
 * exchange correlation functionals. Functional contains both the overall
 * declaration of the functional in terms of semilocal pieces and
 * hybrid/range-separated exchange, as well as the specific implementation of
 * the semilocal density functional and various partials. 
 *
 * Many functionals (e.g., B3LYP) are fixed in definition from the outset, and
 * parameters such as hybrid exchange are fixed. For these functionals,
 * is_alpha_fixed, is_beta_fixed, and is_omega_fixed are all true and the
 * functional will throw if the user tries to adjust these parameters.
 *
 * Some functionals (e.g., PBE0) allow an adjustable amount of global hybrid
 * exchange, according to:
 * 
 * E_xc = a E_x^HF + (1 - a) E_x^LDA + E_c^LDA 
 *
 * For these functionals, is_alpha_fixed is false, and the user may set alpha
 * at runtime.
 *
 * Some functionals (e.g., wPBE0) allow an adjustable amount of global hybrid
 * exchange and LRC exchange, according to:
 *
 * E_xc = a E_x^HF + b E_x^HF,LR(w) + b E_x^LDA,SR(w) + (1 - a - b) E_x^LDA + E_c^LDA 
 *  
 * For these functionals, is_alpha_fixed, is_beta_fixed, and is_omega_fixed are
 * all false, and the user may set these parameters at runtime.
 **/
class Functional {
    
public:

// => Constructors <= //

/**
 * Build a Functional corresponding to a unique ID name.
 * @param name the unique ID name for the DFT functional, e.g., B3LYP
 * @return the desired functional, ready to construct the energy density and
 *  all available partials
 **/
static 
std::shared_ptr<Functional> build(
    const std::string& name);

// => Accessors <= //

// The name of this functional, e.g., B3LYP
std::string name() const;
// A string containing a citation for this functional
std::string citation() const;
// A handy string representation of this object
std::string string() const;
// The functional type: 0 - LSDA, 1 - GGA, 2 - MGGA
int type() const;
// Does this functional depend on the spin density (almost surely)?
bool has_lsda() const;
// Does this functional depend on the spin density gradient?
bool has_gga() const;
/**
 * Maximum derivative level implemented
 * 0 - exc, sufficient for KS(HF)
 * 1 - vxc, sufficient for KS-DFT + gradients
 * 2 - fxc, sufficient for TD-DFT
 * 3 - kxc, sufficient for TD-DFT + gradients
 **/
int deriv() const;

// => Parameters (Advanced/Cognizent users only, e.g., LS developers) <= //

/**
 * Return the parameter value for string id
 * @param id the parameter identifier
 * @return the current parameter value
 * Throws if functional does not a have a parameter corresponding to ID
 **/
double get_param(const std::string& id) const;
/**
 * Set a parameter value for string id
 * @param id the parameter identifier
 * @param val the value to set
 * @result the parameter value is updated
 * Throws if functional does not a have a parameter corresponding to ID
 **/
void set_param(const std::string& id, double val);

// => Generalized KS Considerations <= //

// The global hybrid mixing parameter alpha (amount of E_x^HF to use)
double alpha() const;
// the LRC range-separation mixing parameter beta (amount of E_x^HF,LR(w) to use)
double beta() const;
// The LRC range-separation omega
double omega() const;

// Set the global hybrid mixing parameter alpha (throws if is_alpha_fixed is true)
void set_alpha(double alpha);
// Set the LRC range-separation mixing parameter beta (throws if is_beta_fixed is true)
void set_beta(double beta);
// Set the LRC range-separation parameter omega (throws if is_omega_fixed is true)
void set_omega(double omega);

// Is alpha fixed (true) or can the user set this value as desired (false)?
bool is_alpha_fixed() const;
// Is beta fixed (true) or can the user set this value as desired (false)?
bool is_beta_fixed() const;
// Is omega fixed (true) or can the user set this value as desired (false)?
bool is_omega_fixed() const;

// => Compute Routines <= //

/**
 * Compute the XC energy density and gradients thereof at a set of npoint
 * points.
 *
 * @param rho an (npoint,T), where T is 2 for LSDA and 5 for GGA (throws if
 * column dimension is incorrect for the LSDA/GGA composition of the
 * functional). The entries along each row are:
 *  LSDA: (rhoa,rhob) 
 *  GGA: (rhoa,rhob,sigmaaa,sigmaab,sigmabb)
 * @param deriv the level of derivatives to compute (throws if not
 * implemented). The entries along each row are:
 *  LDSA/0: (e0,) [col shape: 1]
 *  LSDA/1: (e0,
 *           e1rhoa,e1rhob,) [col shape: 3]
 *  LSDA/2: (e0,
 *           e1rhoa,e1rhob,
 *           e2rhoarhoa,e2rhoarhob,e2rhobrhob,) [col shape: 6]
 *  LSDA/3: (e0,
 *           e1rhoa,e1rhob,
 *           e2rhoarhoa,e2rhoarhob,e2rhobrhob,
 *           e3rhoarhoarhoa,e3rhoarhoarhob,e3rhoarhobrhob,e3rhobrhobrhob,) [col shape: 10]
 *  GGA/0: (e0,) [col shape: 1]
 *  GGA/1: (e0,
 *          e1rhoa,e1rhob,e1sigmaaa,e1sigmaab,e1sigmabb) [col shape: 6]
 *  GGA/2: (e0,
 *          e1rhoa,e1rhob,e1sigmaaa,e1sigmaab,e1sigmabb,
 *          e2rhoarhoa,e2rhoarhob,e2rhobrhob,e2rhoasigmaaa,e2rhoasigmaab,e2rhoasigmabb,e2rhoasigmaaa,e2rhoasigmaab,e2rhoasigmabb,
 *              e2sigmaaasigmaaa,e2sigmaaasigmaab,e2sigmaaasigmabb,e2sigmaabsigmaab,e2sigmaabsigmabb,e2sigmabbsigmabb) [col shape: 21]
 *  GGA/3: (e0,
 *          e1rhoa,e1rhob,e1sigmaaa,e1sigmaab,e1sigmabb,
 *          e2rhoarhoa,e2rhoarhob,e2rhobrhob,e2rhoasigmaaa,e2rhoasigmaab,e2rhoasigmabb,e2rhoasigmaaa,e2rhoasigmaab,e2rhoasigmabb,
 *              e2sigmaaasigmaaa,e2sigmaaasigmaab,e2sigmaaasigmabb,e2sigmaabsigmaab,e2sigmaabsigmabb,e2sigmabbsigmabb,
 *          e3rhoarhoarhoa,e3rhoarhoarhob,e3rhoarhobrhob,e3rhobrhobrhob,
 *              e3rhoarhoasigmaaa,e3rhoarhoasigmaab,e3rhoarhoasigmabb,
 *              e3rhoarhobsigmaaa,e3rhoarhobsigmaab,e3rhoarhobsigmabb,
 *              e3rhobrhobsigmaaa,e3rhobrhobsigmaab,e3rhobrhobsigmabb,
 *              e3rhoasigmaaasigmaaa,e3rhoasigmaaasigmaab,e3rhoasigmaaasigmabb,e3rhoasigmaabsigmaab,e3rhoasigmaabsigmabb,e3rhoasigmabbsigmabb,
 *              e3rhobsigmaaasigmaaa,e3rhobsigmaaasigmaab,e3rhobsigmaaasigmabb,e3rhobsigmaabsigmaab,e3rhobsigmaabsigmabb,e3rhobsigmabbsigmabb,
 *              e3sigmaaasigmaaasigmaaa,e3sigmaaasigmaaasigmaab,e3sigmaaasigmaaasigmabb,e3sigmaaasigmaabsigmaab,e3sigmaaasigmaabsigmabb,e3sigmaaasigmabbsigmabb
 *              e3sigmaabsigmaabsigmaab,e3sigmaabsigmaabsigmabb,e3sigmaabsigmabbsigmabb
 *              e3sigmabbsigmabbsigmabb,) [col shape: 46]
 **/
std::shared_ptr<Tensor> compute(
    const std::shared_ptr<Tensor>& rho,
    int deriv) const;

private:

// The implementation class pointer, which the user can never see.
std::shared_ptr<FunctionalImpl> impl_;

};

/**
 * Class DFTBox merges some of the longer/more-verbose parts of semilocal XC
 * functional operations. An example is forming the RKS potential matrix (the
 * DFTBox::rksPotential method below). This involves forming the density and
 * appropriate gradients thereof on a BeckeGrid quadrature object, computing
 * the XC functional values and derivatives on these points, and then
 * integrating against the basis function values to form the spectral RKS XC
 * potential matrix. At the same time, the semilocal XC energy must be
 * accumulated, due to it being nonlinear. And the user often likes to know how
 * many electrons the DFT grid can see, to check the Becke quadrature accuracy.
 * So we provide a set of routines to efficiently and succinctly perform these
 * operations.
 **/
class DFTBox {

public:

/**
 * Compute the RKS XC potential (the semilocal part). 
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param functional the Functional object defining the semilocal DFT functional
 * @param becke the BeckeGrid defining the molecular quadrature
 * @param hash a HashedGrid corresponding to the BeckeGrid 
 * @param pairlist the list of significant pairs describing basis1 and basis2
 * @param D an (nao1, nao2) Tensor providing the alpha OPDM (same as beta OPDM) 
 * @param thre the threshold for grid collocation 
 * @param [optional] a vector of Tensor objects to accumulate into. If not
 *  provided, this will be allocated. Must have same dimensions as return
 * @return a vector of 2 Tensor objects. The first is a (2,)-shape Tensor
 *  containing the total semilocal XC energy and the integral of the alpha
 *  density over space to check the grid accuracy. The second is a
 *  (nao1,nao2)-shape Tensor with the alpha KS potential matrix accumulated
 **/
static
std::vector<std::shared_ptr<Tensor> > rksPotential(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Functional>& functional,
    const std::shared_ptr<BeckeGrid>& becke,
    const std::shared_ptr<HashedGrid>& hash,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    double thre,
    const std::vector<std::shared_ptr<Tensor> >& V = 
          std::vector<std::shared_ptr<Tensor> >());

/**
 * Compute the nuclear gradient of the RHS XC potential (the semilocal part),
 * including contributions from basis set gradients and from BeckeGrid atomic
 * weight gradients.
 *
 * @param resources the list of CPU and/or GPU resources to use
 * @param functional the Functional object defining the semilocal DFT functional
 * @param becke the BeckeGrid defining the molecular quadrature
 * @param hash a HashedGrid corresponding to the BeckeGrid 
 * @param pairlist the list of significant pairs describing basis1 and basis2
 * @param D an (nao1, nao2) Tensor providing the alpha OPDM (same as beta OPDM) 
 * @param thre the threshold for grid collocation 
 * @param G [optional] a (natom,3)-shape Tensor to accumulate the gradient
 *  into. If not provided, this will be allocated.
 * @return a (natom,3)-shape Tensor with the gradient accumulated 
 **/
static
std::shared_ptr<Tensor> rksGrad(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Functional>& functional,
    const std::shared_ptr<BeckeGrid>& becke,
    const std::shared_ptr<HashedGrid>& hash,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    double thre,
    const std::shared_ptr<Tensor>& G = 
          std::shared_ptr<Tensor>());


#if 0

std::vector<std::shared_ptr<Tensor> > uksPotential(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Functional>& functional,
    const std::shared_ptr<BeckeGrid>& becke,
    const std::shared_ptr<HashedGrid>& hash,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& Da,
    const std::shared_ptr<Tensor>& Db,
    double thre,
    const std::vector<std::shared_ptr<Tensor> >& V = 
          std::vector<std::shared_ptr<Tensor> >());

#endif

};

} // namespace lightspeed

#endif
