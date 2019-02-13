#ifndef LS_FUNCTIONAL_IMPL_HPP
#define LS_FUNCTIONAL_IMPL_HPP

#include <lightspeed/dftbox.hpp>

namespace lightspeed { 

/**
 * Class FunctionalImpl provides an abstract interface for classes which
 * actually implement exchange correlation functionals. Functional contains
 * both the overall declaration of the functional in terms of semilocal pieces
 * and hybrid/range-separated exchange, as well as the specific implementation
 * of the semilocal density functional and various partials. 
 **/
class FunctionalImpl {
    
public:

// ==> Abstract Class Superstructure <== //

virtual ~FunctionalImpl() {}

// ==> PIMPL Interface <== //

// => Accessors <= //

// The name of this functional, e.g., B3LYP
virtual std::string name() const = 0;
// A string containing a citation for this functional
virtual std::string citation() const = 0;
// A handy string representation for this functional
virtual std::string string() const = 0;
// The functional type: 0 - LSDA, 1 - GGA, 2 - MGGA
virtual int type() const = 0;
// Does this functional depend on the spin density (almost surely)?
virtual bool has_lsda() const = 0;
// Does this functional depend on the spin density gradient?
virtual bool has_gga() const = 0;
/**
 * Maximum derivative level implemented
 * 0 - exc, sufficient for KS(HF)
 * 1 - vxc, sufficient for KS-DFT + gradients
 * 2 - fxc, sufficient for TD-DFT
 * 3 - kxc, sufficient for TD-DFT + gradients
 **/
virtual int deriv() const = 0;

// => Parameters (Advanced/Cognizent users only, e.g., LS developers) <= //

/**
 * Return the parameter value for string id
 * @param id the parameter identifier
 * @return the current parameter value
 * Throws if functional does not a have a parameter corresponding to ID
 **/
virtual double get_param(const std::string& id) const = 0;
/**
 * Set a parameter value for string id
 * @param id the parameter identifier
 * @param val the value to set
 * @result the parameter value is updated
 * Throws if functional does not a have a parameter corresponding to ID
 **/
virtual void set_param(const std::string& id, double val) = 0;

// => Generalized KS Considerations <= //

// The global hybrid mixing parameter alpha
virtual double alpha() const = 0;
// the LRC range-separation mixing parameter beta
virtual double beta() const = 0;
// The LRC range-separation omega
virtual double omega() const = 0;

// Set the global hybrid mixing parameter alpha (throws if is_alpha_fixed is true)
virtual void set_alpha(double alpha) = 0;
// Set the LRC range-separation mixing parameter beta (throws if is_beta_fixed is true)
virtual void set_beta(double beta) = 0;
// Set the LRC range-separation parameter omega (throws if is_omega_fixed is true)
virtual void set_omega(double omega) = 0;

// Is alpha fixed (true) or can the user set this value as desired (false)?
virtual bool is_alpha_fixed() const = 0;
// Is beta fixed (true) or can the user set this value as desired (false)?
virtual bool is_beta_fixed() const = 0;
// Is omega fixed (true) or can the user set this value as desired (false)?
virtual bool is_omega_fixed() const = 0;

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
virtual std::shared_ptr<Tensor> compute(
    const std::shared_ptr<Tensor>& rho,
    int deriv) const = 0; 

};

} // namespace lightspeed

#endif
