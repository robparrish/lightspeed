#ifndef LS_LR_H
#define LS_LR_H

#include <vector>
#include <map>
#include <set>
#include <string>
#include <memory>

namespace lightspeed { 

class Tensor;
class ResourceList;
class Ewald;
class Basis;
class PairList;
class BeckeGrid;

class LaplaceDenom {

public:

/**
 * Compute a Laplace-factorized low-rank representation of an orbital energy
 * denominator:
 *
 *  D_ijab = 1 / (e_a + e_b - e_i - e_j) 
 *      \approx \sum_w t_i^w t_j^w u_a^w u_b^w
 *
 * This flavor has two hole (occupied) orbitals labeled i and j, and two
 * particle (virtual) orbitals labeled a and b. Such an energy denominator
 * occurs in many places in dynamical correlation methods, such as in MP2,
 * CCSD, etc.
 *
 * @param eps_occ (no,) Tensor with the occupied orbital energies
 * @param eps_vir (nv,) Tensor with the virtual orbital energies
 * @param max_error maximum tolerable absolute error in the representation
 * @return ret - vector of size 2 of Tensor. 
 *  ret[0] is a (nw, no) Tensor (t above)
 *  ret[1] is a (nw, nv) Tensor (u above)
 *
 * The occupied energy levels must be strictly less than the virtual energy
 * levels, else an error will be thrown.
 *
 * In this flavor, the number of Laplace points is computed to provide the
 * smallest possible Laplace factorization that satisfies the error tolerance.
 **/
static 
std::vector<std::shared_ptr<Tensor>> o2v2_denom(
    const std::shared_ptr<Tensor>& eps_occ,
    const std::shared_ptr<Tensor>& eps_vir,
    double max_error
    );

/**
 * Compute a Laplace-factorized low-rank representation of an orbital energy
 * denominator:
 *
 *  D_ijkabc = 1 / (e_a + e_b + e_c - e_i - e_j - e_k) 
 *      \approx \sum_w t_i^w t_j^w t_k^w u_a^w u_b^w u_c^w
 *
 * This flavor has three hole (occupied) orbitals labeled i, j and k, and three
 * particle (virtual) orbitals labeled a, b and c. Such an energy denominator
 * occurs in many places in perturbative triples expansions [e.g., CCSD(T)].
 *
 * @param eps_occ (no,) Tensor with the occupied orbital energies
 * @param eps_vir (nv,) Tensor with the virtual orbital energies
 * @param max_error maximum tolerable absolute error in the representation
 * @return ret - vector of size 2 of Tensor. 
 *  ret[0] is a (nw, no) Tensor (t above)
 *  ret[1] is a (nw, nv) Tensor (u above)
 *
 * The occupied energy levels must be strictly less than the virtual energy
 * levels, else an error will be thrown.
 *
 * In this flavor, the number of Laplace points is computed to provide the
 * smallest possible Laplace factorization that satisfies the error tolerance.
 **/
static 
std::vector<std::shared_ptr<Tensor>> o3v3_denom(
    const std::shared_ptr<Tensor>& eps_occ,
    const std::shared_ptr<Tensor>& eps_vir,
    double max_error
    );

/**
 * Compute a Laplace-factorized low-rank representation of an orbital energy
 * denominator:
 *
 *  D_ijk...abc... = 1 / (e_a + e_b + e_c + ... - e_i - e_j - e_k - ...) 
 *      \approx \sum_w t_i^w t_j^w t_k^w ... u_a^w u_b^w u_c^w ...
 *
 * This flavor has n hole (occupied) orbitals labeled i, j, k,..., and n
 * particle (virtual) orbitals labeled a, b, c.... Such an energy denominator
 * occurs in many places in perturbation theory.
 *
 * @param eps_occ (no,) Tensor with the occupied orbital energies
 * @param eps_vir (nv,) Tensor with the virtual orbital energies
 * @param max_error maximum tolerable absolute error in the representation
 * @param order_denom the number of hole or particle indices of the orbital
 *  energy denominator (e.g., 2 for MP2, 3 for (T), etc).
 * @return ret - vector of size 2 of Tensor. 
 *  ret[0] is a (nw, no) Tensor (t above)
 *  ret[1] is a (nw, nv) Tensor (u above)
 *
 * The occupied energy levels must be strictly less than the virtual energy
 * levels, else an error will be thrown.
 *
 * In this flavor, the number of Laplace points is computed to provide the
 * smallest possible Laplace factorization that satisfies the error tolerance.
 **/
static 
std::vector<std::shared_ptr<Tensor>> onvn_denom(
    const std::shared_ptr<Tensor>& eps_occ,
    const std::shared_ptr<Tensor>& eps_vir,
    double max_error,
    int order_denom
    );

private:

// => Implementation Details - most users should ignore this <= //

struct RKThresh{
    std::string R;
	std::string k;
	double thresh;

    bool operator< (const RKThresh &) const;
};

typedef std::map<double, std::set<RKThresh>> TabelleMap;
typedef std::map<std::string, std::vector<double>> ParamMap;
typedef std::pair< std::vector<double>, std::vector<double> > Params;

const TabelleMap& tabelle() const { return tabelle_; }
const ParamMap& omega() const { return omega_; }
const ParamMap& alpha() const { return alpha_; }

// => helper methods <= //
static std::vector< double > get_eps_min_max(
    const std::shared_ptr<Tensor>& eps_occ,
    const std::shared_ptr<Tensor>& eps_vir    
    );

static Params get_omega_alphas(
    double R, 
    double sc, 
    double max_error
    );

static void fill_tau(
    std::shared_ptr<Tensor>& tau, 
    const std::shared_ptr<Tensor>& eps, 
    const Params params, 
    bool is_vir, 
    int order_denom
    );

// => data tables <= //
TabelleMap tabelle_;
static TabelleMap build_tabelle();
ParamMap omega_;
ParamMap alpha_;
static ParamMap build_omega();
static ParamMap build_alpha();

// => Singleton design pattern <= //
  
static std::shared_ptr<LaplaceDenom> instance() {
    return LaplaceDenom::instance_;
}

static std::shared_ptr<LaplaceDenom> instance_;

/**
 * Constructor, fills in data from source files. Users of this class
 * should use the singleton instance instead.
 **/
LaplaceDenom();

};

class DF {

public:

/**
 * Compute the DF metric matrix (A|B)
 *
 * OMP threaded CPU algorithm.
 *
 * @param resources ResourceList to use
 * @param ewald interaction operator to use for O(r_12) in metric
 * @param auxiliary auxiliary basis set
 * @return (naux, naux) Tensor with metric
 **/
static
std::shared_ptr<Tensor> metric(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<Basis>& auxiliary
    );

/**
 * Compute the AO DF integrals (Q|pq) = (Q|A)^-1/2 (A|pq)
 *
 * OMP threaded CPU algorithm. A 2x speedup in integral generation is achieved
 * if pairlist is symmetric.
 *
 * @param resources ResourceList to use
 * @param ewald interaction operator to use for O(r_12) in metric
 * @param auxiliary auxiliary basis set
 * @param pairlist pairlist information on p and q basis sets
 * @param thre_df_cond relative condition number used in pseudoinversion of DF
 *  metric.
 * @return (naux, naop, naoq) Tensor with symmetrically-fitted integrals
 **/
static
std::shared_ptr<Tensor> ao_df(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<Basis>& auxiliary,
    const std::shared_ptr<PairList>& pairlist,
    double thre_df_cond=1.0E-12
    );

/**
 * Compute the MO DF integrals (Q|ab) = (Q|A)^-1/2 (A|pq) CA_pa CB_qb. The AO
 * integrals are computed once and re-used for as many CA/CB tasks as are
 * requested.
 *
 * OMP threaded CPU algorithm. A 2x speedup in integral generation is achieved
 * if pairlist is symmetric.
 *
 * The first transformation is performed for CA, then for CB. Therefore, it is
 * recommended that norbA <= norbB for best performance in AO-to-MO transform.
 *
 * @param resources ResourceList to use
 * @param ewald interaction operator to use for O(r_12) in metric
 * @param auxiliary auxiliary basis set
 * @param pairlist pairlist information on p and q basis sets
 * @param CAs list of (naop, nmoa) Tensor with MO coefficients
 * @param CBs list of (naop, nmob) Tensor with MO coefficients
 * @param thre_df_cond relative condition number used in pseudoinversion of DF
 *  metric.
 * @return {(naux, nmoa, nmob)} Tensor with symmetrically-fitted MO integrals
 *  for each CA/CB task.
 **/
static 
std::vector<std::shared_ptr<Tensor>> mo_df(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<Basis>& auxiliary,
    const std::shared_ptr<PairList>& pairlist,
    const std::vector<std::shared_ptr<Tensor>>& CAs,
    const std::vector<std::shared_ptr<Tensor>>& CBs,
    double thre_df_cond=1.0E-12
    );

};

class THC {

public:

/**
 * Compute the MO-basis THC grid collocation matrix
 *
 * X_a^P = w_P^pow \phi_p (\vec r_P) C_pa
 *
 * @param resources ResourceList to use
 * @param basis the AO basis set to collocate
 * @param grid the BeckeGrid defining THC grid points and weights
 * @param C (nao, nmo) Tensor with MO coefficients
 * @param thre_grid grid cutoff for orbitals
 * @param weight_pow power of the grid weights in collocation matrix
 * @return (ngrid, nmo) Tensor with collocation matrix
 **/
static
std::shared_ptr<Tensor> mo_thc_X(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Basis>& basis,
    const std::shared_ptr<BeckeGrid>& grid,
    const std::shared_ptr<Tensor>& C,
    double thre_grid=1.0E-14,
    double weight_pow=0.25
    );

/**
 * Compute the MO-basis DF-THC V operator (square root of Z).
 *
 * V_A^P = (A|B)^-1/2 (B|pq) CA_pa CB_qb XA_a^Q XB_b^Q [S_PQ]^-1
 *
 * where,
 *
 * S_PQ = [XA_a^P XA_a^Q] [XB_b^P XB_b^Q]
 *
 * The traditional Z operator can be formed as,
 *
 * Z^PQ = V_A^P V_A^Q
 *
 * with appropriate left/right V operators for the bra/ket indices of the
 * chemists' ERIs.
 *
 * @param resources ResourceList to use
 * @param ewald interaction operator to use for O(r_12) in metric
 * @param auxiliary auxiliary basis set
 * @param pairlist pairlist information on p and q basis sets
 * @param CAs list of (naop, nmoa) Tensor with MO coefficients
 * @param CBs list of (naop, nmob) Tensor with MO coefficients
 * @param XAs list of (ngrid, nmoa) Tensor with MO THC collocations
 * @param XBs list of (ngrid, nmob) Tensor with MO THC collocations
 * @param thre_df_cond relative condition number used in pseudoinversion of DF
 *  metric.
 * @param thre_thc_cond relative condition number used in pseudoinversion of THC
 *  metric.
 * @return {(naux,ngrid)} Tensor with V operator for each CA/CB//XA/XB task
 **/ 
static 
std::vector<std::shared_ptr<Tensor>> mo_thc_V(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<Basis>& auxiliary,
    const std::shared_ptr<PairList>& pairlist,
    const std::vector<std::shared_ptr<Tensor>>& CAs,
    const std::vector<std::shared_ptr<Tensor>>& CBs,
    const std::vector<std::shared_ptr<Tensor>>& XAs,
    const std::vector<std::shared_ptr<Tensor>>& XBs,
    double thre_df_cond=1.0E-12,
    double thre_thc_cond=1.0E-10
    );

};

} // namespace lightspeed

#endif
