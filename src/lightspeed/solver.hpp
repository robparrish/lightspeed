#ifndef LS_SOLVER_HPP    
#define LS_SOLVER_HPP    

#include <memory>
#include <vector>
#include <string>

namespace lightspeed {

class Tensor;

/**
 * Class Storage is an abstract representation for an element of a linear
 * vector space in real double precision. Storage is designed to provide a
 * firewall between the details of many Krylov-type vector space solvers (DIIS,
 * Davidson, etc), and the data type of user applications (Tensor,
 * vector<Tensor>, other sparse storage types, etc). Solver also handles
 * disk/core operations in a manner that is transparent to the developer of the
 * Solver class. It is assumed that at least two copies of a Solver object can
 * fit in core memory, so that we may perform linear algebra and load/store
 * routines without blocking.
 *
 * Storage provides a vector-space API, with the following features:
 *  - initialization (zeros_like)
 *  - scaling (scale)
 *  - dot products (dot)
 *  - vector addition (axpby)
 * Solver-type classes (DIIS, Davidson, etc), should call *ONLY* these
 * functions.
 *
 * Storage provides a utility function API to convert to/from Storage and user
 * data types such as Tensor and vector<shared_ptr<Tensor> >, doing disk
 * conversion on-the-fly if requested. User codes should call *ONLY* these
 * functions.
 *
 * Storage also exposes some of the details about the storage data (including
 * constructor and ways to place data on core and disk). This is intended
 * *ONLY* for users wishing to use the various Solver classes with new data
 * types (such as sparse matrices). This is accomplished by writing new Storage
 * to/from utility conversion routines for the new user data types.
 **/
class Storage {

public:

// ==> Vector-Space API (Solver classes should *ONLY* call these) <== //

/**
 * Return a new Storage object which is the same size as x, and in the same
 * disk/core state.
 * @param x a reference Storage object to take the size and file state from
 *  (input).
 * @return y a new Storage object initialized to 0, which has the same size and
 *  disk/core state as x. If x is disk, y will be disk with a new random
 *  filename.
 **/
static 
std::shared_ptr<Storage> zeros_like(
    const std::shared_ptr<Storage>& x);

/**
 * Scale the Storage object x by the scalar a
 * @param x the Storage to scale (input/output)
 * @param a the scale value
 * @result the data of x is scaled in place
 * @return x to allow chaining
 **/
static
std::shared_ptr<Storage> scale(
    const std::shared_ptr<Storage>& x,
    double a);

/**
 * Return the inner product <x|y> = \sum_I x_I y_I
 * @param x the first Storage object (input)
 * @param y the second Storage object (input)
 * @return the inner product scalar
 *
 * Throws if x and y are not the same size
 **/
static  
double dot( 
    const std::shared_ptr<Storage>& x,
    const std::shared_ptr<Storage>& y);

/**
 * Compute the linear combination y_I <- a x_I + b y_I 
 * @param x the first Storage object (input)
 * @param y the second Storage object (input/output)
 * @param a the scale of x
 * @param b the scale of y
 * @return y to allow chaining
 *
 * Throws if x and y are not the same size
 **/
static 
std::shared_ptr<Storage> axpby(
    const std::shared_ptr<Storage>& x,
    const std::shared_ptr<Storage>& y,
    double a,
    double b);
/**
 * Produce one  copy of a preexistent  object
 * @param other preexistent Storage object (input)
  * @return copy of the Storage object
 **/
static 
std::shared_ptr<Storage> copy(
    const std::shared_ptr<Storage>& other);


public:

// ==> Utility Conversion API (User codes should *ONLY* call these) <== //

/**
 * Convert Tensor T to Storage
 * @param T the Tensor to cast to Storage
 * @param use_disk build the Storage object on disk (true) or core (false)
 * @return the Storage version of T
 **/
static
std::shared_ptr<Storage> from_tensor(
    const std::shared_ptr<Tensor>& T,
    bool use_disk);

/**
 * Convert Storage S to Tensor T (overwrites T)
 * @param S the Storage object to convert (input)
 * @param T the Tensor object to write to (output)
 * @return T is overwritten with the data in S, and T is returned to allow
 *  chaining
 *
 * Throws if S.size() != T.size()
 **/
static 
std::shared_ptr<Tensor> to_tensor(
    const std::shared_ptr<Storage>& S,
    const std::shared_ptr<Tensor>& T);

/**
 * Convert vector<Tensor> T to Storage
 * @param T the vector<Tensor> to cast to Storage
 * @param use_disk build the Storage object on disk (true) or core (false)
 * @return the Storage version of T
 **/
static
std::shared_ptr<Storage> from_tensor_vec(
    const std::vector<std::shared_ptr<Tensor> >& T,
    bool use_disk);

/**
 * Convert Storage S to vector<Tensor> T (overwrites T)
 * @param S the Storage object to convert (input)
 * @param T the vector<Tensor> object to write to (output)
 * @return T is overwritten with the data in S, and T is returned to allow
 *  chaining
 *
 * Throws if S.size() != sum(T.size())
 **/
static 
std::vector<std::shared_ptr<Tensor> > to_tensor_vec(
    const std::shared_ptr<Storage>& S,
    const std::vector<std::shared_ptr<Tensor> >& T);

public:

// ==> Implementation Details <== //

/**
 * These are exposed *ONLY* so users with new data types (e.g., sparse storage
 * types) can build their own Utility Conversion API functions, which they
 * should use from then on in all client codes using the new data type.
 **/

/**
 * Constructor, builds a zeroed Storage object on core or disk
 * @param size the size (total number of double elements) of the Storage object
 * @param is_disk use disk (true) or core (false) storage?
 * @result the object is created and returned.
 *  If is_disk:
 *    core_data_.size() == 0 (not used)
 *    disk_data_ is opened in binary read/write mode to next_filename() and
 *    zeros are written to fill the data file out to size*sizeof(double).
 *  If is_core:
 *    disk_data_ == NULL (not used)
 *    core_data_.size() == size, and is initialized to all zeros.
 **/ 
Storage(
    size_t size,
    bool is_disk);

// Destructor, removes disk file if is_disk
~Storage();

// => Accessors <= //

// The size of this Storage object
size_t size() const { return size_; }
// Is this Storage object on core or disk?
bool is_disk() const { return is_disk_; }
// A handy string representation of the object
std::string string() const;

// => Data Accessors <= //

// A reference to the core data vector, if !is_disk. Throws if is_disk
std::vector<double>& core_data();

// A core version of the data, read from file if is_disk. Throws if !is_disk
std::vector<double> disk_data();
// Saves data to file if is_disk. Throw if !is_disk or if data.size() != size
void to_disk_data(const std::vector<double>& data);

private:

// Input fields
size_t size_;
bool is_disk_;

// The core representation of the data (not used if is_disk)
std::vector<double> core_data_;
// The disk representation of the data (not used if !is_disk)
FILE* disk_data_;
// The filename corresponding to disk_data_ (not used if !is_disk)
// Used in Storage::string to tell the user where this data lives, for
// debugging.
std::string filename_;

// ==> Scratch Path/Random Filename Utilities <== //

public:

/**
 * Set a user-writable location (desirably on a fast local disk with plenty of
 * space) to place disk-based Storage objects.
 * @param the path to a user-writable scratch directory
 **/
static
void set_scratch_path(const std::string& scratch_path) { scratch_path__ = scratch_path; }
    
/**
 * Where are scratch files currently being written?
 * @return the current value of scratch_path__
 **/
static 
std::string scratch_path() { return scratch_path__; }

private:

/**
 * Get a unique scratch file name, suitable for opening with
 * fopen(filename.c_str(),"wb+"). Should be protected by PID and a statically
 * incremented counter.
 *  
 * @return the unique scratch file name
 *
 * TODO: This should be made to be thread-safe (not required in first pass, but
 * leave the TODO in this case). This means that the static file counter must
 * be mutex-protected.
 **/
static
std::string next_scratch_file();

// The common location to place scratch files (default to "./")
static std::string scratch_path__;
  
};

/**!
 * Class DIIS provides a uniform and simple interface for performing Pulay's
 * Direct Inversion of the Iterative Subspace (DIIS) optimization technique.
 * DIIS is a general quasi-Newton method to solve the problem G(S) = 0, where G
 * is some error vector corresponding to some state vector S. 
 *
 * Note: This DIIS object uses the worst-error removal policy, and performs
 * balancing of the DIIS B matrix to attenuate condition problems near
 * convergence.
 *
 * => Example Use (UHF) <=
 *
 * Description:
 *  Use DIIS to extrapolate the Fa and Fb Fock matrices with respect to the
 *  error metrics Ga and Gb, as is often needed in UHF. Uses the vector<Tensor>
 *  user datatype and converts to disk-based Storage to interact with DIIS.
 *
 * Definitions:
 *  Fa/Fb - alpha and beta Fock matrices (Tensor)
 *  Ga/Gb - alpha and beta orbital gradients (Tensor) [usually X(FDS-SDF)X']
 *
 * Code:
 *
 *  // Create a DIIS object with a subspace of size 6.
 *  DIIS diis(6);
 *  for (int iter = 0; iter < 50; iter++) {
 *      // Build Fa,Fb (state vectors)
 *      ...
 *      // Build Ga,Gb (error vectors)
 *      ...
 *      // Add the state and error vectors and extrapolate
 *      // (1) Convert state/error vectors to Storage (using disk)
 *      std::shared_ptr<Storage> SF = Storage::from_tensor_vec({Fa,Fb},true);
 *      std::shared_ptr<Storage> SG = Storage::from_tensor_vec({Ga,Gb},true);
 *      // (2) Perform the DIIS extrapolation (updates the state of the DIIS object)
 *      std::shared_ptr<Storage> SF2 = diis.iterate(SF,SG);
 *      // (3) Extract the extrapolated state vector into Tensor format
 *      Storage::to_tensor_vec(SF2,{Fa,Fb});
 *      // Use the extrapolated Fock matrices to get new orbitals
 *      ...
 *  }
 *
 **/
class DIIS {

public:

/**
 * Construct an empty DIIS object
 * @param max_vectors the maximum number of vectors to hold - vectors added
 *  after this point will overwrite existing vectors according to the
 *  worst-error removal policy
 **/
DIIS(
    size_t max_vectors);         
    
/**
 * Add a state/error vector pair to the DIIS object's state, and return the
 * extrapolated state vector.
 *  
 * @param state the current state vector (input, reference kept)
 * @param state the current error vector (input, reference kept)
 * @return the extrapolated state vector (newly created)
 *
 * NOTE: As Storage is a utility class to be used only in communicating with
 * Solver classes, we do not copy the data out of state/error. Instead, only
 * references are kept. Therefore, writing into state/error Storage objects
 * after this call will cause undefined behavior in the DIIS iterations. The
 * advice is to convert to a Storage intermediate immediately before this call,
 * and then never refer to these Storage intermediates again. This will
 * effectively transfer ownership of these Storage objects to DIIS.
 **/
std::shared_ptr<Storage> iterate(
    const std::shared_ptr<Storage>& state,
    const std::shared_ptr<Storage>& error);

// => Accessors <= //

// The maximum number of vectors in this DIIS object
size_t max_vectors() const { return max_vectors_; }
// The current number of vectors in this DIIS object
size_t current_vectors() const { return state_vecs_.size(); }
// A handy string representation of this DIIS object
std::string string() const;

private:

// The input maximum number of DIIS vectors
size_t max_vectors_;

// The DIIS state vector history
std::vector<std::shared_ptr<Storage> > state_vecs_;
// The DIIS error vector history
std::vector<std::shared_ptr<Storage> > error_vecs_;

// (max_vectors,max_vectors) Tensor to cache <E_I|E_J> error inner products 
// to avoid quadratic recomputation as history is accumulated.
std::shared_ptr<Tensor> E_;

// Utility function to compute the DIIS extrapolation coefficients from the
// current E matrix, using pseudoinversion and balancing.
std::shared_ptr<Tensor> compute_coefs() const;

};

/**
 * Class Davidson uses the (balanced) Davidson-Liu simultaneous expansion
 * method to iteratively solve for the lowest nstate eigenpairs of a large
 * generalized Hermitian eigenvalue problem:
 *
 *  A_IJ C_JK = S_IJ C_JK E_K : C_IK S_IJ C_JL = I_KL
 *
 * Here A_IJ is the "stiffness matrix" (typically the Hamiltonian), S_IJ is the
 * "metric matrix" (often the identity matrix), C_JK are the eigenvectors, and
 * E_K are the eigenvalues.
 *
 * In this implementation of Davidson-Liu, the user constructs a Davidson
 * object with a target number of eigenpairs (nstate), a maximum number of
 * Krylov space history vectors to store (nmax), and a desired residual 2-norm
 * convergence tolerance (convergence). The user then iteratively calls the
 * "add_vectors" method to inform the Davidson object about a new search
 * direction b_I, it's stiffness matrix product Ab_I = A_IJ b_J (often called
 * the "sigma" vector) and it's metric matrix product Sb_I = S_IJ b_J.
 *
 * At this point, the Davidson object solves the eigenproblem in the basis of
 * all search directions in its iterative history. The estimated eigenvalues
 * (the "Ritz vectors") are cached in evecs, and the estimated eigenvalues (the
 * "Ritz values") are cached in evals. The residuals and their 2-norm values
 * are cached in rs and rnorms, respectively.
 *
 * At this point, technically any new vectors can be used to expand the Krylov
 * space, but random choices will usually yield very poor convergence. Under
 * Davidson's criterion, a good choice for search vectors are preconditioned
 * residuals,
 *
 *  d_I = -(\bar A_IJ - E)^{-1} r_J
 *
 * Where \bar A_IJ is a preconditioner matrix (typically the exact or
 * approximate diagonal of A_IJ). In order to allow the user to control the
 * details of the preconditioner, the Davidson object suggests that the user
 * precondition the vectors in gs using the Ritz values in hs to generate the
 * new search directions. The fields gs/hs are different from rs/evals because
 * some of the eigenpairs might be already converged (and thus require no new
 * search directions).
 * 
 * The user now computes the preconditioned residuals using gs/hs and their own
 * scheme for the preconditioner matrix. Next, the user passes the
 * preconditioned residuals to Davidson via the "add_preconditioned" routine.
 * Finally, the user asks the Davidson for the next search direction task (new
 * vectors to compute the stiffness and metric products for) by accessing the
 * cs field. Usually cs simply echoes the preconditioned residuals that the
 * user passed in to add_preconditioned. However, if the Krylov space has
 * exceeded nmax vectors, the Davidson object will collapse the Krylov space to
 * the current eigenvectors, and these will be provided to the user as the new
 * search directions (this helps reduce the susceptibility to roundoff error in
 * long iterative processes, and helps keep the code simple).
 *
 * Note that the user is allowed at any point to ignore the suggestions
 * provided by the Davidson object regarding the residual vectors to
 * precondition (gs/hs) or the new search directions (cs). Advanced users might
 * find a use for this (rank compression, etc), but this will surely affect
 * convergence properties.
 *  
 * At no point does Davidson assume that the search directions should be
 * orthonormal with respect to the given metric matrix (in contrast to
 * Davidson's original formulation). Instead, a generalized eigenproblem is
 * solved within the Krylov subspace, preceded by diagonal "balancing" of the
 * subspace metric matrix to improve condition. This leads naturally to the
 * "balanced Davidson-Liu" (B-DL) approach, in which the norms of the search
 * directions decrease through the iterative process, allowing for enhanced
 * screening in matrix-vector products, with negligible loss in accuracy. 
 *
 * As a technical note, accuracy in B-DL is improved if subspace matrix elements
 * of the form <k|M|l> = b_I^k M_IJ b_J^l = b_I^k Mb_I^l are formed from the
 * sigma vector Mb_I^l where b_I^l is the search direction with the greater
 * norm <l|S|l> = b_I S_IJ b_I^l. This is done for the construction of the
 * off-diagonal elements of the subspace stiffness and metric matrices.
 *
 * TODO: I want to try balanced canonical orthogonalization to solve the
 * subspace eigenproblem. This might need another cutoff parameter.
 **/
class Davidson {

public: 
/**
 * Construct an empty Davidson object
 * @param nstate the number of eigenpairs to target
 * @param nmax the maximum number of Krylov history vectors to accumulate
 *  before a subspace collapse is performed
 * @param convergence the 2-norm of the residual vector before it is declared
 *  to be converged and is not added to the list of vectors to precondition
 *  (gs/hs). When all vectors are converged, the is_converged property becomes
 *  true, and the user should exit the iterations.
 * @param canonical_tolerance is the cutoff to be used during the canonical 
 *  orthogonalization procedure
 * 
 * Throws if nmax < nstate (generally nmax >> nstate for good performance)
 **/
Davidson(
    size_t nstate,     
    size_t nmax,      
    double convergence,
    double canonical_tolerance=1.0e-9,
    const std::vector<std::shared_ptr<Storage> >& moms=std::vector<std::shared_ptr<Storage> >());

// => Accessors <= //

/// Number of states targeted 
size_t nstate() const { return nstate_; }
/// Maximum number of states allowed in history
size_t nmax() const { return nmax_; }
/// Residual norm to converge to
double convergence() const { return convergence_; }
      
/// Get a string representation of this Davidson object
std::string string() const;

// => Subspace Update <= //

/**
 * Add search vectors and necessary matrix-vector products to the Davidson
 * object. 
 *  
 * @param b the search vectors
 * @param Ab the stiffness-vector products Ab_I = A_IJ b_J (the "sigma" vectors)
 * @param Sb the metric-vector products. If the eigenproblem in question is
 *  being done in an orthonormal basis, just pass in b again (due to the use of
 *  pointers, there will be no performance or storage penalty).
 * @result b/Ab/Sb will be added to the Krylov history. The subspace
 *  eigenproblem will be solved in the new Krylov basis, and the eigenpairs will
 *  be provided in evecs/evals. The residuals and their norms will be computed
 *  and provided in rs/rnorms. The convergence of each eigenpair will be
 *  checked, and the residual/Ritz value for any non-converged root will be
 *  placed in gs/hs.
 *
 * Throws if b/Ab/Sb are not all the same size.
 **/
void add_vectors(
    const std::vector<std::shared_ptr<Storage> >& b,
    const std::vector<std::shared_ptr<Storage> >& Ab,
    const std::vector<std::shared_ptr<Storage> >& Sb);

// => Target Eigenbasis Quantities (updated in add_vectors) <= //

/// The current estimate for the eigenvectors
const std::vector<std::shared_ptr<Storage> >& evecs() const { return evecs_; }
/// The current estimate for the eigenvalues
const std::vector<double>& evals() const { return evals_; }
/// The current value for the residuals
const std::vector<std::shared_ptr<Storage> >& rs() const { return rs_; }
/// The current value for the residual 2-norms
const std::vector<double>& rnorms() const { return rnorms_; }

// => Convergence Information (updated in add_vectors) <= //

/// Are all vectors converged?
bool is_converged() const;
/// Max residual 2-norm
double max_rnorm() const;

// => Quantities to Precondition (updated in add_vectors) <= //

/// Vectors to precondition by - (\bar A - h)^-1 (usually a subset of rs)
const std::vector<std::shared_ptr<Storage> >& gs() const { return gs_; }
/// Ritz values to precondition with (usually a subset of evals)
const std::vector<double>& hs() const { return hs_; }

// => Subspace Augmentation <= //

/**
 * Add a vector of preconditioned residuals into the Davidson object to expand
 * the subspace.  Typically these will immediately be written to the new search
 * directions below.
 *
 * @param ds the preconditioned residuals computed by the user from gs/hs,
 *  utilizing a user-specified preconditioning matrix \bar A_IJ
 * @result the cs field is updated with the new search directions (either ds,
 *  or possibly evecs if a subspace collapse has occurred)
 **/
void add_preconditioned(
    const std::vector<std::shared_ptr<Storage> >& ds);

// => New Search Directions <= //

/// New search vectors (updated in add_preconditioned)
const std::vector<std::shared_ptr<Storage> >& cs() const { return cs_; }

private: 

size_t nstate_;
size_t nmax_;   
double convergence_;
double canonical_tolerance_;

std::vector<std::shared_ptr<Storage> > bs_;
std::vector<std::shared_ptr<Storage> > Abs_;
std::vector<std::shared_ptr<Storage> > Sbs_;
// Maximum overlap reference vectors (moms_) and metric (p_)
std::vector<std::shared_ptr<Storage> > moms_;
std::vector<std::shared_ptr<Tensor> > p_;
// Subspace stiffness matrix
std::shared_ptr<Tensor> A_;
// Subspace metric matrix
std::shared_ptr<Tensor> S_; 

std::vector<std::shared_ptr<Storage> > evecs_;
std::vector<double> evals_;
std::vector<std::shared_ptr<Storage> > rs_;
std::vector<double> rnorms_;

std::vector<std::shared_ptr<Storage> > gs_;
std::vector<double> hs_;

std::vector<std::shared_ptr<Storage> > cs_;

};

} // namespace lightspeed

#endif
