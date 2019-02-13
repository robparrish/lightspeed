#ifndef LIGHTSPEED_AM_HPP
#define LIGHTSPEED_AM_HPP

#include <cstddef>
#include <vector>

namespace lightspeed {

/**!
 * Class AngularMomentum describes the ordering and composition of cartesian
 * and spherical (pure) angular momentum shells in Lightspeed.
 * 
 * Typically, a user would generate a vector of AngularMomentum, with one
 * entry each for [0, Lmax]. E.g., to treat up to G functions (L=4), the user
 * would say:
 *
 *  std::vector<AngularMomentum> L_info(4+1);
 *  for (int L = 0; L <= 4; L++) {
 *     L_info[L] = AngularMomentum(L);
 *  }
 *
 * alternatively, there is a static helper method to do this,
 *
 *  std::vector<AngularMomentum> L_info = AngularMomentum::build(4);
 *
 * Now the user may ask the AngularMomentum for L = L for information on how
 * to build the cartesian or pure angular momentum for the given shell.
 * Note: these objects are cheap to build, use wherever needed. 
 *
 * => Cartesian Functions <= 
 *
 * For cartesian functions, we adopt the following convention:
 *
 *  \phi_lmn (\vec r_1) = x_1A^l y_1A^m z_1A^n [\sum_{K}^{nprim} c_K \exp(-e_K r_1A^2)]
 * 
 * That is, angular momentum is applied independently in x,y,and z, and no
 * joint l,m,n-dependent normalization coefficient is applied. Note that this
 * implies that a cartesian d-shell which is normalized in xx, yy, and zz will
 * NOT be normalized in xy, xy, and yz. More generally, a diagonal function
 * (xx, yyy, zzzz, etc) will be normalized (if the underlying shell is
 * normalized), but non-diagonal functions (xy, xxy, xyzz, etc) will not be.  
 *
 * Now, all the remains is to specify a rule for the lexical ordering of l,m,n
 * tuples. These are given for convenience below in the l/m/n methods (and
 * plural variants thereof). Alternatively, users may find it more convenient
 * to use the standard algorithm for generating these:
 *
 *  for (int i = 0, index = 0; i <= L; i++) {
 *      int l = L - i;
 *      for (int j = 0; j <= i; j++, index++) {
 *          int m = i - j;
 *          int n = j;
 *      }
 *  }
 *
 * In English, this first adds angular momentum to l, then to m, then to n, in
 * a lexical manner.
 *
 * The ordering of cartesian prefactors for basis shells of up to G-type are
 * shown below:
 *
 * l/i  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14
 * 0    1
 * 1    x    y    z
 * 2   xx   xy   xz   yy   yz   zz
 * 3  xxx  xxy  xxz  xyy  xyz  xzz  yyy  yyz  yzz  zzz
 * 4 xxxx xxxy xxxz xxyy xxyz xxzz xyyy xyyz xyzz xzzz yyyy yzzz yyzz yzzz zzzz
 * l/i  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14
 *
 * => Pure Functions <= 
 * 
 * To produce pure functions, we invoke the standard real solid harmonics:
 *
 *  \phi_lm = S_lm (\vec r_1A) [\sum_{K}^{nprim} c_K \exp(- e_K r_1A^2)]
 *  = C_lm^lmn \phi_lmn (\vec r_1A)
 * 
 * The last expression demonstrates how the pure basis functions are
 * produced as linear combinations of cartesian basis functions. The sparse
 * matrix C_lm^lmn provides the coefficients of the cartesian basis functions
 * (upper) to transform to the pure basis functions (lower).
 *
 * The first remaining task is to specify a lexical ordering of the pure
 * basis functions. In ian we adopt the convention (c is for cos, s is
 * for sin):
 *  l0,l1c,l1s,l2c,l2s,..., 
 * which is equivalent to the +/- convention sometimes seen,
 *  l,0,l,+1,l,-1,l,+2,l,-2,...
 *
 * The second remaining task is to specify the coefficient matrix C_lm^lmn in a
 * sparse manner, which is accomplished below in the
 * cart_ind/pure_ind/cart_coef methods (and pure variants thereof).
 *
 * For reference, these coefficients are derived by recurrence relations as may
 * be found in many places in the literature. In particular, we have
 * implemented the equations directly from Equations 6.4.70-6.4.73 in Molecular
 * Electronic-Structure Theory by Helgaker, Jorgensen, and Olsen (the Purple
 * Book):
 *
 *  S_0,+0 = 1 
 *  S_l+1,+l+1 = sqrt{2^{\delta_{l0}} \frac{2l+1}{2l+2}} (x S_l,l - (1 - \delta_{l0}) y S_l,-l) 
 *  S_l+1,-l-1 = sqrt{2^{\delta_{l0}} \frac{2l+1}{2l+2}} (y S_l,l + (1 - \delta_{l0}) x S_l,-l) 
 *  S_l+1,m = \frac{(2l+1) z S_l,m - sqrt{(l+m)(l-m)} r^2 S_l-1,m}{\sqrt{(l+m+1)(l-m+1)}} 
 *
 * The explicit solid harmonics up through D-type shells are specified below.
 * The corresponding expressions for F- and G-type shells are explicitly
 * presented in Table 6.3 of the Purple Book.
 *
 *  S_00  = 1
 *  S_10  = z
 *  S_11c = x
 *  S_11c = y
 *  S_20  = 1/2 (3 zz - rr)
 *  S_21c = \sqrt{3} xz
 *  S_21s = \sqrt{3} yz
 *  S_22c = 1/2 \sqrt{3} (xx - yy) 
 *  S_22s = \sqrt{3} xy 
 *
 * - Rob Parrish, 16 February, 2015
 **/
class AngularMomentum {

public:

// => Constructors <= //

/// Main constructor, builds information for a shell with L = L
AngularMomentum(int L);

/// Default constructor
AngularMomentum() {}

/// Return a vector of AngularMomentum in [0,Lmax] (Lmax + 1 entries)
static std::vector<AngularMomentum> build(int Lmax);

// => General Accessors <= //

/// Angular momentum of this type of shell
int L() const { return L_; }
/// Number of cartesian functions in this type of shell
size_t ncart() const { return (L_ + 1L) * (L_ + 2L) / 2L; }
/// Number of pure functions in this type of shell
size_t npure() const { return 2L * L_ + 1L; }

// => Cartesian Shell Information <= //

/// Powers of x in the cart version of this shell, length ncart()
const std::vector<int>& ls() const { return ls_; }
/// Powers of y in the cart version of this shell, length ncart()
const std::vector<int>& ms() const { return ms_; }
/// Powers of z in the cart version of this shell, length ncart()
const std::vector<int>& ns() const { return ns_; }

// => Pure Shell Information <= //

/// Total number of cart to pure transformation coefficients
size_t ncoef() const { return cart_inds_.size(); }
/// Indices of cart functions in transformation, length ncoef()
const std::vector<int>& cart_inds() const { return cart_inds_; }
/// Indices of pure functions in transformation, length ncoef()
const std::vector<int>& pure_inds() const { return pure_inds_; }
/// Coefficients of cart functions in transformation, length ncoef()
const std::vector<double>& cart_coefs() const { return cart_coefs_; }

// => Equivalence (for Python) <= //


bool operator==(const AngularMomentum& other) const
{
    return L_ == other.L_;
}

bool operator!=(const AngularMomentum& other) const 
{
    return !((*this) == other);
}


private:

int L_;

std::vector<int> ls_;
std::vector<int> ms_;
std::vector<int> ns_;

std::vector<int> cart_inds_;
std::vector<int> pure_inds_;
std::vector<double> cart_coefs_;

void build_cart();
void build_pure();

};

} // namespace lightspeed

#endif
