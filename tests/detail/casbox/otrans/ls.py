import lightspeed as ls
import numpy as np
import scipy.linalg as sp

# => Test Routines <= #

def test_casbox_orbital_transformation_det(
    S=0,
    M=6,
    Na=3,
    Nb=3,
    H1=None,
    I1=None,
    ):

    # Random rotation matrix
    A = ls.Tensor.array(np.random.rand(M, M) + 0.3 * np.eye(M))
    Q, R = sp.qr(A)
    Q = ls.Tensor.array(Q)
    # Random permutation matrix (forces pivoting)
    perm = np.random.permutation(list(range(M)))
    Q2 = ls.Tensor((M,)*2)
    for k, k2 in enumerate(perm):
        Q2[k,k2] = 1.0
    # Orbital transformation matrix (some rotation, required pivoting in LU)
    Q3 = ls.Tensor.chain([Q, Q2], [False, False])

    # Integrals in transformed basis
    H2 = ls.Tensor.array(np.einsum('ae,bf,ab->ef', Q3, Q3, H1))
    I2 = ls.Tensor.array(np.einsum('ae,bf,cg,dh,abcd->efgh', Q3, Q3, Q3, Q3, I1))

    casbox1 = ls.CASBox(M, Na, Nb, H1, I1)
    casbox2 = ls.CASBox(M, Na, Nb, H2, I2)

    ebox1 = ls.ExplicitCASBox(casbox1)
    ebox2 = ls.ExplicitCASBox(casbox2)

    evec1 = ebox1.evec(S,0)
    evec2 = ebox2.evec(S,0)

    evec1d = casbox1.CSF_basis(S).transform_CSF_to_det(evec1)
    evec2d = casbox2.CSF_basis(S).transform_CSF_to_det(evec2)

    evec1dp = casbox1.orbital_transformation_det(Q3, evec1d)
    O = evec1dp.vector_dot(evec2d)
    OK = abs(abs(O) - 1.0) < 1.0E-13
    return OK

if __name__ == '__main__':

    dat = np.load('data.npz')
    H1 = ls.Tensor.array(dat['H'])
    I1 = ls.Tensor.array(dat['I'])

    OK = True
    OK &= test_casbox_orbital_transformation_det(S=0, M=H1.shape[0], Na=3, Nb=3, H1=H1, I1=I1)
    OK &= test_casbox_orbital_transformation_det(S=2, M=H1.shape[0], Na=3, Nb=3, H1=H1, I1=I1)
    OK &= test_casbox_orbital_transformation_det(S=1, M=H1.shape[0], Na=3, Nb=2, H1=H1, I1=I1)
    OK &= test_casbox_orbital_transformation_det(S=3, M=H1.shape[0], Na=3, Nb=2, H1=H1, I1=I1)
    OK &= test_casbox_orbital_transformation_det(S=3, M=H1.shape[0], Na=5, Nb=2, H1=H1, I1=I1)
    print('OK' if OK else 'BAD')
