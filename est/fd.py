import lightspeed as ls
import numpy as np

def fd_gradient(
    xyz,
    fval,
    h=0.001,
    npoint=3,
    print_level=1,
    ):

    """ Compute the finite-difference gradient for a function returning a
        Tensor quantity as a function of coordinates XYZ. 

    Params:
        xyz (ls.Tensor, shape (A, 3)) - the central XYZ of the geometry.
        fval (function) - a function that takes an XYZ Tensor and returns a
            Tensor f(XYZ) (which could be either a scalar or tensor quantity)
        h (float) - stepsize
        npoint (int) - number of points in the stencil. 
            npoint = 3 gives:
                f'(x) = [f(x+h) - f(x-h)] / (2h)
            npoint = 5 gives:
                f'(x) = [-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)] / (12h)
            npoint = 7 gives:
                f'(x) = [f(x+3h) - 9f(x+2h) + 45f(x+h) - 45f(x-h) + 9f(x-2h) - f(x-3h)] / (60h)
        print_level (int) - how much to print: 
            0 - nothing. 
            >0 - headers for each perturbation.
    Returns:
        G (ls.Tensor) - the desired finite-difference gradient. If fval returns
            ndim == 0 Tensors (scalars), this is a "gradient-like" (A, 3)
            Tensor. If fval returns ndim != 0 Tensors (e.g., gradients, etc),
            this is a "Hessian-like (A*3, size(fval(XYZ))) Tensor.
    """ 

    # Validity checks
    xyz.ndim_error(2)
    # Not really a restriction, but be sure to know what you are doing to remove this
    xyz.shape_error((xyz.shape[0], 3)) 

    if print_level:
        print('Finite Difference Gradient:')
        print('  npoint = %d' % npoint)
        print('  h = %14.6E' % h)
        print('')
    
    G = None
    for A in range(xyz.shape[0]):
        for d in range(xyz.shape[1]):
            # Finite difference stencil
            if npoint == 3:
                # +h
                if print_level: print('Finite Difference Perturbation %d of %d\n' % (2 * (A*xyz.shape[1] + d) + 0, 2 * (xyz.size)))
                xp1 = ls.Tensor.array(xyz)
                xp1[A,d] += h
                fp1 = fval(xp1)
                # -h
                if print_level: print('Finite Difference Perturbation %d of %d\n' % (2 * (A*xyz.shape[1] + d) + 1, 2 * (xyz.size)))
                xm1 = ls.Tensor.array(xyz)
                xm1[A,d] -= h
                fm1 = fval(xm1)
                # f^Ad 
                fprime = ls.Tensor.array(1.0 / (2.0 * h) * (fp1[...] - fm1[...]))
            elif npoint == 5:
                # +2h
                if print_level: print('Finite Difference Perturbation %d of %d\n' % (4 * (A*xyz.shape[1] + d) + 0, 4 * (xyz.size)))
                xp2 = ls.Tensor.array(xyz)
                xp2[A,d] += 2.0 * h
                fp2 = fval(xp2)
                # +h
                if print_level: print('Finite Difference Perturbation %d of %d\n' % (4 * (A*xyz.shape[1] + d) + 1, 4 * (xyz.size)))
                xp1 = ls.Tensor.array(xyz)
                xp1[A,d] += h
                fp1 = fval(xp1)
                # -h
                if print_level: print('Finite Difference Perturbation %d of %d\n' % (4 * (A*xyz.shape[1] + d) + 2, 4 * (xyz.size)))
                xm1 = ls.Tensor.array(xyz)
                xm1[A,d] -= h
                fm1 = fval(xm1)
                # -2h
                if print_level: print('Finite Difference Perturbation %d of %d\n' % (4 * (A*xyz.shape[1] + d) + 3, 4 * (xyz.size)))
                xm2 = ls.Tensor.array(xyz)
                xm2[A,d] -= 2.0 * h
                fm2 = fval(xm2)
                # f^Ad
                fprime = ls.Tensor.array(1.0 / (12.0 * h) * (- fp2[...] + 8.0 * fp1[...] - 8.0 * fm1[...] + fm2[...]))
            elif npoint == 7:
                # +3h
                if print_level: print('Finite Difference Perturbation %d of %d\n' % (6 * (A*xyz.shape[1] + d) + 0, 6 * (xyz.size)))
                xp3 = ls.Tensor.array(xyz)
                xp3[A,d] += 3.0 * h
                fp3 = fval(xp3)
                # +2h
                if print_level: print('Finite Difference Perturbation %d of %d\n' % (6 * (A*xyz.shape[1] + d) + 1, 6 * (xyz.size)))
                xp2 = ls.Tensor.array(xyz)
                xp2[A,d] += 2.0 * h
                fp2 = fval(xp2)
                # +h
                if print_level: print('Finite Difference Perturbation %d of %d\n' % (6 * (A*xyz.shape[1] + d) + 2, 6 * (xyz.size)))
                xp1 = ls.Tensor.array(xyz)
                xp1[A,d] += h
                fp1 = fval(xp1)
                # -h
                if print_level: print('Finite Difference Perturbation %d of %d\n' % (6 * (A*xyz.shape[1] + d) + 3, 6 * (xyz.size)))
                xm1 = ls.Tensor.array(xyz)
                xm1[A,d] -= h
                fm1 = fval(xm1)
                # -2h
                if print_level: print('Finite Difference Perturbation %d of %d\n' % (6 * (A*xyz.shape[1] + d) + 4, 6 * (xyz.size)))
                xm2 = ls.Tensor.array(xyz)
                xm2[A,d] -= 2.0 * h
                fm2 = fval(xm2)
                # -3h
                if print_level: print('Finite Difference Perturbation %d of %d\n' % (6 * (A*xyz.shape[1] + d) + 5, 6 * (xyz.size)))
                xm3 = ls.Tensor.array(xyz)
                xm3[A,d] -= 3.0 * h
                fm3 = fval(xm3)
                # f^Ad
                fprime = ls.Tensor.array(1.0 / (60.0 * h) * (fp3[...] - 9.0 * fp2[...] + 45.0 * fp1[...] - 45.0 * fm1[...] + 9.0 * fm2[...] - fm3[...]))
            else:
                raise ValueError('Unknown number of points: %r' % npoint)
            # Allocate G if needed (a bit awkward, but we don't know fval shape until first call)
            if G is None:
                if fprime.ndim == 0:
                    # "Gradient like"
                    G = ls.Tensor.zeros_like(xyz)
                else:
                    # "Hessian like"
                    G = ls.Tensor.zeros((xyz.size, fprime.size))
            # Assign gradient
            if fprime.ndim == 0:            
                # "Gradient like"
                G[A,d] = fprime[...]
            else:
                # "Hessian like"
                G[A*xyz.shape[1] + d, :] = np.ravel(fprime)

    return G
    
