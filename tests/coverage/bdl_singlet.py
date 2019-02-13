#!/usr/bin/env /home/rguerrero/anaconda2/bin/python
import time
import unittest
import collections
import numpy as np
import lightspeed as ls
from numpy import linalg as LA

class BDLTest(unittest.TestCase):
    
    def test_BDL(self):
        print("\n=> B-DL Test <=\n")
        self.assertTrue(test_BDL(5,50,1.0E-6,100,"coverage/data/H2O_Singlet.npz"))

def BDL(npH,nguess,nkrylov,Threshold,max_iter):
    Ham=ls.Tensor.array(npH)
    # Guess
    F = ls.Tensor.array(np.diag(Ham))
    guess =  np.argpartition(F,nguess)
    n=npH.shape[0]
    I=np.identity(n)
    
    # Initializing b, Ab and Sb correspondingly
    b=[]
    for i in range(nguess):
        b.append(ls.Tensor.array(I[guess[i],:]))
    Ab = []
    for i in range(nguess):
        Ab.append(ls.Tensor.array(npH.dot(b[i])))
        
    dav = ls.Davidson(nguess, nkrylov, Threshold, 1.0E-6)
    
    print(dav)
    
    converged = False
    
    start_davidson = time.time()
    
    for iter in range(max_iter):
        Rs, Es = dav.add_vectors(b, Ab)

        
        print('Iter: %4d: Max Residual Norm :%11.3E' % (iter, dav.max_rnorm))
        # Check for convergence
        if dav.is_converged:
            converged = True
            break
        
        # Precondition the desired residuals
        
        Ds = [ls.Tensor.array(x) for x in Rs]
        
        for D, E in zip(Ds, Es):
            D.np[...] /= -(F.np[...] - E + 1.0E-7)
            
        b = dav.add_preconditioned(Ds)
        
        Ab = [ls.Tensor.array(Ham.np.dot(x)) for x in b]
        
    end_davidson = time.time()
    # Print Convergence
    print('')
    if converged:
        print('Davidson Converged in %lf seconds\n' % (end_davidson-start_davidson))
    else:
        print('Davidson Failed\n') 
    
    Davidson_energies = [x for x in dav.evals]
    #Davidson_orbitals = [x for x in dav.evecs]
    Davidson_orbitals = [ls.Storage.from_storage(x,b[0]) for x in dav.evecs]

    return Davidson_energies,Davidson_orbitals

# Computing L-2 distance between eigenvectors 
# taking into account the phase
def L2_distance(np_x, ls_x):
    sgn = np_x.dot(ls_x.np)
    return LA.norm(np_x - sgn*ls_x.np)


#Testing against exact diagonalization
def test_BDL(nguess,nkrylov,Threshold,max_iter,RefFile):
    reference = np.load(RefFile)
    
    print('Full space size: %d ' % reference['fullsize'])
    
    n=len(reference['evals'])
    results = BDL(reference['Hamiltonian'],nguess,nkrylov,Threshold,max_iter)
    
    evals_exact = reference['evals'] 
    evecs_exact = reference['evecs'] 
    
    #print "\n=> Test Results <="

    delta_E = []
    for E1, E2 in zip(evals_exact[:nguess], [x for x in results[0] ]):
        delta_E.append(abs(E1-E2));
    tol_eigval = 1.0e-10
    E_passed = np.amax(delta_E) < tol_eigval
    if E_passed:
        print('Eigenvalues : %15.6s ' % E_passed)
    else:
        print('Eigenvalues : %15.6s ' % E_passed)
        print("Max error in Eigenvalues = ", np.amax(delta_E))
    
    delta_V=[]
    for i in range(nguess):
        delta_V.append(L2_distance(evecs_exact[:,i],results[1][i]))
    
    tol_eigvecs = 1.0e-8
    V_passed = np.amax(delta_V) < tol_eigvecs
    if V_passed:
        print('Eigenvectors : %15s ' % V_passed)
    else:
        print('Eigenvectors : %15s ' % V_passed)
        print("Max error in Eigenvectors = ", np.amax(delta_V))

    return (V_passed and E_passed)

# => main function <= #

if __name__ == "__main__":
    print("\n=> B-DL Test <=\n")
    assert(test_BDL(5,50,1.0E-6,100,"data/H2O_Singlet.npz"))
