#!/usr/bin/env python
import unittest
import lightspeed as ls
import numpy as np

class RHF(object):

    options = {
        
        # > Problem Geometry < #

        # Resource List (required)
        'resources' : None,
        # Molecular Geometry (required)
        'molecule' : None,
        # Primary Basis Set (required)
        'basis' : None,
        # MinAO Basis Set (required)
        'minbasis' : None, 

        # External Potential Matrix (optional)
        'V_ext' : None,

        # > Number of electrons < #

        # Total number of doubly-occupied orbitals (1st priority)
        'nocc' : None,
        # Total molecular charge (2nd priority)
        'charge' : None,
        # molecule.charge (3rd priority)
        
        # > Numerical Thresholds < #

        # Schwarz-sieve threshold for pq pairs
        'thre_pq' : 1.E-14,
        # Threshold above which J/K builds will be double precision
        'thre_dp' : 1.E-6,
        # Threshold above which J/K builds will be single precision
        'thre_sp' : 1.E-14,
        # Canonical orthogonalization threshold
        'thre_canonical' : 1.0E-6,
        
        # > DIIS Options < #

        # Maximum number of vectors to DIIS with
        'diis_max_vecs' : 6,
        # Should DIIS use disk or not to save memory?
        'diis_use_disk' : False,
    
        # > Convergence Options < #

        # Maximum number of SCF iterations before failure
        'maxiter' : 50,
        # Maximum allowed energy change at convergence
        'e_convergence' : 1.E-6,
        # Maximum allowed element in orbital gradient at convergence 
        'g_convergence' : 1.E-7,

    }

    # => CPHF Default Options <= #

    cphf_options = {

        # > Numerical Thresholds < #

        # Threshold above which J/K builds will be double precision
        'thre_dp' : 1.E-6,
        # Threshold above which J/K builds will be single precision
        'thre_sp' : 1.E-14,

        # > DIIS Options < #

        # Maximum number of vectors to DIIS with
        'diis_max_vecs' : 6,
        # Should DIIS use disk or not to save memory?
        'diis_use_disk' : False,

        # > Convergence Options < #

        # Maximum number of CPHF iterations
        'maxiter'  : 50,
        # Maximum allowed preconditioned residual in CPHF at convergence
        'convergence' : 1.E-6,

    }

    def __init__(
        self,
        options, # A dictionary of user options
        ):

        # => Default Options <= #

        self.options = RHF.options.copy()
    
        # => Override Options <= #

        for key, val in options.items():
            if not key in list(self.options.keys()):
                raise ValueError('Invalid user option: %s' % key)
            self.options[key] = val 

        # => Useful Registers <= #

        self.sizes = {}
        self.scalars = {}
        self.tensors = {}

        # => Title Bars <= #

        #print '==> RHF <==\n'

        # => Problem Geometry <= #

        self.resources = self.options['resources']
        self.molecule = self.options['molecule']
        self.basis = self.options['basis']
        self.minbasis = self.options['minbasis']

        #print self.resources
        #print self.molecule
        #print self.basis
        #print self.minbasis

        self.sizes['natom'] = self.molecule.natom
        self.sizes['nao'] = self.basis.nao
        self.sizes['nmin'] = self.minbasis.nao

        # => Nuclear Repulsion <= #

        self.scalars['Enuc'] = ls.IntBox.chargeEnergySelf(
            self.resources,
            ls.Ewald.coulomb(),
            self.molecule.xyzZ)
        #print 'Nuclear Repulsion Energy = %20.14f\n' % self.scalars['Enuc']

        # => Integral Thresholds <= #

        #print 'Integral Thresholds:'
        #print '  Threshold PQ = %11.3E' % self.options['thre_pq']
        #print '  Threshold DP = %11.3E' % self.options['thre_dp']
        #print '  Threshold SP = %11.3E' % self.options['thre_sp']
        #print ''

        # => PairList <= #

        self.pairlist = ls.PairList.build_schwarz(
            self.basis,
            self.basis,
            True,
            self.options['thre_pq'])
        #print self.pairlist

        # => One-Electron Integrals <= #

        #print 'One-Electron Integrals:\n'
        self.tensors['S'] = ls.IntBox.overlap(
            self.resources,
            self.pairlist)
        self.tensors['T'] = ls.IntBox.kinetic(
            self.resources,
            self.pairlist)
        self.tensors['V'] = ls.IntBox.potential(
            self.resources,
            ls.Ewald.coulomb(),
            self.pairlist,
            self.molecule.xyzZ)
        self.tensors['H'] = ls.Tensor.array(self.tensors['T'].np + self.tensors['V'].np)

        # => External Potential <= #
    
        if self.options['V_ext']:
            #print 'Using External Potential\n'
            self.tensors['H'].np[...] += self.options['V_ext']

        # => Canonical Orthogonalization <= #

        #print 'Canonical Orthogonalization:'
        #print '  Threshold = %11.3E' % self.options['thre_canonical']
        self.tensors['X'] = ls.Tensor.canonical_orthogonalize(self.tensors['S'], self.options['thre_canonical'])
        self.sizes['nmo'] = self.tensors['X'].shape[1]
        #print '  Nmo       = %11d' % self.sizes['nmo']
        #print '  Ndiscard  = %11d' % (self.sizes['nao'] - self.sizes['nmo'])
        #print ''

        # => SAD Guess <= #

        #print 'SAD Guess:\n'
        self.tensors['CSAD'] = ls.SAD.orbitals(
            self.resources, 
            self.molecule,
            self.basis,
            self.minbasis)
        self.tensors['DSAD'] = ls.Tensor.chain([self.tensors['CSAD'],self.tensors['CSAD']],[False,True])

        # => DIIS <= #

        diis = ls.DIIS(
            self.options['diis_max_vecs'],
            )
        #print diis
        
        # => Determine number of electrons (and check for integral singlet) <= #

        if self.options['nocc'] is None:
            if self.options['charge'] is None:
                charge = self.molecule.charge
            else:
                charge = self.options['charge']
            nelec = self.molecule.nuclear_charge - charge
            nalpha = 0.5 * nelec
            if nalpha != round(nalpha):
                raise ValueError('Cannot do fractional electrons. Possibly charge/multiplicity are wrong.') 
            self.sizes['nocc'] = int(nalpha)
        else:
            self.sizes['nocc'] = int(self.options['nocc'])

        self.sizes['nvir'] = self.sizes['nmo'] - self.sizes['nocc']
        #print 'Orbital Spaces:'
        #print '  Nocc = %d' % self.sizes['nocc']
        #print '  Nvir = %d' % self.sizes['nvir']
        #print '' 

        # ==> SCF Iterations <== #

        #print 'Convergence Options:'
        #print '  Max Iterations = %11d' % self.options['maxiter']
        #print '  E Convergence  = %11.3E' % self.options['e_convergence']
        #print '  G Convergence  = %11.3E' % self.options['g_convergence']
        #print ''

        self.tensors['D'] = ls.Tensor.array(self.tensors['DSAD'])
        Eold = 0.
        self.scf_converged = False
        #print 'SCF Iterations:\n'
        #print '%4s: %24s %11s %11s' % ('Iter', 'Energy', 'dE', 'dG')
        for iter in range(self.options['maxiter']):
            
            # => Compute J/K Matrices <= #

            self.tensors['J'] = ls.IntBox.coulomb(
                self.resources,
                ls.Ewald.coulomb(),
                self.pairlist,
                self.pairlist,
                self.tensors['D'],
                self.options['thre_sp'],    
                self.options['thre_dp'],    
                )
            self.tensors['K'] = ls.IntBox.exchange(
                self.resources,
                ls.Ewald.coulomb(),
                self.pairlist,
                self.pairlist,
                self.tensors['D'],
                True,
                self.options['thre_sp'],    
                self.options['thre_dp'],    
                )

            # => Build Fock Matrix <= #            

            self.tensors['F'] = ls.Tensor.array(self.tensors['H'])
            self.tensors['F'].np[...] += 2. * self.tensors['J'].np
            self.tensors['F'].np[...] -= 1. * self.tensors['K'].np

            # => Compute Energy <= #

            self.scalars['Escf'] = self.scalars['Enuc'] + \
                self.tensors['D'].vector_dot(self.tensors['H']) + \
                self.tensors['D'].vector_dot(self.tensors['F']) 
            dE = self.scalars['Escf'] - Eold
            Eold = self.scalars['Escf']

            # => Compute Orbital Gradient <= #

            G1 = ls.Tensor.chain([self.tensors[x] for x in ['X', 'S', 'D', 'F', 'X']], [True] + [False]*4)
            G1.antisymmetrize()
            G1[...] *= 2.0
            self.tensors['G'] = G1
            dG = np.max(np.abs(self.tensors['G']))

            # => Print Iteration <= #
            
            #print '%4d: %24.16E %11.3E %11.3E' % (iter, self.scalars['Escf'], dE, dG)
    
            # => Check Convergence <= #
    
            if iter > 0 and abs(dE) < self.options['e_convergence'] and dG < self.options['g_convergence']:
                self.scf_converged = True
                break

            # => DIIS Fock Matrix <= #

            if iter > 0:
                self.tensors['F'] = diis.iterate(
                    self.tensors['F'], 
                    self.tensors['G'],
                    use_disk=self.options['diis_use_disk']
                    )

            # => Diagonalize Fock Matrix <= #

            F2 = ls.Tensor.chain([self.tensors[x] for x in ['X', 'F', 'X']],[True,False,False])
            e2, U2 = ls.Tensor.eigh(F2)
            C2 = ls.Tensor.chain([self.tensors['X'], U2],[False,False])
            e2.name = 'eps'
            C2.name = 'C'
            self.tensors['C'] = C2
            self.tensors['eps'] = e2
            
            # => Aufbau Occupy Orbitals <= #

            self.tensors['Cocc'] = ls.Tensor((self.sizes['nao'], self.sizes['nocc']),'Cocc')
            self.tensors['Cocc'].np[...] = self.tensors['C'].np[:,:self.sizes['nocc']]

            # => Compute Density Matrix <= #

            self.tensors['D'] = ls.Tensor.chain([self.tensors['Cocc'], self.tensors['Cocc']], [False, True])

        # => Print Convergence <= #
        
        #print ''
        #if self.scf_converged:
            #print 'SCF Converged\n'
        #else:
            #print 'SCF Failed\n'
        
        # => Print Final Energy <= #
        
        #print 'SCF Energy = %24.16E\n' % self.scalars['Escf']

        # => Cache the Orbitals/Eigenvalues (for later use) <= #

        self.tensors['Cvir'] = ls.Tensor((self.sizes['nao'], self.sizes['nvir']), 'Cvir')
        self.tensors['Cvir'].np[...] = self.tensors['C'].np[:,self.sizes['nocc']:]
        
        self.tensors['eps_occ'] = ls.Tensor((self.sizes['nocc'],), 'eps_occ')
        self.tensors['eps_occ'].np[...] = self.tensors['eps'].np[:self.sizes['nocc']]
        
        self.tensors['eps_vir'] = ls.Tensor((self.sizes['nvir'],), 'eps_vir')
        self.tensors['eps_vir'].np[...] = self.tensors['eps'].np[self.sizes['nocc']:]

        # => Print Orbital Energies <= #

        #print self.tensors['eps_occ']
        #print self.tensors['eps_vir']

        # => Trailer Bars <= #

        #print '"I love it when a plan comes together!"'
        #print '        --LTC John "Hannibal" Smith\n'
        #print '==> End RHF <==\n'

    # Compute the RHF Nuclear Gradient 
    def compute_gradient(
        self,
        ):

        # => Density Matrices <= #

        # OPDM 
        D = self.tensors['D']
        # Energy-weighted OPDM
        C1 = self.tensors['Cocc']
        C2 = ls.Tensor.zeros_like(C1)
        C2.np[...] = np.einsum('pi,i->pi', C1, self.tensors['eps_occ'])
        W = ls.Tensor.chain([C1,C2],[False,True])


        #print D
        # => Gradient Contributions <= # 
        
        keys = [
            'Nuc',
            'S',
            'T',
            'V',
            'J',
            'K',
            ]
        grads = {}

        # Nuclear repulsion energy gradient
        grads['Nuc'] = ls.IntBox.chargeGradSelf(
            self.resources,
            ls.Ewald.coulomb(),
            self.molecule.xyzZ)
        # Overlap integral gradient
        grads['S'] = ls.IntBox.overlapGrad(
            self.resources,
            self.pairlist,
            W,
            )
        grads['S'].np[...] *= -2.
        # Kinetic integral gradient
        grads['T'] = ls.IntBox.kineticGrad(
            self.resources,
            self.pairlist,
            D,
            )
        grads['T'].np[...] *= 2.
        # Potential integral gradient
        grads['V'] = ls.IntBox.potentialGrad(
            self.resources,
            ls.Ewald.coulomb(),
            self.pairlist,
            D,
            self.molecule.xyzZ,
            )
        grads['V'].np[...] *= 2.
        # Coulomb integral gradient
        grads['J'] = ls.IntBox.coulombGrad(
            self.resources,
            ls.Ewald.coulomb(),
            self.pairlist,
            D,
            D,
            self.options['thre_sp'],
            self.options['thre_dp'],
            )
        grads['J'].np[...] *= 2.
        # Exchange integral gradient
        grads['K'] = ls.IntBox.exchangeGrad(
            self.resources,
            ls.Ewald.coulomb(),
            self.pairlist,
            D,
            D,
            True, 
            True,
            True,
            self.options['thre_sp'],
            self.options['thre_dp'],
            )
        grads['K'].np[...] *= -1.

        G = ls.Tensor.zeros_like(grads['Nuc'])
        G.name = 'G (RHF)'
        for key in keys:
            G.np[...] += grads[key]

        #print '==> RHF Gradient Contributions <==\n'
        #for key in keys:
            #print grads[key]
        #print G
    
        return G

    # Solve the CPHF equations for a trial vector G
    # 4 [(e_a-e_i) + 4 (ia|jb) - (ib|ja) - (ij|ba)] X_jb = G_ia
    def compute_cphf(
        self,
        G, # A tensor which is nocc x nvir
        options = {}, # Override options
        ):

        # => Override Options <= #

        options2 = options
        options = RHF.cphf_options.copy()
        for key, val in options2.items():
            if not key in list(options.keys()):
                raise ValueError('Invalid user option: %s' % key)
            options[key] = val 

        # Check that G has correct shape
        G.shape_error((self.sizes['nocc'],self.sizes['nvir']))

        # => Header <= #

        #print '==> CPHF <==\n'

        # => DIIS <= #

        diis = ls.DIIS(
            options['diis_max_vecs'],
            )
        #print diis

        # => Preconditioner <= #

        # Fock-Matrix Contribution (Preconditioner)
        F = ls.Tensor((self.sizes['nocc'], self.sizes['nvir']), 'F')
        F.np[...] += np.einsum('i,a->ia',  np.ones(self.sizes['nocc']), self.tensors['eps_vir'])
        F.np[...] -= np.einsum('i,a->ia',  self.tensors['eps_occ'], np.ones(self.sizes['nvir']))
    
        # => Initial Guess (UCHF Result) <= #

        X = ls.Tensor.array(G)
        X.np[...] /= 4. * F.np[...] 

        # ==> Iterations <== #

        converged = False
        #print 'CPHF Iterations:\n'
        #print '%4s: %11s' % ('Iter', 'dR')
        for iter in range(options['maxiter']+1):

            # => Compute the Residual <= #

            R = ls.Tensor.array(G)
            # One-electron term
            R.np[...] -= 4. * F.np[...] * X.np[...]         
            # Two-electron terms
            D = ls.Tensor.chain([self.tensors['Cocc'],X,self.tensors['Cvir']],[False,False,True])
            D.symmetrize()
            I = ls.Tensor.zeros_like(D)
            I.np[...] += 16. * ls.IntBox.coulomb(
                self.resources,
                ls.Ewald.coulomb(),
                self.pairlist,
                self.pairlist,
                D,
                options['thre_sp'],    
                options['thre_dp'],    
                ).np
            I.np[...] -= 8. * ls.IntBox.exchange(
                self.resources,
                ls.Ewald.coulomb(),
                self.pairlist, 
                self.pairlist,
                D,
                True,
                options['thre_sp'],    
                options['thre_dp'],    
                ).np
            R.np[...] -= ls.Tensor.chain([self.tensors['Cocc'],I,self.tensors['Cvir']],[True,False,False])
            
            # => Precondition the Residual <= #

            R.np[...] /= 4. * F.np 
            X.np[...] += R.np # Approximate NR-Step

            # => Check Convergence <= #

            Rmax = np.max(np.abs(R))             
            #print '%4d: %11.3E' % (iter, Rmax)
            if Rmax < options['convergence']:
                converged = True
                break

            # => Perform DIIS <= #
            
            X = diis.iterate(X,R,use_disk=options['diis_use_disk'])
            
        # => Print Convergence <= #
        
        #print ''
        if converged:
            #print 'CPHF Converged\n'
            pass
        else:
            #print 'CPHF Failed\n'
            return False

        # => Trailer <= #

        #print '==> End CPHF <==\n'

        return X

    @staticmethod
    def build(
        resources,
        molecule,
        basis=None,
        minbasis=None,
        basisname='cc-pvdz',
        minbasisname='cc-pvdz-minao',
        options={},
        ):
    
        if basis is None:
            basis = ls.Basis.from_gbs_file(
                molecule,
                basisname)
    
        if minbasis is None:
            minbasis = ls.Basis.from_gbs_file(
                molecule,
                minbasisname)
        
        options.update({
            'resources' : resources,
            'molecule'  : molecule,
            'basis'     : basis,
            'minbasis'  : minbasis,
            })
    
        rhf = RHF(options)
        return rhf

# => UnitTest <= #

class RHFTest(unittest.TestCase):

    def test_rhf(self):

        print("\n=> RHF Test <=\n")
        print_title()
        self.assertTrue(run_rhf('coverage/data/', 'h2o', 'sto-3g'))

# => run RHF <= #

def run_rhf(run_dir, run_xyz, run_basis):

    e_tol = 1.0E-11
    g_tol = 1.0E-07

    res = ls.ResourceList.build(1024**2,1024**2)
    #res = ls.ResourceList.build_cpu()
    mol = ls.Molecule.from_xyz_file(run_dir + run_xyz + '.xyz')

    rhf = RHF.build(
        res,
        mol,
        basisname=run_basis,
        minbasisname='cc-pvdz-minao')

    if (not rhf.scf_converged):
        print("  Error: RHF SCF not converged!")
        return False

    # PSI4 reference energy
    e_ref = -74.9420799265908784
    e_diff = abs(rhf.scalars['Escf'] - e_ref)

    # PSI4 reference gradient
    g_ref = [[-0.000000000000, -0.097441382409, -0.000000000000],
             [ 0.086300060908,  0.048720691204, -0.000000000000],
             [-0.086300060908,  0.048720691204,  0.000000000000]]
    rhf.grad = rhf.compute_gradient()
    g_diff = np.max(np.abs(rhf.grad.np - g_ref))

    print("  %-15s%15.6e%15.6e" % \
            (run_xyz.upper()+'/'+run_basis, e_diff, g_diff))

    if (e_diff > e_tol):
        print("  %-6s diff=%15.6e\n" % ("E_RHF", e_diff))
        return False
    if (g_diff > g_tol):
        print("  %-6s diff=%15.6e\n" % ("G_RHF", g_diff))
        return False
    return True

def print_title():

    print("  Task: RHF energy and gradient")
    print("\n  %-15s%15s%15s" % \
            ("Mol", "E_Diff", "G_Diff"))
    print("  %s" % ('-'*45))

# => main function <= #

if __name__ == '__main__':
    print("\n=> RHF Test <=\n")
    print_title()
    assert(run_rhf('data/', 'h2o', 'sto-3g'))
