#!/usr/bin/env python
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
        'g_convergence' : 1.E-5,

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

        print('==> RHF <==\n')

        # => Problem Geometry <= #

        self.resources = self.options['resources']
        self.molecule = self.options['molecule']
        self.basis = self.options['basis']
        self.minbasis = self.options['minbasis']

        print(self.resources)
        print(self.molecule)
        print(self.basis)
        print(self.minbasis)

        self.sizes['natom'] = self.molecule.natom
        self.sizes['nao'] = self.basis.nao
        self.sizes['nmin'] = self.minbasis.nao

        # => Nuclear Repulsion <= #

        self.scalars['Enuc'] = self.molecule.nuclear_repulsion_energy()
        print('Nuclear Repulsion Energy = %20.14f\n' % self.scalars['Enuc'])

        # => Integral Thresholds <= #

        print('Integral Thresholds:')
        print('  Threshold PQ = %11.3E' % self.options['thre_pq'])
        print('  Threshold DP = %11.3E' % self.options['thre_dp'])
        print('  Threshold SP = %11.3E' % self.options['thre_sp'])
        print('')

        # => PairList <= #

        self.pairlist = ls.PairList.build_schwarz(
            self.basis,
            self.basis,
            True,
            self.options['thre_pq'])
        print(self.pairlist)

        # => One-Electron Integrals <= #

        print('One-Electron Integrals:\n')
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

        # => Canonical Orthogonalization <= #

        print('Canonical Orthogonalization:')
        print('  Threshold = %11.3E' % self.options['thre_canonical'])
        self.tensors['X'] = ls.Tensor.canonical_orthogonalize(self.tensors['S'], self.options['thre_canonical'])
        self.sizes['nmo'] = self.tensors['X'].shape[1]
        print('  Nmo       = %11d' % self.sizes['nmo'])
        print('  Ndiscard  = %11d' % (self.sizes['nao'] - self.sizes['nmo']))
        print('')

        # => SAD Guess <= #

        print('SAD Guess:\n')
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
        print(diis)
        
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
        print('Orbital Spaces:')
        print('  Nocc = %d' % self.sizes['nocc'])
        print('  Nvir = %d' % self.sizes['nvir'])
        print('') 

        # ==> SCF Iterations <== #

        print('Convergence Options:')
        print('  Max Iterations = %11d' % self.options['maxiter'])
        print('  E Convergence  = %11.3E' % self.options['e_convergence'])
        print('  G Convergence  = %11.3E' % self.options['g_convergence'])
        print('')

        self.tensors['D'] = ls.Tensor.array(self.tensors['DSAD'])
        Eold = 0.
        converged = False
        print('SCF Iterations:\n')
        print('%4s: %24s %11s %11s' % ('Iter', 'Energy', 'dE', 'dG'))
        for iter in range(self.options['maxiter']):
            
            # => Compute J/K Matrices <= #

            import time
            start = time.time()
            self.tensors['J'] = ls.IntBox.coulomb(
                self.resources,
                ls.Ewald.coulomb(),
                self.pairlist,
                self.pairlist,
                self.tensors['D'],
                self.options['thre_sp'],    
                self.options['thre_dp'],    
                )
            print('J: %11.3E' % (time.time() - start))
            start = time.time()
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
            print('K: %11.3E' % (time.time() - start))

            start = time.time()

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
            
            print('%4d: %24.16E %11.3E %11.3E' % (iter, self.scalars['Escf'], dE, dG))
    
            # => Check Convergence <= #
    
            if iter > 0 and abs(dE) < self.options['e_convergence'] and dG < self.options['g_convergence']:
                converged = True
                break

            # => DIIS Fock Matrix <= #

            if iter > 0:
                self.tensors['F'] = diis.iterate(self.tensors['F'], self.tensors['G'])

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
            print('T: %11.3E' % (time.time() - start))

        # => Print Convergence <= #
        
        print('')
        if converged:
            print('SCF Converged\n')
        else:
            print('SCF Failed\n')
        
        # => Print Final Energy <= #
        
        print('SCF Energy = %24.16E\n' % self.scalars['Escf'])

        # => Cache the Orbitals/Eigenvalues (for later use) <= #

        self.tensors['Cvir'] = ls.Tensor((self.sizes['nao'], self.sizes['nvir']), 'Cvir')
        self.tensors['Cvir'].np[...] = self.tensors['C'].np[:,self.sizes['nocc']:]
        
        self.tensors['eps_occ'] = ls.Tensor((self.sizes['nocc'],), 'eps_occ')
        self.tensors['eps_occ'].np[...] = self.tensors['eps'].np[:self.sizes['nocc']]
        
        self.tensors['eps_vir'] = ls.Tensor((self.sizes['nvir'],), 'eps_vir')
        self.tensors['eps_vir'].np[...] = self.tensors['eps'].np[self.sizes['nocc']:]

        # => Print Orbital Energies <= #

        print(self.tensors['eps_occ'])
        print(self.tensors['eps_vir'])

        # => Trailer Bars <= #

        print('"I love it when a plan comes together!"')
        print('        --LTC John "Hannibal" Smith\n')
        print('==> End RHF <==\n')


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

# => Test Utilities <= #

def run_h2o_sto3g():

    mol = ls.Molecule.from_xyz_str("""
        O 0.000000000000  0.000000000000 -0.129476890157
        H 0.000000000000 -1.494186750504  1.027446102928
        H 0.000000000000  1.494186750504  1.027446102928""", 
        name='h2o',
        scale=1.0)

    rhf = RHF.build(
        ls.ResourceList.build_cpu(),
        mol,
        basisname='sto-3g',
        minbasisname='cc-pvdz-minao')

run_h2o_sto3g()
