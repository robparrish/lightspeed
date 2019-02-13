import lightspeed as ls
from . import options
from . import geometry
import numpy as np
import scipy.special as sp # For erfc
import time

class RHF(object):

    @staticmethod
    def default_options():
        if hasattr(RHF, '_default_options'): return RHF._default_options.copy()
        opt = options.Options() 

        # > Print Control < #

        opt.add_option(
            key='print_level',
            value=2,
            allowed_types=[int],
            doc='Level of detail to print (0 - nothing, 1 - minimal [dynamics], 2 - standard [SP])')

        # > Problem Geometry < #

        opt.add_option(
            key="geometry",
            required=True,
            allowed_types=[geometry.Geometry],
            doc='EST problem geometry')

        # > Number of Electrons < #

        opt.add_option(
            key='npair',
            required=False,
            allowed_types=[int],
            doc='Total number of electron pairs (taken verbatim) [1st priority]')
        opt.add_option(
            key='charge',
            required=False,
            allowed_types=[int],
            doc='Total molecular charge (Z - charge - ECP nelec) [2nd priority]')
        # qm_molecule.charge (Z - charge - ECP nelec) [3rd priority]
    
        # > Numerical Thresholds < #
    
        opt.add_option(
            key='thre_dp',
            value=1.0E-6,
            required=True,
            allowed_types=[float],
            doc='Threshold above which J/K builds will be double precision')
        opt.add_option(
            key='thre_sp',
            value=1.0E-14,
            required=True,
            allowed_types=[float],
            doc='Threshold above which J/K builds will be single precision')
        opt.add_option(
            key='thre_canonical',
            value=1.0E-6,
            required=True,
            allowed_types=[float],
            doc='Canonical orthogonalization threshold') 

        # > Guess Options < #
    
        opt.add_option( 
            key='sad_qocc',
            required=False,
            allowed_types=[list],
            doc='SAD specific orbital electronic charges in electron pairs [1st priority]')
        opt.add_option( 
            key='sad_qatom',
            required=False,
            allowed_types=[list],
            doc='SAD specific atomic electronic charges in electron pairs [2nd priority]')
        opt.add_option( 
            key='sad_qtotal',
            required=False,
            allowed_types=[list],
            doc='SAD total charge in electron pairs [3rd priority]')

        # > FOMO Parameters < #

        opt.add_option(
            key='fomo',
            value=False,
            required=True,
            allowed_types=[bool],
            doc='Do FOMO or not?')
        opt.add_option(
            key='fomo_method',
            value='gaussian',
            required=True,
            allowed_types=[str],
            allowed_values=['gaussian', 'fermi'],
            doc='FOMO occupation method')
        opt.add_option(
            key='fomo_temp',
            value=0.3,
            required=True,
            allowed_types=[float],
            doc='FOMO electronic temperature parameter (kT in au)')
        opt.add_option(
            key='fomo_nocc',
            value=0,
            required=True,
            allowed_types=[int],
            doc='FOMO number of closed (doubly occupied) orbitals')
        opt.add_option(
            key='fomo_nact',
            value=0,
            required=True,
            allowed_types=[int],
            doc='FOMO number of active (fractionally occupied) orbitals')
        
        # > DIIS Options < #    

        opt.add_option(
            key='diis_max_vecs',
            value=6,
            required=True,
            allowed_types=[int],
            doc='Maximum number of vectors in the DIIS history') 
        opt.add_option( 
            key='diis_use_disk',
            value=False,
            required=True,
            allowed_types=[bool],
            doc='Should DIIS use disk storage to save core memory?')
        opt.add_option(
            key='diis_flush_niter',
            value=40,
            required=True,
            allowed_types=[bool],
            doc='Number of SCF iterations between flushes of DIIS object - helps with stagnated convergence')

        # > Convergence Options < #

        opt.add_option(
            key='maxiter',
            value=50,
            required=True,
            allowed_types=[int],
            doc='Maximum number of SCF iterations before convergence failure')
        opt.add_option(
            key='e_convergence',
            value=1.E-6,
            required=True,
            allowed_types=[float],
            doc='Maximum allowed energy change between iterations at SCF convergence')
        opt.add_option(
            key='g_convergence',
            value=1.E-5,
            required=True,
            allowed_types=[float],
            doc='Maximum allowed element in the orbital gradient at SCF convergence')
        opt.add_option(
            key='incremental_niter',
            value=8,
            required=True,
            allowed_types=[int],
            doc='Number of iterations between flushes of incremental Fock builds')

        # > DFT Options < #

        opt.add_option(
            key='dft_functional',
            required=False,
            allowed_types=[str],
            doc='Name of the DFT functional, e.g., B3LYP (None indicates RHF)')
        opt.add_option(
            key='dft_grid_name',
            value='SG0',
            required=False,
            allowed_types=[str],
            allowed_values=['SG0', 'SG1'],
            doc='Name of the DFT grid, e.g., SG0 or SG1 (None uses simple grid)')
        opt.add_option(
            key='dft_grid_atomic_scheme',
            value='FLAT',
            required=True,
            allowed_types=[str],
            allowed_values=['FLAT', 'BECKE', 'AHLRICHS'],
            doc='DFT grid atomic weight scheme for simple grid') 
        opt.add_option(
            key='dft_grid_radial_scheme',
            value='BECKE',
            required=True,
            allowed_types=[str],
            allowed_values=['BECKE', 'MULTIEXP'],
            doc='DFT grid radial weight scheme for simple grid') 
        opt.add_option(
            key='dft_grid_nradial',
            value=99,
            required=True,
            allowed_types=[int],
            doc='DFT grid number of radial quadrature points for simple grid')
        opt.add_option(
            key='dft_grid_nspherical',
            value=302,
            required=True,
            allowed_types=[int],
            doc='DFT grid number of spherical quadrature points for simple grid')
        opt.add_option(
            key='dft_thre_grid',
            value=1.0E-14,
            required=True,
            allowed_types=[float],
            doc='DFT grid threshold')
        opt.add_option(
            key='dft_grid_R',
            value=5.0,
            required=True,
            allowed_types=[float],
            doc='DFT grid hash cell spacing')

        # > Gradient Thresholds < #

        opt.add_option(
            key='grad_thre_dp',
            value=1.0E-6,
            required=True,
            allowed_types=[float],
            doc='Threshold above which J/K grad builds will be double precision')
        opt.add_option(
            key='grad_thre_sp',
            value=1.0E-14,
            required=True,
            allowed_types=[float],
            doc='Threshold above which J/K grad builds will be single precision')
        opt.add_option(
            key='grad_dft_thre_grid',
            value=1.0E-14,
            required=True,
            allowed_types=[float],
            doc='DFT grid threshold for gradients')

        # > CPHF Options < #

        opt.add_option(
            key='cphf_maxiter',
            value=50,
            required=True,
            allowed_types=[int],
            doc='Maximum number of CPHF iterations before convergence failure')
        opt.add_option(
            key='cphf_r_convergence',
            value=1.E-5,
            required=True,
            allowed_types=[float],
            doc='Maximum allowed element in the reisdual at CPHF convergence')
        opt.add_option(
            key='cphf_incremental_niter',
            value=8,
            required=True,
            allowed_types=[int],
            doc='Number of iterations between flushes of incremental Fock builds')
        opt.add_option(
            key='cphf_diis_max_vecs',
            value=6,
            required=True,
            allowed_types=[int],
            doc='Maximum number of vectors in the DIIS history') 
        opt.add_option( 
            key='cphf_diis_use_disk',
            value=False,
            required=True,
            allowed_types=[bool],
            doc='Should DIIS use disk storage to save core memory?')
        opt.add_option(
            key='cphf_thre_dp',
            value=1.0E-6,
            required=True,
            allowed_types=[float],
            doc='Threshold above which J/K grad builds will be double precision')
        opt.add_option(
            key='cphf_thre_sp',
            value=1.0E-14,
            required=True,
            allowed_types=[float],
            doc='Threshold above which J/K grad builds will be single precision')

        RHF._default_options = opt
        return RHF._default_options.copy()

    def __init__(
        self,
        options,
        ):
        
        self.options = options

        # Cache some useful attributes
        self.print_level = self.options['print_level']
        self.geometry = self.options['geometry']
        self.resources = self.geometry.resources
        self.ewald = self.geometry.ewald
        self.qm_molecule = self.geometry.qm_molecule
        self.basis = self.geometry.basis
        self.minbasis = self.geometry.minbasis
        self.pairlist = self.geometry.pairlist
    
        # Some useful registers
        self.sizes = {}
        self.scalars = {}
        self.tensors = {}

    @staticmethod
    def from_options(**kwargs):
        """ Return an instance of this class with default options updated from values in kwargs. """
        return RHF(RHF.default_options().set_values(kwargs))

    def initialize(self):

        """ Initialize the RHF object to the point that an intelligent guess
        for the D matrix can be performed. Users wishing to perform a
        corresponding orbital transformation or other guess that needs to know
        the X matrix should call this function, compute the guess, and then
        call compute_energy, passing the guess in to compute_energy. Users OK
        with a SAD guess or not needing X for the guess can just call
        compute_energy, which will ensure this function is called.
        """

        # => Title Bars <= #

        if self.print_level: 
            print('==> RHF <==\n')

        # => Problem Geometry <= #

        if self.print_level > 1:
            print(self.geometry)

        # => Some Useful Sizes <= #

        self.sizes['natom'] = self.qm_molecule.natom
        self.sizes['nao'] = self.basis.nao
        self.sizes['nmin'] = self.minbasis.nao

        # => PairList <= #

        if self.print_level > 1: 
            print(self.pairlist)

        # => One-Electron Integrals <= #

        if self.print_level > 1: 
            print('One-Electron Integrals:\n')
        self.tensors['S'] = ls.IntBox.overlap(
            self.resources,
            self.pairlist)
        self.tensors['T'] = ls.IntBox.kinetic(
            self.resources,
            self.pairlist)

        # => Canonical Orthogonalization <= #

        if self.print_level > 1:
            print('Canonical Orthogonalization:')
            print('  Threshold = %11.3E' % self.options['thre_canonical'])
        self.tensors['X'] = ls.Tensor.canonical_orthogonalize(self.tensors['S'], self.options['thre_canonical'])
        self.sizes['nmo'] = self.tensors['X'].shape[1]
        if self.print_level > 1:
            print('  Nmo       = %11d' % self.sizes['nmo'])
            print('  Ndiscard  = %11d' % (self.sizes['nao'] - self.sizes['nmo']))
            print('')

    def compute_energy(
        self,
        Dguess=None,
        Cocc_mom=None,
        Cact_mom=None,
        ):

        """ Compute the RHF wavefunction and energy, using Roothaan/DIIS.
            Modifies the structure of this RHF object with the RHF
            wavefunction, Fock matrix elements, energies, etc.

        Params:
            Dguess (ls.Tensor) - guess to OPDM in AO basis. Overrides SAD guess.
            Cocc_mom (ls.Tensor) - Reference orbitals for MOM orbital
                selection. Overrides aufbau occupation.
            Cact_mom (ls.Tensor) - Reference orbitals for MOM orbital
                selection. Overrides aufbau occupation. Not referred to if
                conventional RHF. Both Cocc_mom and Cact_mom must be provided
                if using MOM for FOMO-RHF.
        Returns:
            converged (bool) - did the SCF procedure converge (True) or not (False).
        """

        # => Call initialize if not already initialized <= #

        if 'X' not in self.tensors: 
            self.initialize()

        # => Guess D Matrix <= #

        if Dguess:
            Dguess.shape_error((self.sizes['nao'],)*2)
        if Dguess:
            if self.print_level > 1: print('Input Guess:\n')
            self.tensors['D'] = Dguess
        else:
            if self.print_level > 1: print('SAD Guess:\n')
            # How many core orbitals per atom are represented by ECPs?    
            necps = None
            if self.geometry.ecp:
                necps = [x // 2 for x in self.geometry.ecp.nelecs]
            Csad = ls.SAD.orbitals(
                self.resources, 
                self.qm_molecule,
                self.basis,
                self.minbasis,
                Qocc=self.options['sad_qocc'],
                Qatom=self.options['sad_qatom'],
                Qtotal=self.options['sad_qtotal'],
                necps=necps,
                )
            self.tensors['D'] = ls.Tensor.chain([Csad, Csad],[False,True])

        # => DFT <= #

        self.dft_functional = None
        self.dft_grid = None
        self.dft_hash = None 
        if self.options['dft_functional']:
            self.is_dft = True
            self.dft_functional = ls.Functional.build(self.options['dft_functional'])
            self.dft_grid = ls.BeckeGrid.build(
                self.resources,
                self.qm_molecule,
                options = {
                    'dft_grid_name' : self.options['dft_grid_name'],
                    'dft_grid_atomic_scheme' : self.options['dft_grid_atomic_scheme'],
                    'dft_grid_radial_scheme' : self.options['dft_grid_radial_scheme'],
                    'dft_grid_nradial' : self.options['dft_grid_nradial'],
                    'dft_grid_nspherical' : self.options['dft_grid_nspherical'],
                })
            self.dft_hash = ls.HashedGrid(
                self.dft_grid.xyz,
                self.options['dft_grid_R'])
            if self.print_level > 1:
                print(self.dft_functional)
                print(self.dft_grid)
                print(self.dft_hash)
        else:
            self.is_dft = False

        # => External Energy/Potential <= #

        if self.print_level:
            print('External Environment:')
        self.scalars['Enuc'] = self.geometry.compute_nuclear_repulsion_energy()
        self.scalars['Eext'] = self.geometry.compute_external_energy()
        self.tensors['Vext'] = self.geometry.compute_external_potential(self.pairlist)
        if self.print_level:
            print('  Enuc = %24.16E' % self.scalars['Enuc'])
            print('  Eext = %24.16E' % self.scalars['Eext'])
            print('')

        # => Core Hamiltonian <= #
        
        self.tensors['H'] = ls.Tensor.array(self.tensors['T'])
        self.tensors['H'][...] += self.tensors['Vext']

        # => Determine number of electrons (and check for integral singlet) <= #

        charge = None
        charge_method = None
        if self.options['npair'] is None:
            if self.options['charge'] is None:
                charge = self.qm_molecule.charge
                charge_method = 'molecule'
            else:
                charge = self.options['charge']
                charge_method = 'charge option'
            nelec = self.qm_molecule.nuclear_charge - charge
            # Account for ECP-modeled electrons
            if self.geometry.ecp:
                nelec -= self.geometry.ecp.nelec
            nalpha = 0.5 * nelec
            if nalpha != round(nalpha):
                raise ValueError('Cannot do fractional electrons. Possibly charge/multiplicity are wrong.') 
            self.sizes['npair'] = int(nalpha)
        else:
            self.sizes['npair'] = int(self.options['npair'])
            charge_method = 'npair option'
        if self.print_level > 1:
            print('Charge Determination:')
            print('  Method = %s' % charge_method)
            print('  Charge = %r' % charge)
            print('  Npair  = %d' % self.sizes['npair'])
            print('')

        # => FOMO <= #

        if self.options['fomo']:
            self.is_fomo = True
            self.sizes['nocc'] = self.options['fomo_nocc']
            self.sizes['nact'] = self.options['fomo_nact']
            self.sizes['nvir'] = self.sizes['nmo'] - self.sizes['nocc'] - self.sizes['nact']
            self.sizes['nclsd'] = self.sizes['nocc']
            self.sizes['nfrac'] = self.sizes['npair'] - self.sizes['nclsd']
        else:
            self.is_fomo = False
            self.sizes['nocc'] = self.sizes['npair']
            self.sizes['nvir'] = self.sizes['nmo'] - self.sizes['npair']
            self.sizes['nact'] = 0
            self.sizes['nclsd'] = self.sizes['npair']
            self.sizes['nfrac'] = 0

        if self.print_level > 1:
            if self.options['fomo']:
                print('FOMO:')
                print('  FOMO Method = %14s' % self.options['fomo_method'])
                print('  FOMO Temp   = %14.6E' % self.options['fomo_temp'])
                print('')

                print('Orbital Spaces:')
                print('  Nmo   = %d' % self.sizes['nmo'])
                print('  Nocc  = %d' % self.sizes['nocc'])
                print('  Nact  = %d' % self.sizes['nact'])
                print('  Nvir  = %d' % self.sizes['nvir'])
                print('  Npair = %d' % self.sizes['npair'])
                print('  Nclsd = %d' % self.sizes['nclsd'])
                print('  Nfrac = %d' % self.sizes['nfrac'])
                print('') 
            else:
                print('Orbital Spaces:')
                print('  Nocc  = %d' % self.sizes['nocc'])
                print('  Nvir  = %d' % self.sizes['nvir'])
                print('') 

        # => MOM Considerations <= #

        is_mom = Cocc_mom or Cact_mom
        if Cocc_mom:
            Cocc_mom.shape_error((self.sizes['nao'], self.sizes['nocc']))
            CSocc_mom = ls.Tensor.chain([Cocc_mom, self.tensors['S']], [True, False])
        if Cact_mom:
            Cact_mom.shape_error((self.sizes['nao'], self.sizes['nact']))
            CSact_mom = ls.Tensor.chain([Cact_mom, self.tensors['S']], [True, False])
        if self.print_level > 1:
            print('Orbital Occupations:')
            if is_mom:
                print('  Method  = MOM')
                print('  MOM occ = %s' % ('True' if Cocc_mom else 'False'))
                print('  MOM act = %s' % ('True' if Cact_mom else 'False'))
            else: 
                print('  Method = Aufbau')
            print('')
        if is_mom and self.is_fomo and not Cact_mom:
            raise ValueError('Must include Cact_mom if MOM used with FOMO')

        # => Integral Thresholds <= #

        if self.print_level > 1:
            print('Integral Thresholds:')
            print('  Threshold DP = %11.3E' % self.options['thre_dp'])
            print('  Threshold SP = %11.3E' % self.options['thre_sp'])
            print('')
        
        # => DIIS <= #

        diis = ls.DIIS(
            self.options['diis_max_vecs'],
            )
        if self.print_level > 1: 
            print(diis)

        # ==> SCF Iterations <== #

        if self.print_level > 1:
            print('Convergence Options:')
            print('  Max Iterations = %11d' % self.options['maxiter'])
            print('  Inc Iterations = %11d' % self.options['incremental_niter'])
            print('  E Convergence  = %11.3E' % self.options['e_convergence'])
            print('  G Convergence  = %11.3E' % self.options['g_convergence'])
            print('')

        J = ls.Tensor.zeros_like(self.tensors['D'])
        K = ls.Tensor.zeros_like(self.tensors['D'])
        wK = ls.Tensor.zeros_like(self.tensors['D'])

        Dold = ls.Tensor.zeros_like(self.tensors['D'])

        a = 1.0 if not self.is_dft else self.dft_functional.alpha
        b = 0.0 if not self.is_dft else self.dft_functional.beta

        if self.is_dft and b != 0.0 and (not self.ewald.is_coulomb):
            raise ValueError("Long range corrected DFT with non-Coulomb Ewald operators ill defined.")

        start = time.time()
        Eold = 0.0
        self.scf_converged = False
        if self.print_level:
            print('SCF Iterations:\n')
            print('%4s: %24s %11s %11s %8s' % ('Iter', 'Energy', 'dE', 'dG', 'Time[s]'))
        for iteration in range(self.options['maxiter']):

            # => Incremental D Matrix <= #

            if iteration % self.options['incremental_niter'] == 0:
                Dold.zero()
                J.zero()
                K.zero()
                wK.zero()
            Dd = ls.Tensor.array(self.tensors['D'])
            Dd[...] -= Dold
            Dold = self.tensors['D']
            
            # => Compute J Matrix <= #

            J = ls.IntBox.coulomb(
                self.resources,
                self.ewald,
                self.pairlist,
                self.pairlist,
                Dd,
                self.options['thre_sp'],    
                self.options['thre_dp'],    
                J,
                )

            # => Compute K Matrix <= #

            if self.is_dft and a == 0.0:
                pass
            else:
                K = ls.IntBox.exchange(
                    self.resources,
                    self.ewald,
                    self.pairlist,
                    self.pairlist,
                    Dd,
                    True,
                    self.options['thre_sp'],    
                    self.options['thre_dp'],    
                    K,
                    )

            # => Compute Kw Matrix (Long-Range) <= #

            if self.is_dft and b != 0.0:
                wK = ls.IntBox.exchange(
                    self.resources,
                    ls.Ewald([1.0],[self.dft_functional.omega]),
                    self.pairlist,
                    self.pairlist,
                    Dd,
                    True,
                    self.options['thre_sp'],    
                    self.options['thre_dp'],    
                    wK,
                    )

            # => Compute XC Matrix <= #

            if self.is_dft:
                ret = ls.DFTBox.rksPotential(
                    self.resources,
                    self.dft_functional,
                    self.dft_grid,
                    self.dft_hash,
                    self.pairlist,
                    self.tensors['D'],
                    self.options['dft_thre_grid'],
                    ) 
                Vxc = ret[1]
                Exc = ret[0][0]
                Qgrid = ret[0][1]
            else:
                Vxc = ls.Tensor.zeros_like(J)
                Exc = 0.0
                Qgrid = float(self.sizes['nocc'])

            # => Build Fock Matrix <= #            

            self.tensors['F'] = ls.Tensor.array(self.tensors['H'])
            self.tensors['F'].np[...] += 2. * J.np
            self.tensors['F'].np[...] -= a * K.np
            self.tensors['F'].np[...] -= b * wK.np
            self.tensors['F'].np[...] += 1. * Vxc.np

            # => Compute FOMO Entropy Contribution to Energy <= #

            if self.is_fomo and iteration > 0:
                self.scalars['TSfomo'] = 2.0 * RHF.TS_fomo(
                    self.options['fomo_method'],
                    self.options['fomo_temp'],
                    self.tensors['n'][self.sizes['nclsd']:self.sizes['nclsd']+self.sizes['nact']],
                    self.tensors['eps'][self.sizes['nclsd']:self.sizes['nclsd']+self.sizes['nact']],
                    self.scalars['Emu']
                    )
            else:
                self.scalars['TSfomo'] = 0.0

            # => Compute Energy <= #

            self.scalars['Escf'] = self.scalars['Eext'] + \
                 2.0 * self.tensors['D'].vector_dot(self.tensors['H']) + \
                 2.0 * self.tensors['D'].vector_dot(J) + \
                -a * self.tensors['D'].vector_dot(K) + \
                -b * self.tensors['D'].vector_dot(wK) + \
                Exc + \
                self.scalars['TSfomo']
            dE = self.scalars['Escf'] - Eold
            Eold = self.scalars['Escf']

            # => Compute Orbital Gradient <= #

            G1 = ls.Tensor.chain([self.tensors[x] for x in ['X', 'S', 'D', 'F', 'X']], [True] + [False]*4)
            G1.antisymmetrize()
            G1[...] *= 2.0
            dG = np.max(np.abs(G1))

            # => Print Iteration <= #
            
            stop = time.time()
            if self.print_level:
                print('%4d: %24.16E %11.3E %11.3E %8.3f' % (iteration, self.scalars['Escf'], dE, dG, stop-start))
            start = stop
    
            # => Check Convergence <= #
    
            if iteration > 0 and abs(dE) < self.options['e_convergence'] and dG < self.options['g_convergence']:
                self.scf_converged = True
                break

            # This ensures that we do not extrapolate/diagonalize if maxiter is reached
            if iteration >= self.options['maxiter'] - 1:
                break

            # => Flush DIIS if Stagnated <= #

            if iteration > 0 and iteration % self.options['diis_flush_niter'] == 0:
                diis = ls.DIIS(
                    self.options['diis_max_vecs'],
                    )
    
            # => DIIS Fock Matrix <= #

            if iteration > 0: # First iteration might not be idempotent
                self.tensors['F'] = diis.iterate(self.tensors['F'], G1)

            # => Diagonalize Fock Matrix <= #

            F2 = ls.Tensor.chain([self.tensors[x] for x in ['X', 'F', 'X']],[True,False,False])
            e2, U2 = ls.Tensor.eigh(F2)
            C2 = ls.Tensor.chain([self.tensors['X'], U2],[False,False])
            e2.name = 'eps'
            C2.name = 'C'
            self.tensors['C'] = C2
            self.tensors['eps'] = e2

            # => Order Orbitals According to MOM <= #

            if is_mom:
                # Active orbitals are first priority, and must be MOM'd if MOM
                if self.is_fomo:
                    nact_mom = ls.Tensor.array(np.sum(ls.Tensor.chain([CSact_mom, self.tensors['C']], [False, False])[...]**2,0))
                    inds_act = [x for x in np.argsort(-nact_mom[...])][:self.sizes['nact']] 
                    dQact_mom = np.sum(nact_mom) - np.sum(nact_mom[inds_act])
                else:
                    inds_act = []
                    dQact_mom = 0.0
                # Occupied orbitals are second priority, and MOM is optional (active-only MOM)
                if Cocc_mom:
                    nocc_mom = ls.Tensor.array(np.sum(ls.Tensor.chain([CSocc_mom, self.tensors['C']], [False, False])[...]**2,0))
                    inds_occ = [x for x in np.argsort(-nocc_mom[...]) if x not in inds_act][:self.sizes['nocc']] 
                    dQocc_mom = np.sum(nocc_mom) - np.sum(nocc_mom[inds_occ])
                else:
                    inds_occ = [x for x in range(self.sizes['nmo']) if x not in inds_act][:self.sizes['nocc']]
                    dQocc_mom = 0.0
                # Virtual orbitals are third priority, and there is no MOM concept
                inds_vir = [x for x in range(self.sizes['nmo']) if x not in inds_occ and x not in inds_act]
                # Aufbau order withinn each set
                inds_occ.sort()
                inds_act.sort()
                inds_vir.sort()
                inds_all = inds_occ + inds_act + inds_vir
                C2 = ls.Tensor.zeros_like(self.tensors['C'])
                C2[...] = self.tensors['C'][:,inds_all]
                self.tensors['C'] = C2
                eps2 = ls.Tensor.zeros_like(self.tensors['eps'])
                eps2[...] = self.tensors['eps'][inds_all]
                self.tensors['eps'] = eps2

            # => Orbital Occupation Numbers (FOMO or RHF) <= #

            if self.is_fomo:
                # Closed orbitals
                self.tensors['n'] = ls.Tensor((self.sizes['nmo'],),'n')
                self.tensors['n'][:self.sizes['nclsd']] = 1.0
                # Fractional orbitals
                fact, mu = RHF.occupy_fomo(
                    self.options['fomo_method'],
                    self.options['fomo_temp'],
                    self.sizes['nfrac'],
                    self.tensors['eps'][self.sizes['nclsd']:self.sizes['nclsd']+self.sizes['nact']],
                    )
                self.tensors['n'][self.sizes['nclsd']:self.sizes['nclsd']+self.sizes['nact']] = fact
                self.scalars['Emu'] = mu
            else:
                self.tensors['n'] = ls.Tensor((self.sizes['nmo'],),'n')
                self.tensors['n'][:self.sizes['npair']] = 1.0
                # TODO: Is this right? (not that it really matters)
                self.scalars['Emu'] = 0.5 * np.sum(self.tensors['eps'][self.sizes['npair']-1:self.sizes['npair']])

            # => Compute Density Matrix <= #

            Ctemp = ls.Tensor.array(self.tensors['C'][:,:self.sizes['nocc']+self.sizes['nact']])
            Ctemp[...] *= np.outer(np.ones((self.sizes['nao'],)),np.sqrt(self.tensors['n'][:self.sizes['nocc']+self.sizes['nact']]))
            self.tensors['D'] = ls.Tensor.chain([Ctemp,Ctemp],[False,True])

        # => Print Convergence <= #
        
        if self.print_level:
            print('')
            if self.scf_converged:
                print('SCF Converged\n')
            else:
                print('SCF Failed\n')
        
        # => Print Final Energy <= #
        
        if self.print_level:
            print('SCF Energy = %24.16E\n' % self.scalars['Escf'])

        # => Print FOMO Energy Contribution <= #

        if self.print_level and self.is_fomo:
            print('SCF Internal Energy (E)     = %24.16E' % (self.scalars['Escf'] - self.scalars['TSfomo']))
            print('SCF Entropy Term (-T * S)   = %24.16E' % (self.scalars['TSfomo']))
            print('SCF Free Energy (E - T * S) = %24.16E' % (self.scalars['Escf']))
            print('')

        # => Cache the Orbitals/Eigenvalues (for later use) <= #

        self.tensors['Cocc'] = ls.Tensor((self.sizes['nao'], self.sizes['nocc']),'Cocc')
        self.tensors['eps_occ'] = ls.Tensor((self.sizes['nocc'],), 'eps_occ')
        self.tensors['nocc'] = ls.Tensor((self.sizes['nocc'],), 'nocc')
        if self.sizes['nocc']:
            self.tensors['Cocc'].np[...] = self.tensors['C'].np[:,:self.sizes['nocc']]
            self.tensors['eps_occ'].np[...] = self.tensors['eps'].np[:self.sizes['nocc']]
            self.tensors['nocc'].np[...] = self.tensors['n'].np[:self.sizes['nocc']]

        self.tensors['Cvir'] = ls.Tensor((self.sizes['nao'], self.sizes['nvir']), 'Cvir')
        self.tensors['eps_vir'] = ls.Tensor((self.sizes['nvir'],), 'eps_vir')
        self.tensors['nvir'] = ls.Tensor((self.sizes['nvir'],), 'nvir')
        if self.sizes['nvir']:
            self.tensors['Cvir'].np[...] = self.tensors['C'].np[:,self.sizes['nocc']+self.sizes['nact']:]
            self.tensors['eps_vir'].np[...] = self.tensors['eps'].np[self.sizes['nocc']+self.sizes['nact']:]
            self.tensors['nvir'].np[...] = self.tensors['n'].np[self.sizes['nocc']+self.sizes['nact']:]

        self.tensors['Cact'] = ls.Tensor((self.sizes['nao'], self.sizes['nact']), 'Cact')
        self.tensors['eps_act'] = ls.Tensor((self.sizes['nact'],), 'eps_act')
        self.tensors['nact'] = ls.Tensor((self.sizes['nact'],), 'nact')
        if self.sizes['nact']:
            self.tensors['Cact'].np[...] = self.tensors['C'].np[:,self.sizes['nocc']:self.sizes['nocc']+self.sizes['nact']]
            self.tensors['eps_act'].np[...] = self.tensors['eps'].np[self.sizes['nocc']:self.sizes['nocc']+self.sizes['nact']]
            self.tensors['nact'].np[...] = self.tensors['n'].np[self.sizes['nocc']:self.sizes['nocc']+self.sizes['nact']]

        # => Print Fractional Orbitals or HOMO/LUMO <= #
        
        if self.print_level > 1 and self.is_fomo:
            enp = self.tensors['eps'].np
            nnp = self.tensors['n'].np
            start = self.sizes['nocc'] - min(self.sizes['nocc'], 2)
            stop = self.sizes['nocc'] + self.sizes['nact'] + min(self.sizes['nvir'], 2)
            print('FOMO Orbitals:')
            print('  %-5s %12s %12s' % ('Orb', 'Energy', 'Occupation'))
            for p in range(start, stop):    
                if p == self.sizes['nocc']: print(('  ' + '-' * 31))
                if p == (self.sizes['nocc'] + self.sizes['nact']): print(('  ' + '-' * 31))
                print('  %-5d %12.6f %12.6f' % (p, enp[p], nnp[p]))
            print('')

        if self.print_level > 1 and not self.is_fomo:
            if self.sizes['nocc']: print('HOMO Level: %12.6f' % self.tensors['eps'][self.sizes['nocc']-1])
            if self.sizes['nvir']: print('LUMO Level: %12.6f' % self.tensors['eps'][self.sizes['nocc']])
            if self.sizes['nocc'] or self.sizes['nvir']: print('')

        # => Print MOM Spanning Loss Metric <= #

        if self.print_level >= 1 and is_mom:
            print('MOM Loss Metric:')
            print('  Loss occ: %.3f' % dQocc_mom)
            if self.is_fomo:
                print('  Loss act: %.3f' % dQact_mom)
            print('')

        # TODO: Print orbitals
        
        # => Print Integrated Grid Density <= #

        if self.print_level > 1 and self.is_dft:
            print('Becke Grid Density:')
            print('  Expected = %14.6f' % (float(self.sizes['npair'])))
            print('  Observed = %14.6f' % (Qgrid))
            print('  Error    = %14.3E' % (Qgrid - float(self.sizes['npair'])))
            print('')
    
        # => Trailer Bars <= #

        if self.print_level > 1:
            print('"I love it when a plan comes together!"')
            print('        --LTC John "Hannibal" Smith\n')

        if self.print_level:
            print('==> End RHF <==\n')

        # Return if the SCF procedure converged
        return self.scf_converged

    def compute_gradient(
        self,
        ):

        if self.is_fomo:
            raise RuntimeError('FOMO-RHF gradient not implemented')

        if self.is_dft and self.dft_functional.beta != 0.0 and (not self.ewald.is_coulomb):
            raise ValueError("Long range corrected DFT with non-Coulomb Ewald operators ill defined.")

        # => Density Matrices <= #

        # OPDM 
        D = self.tensors['D']
        # Energy-weighted OPDM
        C1 = self.tensors['Cocc']
        C2 = ls.Tensor.zeros_like(C1)
        C2.np[...] = np.einsum('pi,i->pi', C1, self.tensors['eps_occ'])
        W = ls.Tensor.chain([C1,C2],[False,True])

        # => Gradient Contributions <= # 
        
        keys = [
            'S',
            'T',
            'J',
            'K',
            'Kw',
            'XC',
            ]
        grads = {}

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
        # Coulomb integral gradient
        grads['J'] = ls.IntBox.coulombGrad(
            self.resources,
            self.ewald,
            self.pairlist,
            D,
            D,
            self.options['grad_thre_sp'],
            self.options['grad_thre_dp'],
            )
        grads['J'].np[...] *= 2.
        # Exchange integral gradient
        if self.is_dft and self.dft_functional.alpha == 0.0:
            grads['K'] = ls.Tensor.zeros_like(grads['J'])
        else:
            grads['K'] = ls.IntBox.exchangeGrad(
                self.resources,
                self.ewald,
                self.pairlist,
                D,
                D,
                True, 
                True,
                True,
                self.options['grad_thre_sp'],
                self.options['grad_thre_dp'],
                )
            grads['K'].np[...] *= -1.
        if self.is_dft:
            grads['K'].scale(self.dft_functional.alpha)
        # Exchange integral gradient (LR)
        if self.is_dft and self.dft_functional.beta != 0.0:
            grads['Kw'] = ls.IntBox.exchangeGrad(
                self.resources,
                ls.Ewald([1.0],[self.dft_functional.omega]),
                self.pairlist,
                D,
                D,
                True, 
                True,
                True,
                self.options['grad_thre_sp'],
                self.options['grad_thre_dp'],
                )
            grads['Kw'].scale(-self.dft_functional.beta)
        else:
            grads['Kw'] = ls.Tensor.zeros_like(grads['J'])

        # XC gradient
        if self.is_dft:
            grads['XC'] = ls.DFTBox.rksGrad(
                self.resources,
                self.dft_functional,
                self.dft_grid,
                self.dft_hash,
                self.pairlist,
                self.tensors['D'],
                self.options['grad_dft_thre_grid'],
                ) 
        else:
            grads['XC'] = ls.Tensor.zeros_like(grads['K'])
        
        # Assemble electronic gradient
        G = ls.Tensor.zeros_like(grads['S'])
        G.name = 'G (RHF)'
        for key in keys:
            G.np[...] += grads[key]

        # print '==> RHF Gradient Contributions <==\n'
        # for key in keys:
        #     grads[key].name = key
        #     print grads[key]
        # print G

        # Need total density for Geometry
        Dt = ls.Tensor.array(D)
        Dt[...] *= 2.0

        # Have the Geometry object handle external gradient
        return self.geometry.compute_gradient(
            self.pairlist,
            Dt,
            G) 

    # => CPHF Utility <= #

    def compute_cphf(
        self,
        G,
        ):

        # => Validity Checks <= #

        if self.is_dft: raise RuntimeError("Cannot solve CPKS")

        # Check that G is (nmo, nmo)
        G.shape_error((self.sizes['nmo'],)*2) 

        # => Header <= #

        if self.print_level:
            print('==> CPHF <==\n')

        # => DIIS <= #

        diis = ls.DIIS(
            self.options['cphf_diis_max_vecs'],
            )
        if self.print_level > 1:
            print(diis)

        if self.print_level > 1:
            print('Convergence Options:')
            print('  Max Iterations = %11d' % self.options['cphf_maxiter'])
            print('  Inc Iterations = %11d' % self.options['cphf_incremental_niter'])
            print('  R Convergence  = %11.3E' % self.options['cphf_r_convergence'])
            print('')

        # => Preconditioner <= #

        # Fock-Matrix Contribution (Preconditioner)
        F = ls.Tensor((self.sizes['nmo'], self.sizes['nmo']), 'F')
        F[...] -= np.outer(self.tensors['eps'], np.ones(self.sizes['nmo']))
        F[...] += np.outer(np.ones(self.sizes['nmo']), self.tensors['eps'])
        F[...] += 1.2345678 * np.eye(self.sizes['nmo']) # Prevents division by zero

        # => Particle Number Difference <= #

        N = ls.Tensor((self.sizes['nmo'], self.sizes['nmo']), 'N')
        N[...] += np.outer(self.tensors['n'], np.ones(self.sizes['nmo']))
        N[...] -= np.outer(np.ones(self.sizes['nmo']), self.tensors['n'])

        # => Upper-Triangular Mask <= #

        M = ls.Tensor.array(1.0 - np.tri(self.sizes['nmo'], self.sizes['nmo']))

        # => FOMO Intermediates <= #

        if self.is_fomo:
            fomoAinvB = self.fomoAinvB
            fomoJK = self.fomoJK
    
        # => Initial Guess (UCHF Result) <= #

        Z = ls.Tensor.array(G)
        Z.np[...] /= F.np[...] 
        Z[...] *= M[...] # Mask

        # ==> CPHF Iterations <== #

        Dold = ls.Tensor.zeros_like(self.tensors['S'])
        T = ls.Tensor.zeros_like(self.tensors['S'])

        start = time.time()
        converged = False
        if self.print_level:
            print('CPHF Iterations:\n')
            print('%4s: %11s %8s' % ('Iter', 'dR', 'Time[s]'))
        for iteration in range(self.options['cphf_maxiter']):

            # Residual 
            R = ls.Tensor.array(G)
    
            # One-electron term
            R[...] -= F[...] * Z[...]         
    
            # MO-to-AO transform of trial Z vector
            D = ls.Tensor.chain([self.tensors['C'],Z,self.tensors['C']],[False,False,True])
            D.symmetrize()

            # Incremental CPHF
            if iteration % self.options['cphf_incremental_niter'] == 0:
                Dold.zero()
                T.zero()
            Dd = ls.Tensor.array(D)
            Dd[...] -= Dold
            Dold = D

            # J/K Terms
            J = ls.IntBox.coulomb(
                self.resources,
                self.ewald,
                self.pairlist,
                self.pairlist,
                Dd,
                self.options['cphf_thre_sp'],    
                self.options['cphf_thre_dp'],    
                )
            K = ls.IntBox.exchange(
                self.resources,
                self.ewald,
                self.pairlist, 
                self.pairlist,
                Dd,
                True,
                self.options['cphf_thre_sp'],    
                self.options['cphf_thre_dp'],    
                )
            T[...] += 4.0 * J[...]
            T[...] -= 2.0 * K[...]

            # FOMO terms
            if self.is_fomo:
                i1 = ls.Tensor.einsum('epq,pq->e', fomoJK, Dd)
                i2 = ls.Tensor.einsum('ef,e->f', fomoAinvB, i1)
                T[...] += 2.0 * ls.Tensor.einsum('epq,e->pq', fomoJK, i2)[...] # Definitely 2.0

            # Residual Assembly
            R.np[...] -= N[...] * ls.Tensor.chain([self.tensors['C'],T,self.tensors['C']],[True,False,False]).np

            # Precondition the residual
            R.np[...] /= F.np 
            R.np[...] *= M.np # Mask the residual
            Z.np[...] += R.np # Approximate NR-Step

            # => Check Convergence <= #

            stop = time.time()
            Rmax = np.max(np.abs(R))             
            if self.print_level:
                print('%4d: %11.3E %8.3f' % (iteration, Rmax, stop-start))
            if Rmax < self.options['cphf_r_convergence']:
                converged = True
                break
            start = stop

            # => Perform DIIS <= #
            
            Z = diis.iterate(Z,R,use_disk=self.options['cphf_diis_use_disk'])

        # => Print Convergence <= #
        
        if self.print_level:
            print('')
            if converged:
                print('CPHF Converged\n')
            else:
                print('CPHF Failed\n')

        # => Trailer <= #

        if self.print_level:
            print('==> End CPHF <==\n')

        return Z

    def compute_cphf_ov(
        self,
        G,
        ):

        # => Validity Checks <= #

        if self.is_dft: raise RuntimeError("Cannot solve CPKS")
        if self.is_fomo: raise RuntimeError("This is FOMO-HF: must use full compute_cphf")

        # Check that G is (nocc, nvir)
        G.shape_error((self.sizes['nocc'], self.sizes['nvir']))

        # => Header <= #

        if self.print_level:
            print('==> CPHF (Occ/Vir) <==\n')

        # => DIIS <= #

        diis = ls.DIIS(
            self.options['cphf_diis_max_vecs'],
            )
        if self.print_level > 1:
            print(diis)

        if self.print_level > 1:
            print('Convergence Options:')
            print('  Max Iterations = %11d' % self.options['cphf_maxiter'])
            print('  Inc Iterations = %11d' % self.options['cphf_incremental_niter'])
            print('  R Convergence  = %11.3E' % self.options['cphf_r_convergence'])
            print('')

        # => Preconditioner <= #

        # Fock-Matrix Contribution (Preconditioner)
        F = ls.Tensor((self.sizes['nocc'], self.sizes['nvir']), 'F')
        F[...] -= np.outer(self.tensors['eps_occ'], np.ones(self.sizes['nvir']))
        F[...] += np.outer(np.ones(self.sizes['nocc']), self.tensors['eps_vir'])

        # => Initial Guess (UCHF Result) <= #

        Z = ls.Tensor.array(G)
        Z.np[...] /= F.np[...] 

        # ==> CPHF Iterations <== #

        Dold = ls.Tensor.zeros_like(self.tensors['S'])
        T = ls.Tensor.zeros_like(self.tensors['S'])

        start = time.time()
        converged = False
        if self.print_level:
            print('CPHF Iterations:\n')
            print('%4s: %11s %8s' % ('Iter', 'dR', 'Time[s]'))
        for iteration in range(self.options['cphf_maxiter']):

            # Residual 
            R = ls.Tensor.array(G)
    
            # One-electron term
            R[...] -= F[...] * Z[...]         
    
            # MO-to-AO transform of trial Z vector
            D = ls.Tensor.chain([self.tensors['Cocc'],Z,self.tensors['Cvir']],[False,False,True])
            D.symmetrize()

            # Incremental CPHF
            if iteration % self.options['cphf_incremental_niter'] == 0:
                Dold.zero()
                T.zero()
            Dd = ls.Tensor.array(D)
            Dd[...] -= Dold
            Dold = D

            # J/K Terms
            J = ls.IntBox.coulomb(
                self.resources,
                self.ewald,
                self.pairlist,
                self.pairlist,
                Dd,
                self.options['cphf_thre_sp'],    
                self.options['cphf_thre_dp'],    
                )
            K = ls.IntBox.exchange(
                self.resources,
                self.ewald,
                self.pairlist, 
                self.pairlist,
                Dd,
                True,
                self.options['cphf_thre_sp'],    
                self.options['cphf_thre_dp'],    
                )
            T[...] += 4.0 * J[...]
            T[...] -= 2.0 * K[...]

            # Residual Assembly
            R.np[...] -= ls.Tensor.chain([self.tensors['Cocc'],T,self.tensors['Cvir']],[True,False,False]).np

            # Precondition the residual
            R.np[...] /= F.np 
            Z.np[...] += R.np # Approximate NR-Step

            # => Check Convergence <= #

            stop = time.time()
            Rmax = np.max(np.abs(R))             
            if self.print_level:
                print('%4d: %11.3E %8.3f' % (iteration, Rmax, stop-start))
            if Rmax < self.options['cphf_r_convergence']:
                converged = True
                break
            start = stop

            # => Perform DIIS <= #
            
            Z = diis.iterate(Z,R,use_disk=self.options['cphf_diis_use_disk'])

        # => Print Convergence <= #
        
        if self.print_level:
            print('')
            if converged:
                print('CPHF Converged\n')
            else:
                print('CPHF Failed\n')

        # => Trailer <= #

        if self.print_level:
            print('==> End CPHF <==\n')

        return Z

    def compute_hf_based_gradient(
        self,
        D,
        X,
        GI,
        ):

        """ Compute and assemble the gradient for a post-HF method based on
            HF or FOMO-HF orbitals.

        Params:
            D (ls.Tensor) - total OPDM in the AO basis (used for contraction
                with T+Vext integral derivatives)
            X (ls.Tensor) - orbital Lagrangian in the MO basis, defined as
                X_{tu} = \sum_{mu} \pdiff{E}{C_mu,t} C_mu,u. This is used to
                form the orbital gradient, solve the Z-vector equations, and to
                contract with the S integral derivatives)
            GI (ls.Tensor) - Gradient on the QM molecule from the
                two-particle density matrix [D2_pqrs (pq|rs)^z]
        Returns:
            grad (ls.Tensor) - total gradient for self.geometry.molecule,
                including all response, one-electron, two-electron, and
                environmental pieces.
        """

        if self.is_dft: raise RuntimeError("Cannot solve CPKS")

        # Form Z-Vector RHS: (X - X')
        G = ls.Tensor.array(X)
        G.antisymmetrize() # 0.5 * (X - X')
        G[...] *= 2.0 # Ed
        G[...] = np.triu(G, k=1) # Restrict to upper triangle

        # Solve Z-Vector equations
        Z = self.compute_cphf(G)

        # Symmetrize Z and place in AO basis
        Z.symmetrize()
        Z2 = ls.Tensor.chain([self.tensors['C'], Z, self.tensors['C']], [False, False, True])

        # Full MO-basis X matrix
        X3 = ls.Tensor.zeros_like(X)

        # Augment Z2 and X with FOMO piece
        if self.is_fomo:
            fomoAinvB = self.fomoAinvB
            fomoJK = self.fomoJK
            i1 = ls.Tensor.einsum('epq,pq->e', fomoJK, Z2)
            i2 = ls.Tensor.einsum('ef,e->f', fomoAinvB, i1) # Ed's z_g
            Z2[...] += 1.0 * ls.Tensor.chain([self.tensors['Cact'], ls.Tensor.array(np.diag(i2)), self.tensors['Cact']], [False, False, True]).np 
            # Weird diagonal term (typo in Ed's paper - there should be no occupation number)
            diag = 2.0 * self.tensors['eps_act'][...] * i2[...]
            for ind in range(self.sizes['nact']):
                ind2 = ind + self.sizes['nocc'] 
                X3[ind2, ind2] += diag[ind]
                
        # Lagrangian Fock contributions
        JK = ls.IntBox.coulomb(
            self.resources,
            self.ewald,
            self.pairlist,
            self.pairlist,
            Z2,
            self.options['thre_sp'],    
            self.options['thre_dp'],    
            )
        JK[...] *= -2.0
        JK = ls.IntBox.exchange(
            self.resources,
            self.ewald,
            self.pairlist, 
            self.pairlist,
            Z2,
            True,
            self.options['thre_sp'],    
            self.options['thre_dp'],    
            JK,
            )
        JK[...] *= -4.0

        JK2 = ls.Tensor.chain([self.tensors['C'], JK, self.tensors['C']], [True, False, False])
    
        X3[...] += 2.0 * np.outer(self.tensors['eps'], np.ones(self.sizes['nmo'])) * Z[...] # Definite + 2.0
        X3[...] += 0.5 * np.outer(np.ones(self.sizes['nmo']), self.tensors['n']) * JK2[...] # Definite + 0.5
        X3[...] += X
        X3.symmetrize()

        GJ = ls.IntBox.coulombGrad(
            self.resources,
            self.ewald,
            self.pairlist,
            Z2,
            self.tensors['D'],
            self.options['thre_sp'],
            self.options['thre_dp'],
            )
        GJ.name = 'GJ'
        GJ[...] *= 2.0
        GK = ls.IntBox.exchangeGrad(
            self.resources,
            self.ewald,
            self.pairlist,
            Z2,
            self.tensors['D'],
            True,
            True,
            False,
            self.options['thre_sp'],
            self.options['thre_dp'],
            )
        GK.name = 'GK'
        GK[...] *= -1.0

        D2 = ls.Tensor.array(D)
        D2[...] += Z2

        GT = ls.IntBox.kineticGrad(
            self.resources,
            self.pairlist,
            D2,
            )
        GT.name = 'GT'
        GT[...] *= 1.0

        X2 = ls.Tensor.chain([self.tensors['C'], X3, self.tensors['C']], [False, False, True])
        GS = ls.IntBox.overlapGrad(
            self.resources,
            self.pairlist,
            X2,
            )
        GS.name = 'GS'
        GS[...] *= -0.5

        # print GJ
        # print GK
        # print GT
        # print GS
    
        G = ls.Tensor.array(GJ[...] + GK[...] + GT[...] + GS[...] + GI[...])

        return self.geometry.compute_gradient(
            self.pairlist,
            D2,
            G)

    def compute_hf_based_coupling(
        self,
        D,
        X,
        GI,
        dE,
        doETF=False,
        ):

        """ Compute and assemble the derivative coupling for a post-HF method
            based on HF or FOMO-HF orbitals.

        Params:
            D (ls.Tensor) - transition OPDM in the AO basis (used for contraction
                with T+Vext integral derivatives *and* with (p|q^z), so do not
                symmetrize this object)
            X (ls.Tensor) - orbital Lagrangian in the MO basis, defined as
                X_{tu} = \sum_{mu} \pdiff{E}{C_mu,t} C_mu,u. This is used to
                form the orbital gradient, solve the Z-vector equations, and to
                contract with the S integral derivatives)
            GI (ls.Tensor) - Gradient on the QM molecule from the
                two-particle transition density matrix [D2_pqrs (pq|rs)^z]
            dE (float) - State energy difference (EA - EB), used as energy
                denominator in weighting gradient contributions.
            doETF (bool) - apply Joe Subotnik's "Electronic Translation Factor"
                (ETF) approach as in JCP 135, 234015 (2011).
        Returns:
            grad (ls.Tensor) - total gradient for self.geometry.molecule,
                including all response, one-electron, two-electron, and
                environmental pieces.
        """
        
        # TODO: Clean this up

        if self.is_dft: raise RuntimeError("Cannot solve CPKS")

        # Form Z-Vector RHS: (X - X')
        G = ls.Tensor.array(X)
        G.antisymmetrize() # 0.5 * (X - X')
        G[...] *= 2.0 # Ed
        G[...] = np.triu(G, k=1) # Restrict to upper triangle

        # Solve Z-Vector equations
        Z = self.compute_cphf(G)

        # Symmetrize Z and place in AO basis
        Z.symmetrize()
        Z2 = ls.Tensor.chain([self.tensors['C'], Z, self.tensors['C']], [False, False, True])

        # Full MO-basis X matrix
        X3 = ls.Tensor.zeros_like(X)

        # Augment Z2 and X with FOMO piece
        if self.is_fomo:
            fomoAinvB = self.fomoAinvB
            fomoJK = self.fomoJK
            i1 = ls.Tensor.einsum('epq,pq->e', fomoJK, Z2)
            i2 = ls.Tensor.einsum('ef,e->f', fomoAinvB, i1) # Ed's z_g
            Z2[...] += 1.0 * ls.Tensor.chain([self.tensors['Cact'], ls.Tensor.array(np.diag(i2)), self.tensors['Cact']], [False, False, True]).np 
            # Weird diagonal term (typo in Ed's paper - there should be no occupation number)
            diag = 2.0 * self.tensors['eps_act'][...] * i2[...]
            for ind in range(self.sizes['nact']):
                ind2 = ind + self.sizes['nocc'] 
                X3[ind2, ind2] += diag[ind]
                
        # Lagrangian Fock contributions
        JK = ls.IntBox.coulomb(
            self.resources,
            self.ewald,
            self.pairlist,
            self.pairlist,
            Z2,
            self.options['thre_sp'],    
            self.options['thre_dp'],    
            )
        JK[...] *= -2.0
        JK = ls.IntBox.exchange(
            self.resources,
            self.ewald,
            self.pairlist, 
            self.pairlist,
            Z2,
            True,
            self.options['thre_sp'],    
            self.options['thre_dp'],    
            JK,
            )
        JK[...] *= -4.0

        JK2 = ls.Tensor.chain([self.tensors['C'], JK, self.tensors['C']], [True, False, False])
    
        X3[...] += 2.0 * np.outer(self.tensors['eps'], np.ones(self.sizes['nmo'])) * Z[...] # Definite + 2.0
        X3[...] += 0.5 * np.outer(np.ones(self.sizes['nmo']), self.tensors['n']) * JK2[...] # Definite + 0.5
        X3[...] += X
        # Diagonal needs a factor of 0.5 in Ed's definition
        X3np = X3.np
        for p in range(self.sizes['nmo']):
            X3np[p] *= 0.5
        X3.symmetrize()

        GJ = ls.IntBox.coulombGrad(
            self.resources,
            self.ewald,
            self.pairlist,
            Z2,
            self.tensors['D'],
            self.options['thre_sp'],
            self.options['thre_dp'],
            )
        GJ.name = 'GJ'
        GJ[...] *= 2.0
        GK = ls.IntBox.exchangeGrad(
            self.resources,
            self.ewald,
            self.pairlist,
            Z2,
            self.tensors['D'],
            True,
            True,
            False,
            self.options['thre_sp'],
            self.options['thre_dp'],
            )
        GK.name = 'GK'
        GK[...] *= -1.0

        D2 = ls.Tensor.array(D)
        D2[...] += Z2
        D2.symmetrize() # D is not intrinsically symmetric

        GT = ls.IntBox.kineticGrad(
            self.resources,
            self.pairlist,
            D2,
            )
        GT.name = 'GT'
        GT[...] *= 1.0

        X2 = ls.Tensor.chain([self.tensors['C'], X3, self.tensors['C']], [False, False, True])
        GS = ls.IntBox.overlapGrad(
            self.resources,
            self.pairlist,
            X2,
            )
        GS.name = 'GS'
        GS[...] *= -1.0

        if doETF:
            # D1_pq [(p^z|q) + (p|q^z)] / 2
            GS2 = ls.IntBox.overlapGrad(
                self.resources, 
                self.pairlist,
                D)
            GS2[...] *= 0.5 # Symmetrization
            GS2.name = 'GS2'
            GS2[...] *= dE
        else:
            # D1_pq (p|q^z)
            GS2 = ls.IntBox.overlapGradAdv(
                self.resources, 
                self.pairlist,
                D)[1]
            GS2.name = 'GS2'
            GS2[...] *= dE

        G = ls.Tensor.array(GJ[...] + GK[...] + GS[...] + GS2[...] + GT[...] + GI[...])

        G = self.geometry.compute_gradient(
            self.pairlist,
            D2,
            G,
            False)

        G[...] /= dE

        return G

    def compute_fomo_cphf_integrals(self):
        """ Compute the needed AinvB and JK intermediates for FOMO-CPHF. """
        if not self.is_fomo:
            raise ValueError('This RHF is not FOMO')
        if self.is_dft: raiseValueError('Cannot do FOMO + DFT response')
        
        # Form active-space JK matrices (2J - K)
        nao = self.tensors['Cact'].shape[0]
        nact = self.tensors['Cact'].shape[1]
        JKact = ls.Tensor((nact, nao, nao))
        for i in range(nact):
            Di = ls.Tensor.array(np.outer(self.tensors['Cact'][:,i], self.tensors['Cact'][:, i]))
            JK = ls.IntBox.coulomb( 
                self.resources,
                self.ewald,
                self.pairlist,
                self.pairlist,
                Di,
                self.options['thre_sp'],
                self.options['thre_dp'],
                )
            JK[...] *= -2.0
            ls.IntBox.exchange(
                self.resources,
                self.ewald,
                self.pairlist,
                self.pairlist,
                Di,
                True,
                self.options['thre_sp'],
                self.options['thre_dp'],
                JK
                )
            JK[...] *= -1.0
            JKact[i,:,:] = JK

        # JK2act[f,g] = 2 (ff|gg) - (fg|fg) 
        JK2act = ls.Tensor.zeros((nact,nact))
        for i in range(nact):
            Di = ls.Tensor.array(np.outer(self.tensors['Cact'][:,i], self.tensors['Cact'][:, i]))
            for j in range(nact):
                JK2act[i,j] = np.sum(Di * JKact[j, :, :])
        JK2act.symmetrize()
                
        # Compute a and b arrays from FOMO occupation scheme
        mu = self.scalars['Emu']
        if self.options['fomo_method'] == 'fermi':
            a = RHF.fermi_a(self.options['fomo_temp'], mu, self.tensors['eps_act'])
        elif self.options['fomo_method'] == 'gaussian':
            a = RHF.gaussian_a(self.options['fomo_temp'], mu, self.tensors['eps_act'])
        else:
            raise RuntimeError('Invalid fomo_method: %s' % self.options['fomo_method'])
        b = ls.Tensor.array(a / np.sum(a))
                
        # B matrix
        B = ls.Tensor.zeros((nact, nact))
        B[...] += np.diag(a)
        B[...] -= np.outer(a, b)

        # A matrix
        A = ls.Tensor.zeros((nact, nact))
        A[...] += np.eye(nact)
        A[...] -= ls.Tensor.chain([B, JK2act], [False, False])

        # Solution by LU decomposition
        AinvB = ls.Tensor.array(np.linalg.solve(A, B))
        # print np.linalg.cond(A)
        
        # Cache the values
        self._fomoJK = JKact
        self._fomoAinvB = AinvB

    @property
    def fomoAinvB(self):
        if not hasattr(self, '_fomoAinvB'): self.compute_fomo_cphf_integrals()
        return self._fomoAinvB

    @property
    def fomoJK(self):
        if not hasattr(self, '_fomoJK'): self.compute_fomo_cphf_integrals()
        return self._fomoJK

    # => FOMO Helper functions <= #

    @staticmethod
    def fermi_fon(
        kT,
        mu,
        eps,
        ):

        focc = ls.Tensor.array(1.0 / (1.0 + np.exp((1.0 / kT) * (eps[...] - mu))))
        N = np.sum(focc)
        return focc, N

    @staticmethod
    def fermi_a(
        kT,
        mu,
        eps,
        ):
    
        return ls.Tensor.array(-(1.0 / kT) * np.exp((1.0 / kT) * (eps[...] - mu)) / (1.0 + np.exp((1.0 / kT) * (eps[...] - mu)))**2)

    @staticmethod
    def gaussian_fon(
        kT,
        mu,
        eps,
        ):

        focc = ls.Tensor.array(0.5 * sp.erfc(1.0 / (np.sqrt(2.0) * kT) * (eps[...] - mu)))
        N = np.sum(focc)
        return focc, N

    @staticmethod
    def gaussian_a(
        kT,
        mu,
        eps,
        ):

        return ls.Tensor.array(-1.0 / (np.sqrt(2.0 * np.pi) * kT) * np.exp(-(eps[...] - mu)**2 / (2.0 * kT**2)))

    @staticmethod
    def occupy_fomo(
        method,
        kT,
        N,
        eps,
        ):

        step = None
        if method == 'fermi':
            step = RHF.fermi_fon
        elif method == 'gaussian':
            step = RHF.gaussian_fon
        else:
            raise ValueError("Incorrect fomo_method: " + method)

        # Initial guess for Fermi level
        mu = 0.5 * (eps[N-1] + eps[N])

        # Target number of electrons
        N = float(N)
        focc, N2 = step(kT,mu,eps)

        # Bracket
        if N2 == N:
            return focc, mu
        elif N2 > N:
            muR = mu
            DF = 1.0
            while True:
                muL = mu - DF
                foccL, N2 = step(kT,muL,eps)
                if N2 < N:
                    break
                DF *= 2.0
        else:
            muL = mu
            DF = 1.0
            while True:
                muR = mu + DF
                foccR, N2 = step(kT,muR,eps)
                if N2 > N:
                    break
                DF *= 2.0

        # Bisection
        iteration = 0 
        while True:
            mu = 0.5 * (muL + muR)
            focc, N2 = step(kT,mu,eps)
            if N2 == N:
                break
            elif N2 > N:
                muR = mu
            else:
                muL = mu
            if abs(muL - muR) < 1.0E-17 * (abs(muL) + abs(muR)) or iteration > 100:
                break
            iteration += 1

        return focc, mu

    @staticmethod
    def fermi_S(
        kT,
        n,
        eps,
        mu,
        ):

        n2 = n[np.logical_and(n[...] > 0.0, n[...] < 1.0)]
        return np.sum(- n2 * np.log(n2) - (1.0 - n2) * np.log(1.0 - n2))

    @staticmethod
    def gaussian_S(
        kT,
        n,
        eps,
        mu,
        ):

        return np.sum(1.0 / np.sqrt(2.0 * np.pi) * np.exp(-(eps[...] - mu)**2 / (2.0 * kT**2)))

    @staticmethod
    def TS_fomo(
        method,
        kT,
        n,
        eps,
        mu,
        ):

        if method == 'fermi':
            return - kT * RHF.fermi_S(kT, n, eps, mu)
        elif method == 'gaussian':
            return - kT * RHF.gaussian_S(kT, n, eps, mu)
        else:
            raise ValueError("Incorrect fomo_method: " + method)

    # => Density Matrix Guess <= #

    @staticmethod
    def guess_density(
        rhf_old,
        rhf_new,
        ):

        """ Compute a symmetrically-orthogonalized least squares or
        corresponding orbital (equivalent) guess density matrix between an old
        RHF object, and a new one which has just been initialized. This is
        performed as follows:
            
        (1) The orbitals are projected from the old basis to the new basis by
        least squares:

            C_{mu'p}' = S_{mu' nu'}^{-1} S_{nu' lambda} C_{lambda p}

        (2) The orbitals are symmetrically orthogonalized in the new basis:

            C_{mu'p} = C_{mu'q}' (C_{nu'q}' S_{nu'lambda'} C_{lambda'p})^{-1/2}

        (3) The new-basis OPDM is constructed as:

            D_{mu' nu'} = C_{mu' p} n_{p} C_{nu' p}

        Importantly, the symmetric orthogonalization of step (2) is performed
        in only the occ + act spaces of a (FOMO)-RHF.
            
        Params:
            rhf_old (psiw.RHF) - an old RHF object, after compute_energy()
            rhf_new (psiw.RHF) - a new RHF object, after initialize()
        Returns:
            D (ls.Tensor of shape (nao2, nao2)) - guess OPDM in the basis of rhf_new, with the same
                idempotency/natural orbital structure as the OPDM in rhf_old.
            Cocc_mom (ls.Tensor of shape (nao2, nocc1)) - reference occupied
                orbitals in the basis of rhf_new, for use in MOM occupation selection.
            Cact_mom (ls.Tensor of shape (nao2, nact1)) - reference active
                orbitals in the basis of rhf_new, for use in MOM occupation selection.
        """

        # Build the overlap integrals (mu|nu') in the AO basis
        pairlist = ls.PairList.build_schwarz(rhf_old.basis, rhf_new.basis, False, rhf_old.pairlist.thre)
        SAB = ls.IntBox.overlap(rhf_old.resources, pairlist)

        # Get the reference occupied orbitals (occ + frac)
        if rhf_old.is_fomo:
            CA = ls.Tensor.array(np.hstack((rhf_old.tensors['Cocc'], rhf_old.tensors['Cact'])))
        else:
            CA = rhf_old.tensors['Cocc']

        # Compute the metric (p|q')
        MAB = ls.Tensor.chain([CA, SAB, rhf_new.tensors['X']], [True, False, False])

        # Compute the metric square (p|q')(q'|r) 
        G = ls.Tensor.chain([MAB, MAB], [False, True])
        
        # Compute the metric square inverse
        Gm12 = ls.Tensor.power(G, -1.0/2.0, 1.0E-12)
        
        # Compute the new orthonormal orbitals
        CB = ls.Tensor.chain([rhf_new.tensors['X'], MAB, Gm12], [False, True, False])

        # MOM reference orbitals in new basis
        CoccB_mom = ls.Tensor.zeros([rhf_new.sizes['nao'], rhf_old.sizes['nocc']])
        CactB_mom = ls.Tensor.zeros([rhf_new.sizes['nao'], rhf_old.sizes['nact']])
        if rhf_old.sizes['nocc']: CoccB_mom[...] = CB[:,:rhf_old.sizes['nocc']]
        if rhf_old.sizes['nact']: CactB_mom[...] = CB[:,rhf_old.sizes['nocc']:]

        # Weight the fractional orbitals by sqrt(nfrac)
        if rhf_old.is_fomo:
            CB[:,rhf_old.sizes['nocc']:] *= np.outer(np.ones(CB.shape[0]), np.sqrt(rhf_old.tensors['nact']))

        # Compute the new OPDM
        DB = ls.Tensor.chain([CB, CB], [False, True])

        # Return data
        return DB, CoccB_mom, CactB_mom

    @staticmethod
    def orbital_health(
        rhfA,
        rhfB,
        ):

        """ Diagnose if two RHF objects have coincident occ/act orbital spaces.
        
        This routine looks at the singular values of the orbital overlaps
        between the occ/act spaces and returns the ratio of smallest/largest
        singular values in each space. These metrics should be near 1 for
        healthy orbital spaces (even accounting for mild changes in geometry),
        and will decay to near 0 if an orbital flips between the occ/act/vir
        spaces in the two RHF objects.

        The singular values in the orbital spaces are returned for further
        diagnostics. If the kocc/kact metrics are near 0, inspecting the
        singular values can provide insights into how many orbitals flipped (k
        nearly zero singular values) or if the system is catestrophically
        non-overlapping (all nearly zero singlar values). The latter case can
        happen if the molecular geomtries are a very large distance apart.

        Note that geometry changes often cause the singular values for core
        orbitals (1s) to decay very rapidly. For CASCI-type purposes, healthy
        metrics in the active space are usually much more important than
        healthy metrics in the core space.

        Params:
            rhfA (psiw.RHF) - bra-side RHF object
            rhfB (psiw.RHF) - ket-side RHF object
        Returns:
            kocc (float) - ratio of smallest to largest singular value in
                overlap metric in occupied orbitals. Should be near 1 for
                healthy orbital spaces, will be near 0 if orbitals have
                flipped.
            kact (float) - ratio of smallest to largest singular value in
                overlap metric in FOMO active orbitals. Should be near 1 for
                healthy orbital spaces, will be near 0 if orbitals have
                flipped.
            socc (ls.Tensor of shape (nocc,)) - singular values of occupied
                orbital overlap matrix <i|j>.
            sact (ls.Tensor of shape (nact,)) - singular values of FOMO active
                orbital overlap matrix <t|u>.
        """

        # Only compare if orbital sizes are coincident
        if rhfA.sizes['nocc'] != rhfB.sizes['nocc']: raise ValueError('nocc must be same size')
        if rhfA.sizes['nact'] != rhfB.sizes['nact']: raise ValueError('nact must be same size')

        # Build the overlap integrals (mu|nu') in the AO basis
        pairlist = ls.PairList.build_schwarz(rhfA.basis, rhfB.basis, False, rhfA.pairlist.thre)
        SAB = ls.IntBox.overlap(rhfB.resources, pairlist)

        if rhfA.sizes['nocc']:
            Socc = ls.Tensor.chain([rhfA.tensors['Cocc'], SAB, rhfB.tensors['Cocc']], [True, False, False])
            Uocc, socc, Vsocc = ls.Tensor.svd(Socc)
            kocc = np.min(socc) / np.max(socc)
        else:
            kocc = 1.0
            socc = ls.Tensor.zeros((0,))

        if rhfA.sizes['nact']:
            Sact = ls.Tensor.chain([rhfA.tensors['Cact'], SAB, rhfB.tensors['Cact']], [True, False, False])
            Uact, sact, Vsact = ls.Tensor.svd(Sact)
            kact = np.min(sact) / np.max(sact)
        else:
            kact = 1.0
            sact = ls.Tensor.zeros((0,))

        return kocc, kact, socc, sact

    def save_molden_file(
        self,
        filename,
        ):

        """ Save a Molden file for the RHF orbitals. 

        Params:
            filename (str) - File path (including .molden to write molden file to)
        Result:
            A Molden file with all RHF orbitals is written to filename. Orbital
            occupations are taken from RHF or FOMO-RHF occupations.
        """

        ls.Molden.save_molden_file(
            filename,
            self.qm_molecule,
            self.basis,
            self.tensors['C'],  
            self.tensors['eps'],
            self.tensors['n'],
            True,
            ) 


def test():

    resources = ls.ResourceList.build()

    molecule = ls.Molecule.from_xyz_file('test/h2o.xyz')

    geom = geometry.Geometry.build(
        resources=resources,
        molecule=molecule,
        basisname='6-31g',
        ewald=ls.Ewald([1.0, -1.0], [-1.0, 0.33]),
        )

    ref = RHF(RHF.default_options().set_values({
        'geometry' : geom,
        'maxiter' : 2,
        }))
    ref.compute_energy()

    G = ls.Tensor((ref.sizes['nmo'],)*2)
    G[:ref.sizes['nocc'],ref.sizes['nocc']:] = ls.Tensor.chain([ref.tensors[x] for x in ['Cocc', 'F', 'Cvir']], [True, False, False])
    G[...] *= -2.0

    Z = ref.compute_cphf(G)

    Z.antisymmetrize() 
    Z[...] *= 2.0

if __name__ == '__main__':

    test()
    exit()

    # import time
    # start = time.time()
    # RHF.default_options().copy()
    # print '%11.3E' % (time.time() - start)
    # start = time.time()
    # RHF.default_options().copy()
    # print '%11.3E' % (time.time() - start)

    resources = ls.ResourceList.build()
    
    molecule = ls.Molecule.from_xyz_file('test/h2o.xyz')

    geom = geometry.Geometry.build(
        resources=resources,
        molecule=molecule,
        basisname='sto-3g',
        )

    ref = RHF(RHF.default_options().set_values({
        'geometry' : geom,
        'fomo' : True,
        'fomo_method' : 'fermi',
        'fomo_temp' : 0.3,
        'fomo_nocc' : 2,
        'fomo_nact' : 4,
        }))
    ref.compute_energy()

    ref.compute_fomo_cphf(ls.Tensor.array(np.random.rand(7,7)))

    # import qmmm
    # qmmm2 = qmmm.QMMM.from_prmtop(
    #     prmtopfile='test/system.prmtop',
    #     inpcrdfile='test/system.rst',
    #     qmindsfile='test/system.qm',
    #     charge=-1.0,
    #     )

    # geom = geometry.Geometry.build(
    #     resources=resources,
    #     qmmm=qmmm2,
    #     basisname='6-31gss',
    #     )

    # ref = RHF(RHF.default_options().set_values({
    #     'geometry' : geom,
    #     'dft_functional' : 'B3LYP',
    #     'dft_grid_name' : 'SG0',
    #     }))
    # ref.compute_energy()

    
        
    
