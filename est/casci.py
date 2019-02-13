import lightspeed as ls
import numpy as np
from . import options
from . import geometry
from . import rhf
# import time

class CASCI(object):

    """ Class CASCI handles complete active space configuration interaction.

    This code takes either a psiw.RHF reference object, or a psiw.Geometry and
    C matrix (orbital coefficients) determined by some other procedure, and
    handles the formation of needed integrals and solution of the CASCI problem
    for a number of roots in different spin blocks. This code interacts
    primarily with ls.IntBox to obtain the required potential matrix elements
    and with ls.CASBox/ls.ExplicitCASBox/ls.Davidson to obtain the solution to
    the CASCI eigenproblem. 

    This class also has efficient codes for the analytical computation of the
    CASCI gradient, non-adiabatic coupling vector, and many-body overlap
    matrix. The latter is particularly efficient, and has a number of
    diabatization/correspondence matching options.
    """

    @staticmethod
    def default_options():
        """ CASCI default options. """
        if hasattr(CASCI, '_default_options'): return CASCI._default_options.copy()
        opt = options.Options() 

        # > Print Control < #

        opt.add_option(
            key='print_level',
            value=2,
            allowed_types=[int],
            doc='Level of detail to print (0 - nothing, 1 - minimal [dynamics], 2 - standard [SP])')

        # > Problem Geometry < #

        opt.add_option(
            key="reference",
            required=False,
            allowed_types=[rhf.RHF],
            doc="Reference RHF, used for geometry, orbitals, and for gradient terms")
        opt.add_option(
            key="geometry",
            required=False,
            allowed_types=[geometry.Geometry],
            doc='EST problem geometry (instead of reference)')
        opt.add_option(
            key="C",
            required=False,
            allowed_types=[ls.Tensor],
            doc="Orbital coefficients (nao, nmo) Tensor (instead of reference)")

        # > Active Space Sizes < #

        opt.add_option(
            key="nocc",
            required=True,
            allowed_types=[int],
            doc="Number of closed (doubly occupied) orbitals")
        opt.add_option(
            key="nact",
            required=True,
            allowed_types=[int],
            doc="Number of active (fractionally occupied) orbitals")
        opt.add_option(
            key="nalpha",
            required=True,
            allowed_types=[int],
            doc="Number of active alpha electrons")
        opt.add_option(
            key="nbeta",
            required=True,
            allowed_types=[int],
            doc="Number of active beta electrons")

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

        # > Desired States < #

        opt.add_option(
            key='S_inds',
            required=True,
            allowed_types=[list],
            doc="List of spin indices (e.g., [0, 2, 4] for singlet, triplet, quintet")
        opt.add_option(
            key='S_nstates',
            required=True,
            allowed_types=[list],
            doc="List of number of states requested in each spin index (e.g., [2, 2, 1])")
        opt.add_option(
            key='alpha',
            value=1.0,
            required=False,
            allowed_types=[float],
            doc="Shift value for alpha CAS correction to energies and gradient, applied if less than 1")

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
        opt.add_option(
            key='dft_cas_flavor',
            required=False,
            allowed_types=[str],
            allowed_values=['Ed', 'new'],
            doc='Flavor of the CAS-DFT to use, e.g., Ed or new')

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

        # > Explicit/Direct CI < #

        opt.add_option(
            key='algorithm',
            value='explicit',
            required=True,
            allowed_types=[str],
            allowed_values=['explicit', 'direct'],
            doc='Use explicit diagonalization or direct method for CI?')

        # > Davidson Options < #

        opt.add_option(
            key='maxiter',
            value=70,
            required=True,
            allowed_types=[int],
            doc='Maximum number of Davidson iterations')
        opt.add_option(
            key='nmax',
            value=50,
            required=True,
            allowed_types=[int],
            doc='Maximum Davidson subspace size')
        opt.add_option(
            key='r_convergence',
            value=1.0E-6,
            required=True,
            allowed_types=[float],
            doc='Davidson convergence criterion (2-norm of residual)')
        opt.add_option(
            key='norm_cutoff',
            value=1.0E-6,
            required=True,
            allowed_types=[float],
            doc='Davidson conditioning cutoff')
        opt.add_option(
            key='use_disk',
            value=False,
            required=True,
            allowed_types=[bool],
            doc='Use disk in Davidson storage?')
        opt.add_option(
            key='nguess_per_root',
            value=2,
            required=True,
            allowed_types=[int],
            doc='Number of guess vectors to use per desired root')

        CASCI._default_options = opt
        return CASCI._default_options.copy()

    def __init__(
        self,
        options,
        ):

        """ CASCI initialization - no computational effort performed. 

        Note that the user must provide *either* a 'reference' psiw.RHF object
        or a 'geometry' psiw.Geometry object and 'C' orbital coefficient
        Tensor. An error is thrown if this is not done.
        """
        
        self.options = options

        # Cache some useful attributes
        self.print_level = self.options['print_level']
        self.reference = self.options['reference']
        self.geometry = self.reference.geometry if self.reference else self.options['geometry']
        self.resources = self.geometry.resources
        self.qm_molecule = self.geometry.qm_molecule
        self.basis = self.geometry.basis
        self.pairlist = self.geometry.pairlist

        # > Validity Checks < #
    
        if self.options['reference'] and self.options['C']:
            raise ValueError('CASCI: can only set one of reference/C options')
        if self.options['reference'] and self.options['geometry']:
            raise ValueError('CASCI: can only set one of reference/geometry options')
    
        # Some useful registers
        self.sizes = {}
        self.scalars = {}
        self.tensors = {}

    @staticmethod
    def from_options(**kwargs):
        """ Return an instance of this class with default options updated from values in kwargs. """
        return CASCI(CASCI.default_options().set_values(kwargs))

    def compute_energy(self):

        """ Solves the CASCI problem to produce the CASCI eigenvectors and
            eigenvalues.

        Result:
            This CASCI object is updated with the solution to the CASCI
            problem. The standard 'sizes', 'scalars', 'tensors' dictionaries
            are updated as detailed below, and a few extra fields are added to
            this object as detailed below:  

        sizes:
            natom - number of atoms in the qm_molecule
            nao - number of atomic orbitals in the basis
            nmo - number of molecular orbitals
            nocc - number of core/doubly occupied orbitals
            nact - number of active CASCI orbitals
            nvir - number of virtual/doubly unoccupied orbitals
            nalpha - number of alpha electrons in active space
            nbeta - number of beta electrons in active space

        scalars:
            Enuc - energy of qm_molecule
            Eext - self energy of external environment (often Enuc)
            Ecore - core determinant energy
            
        tensors:
            C (nao, nmo) - molecular orbital coefficients, packed as nocc,
                nact, nvir. 
            Cocc (nao, nocc) - core molecular orbital coefficients
            Cact (nao, nact) - active molecular orbital coefficients
            Cvir (nao, nvir) - virtual molecular orbital coefficients
            H (nao, nao) - core determinant Fock matrix in AO basis
            Hact (nact, nact) - core determinant Fock matrix in active MO basis
            I (nact, nact, nact, nao) - (tu|vp) integrals in mixed active/AO
                basis. This object is rather expensive to form, and there is no
                penalty for keeping it in (tu|vp) notation, which can be reused
                for the orbital gradient (needed for gradient, coupling, etc
                later).
            Iact (nact,)*4 - (tu|vw) integrals in active MO basis.
        
        fields:
            evecs (dict of list of Tensor) - eigenvectors for each spin and
                state index requested. The eigenvectors are stored in the CSF
                basis determined by ls.CASBox for this problem.
            evals (dict of list of float) - eigenvalues for each spin and state
                index requested.
            converged (dict of bool) - did all eigenstates converge in each
                spin block?

            These are addressed by spin and state index, e.g., 
            V = cas.evecs[0][1] grabs the 1-th state in the S=0 (singlet) spin
            block.
        """
        
        # => Title Bars <= #

        if self.print_level:
            print('==> CASCI <==\n')

        # => SCF Quantities <= #

        self.tensors['C'] = self.reference.tensors['C'] if self.reference else self.options['C']
        self.sizes['natom'] = self.qm_molecule.natom
        self.sizes['nao'] = self.basis.nao

        # => Integral Thresholds <= #

        if self.print_level > 1:
            print('Integral Thresholds:')
            print('  Threshold DP = %11.3E' % self.options['thre_dp'])
            print('  Threshold SP = %11.3E' % self.options['thre_sp'])
            print('')

        # => Orbital Spaces <= #

        self.sizes['nmo'] = self.tensors['C'].shape[1]
        self.sizes['nocc'] = self.options['nocc']
        self.sizes['nact'] = self.options['nact']
        self.sizes['nalpha'] = self.options['nalpha']
        self.sizes['nbeta'] = self.options['nbeta']
        self.sizes['nvir'] = self.sizes['nmo'] - self.sizes['nocc'] - self.sizes['nact']
        if self.print_level > 1:
            print('Orbital Spaces:')
            for key in ['nmo', 'nocc', 'nact', 'nalpha', 'nbeta', 'nvir']:
                print('  %-6s = %5d' % (key, self.sizes[key]))
            print('')

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

        # => Orbital Coefficients <= #

        self.tensors['Cocc'] = ls.Tensor([self.sizes['nao'], self.sizes['nocc']])
        self.tensors['Cact'] = ls.Tensor([self.sizes['nao'], self.sizes['nact']])
        self.tensors['Cvir'] = ls.Tensor([self.sizes['nao'], self.sizes['nvir']])
        off = 0
        if self.sizes['nocc']:
            self.tensors['Cocc'].np[...] = self.tensors['C'].np[:,off:(off+self.sizes['nocc'])]
        off += self.sizes['nocc']
        if self.sizes['nact']:
            self.tensors['Cact'].np[...] = self.tensors['C'].np[:,off:(off+self.sizes['nact'])]
        off += self.sizes['nact']
        if self.sizes['nvir']:
            self.tensors['Cvir'].np[...] = self.tensors['C'].np[:,off:(off+self.sizes['nvir'])]
        off += self.sizes['nvir']

        # => One-Electron Integrals <= #

        if self.print_level > 1: 
            print('One-Electron Integrals:\n')
        T = ls.IntBox.kinetic(
            self.resources,
            self.pairlist)

        # => External Energy/Potential <= #

        if self.print_level:
            print('External Environment:')
        self.scalars['Enuc'] = self.geometry.compute_nuclear_repulsion_energy()
        self.scalars['Eext'] = self.geometry.compute_external_energy()
        Vext = self.geometry.compute_external_potential(self.pairlist)
        if self.print_level:
            print('  Enuc = %24.16E' % self.scalars['Enuc'])
            print('  Eext = %24.16E' % self.scalars['Eext'])
            print('')

        # => Core Hamiltonian <= #
        
        H = ls.Tensor.array(T)
        H[...] += Vext

        # => Core Determinant Fock Matrix <= #

        a = 1.0 if not self.is_dft else self.dft_functional.alpha
        b = 1.0 if not self.is_dft else self.dft_functional.beta

        if self.print_level > 2:
            print('Core Determinant:\n')
        Eocc = self.scalars['Eext']
        Focc = ls.Tensor.array(H)
        Focc.name = 'Focc'
        if self.sizes['nocc']:
            Docc = ls.Tensor.chain([self.tensors['Cocc']]*2, [False, True])
            Jocc = ls.IntBox.coulomb(
                self.resources,
                ls.Ewald.coulomb(),
                self.pairlist,
                self.pairlist,
                Docc,
                self.options['thre_sp'],    
                self.options['thre_dp'],    
                )
            Kocc = ls.IntBox.exchange(
                self.resources,
                ls.Ewald.coulomb(),
                self.pairlist,
                self.pairlist,
                Docc,
                True,
                self.options['thre_sp'],    
                self.options['thre_dp'],    
                )
            if self.dft_functional:
                Kwocc = ls.IntBox.exchange(
                    self.resources,
                    ls.Ewald([1.0],[self.dft_functional.omega]),
                    self.pairlist,
                    self.pairlist,
                    Docc,
                    True,
                    self.options['thre_sp'],    
                    self.options['thre_dp'],    
                    )
            else:
                Kwocc = ls.Tensor.zeros_like(Kocc)
            if self.dft_functional:
                ret = ls.DFTBox.rksPotential(
                    self.resources,
                    self.dft_functional,
                    self.dft_grid,
                    self.dft_hash,
                    self.pairlist,
                    Docc,
                    self.options['dft_thre_grid'],
                    ) 
                XCocc = ret[1]
                EXCocc = ret[0][0]
                Qocc = ret[0][1]
            else:
                XCocc = ls.Tensor.zeros_like(Jocc)
                EXCocc = 0.0
                Qocc = float(self.sizes['nocc'])
            Focc[...] += 2. * Jocc.np
            Focc[...] -= a * Kocc.np
            Focc[...] -= b * Kwocc.np
            Focc[...] += 1. * XCocc.np
            Eocc += \
                 2.0 * Docc.vector_dot(H) + \
                 2.0 * Docc.vector_dot(Jocc) + \
                -a * Docc.vector_dot(Kocc) + \
                -b * Docc.vector_dot(Kwocc) + \
                 1.0 * EXCocc 
        self.scalars['Ecore'] = Eocc

        if self.is_dft:
            if self.options['dft_cas_flavor'] == 'Ed':
                Fcore = ls.Tensor.array(H)
                Fcore[...] += 2. * Jocc.np
                Fcore[...] -= 1 * Kocc.np
                Fcore.name = 'Fcore'
            elif self.options['dft_cas_flavor'] == 'new':
                Fcore = ls.Tensor.array(Focc)
                Fcore.name = 'Fcore'
        else: 
            Fcore = ls.Tensor.array(Focc)
            Fcore.name = 'Fcore'

        # Cache the Core Fock matrix in AO basis 
        # This object is quite expensive and needed for the orbital gradient later
        self.tensors['H'] = Fcore

        # The active-basis core Fock matrix (target for CASCI energy)
        self.tensors['Hact'] = ls.Tensor.chain([self.tensors['Cact'],Fcore,self.tensors['Cact']],[True,False,False])

        if self.print_level:
            print('Core Energy = %20.14f\n' % self.scalars['Ecore'])
            
        # => Integral Transform (Active Orbital Basis) <= #

        if self.print_level > 1:
            print('Integral Transform:\n')

        # The (t,u,v,p) integrals are expensive, and needed for the orbital gradient
        self.tensors['I'] = ls.IntBox.eriJ(
            self.resources,
            ls.Ewald.coulomb(),
            self.pairlist,
            self.pairlist,
            self.tensors['Cact'],
            self.tensors['Cact'],
            self.tensors['Cact'],
            self.tensors['C'],
            self.options['thre_sp'],    
            self.options['thre_dp'],    
            )
        # The active-basis ERIs (target for CASCI energy)
        self.tensors['Iact'] = ls.Tensor.array(self.tensors['I'][:,:,:,self.sizes['nocc']:self.sizes['nocc']+self.sizes['nact']])
        # Symmetrize active-basis ERIs
        self.tensors['Iact'][...] *= 0.125
        self.tensors['Iact'][...] += ls.Tensor.array(np.einsum('ijkl->ijlk', self.tensors['Iact']))
        self.tensors['Iact'][...] += ls.Tensor.array(np.einsum('ijkl->jikl', self.tensors['Iact']))
        self.tensors['Iact'][...] += ls.Tensor.array(np.einsum('ijkl->klij', self.tensors['Iact']))

        # => CASBox Object <= #

        self.casbox = ls.CASBox(
            self.sizes['nact'], 
            self.sizes['nalpha'],
            self.sizes['nbeta'],
            self.tensors['Hact'],
            self.tensors['Iact'],
            )
        if self.print_level > 1:
            print(self.casbox)
        # Will only be touched if explicit algorithm, but then will remember eigenbasis
        self.explicit_casbox = ls.ExplicitCASBox(self.casbox)

        # => State Taskings <= #

        if self.print_level > 1:
            print('CASCI Requested States:')
            print('%2s %6s %11s %11s' % ('S', 'Nstate', 'Ndet', 'NCSF'))
            for S, nS in zip(self.options['S_inds'], self.options['S_nstates']):
                print('%2d %6d %11d %11d' % (
                    S, 
                    nS,
                    self.casbox.CSF_basis(S).total_ndet,
                    self.casbox.CSF_basis(S).total_nCSF))
            print('')

        # => Compute States <= #

        if self.print_level > 1:
            print('CASCI Algorithm: %s' % self.options['algorithm'])
            print('')

        self.converged = {}
        self.evals = {}
        self.evecs = {}
        for S_ind, S_nstate in zip(*[self.options[x] for x in ['S_inds', 'S_nstates']]):
            if self.options['algorithm'] == 'explicit':
                self.compute_states_explicit(S_ind, S_nstate)
            if self.options['algorithm'] == 'direct':
                self.compute_states_direct(S_ind, S_nstate)
        
        # => Apply alpha CAS Correction <= # 

        self.alpha = self.options['alpha']
        if self.alpha < 1.0:

            # deepcopy energies for coupling
            self.orig_evals = {}            
            for key, vals in self.evals.items():
                nvals = [v for v in  vals]
                self.orig_evals[key] = nvals

            for S_ind, S_nstate in zip(*[self.options[x] for x in ['S_inds', 'S_nstates']]):
                # obtain state-averaged energy
                weight = 1.0 / float(S_nstate)
                SA_E = weight * np.sum(self.evals[S_ind])
                # apply shift to states
                for state_ind in range(S_nstate):
                    self.evals[S_ind][state_ind] *= self.alpha
                    self.evals[S_ind][state_ind] += (1.0 - self.alpha) * SA_E

                if self.print_level:
                    title = 'S=%d' % S_ind
                    print('alpha corrected CASCI %s Energies:\n' % title)
                    print('%4s: %24s' % ('I', 'Total E'))
                    for Eind, E in enumerate(self.evals[S_ind]):
                        print('%4d: %24.16E' % (Eind, E))
                    print('')
                    print('State Averaged Energy: %24.16E' % SA_E)
                    print('')

        # => Print Amplitude Summary <= #

        if self.print_level > 1:
            print('CASCI Amplitudes:\n')
            for S_ind in self.options['S_inds']:
                for S_ind2, state in enumerate(self.evecs[S_ind]):
                    print('State S=%d I=%d:' % (S_ind, S_ind2))
                    print(self.casbox.amplitude_string(
                        S_ind,  
                        state,
                        0.0001,
                        60,
                        self.sizes['nocc']))

        # => Print Transition Info <= #

        if self.print_level > 1:
            print(self.transition_string, end=' ')

        # => Trailer Bars <= #

        if self.print_level > 1:
            print('"Will somebody get this big walking carpet out of my way?"')
            print('    --Princess Leia\n')

        if self.print_level:
            print('==> End CASCI <==\n')

    def compute_states_explicit(
        self,
        Sindex,
        nstate,
        ):

        """ Compute CASCI eigenvectors/eigenvalues using explicit CI.

        This is called by compute_energy - users should not call this routine.

        Params:
            Sindex (int) - spin index.
            nstate (int) - number of states to compute.
        Result:
            The CASCI object is updated with the eigenvectors/eigenvalues for
            this spin block. Particular fields updated are:
                self.evecs[Sindex] - list of Tensors of eigenvectors
                self.evals[Sindex] - list of float of eigenvalues
                self.converged[Sindex] - list of bool of converged or not
        """

        # => Title <= #

        title = 'S=%d' % Sindex
        if self.print_level:
            print('=> %s States <=\n' % title)

        self.converged[Sindex] = True
        self.evecs[Sindex] = [self.explicit_casbox.evec(Sindex, x) for x in range(nstate)]
        self.evals[Sindex] = [self.explicit_casbox.eval(Sindex, x) + self.scalars['Ecore'] for x in range(nstate)]

        if self.print_level:
            print('CASCI %s Energies:\n' % title)
            print('%4s: %24s' % ('I', 'Total E'))
            for Eind, E in enumerate(self.evals[Sindex]):
                print('%4d: %24.16E' % (Eind, E))
            print('')

        if self.print_level:
            print('=> End %s States <=\n' % title)

    def compute_states_direct(
        self,
        Sindex,
        nstate,
        ):

        """ Compute CASCI eigenvectors/eigenvalues using Davidson/Direct CI. 

        This is called by compute_energy - users should not call this routine.

        Params:
            Sindex (int) - spin index.
            nstate (int) - number of states to compute.
        Result:
            The CASCI object is updated with the eigenvectors/eigenvalues for
            this spin block. Particular fields updated are:
                self.evecs[Sindex] - list of Tensors of eigenvectors
                self.evals[Sindex] - list of float of eigenvalues
                self.converged[Sindex] - list of bool of converged or not
        """

        # => Title <= #

        title = 'S=%d' % Sindex
        if self.print_level:
            print('=> %s States <=\n' % title)

        # => Guess Vectors <= #

        nguess = min(self.casbox.CSF_basis(Sindex).total_nCSF, nstate * self.options['nguess_per_root'])
        if self.print_level > 1:
            print('Guess Details:') 
            print('  Nguess: %11d' % nguess)
            print('')
        Cs = self.casbox.guess_evangelisti(Sindex, nguess)

        # => Convergence Info <= #

        if self.print_level > 1:
            print('Convergence:')
            print('  Maxiter: %11d' % self.options['maxiter'])
            print('')

        # => Davidson Object <= #

        dav = ls.Davidson(
            nstate,
            self.options['nmax'],
            self.options['r_convergence'],
            self.options['norm_cutoff'],
            )
        if self.print_level > 1:
            print(dav)

        # ==> Davidson Iterations <== # 

        if self.print_level:
            print('%s CASCI Iterations:\n' % title)
            print('%4s: %11s' % ('Iter', '|R|'))
        converged = False
        for iter in range(self.options['maxiter']):

            # Sigma Vector Builds
            Ss = [self.casbox.sigma(self.resources, Sindex, C) for C in Cs]

            # Add vectors/diagonalize Davidson subspace
            Rs, Es = dav.add_vectors(Cs,Ss, self.options['use_disk'])

            # Print Iteration Trace
            if self.print_level:
                print('%4d: %11.3E' % (iter, dav.max_rnorm))

            # Check for convergence
            if dav.is_converged:
                converged = True
                break

            # Precondition the desired residuals
            Ds = [self.casbox.apply_evangelisti(Sindex, R, E) for R, E in zip(Rs, Es)]
    
            # Get new trial vectors
            Cs = dav.add_preconditioned(Ds)

        # => Output Quantities <= #
        
        # Print Convergence
        if self.print_level:
            print('')
            if converged:
                print('%s CASCI Converged\n' % title)
            else:
                print('%s CASCI Failed\n' % title)

        # Cache quantities
        self.converged[Sindex] = converged
        self.evals[Sindex] = [x + self.scalars['Ecore'] for x in dav.evals]
        self.evecs[Sindex] = [ls.Storage.from_storage(x, Ss[0]) for x in dav.evecs]

        # Print residuals
        if self.print_level:
            print('CASCI %s Residuals:\n' % title)
            print('%4s: %11s' % ('I', '|R|'))
            for rind, r in enumerate(dav.rnorms):
                print('%4d: %11.3E %s' % (rind, r, 'Converged' if r < self.options['r_convergence'] else 'Not Converged'))
            print('')
    
        # Print energies
        if self.print_level:
            print('CASCI %s Energies:\n' % title)
            print('%4s: %24s' % ('I', 'Total E'))
            for Eind, E in enumerate(self.evals[Sindex]):
                print('%4d: %24.16E' % (Eind, E))
            print('')

        if self.print_level:
            print('=> End %s States <=\n' % title)

    # => OPDM/Natural Orbital Computation (Negligible Effort) <= #

    def opdm_ao(
        self,
        S,
        indexA,
        indexB,
        spin_sum=True,
        ):

        """ Return the OPDM in the AO basis for a pair of states.
        
        D_pq^AB = <A|p^+ q|B>

        If indexA == indexB, a one particle density matrix will be returned. If
        indexA != indexB, a transition OPDM will be returned.

        If indexA == indexB and spin_sum is True, the density of the core
        determinant will be added to the returned OPDM.

        Params:
            S (int) - spin index.
            indexA (int) - index of state A.
            indexB (int) - index of state B.
            spin_sum (bool) - sum alpha and beta spins (True) or subtract beta
                spins from alpha spins (False).
        Returns:
            D (ls.Tensor of shape (nao, nao)) - the requested OPDM.
        """

        D1act = self.casbox.opdm(
            self.resources, 
            S, 
            self.evecs[S][indexA], 
            self.evecs[S][indexB], 
            spin_sum,
            )

        D1ao = ls.Tensor.chain([self.tensors['Cact'], D1act, self.tensors['Cact']], [False, False, True])
        # Add core density if appropriate
        if spin_sum and indexA == indexB:
            D1ao[...] += 2.0 * ls.Tensor.chain([self.tensors['Cocc'],]*2, [False, True])[...] 

        return D1ao

    def natural_orbitals(
        self,
        S,
        index,
        include_occ=True,
        include_vir=True,
        ):

        """ Compute the natural orbital coefficients and occupations for a CASCI state.

        This routine only involves diagonalization in the active space, and is
        usually of negligible computational effort.

        Params:
            S (int) - the spin index
            index (ind) - the state index
            include_occ (bool) - include the core orbitals or not?
            include_vir (bool) - include the virtual orbitals or not?
        Returns:
            C (ls.Tensor of shape (nao, nmo)) - the natural orbital coefficients.
            n (ls.Tensor of shape (nmo,)) - the natural orbital occupations.
            If include_occ, Cocc/nocc is appendend before the active MO indices.
            If include_vir, Cvir/nvir is appendend after the active MO indices.
        """

        # OPDM (active space)
        D1act = self.casbox.opdm(
            self.resources, 
            S, 
            self.evecs[S][index], 
            self.evecs[S][index], 
            True,
            )

        # Eigendecomposition (active space only)
        n1act2, U1act2 = ls.Tensor.eigh(D1act)

        # Order descending
        n1act = ls.Tensor.array(n1act2[::-1])
        U1act = ls.Tensor.array(U1act2[:,::-1])
        
        # AO-basis natural orbitals (active space only)
        C1act = ls.Tensor.chain([self.tensors['Cact'], U1act], [False, False])

        # Return orbitals including any requested occ/vir garbage
        if include_occ or include_vir:
            Cs = []
            if include_occ: Cs.append(self.tensors['Cocc'])
            Cs.append(C1act)
            if include_vir: Cs.append(self.tensors['Cvir'])
            ns = []
            if include_occ: ns.append(2.0 * np.ones((self.sizes['nocc'],)))
            ns.append(n1act)
            if include_vir: ns.append(0.0 * np.ones((self.sizes['nvir'],)))
            C1 = ls.Tensor.array(np.hstack(Cs))
            n1 = ls.Tensor.array(np.hstack(ns))
            return C1, n1
        else:
            return C1act, n1act
            
    def save_molden_file(
        self,
        filename,
        S,
        index,
        include_occ=True,
        include_vir=True,
        ):

        """ Save a Molden file for the CASCI natural orbitals for a given state.

        The orbital energy fields are filled with 0, 1, 2, ..., as there is no
        strict orbital energy definition for CASCI.
    
        Params:
            filename (str) - File path (including .molden to write molden file to)
            S (int) - the spin index
            index (ind) - the state index
            include_occ (bool) - include the core orbitals or not?
            include_vir (bool) - include the virtual orbitals or not?
        Result:
            A Molden file with all requested CASCI orbitals is written to filename. 
        """

        C, n = self.natural_orbitals(
            S,
            index,
            include_occ,
            include_vir,    
            )

        ls.Molden.save_molden_file(
            filename,
            self.qm_molecule,
            self.basis,
            C,
            ls.Tensor.array(np.arange(n.size)),
            n,
            True,
            ) 

    def save_molden_files(
        self,
        fileroot,
        include_occ=True,
        include_vir=True,
        ):

        """ Save all Molden files for all CASCI natural orbitals for this
            CASCI object.

        The orbital energy fields are filled with 0, 1, 2, ..., as there is no
        strict orbital energy definition for CASCI.

        Params:
            fileroot (str) - File path, without state indexing or .molden file
                extension.
            include_occ (bool) - include the core orbitals or not?
            include_vir (bool) - include the virtual orbitals or not?
        Result:
            A set Molden file with all requested CASCI orbitals is written to
            filename. For each S, index spin/state index pair, the filename
            used is 'fileroot-S%d-%d.molden' % (S, index).
        """

        for S_ind, S_nstate in zip(*[self.options[x] for x in ['S_inds', 'S_nstates']]):
            for ind in range(S_nstate):
                self.save_molden_file(
                    '%s-S%d-%d.molden' % (fileroot, S_ind, ind),
                    S_ind,
                    ind,
                    include_occ,
                    include_vir,
                    )

    # => Gradient Code <= #

    def compute_gradient(
        self,
        S,
        index,
        ):

        """ Compute the analytical gradient for a given spin and state index.

        This requires forming the orbital gradient, solving the CPHF orbital
        response equations, and contracting the relaxed density matrices and
        Lagrangian with the appropriate integral derivatives.

        This cannot be used unless the CASCI orbitals were generated from a
        psiw.RHF object (which knows how to solve CPHF).

        Params:
            S (int) - the spin index
            index (ind) - the state index
        Returns:
            G (Tensor of shape (natom_full, 3)) - the full gradient for the
                given state, including any external potential terms.
        """

        if self.is_dft: raise ValueError('Cannot do DFT core yet')
        if not self.reference: raise ValueError('Cannot compute gradient without a reference')

        # OPDM/TPDM/2-Cumulant (all symmetrized)
        D1 = self.casbox.opdm(self.resources, S, self.evecs[S][index], self.evecs[S][index], True)
        D2 = self.casbox.tpdm(self.resources, S, self.evecs[S][index], self.evecs[S][index], True)

        # Apply alpha-CAS Correction to OPDM/TPDM (if necessary)
        self.alpha = self.options['alpha']
        if self.alpha < 1.0:
            S_nstates = self.options['S_nstates'][S]
            weight = (1.0 - self.alpha) / float(S_nstates)
            avgD1 = np.zeros_like(D1[...])
            avgD2 = np.zeros_like(D2[...])
            for x in range(S_nstates):
                if x != index:
                    avgD1 += weight * self.casbox.opdm(self.resources, S, self.evecs[S][x], self.evecs[S][x], True)[...]
                    avgD2 += weight * self.casbox.tpdm(self.resources, S, self.evecs[S][x], self.evecs[S][x], True)[...]
            D1[...] = (self.alpha + weight) * D1[...] + avgD1
            D2[...] = (self.alpha + weight) * D2[...] + avgD2

        L2 = CASCI.compute_cumulant(D1, D2)

        # Form Xtilde
        X = self.compute_orbital_lagrangian(D1, D2, L2)

        # Total AO-basis OPDM
        D1ao = ls.Tensor.chain([self.tensors['Cocc']]*2, [False, True])
        D1ao[...] *= 2.0
        D1ao[...] += ls.Tensor.chain([self.tensors['Cact'], D1, self.tensors['Cact']], [False, False, True])

        # Wedge Coulomb integral gradient 
        GJ = ls.IntBox.coulombGrad(
            self.resources,
            ls.Ewald.coulomb(),
            self.pairlist,
            D1ao,
            D1ao,
            self.options['thre_sp'],
            self.options['thre_dp'],
            )
        GJ[...] *= +0.5
        GJ.name = 'GJ'
        # Wedge exchange integral gradient 
        GK = ls.IntBox.exchangeGrad(
            self.resources,
            ls.Ewald.coulomb(),
            self.pairlist,
            D1ao,
            D1ao,
            True,
            True,
            True,
            self.options['thre_sp'],
            self.options['thre_dp'],
            )
        GK[...] *= -0.25
        GK.name = 'GK'
        # Nonseparable cumulant gradient
        GI = ls.IntBox.eriGradJ(
            self.resources,
            ls.Ewald.coulomb(),
            self.pairlist,
            self.tensors['Cact'],
            L2,
            self.options['thre_sp'],
            self.options['thre_dp'],
            )
        GI[...] *= 1.0
        GI.name = 'GI'
        G = ls.Tensor.array(GJ[...] + GK[...] + GI[...])

        return self.reference.compute_hf_based_gradient(D1ao, X, G)

    # => Derivative Coupling Code <= #

    def compute_coupling(
        self,
        S, 
        indexA,
        indexB,
        doETF=False, # Do Joe Subotnik ETF?
        ):

        """ Compute the analytical non-adiabatic coupling matrix element for a
            given spin and state index pair.

        This requires forming the orbital gradient, solving the CPHF orbital
        response equations, and contracting the relaxed density matrices and
        Lagrangian with the appropriate integral derivatives.

        This cannot be used unless the CASCI orbitals were generated from a
        psiw.RHF object (which knows how to solve CPHF).

        Params:
            S (int) - the spin index
            indexA (ind) - the state index for the bra
            indexB (ind) - the state index for the key
            doETF (bool) - apply Joe Subotnik's prescription to keep the
                coupling vector translationally invariant?
        Returns:
            G (Tensor of shape (natom_full, 3)) - the full coupling for the
                given state, including any external potential terms.
        """

        # TODO: Clean this up

        if self.is_dft: raise ValueError('Cannot do DFT core yet')
        if not self.reference: raise ValueError('Cannot compute gradient without a reference')
        if indexA == indexB: raise ValueError('indexA == indexB: you want a gradient?')

        # Energy difference between states
        # if alpha correction is applied, use un-corrected energies
        if self.alpha < 1.0:
            dE = self.orig_evals[S][indexB] - self.orig_evals[S][indexA]
        else:
            dE = self.evals[S][indexB] - self.evals[S][indexA]
        
        # Manually compute OPDM/TPDM with correct symmetrization
        D1 = self.casbox.opdm(self.resources, S, self.evecs[S][indexA], self.evecs[S][indexB], True)
        D1s = ls.Tensor.array(D1)
        D1s.symmetrize()
        D2 = self.casbox.tpdm(self.resources, S, self.evecs[S][indexA], self.evecs[S][indexB], False)
        D2s = ls.Tensor.zeros_like(D2)
        D2s[...] += 0.25 * ls.Tensor.permute('rstu->rstu', D2).np
        D2s[...] += 0.25 * ls.Tensor.permute('srut->rstu', D2).np
        D2s[...] += 0.25 * ls.Tensor.permute('turs->rstu', D2).np
        D2s[...] += 0.25 * ls.Tensor.permute('utsr->rstu', D2).np

        # Active-space Fock Matrix (from D1s)
        D1ao = ls.Tensor.chain([self.tensors['Cact'], D1s, self.tensors['Cact']], [False, True, True])
        JK1ao = ls.IntBox.coulomb(
            self.resources,
            ls.Ewald.coulomb(),
            self.pairlist,
            self.pairlist,
            D1ao,
            self.options['thre_sp'],    
            self.options['thre_dp'],    
            )
        JK1ao[...] *= -2.0
        JK1ao = ls.IntBox.exchange(
            self.resources,
            ls.Ewald.coulomb(),
            self.pairlist,
            self.pairlist,
            D1ao,
            True,
            self.options['thre_sp'],    
            self.options['thre_dp'],    
            JK1ao,
            )
        JK1ao[...] *= -1.0
        JK2 = ls.Tensor.chain([self.tensors['C'], JK1ao, self.tensors['C']], [True, False, False])

        # Xtilde
        X = ls.Tensor.zeros((self.sizes['nmo'],)*2)

        # X_pq <- 2 Fact_pr I_qi
        X[:,:self.sizes['nocc']] += 2.0 * JK2[:,:self.sizes['nocc']]

        # X_pt <- 2 Fcore_pu D1s_tu
        H2 = ls.Tensor.chain([self.tensors[x] for x in ['C', 'H', 'Cact']], [True, False, False])
        X[:,self.sizes['nocc']:self.sizes['nocc']+self.sizes['nact']] += 2.0 * ls.Tensor.chain([H2, D1s], [False, False]).np
        
        # X_pt <- 4 (pu|vw) D2s_tuvw
        X[:, self.sizes['nocc']:self.sizes['nocc']+self.sizes['nact']] += 4.0 * ls.Tensor.einsum('pqrs,pqrt->st', self.tensors['I'], D2s).np

        # X_tu <- dE D1_tu
        X[self.sizes['nocc']:self.sizes['nocc']+self.sizes['nact'],self.sizes['nocc']:self.sizes['nocc']+self.sizes['nact']] += dE * D1[...]

        # Core density matrix (1 normalized like RHF)
        D1core = ls.Tensor.chain([self.tensors['Cocc']]*2, [False, True])
        # D1core[...] *= 2.0

        # TPDM gradient
        GJ = ls.IntBox.coulombGrad(
            self.resources,
            ls.Ewald.coulomb(),
            self.pairlist,
            D1ao,
            D1core,
            self.options['thre_sp'],
            self.options['thre_dp'],
            )
        GJ[...] *= 2.0
        GK = ls.IntBox.exchangeGrad(
            self.resources,
            ls.Ewald.coulomb(),
            self.pairlist,
            D1ao,
            D1core,
            False,
            False,
            False,
            self.options['thre_sp'],
            self.options['thre_dp'],
            )
        GK[...] *= -1.0
        GI = ls.IntBox.eriGradJ(
            self.resources,
            ls.Ewald.coulomb(),
            self.pairlist,
            self.tensors['Cact'],
            D2s,
            self.options['thre_sp'],
            self.options['thre_dp'],
            )
        GI[...] *= 1.0

        G = ls.Tensor.zeros_like(GI)
        G[...] += GJ
        G[...] += GK
        G[...] += GI

        # AO basis transition OPDM (*not* symmetrized)
        D1ao = ls.Tensor.chain([self.tensors['Cact'], D1, self.tensors['Cact']], [False, False, True])

        return self.reference.compute_hf_based_coupling(D1ao, X, G, dE, doETF)

    @staticmethod
    def compute_overlap(
        casciA,
        casciB,
        S,
        indsA=None,
        indsB=None,
        orbital_coincidence='none',
        state_coincidence='none',
        ):
        
        """ Compute the overlap between all states in spin block S in two CASCI objects.

        This method requires forming the delayed overlap matrix between the two
        geometries, as well as some linear algebra in the core space and in the
        active configuration space. Generally this method requires negligible
        cost relative to the cost of compute_energy or of any gradient or
        coupling vector.

        This method includes accounting for 'coincidence,' the idea that the
        involved orbitals or states in the two CASCI objects qualitatively span
        the same 1- or N-particle spaces. Coincidence can be enforced on
        several levels within this method by setting the orbital_coincidence or
        state_coincidence keywords. Activating these uses SVD-based approaches
        to force the states into correspondence (e.g., corresponding orbital
        transformation).

        Params:
            casciA (CASCI) - the bra-side CASCI object
            casciB (CASCI) - the ket-side CASCI object
            S (int) - the spin index
            indsA (list of int or None) - indices of eigenvectors in bra-side
                CASCI object. All eigenvectors are included if None.
            indsB (list of int or None) - indices of eigenvectors in ket-side
                CASCI object. All eigenvectors are included if None.
            orbital_coincidence (str) - orbital coincidence assumptions/treatment. 
                'none' - no orbital coincidence assumptions, compute overlap matrix exactly.
                'core' - coincident core assumption, assumes core overlaps
                    exactly and that core-active overlaps are zero.
                'full' - assumes that core and active space are separately
                    coincident. A corresponding orbital SVD is still performed
                    in the active space to handle ordering/phase issues, but
                    the singular values of this transformation are set to 1.
            state_coincidence (str) - state coincidence assumptions/treatment.
                'none' - no state coincidence assumptions, return raw state overlap matrix.
                'full' - assumes that the involved CASCI states are fully
                    coincident. An SVD of the state overlap matrix is
                    performed, the singular values are clamped to 1, and the
                    overlap matrix is reconstructed. This guarantees that the
                    returned overlap matrix is an orthogonal matrix to the
                    machine precision.
        Returns:
            O (ls.Tensor, shape (nstateA, nstateB)) - the overlap matrix elements
        """

        # TODO: Benchmark these overlaps

        # Build the overlap integrals (p|q') in the AO basis
        pairlist = ls.PairList.build_schwarz(casciA.basis, casciB.basis, False, casciA.pairlist.thre)
        Sao = ls.IntBox.overlap(casciA.resources, pairlist)

        if orbital_coincidence == 'core' or orbital_coincidence == 'full':
            # Build the metric in the active orbitals only (make coincident later)
            Mtu = ls.Tensor.chain([casciA.tensors['Cact'], Sao, casciB.tensors['Cact']], [True, False, False])
            # Assume the overlap from the core is 1.0
            detC = 1.0 
        elif orbital_coincidence == 'none':
            # Build the orbital overlap integrals in the occupied and active blocks of the MO basis
            Sij = ls.Tensor.chain([casciA.tensors['Cocc'], Sao, casciB.tensors['Cocc']], [True, False, False])
            Siu = ls.Tensor.chain([casciA.tensors['Cocc'], Sao, casciB.tensors['Cact']], [True, False, False])
            Stj = ls.Tensor.chain([casciA.tensors['Cact'], Sao, casciB.tensors['Cocc']], [True, False, False])
            Stu = ls.Tensor.chain([casciA.tensors['Cact'], Sao, casciB.tensors['Cact']], [True, False, False])
            # Compute the overlap inverse and determinant in the core space
            detC = Sij.invert_lu()
            # Compute the modified metric integrals M_{tu} = S_{tu} - S_{tj} S_{ij}^{-1} S_{iu}
            Mtu = ls.Tensor.array(Stu)
            if detC != 0.0:
                Mtu[...] -= ls.Tensor.chain([Stj, Sij, Siu], [False, False, False])
        else:
            raise RuntimeError('Unknown value of orbital_coincidence: %s' % orbital_coincidence)

        # SVD of active space metric integrals (corresponding orbital transform)
        U, m, V = ls.Tensor.svd(Mtu)
        # Clamp metric singular values to 1.0 if full orbital_coincidence treatment
        if orbital_coincidence == 'full':
            m[...] = 1.0
        V = V.transpose() 
        # print np.max(np.abs(np.dot(np.dot(U,np.diag(m)),V) - M[...]))

        # Form the many-body metric (in det basis)
        Odet = casciA.casbox.metric_det(m)

        # Which state inds to use?
        indsA2 = indsA if indsA else list(range(len(casciA.evecs[S])))
        indsB2 = indsB if indsB else list(range(len(casciB.evecs[S])))

        # Transform CI amplitudes to bi-orthogonal space (in det basis)
        evecsA = []
        for indexA in indsA2:
            evecA = casciA.evecs[S][indexA]
            evecsA.append(casciA.casbox.orbital_transformation_det(
                U,
                casciA.casbox.CSF_basis(S).transform_CSF_to_det(evecA)))
        evecsB = []
        for indexB in indsB2:
            evecB = casciB.evecs[S][indexB]
            evecsB.append(casciB.casbox.orbital_transformation_det(
                V,
                casciB.casbox.CSF_basis(S).transform_CSF_to_det(evecB)))

        # Compute the overlaps as weighted dot products (in det basis)
        O = ls.Tensor((len(casciA.evecs[S]), len(casciB.evecs[S])))
        for A, evecA in enumerate(evecsA):
            for B, evecB in enumerate(evecsB):
                O[A, B] = np.sum(evecA[...] * evecB[...] * Odet)

        # Account for the core overlap
        O[...] *= detC**2

        # Force state coincidence
        if state_coincidence == 'full':
            U, s, V = ls.Tensor.svd(O)
            O = ls.Tensor.chain([U, V], [False, False])
        elif state_coincidence == 'none':
            pass
        else:
            raise RuntimeError('Unknown value of state_coincidence: %s' % state_coincidence)

        return O

    @staticmethod
    def orbital_health(
        casciA,
        casciB,
        ):

        """ Diagnose if two CASCI objects have coincident occ/act orbital spaces.
        
        This routine looks at the singular values of the orbital overlaps
        between the occ/act spaces and returns the ratio of smallest/largest
        singular values in each space. These metrics should be near 1 for
        healthy orbital spaces (even accounting for mild changes in geometry),
        and will decay to near 0 if an orbital flips between the occ/act/vir
        spaces in the two CASCI objects.

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
            casciA (psiw.CASCI) - bra-side CASCI object
            casciB (psiw.CASCI) - ket-side CASCI object
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
        if casciA.sizes['nocc'] != casciB.sizes['nocc']: raise ValueError('nocc must be same size')
        if casciA.sizes['nact'] != casciB.sizes['nact']: raise ValueError('nact must be same size')

        # Build the overlap integrals (mu|nu') in the AO basis
        pairlist = ls.PairList.build_schwarz(casciA.basis, casciB.basis, False, casciA.pairlist.thre)
        SAB = ls.IntBox.overlap(casciB.resources, pairlist)

        if casciA.sizes['nocc']:
            Socc = ls.Tensor.chain([casciA.tensors['Cocc'], SAB, casciB.tensors['Cocc']], [True, False, False])
            Uocc, socc, Vsocc = ls.Tensor.svd(Socc)
            kocc = np.min(socc) / np.max(socc)
        else:
            kocc = 1.0
            socc = ls.Tensor.zeros((0,))

        if casciA.sizes['nact']:
            Sact = ls.Tensor.chain([casciA.tensors['Cact'], SAB, casciB.tensors['Cact']], [True, False, False])
            Uact, sact, Vsact = ls.Tensor.svd(Sact)
            kact = np.min(sact) / np.max(sact)
        else:
            kact = 1.0
            sact = ls.Tensor.zeros((0,))

        return kocc, kact, socc, sact

    # => OPDM/TPDM/Cumulant/Orbital Gradient <= #
        
    def compute_orbital_lagrangian(
        self,
        D1act,
        D2act,
        L2act,
        ):

        """ Compute the orbital Lagrangian: for standard CASCI this is:
        
        X_pq = 2 h_pr D1_qr + 4 (pr|st) D2_qrst

        where D1 and D2 are fully symmetrized (transition) density matrices

        Params:
            D1act (nmo,)*2 Tensor- the symmetrized (transition) OPDM in the active basis
            D2act (nmo,)*4 Tensor - the symmetrized (transition) TPDM in the active basis
            L2act (nmo,)*4 Tensor - the symmetrized (transition) 2 cumulant in the active basis
        Returns:
            X - the (nmo, nmo) Tensor of the orbital Lagrangian matrix
        """

        if self.is_dft: raise ValueError('Cannot handle DFT core yet')

        D1ao = ls.Tensor.chain([self.tensors['Cact'],D1act,self.tensors['Cact']], [False,False,True])

        J1ao = ls.IntBox.coulomb(
            self.resources,
            ls.Ewald.coulomb(),
            self.pairlist,
            self.pairlist,
            D1ao,
            self.options['thre_sp'],    
            self.options['thre_dp'],    
            )
        K1ao = ls.IntBox.exchange(
            self.resources,
            ls.Ewald.coulomb(),
            self.pairlist,
            self.pairlist,
            D1ao,
            True,
            self.options['thre_sp'],    
            self.options['thre_dp'],    
            )
        F1ao = ls.Tensor.array(self.tensors['H'])
        F1ao.np[...] += 1.0 * J1ao.np[...]
        F1ao.np[...] -= 0.5 * K1ao.np[...]
    
        F1 = ls.Tensor.chain([self.tensors['C'],F1ao,self.tensors['C']],[True,False,False])
        D1 = ls.Tensor.zeros_like(F1)
        for k in range(self.sizes['nocc']):
            D1[k,k] = 2.0
        D1[self.sizes['nocc']:self.sizes['nocc']+self.sizes['nact'],self.sizes['nocc']:self.sizes['nocc']+self.sizes['nact']] = D1act

        X = ls.Tensor.array(2.0 * ls.Tensor.chain([F1,D1],[False,False]).np)
        X[:,self.sizes['nocc']:self.sizes['nocc']+self.sizes['nact']] += 4.0 * ls.Tensor.einsum('wvut,wvup->tp', self.tensors['I'], L2act).np

        return X

    @staticmethod
    def compute_cumulant(
        D1act,
        D2act,
        ):

        """ Compute the 2-cumulant from the 1-pdm and 2-pdm. Active space indices only.

        L_pqrs = D2_pqrs - 0.5 * D1_pq * D1_rs + 0.25 * D1_pr * D1_qs

        Params:
            D1act (Tensor of shape (nact,)*2) - 1-pdm in active space
            D2act (Tensor of shape (nact,)*4) - 1-pdm in active space
        Returns:
            L2act (Tensor of shape (nact,)*4) - 2-cumulant in active space
        """

        L2act = ls.Tensor.array(D2act)
        L2act[...] -= 0.50 * np.einsum('pq,rs->pqrs', D1act, D1act)
        L2act[...] += 0.25 * np.einsum('pr,qs->pqrs', D1act, D1act)
        return L2act

    # => Dipoles and Oscillator Strengths <= #

    @property
    def dipoles(self):
        """ A dict mapping from S index to [X, Y, Z] total/transition dipole Tensors """
        if not hasattr(self, '_dipoles'):
            self._dipoles = {}
            for S in self.options['S_inds']:
                self._dipoles[S] = self.compute_dipoles(S)
        return self._dipoles

    @property
    def oscillator_strengths(self):
        """ A dict mapping from S index to O dipole-based oscillator strength Tensor """
        if not hasattr(self, '_oscillator_strengths'):
            self._oscillator_strengths = {}
            for S in self.options['S_inds']:
                self._oscillator_strengths[S] = self.compute_oscillator_strengths(S, self.dipoles[S])
        return self._oscillator_strengths
        
    def compute_dipoles(
        self,
        S, 
        x0=0.0,
        y0=0.0,
        z0=0.0,
        ):

        """Calculates total and transition dipole moments for all states in
            spin block S.

        Users should call the 'dipoles' property above.

        Params:
            S (int) - spin block index
            x0 (float) - origin
            y0 (float) - origin
            z0 (float) - origin

        Returns:
            XYZ ((nstates, nstates) ls.Tensor.array) - transition dipole matrix

        """

        # xyz dipole matrix in AO basis
        XYZ = ls.IntBox.dipole(
            self.resources,
            self.pairlist,
            x0,
            y0,
            z0,
            )

        # Gives xyz in active MO basis
        XYZ2 = [ls.Tensor.chain([self.tensors['Cact'], x, self.tensors['Cact']], [True, False, False]) for x in XYZ]
        evecs = self.evecs[S]
        nstate = len(evecs)

        XYZ3 = [ls.Tensor((nstate, nstate)) for x in range(3)] 
        for A, evecA in enumerate(evecs):
            for B, evecB in enumerate(evecs):
                if A > B: continue
                Dact = self.casbox.opdm(
                    self.resources,
                    S,
                    evecA,
                    evecB,
                    True,     
                    )
                
                XYZ3[0][A, B] = XYZ3[0][B, A] = Dact.vector_dot(XYZ2[0])
                XYZ3[1][A, B] = XYZ3[1][B, A] = Dact.vector_dot(XYZ2[1])
                XYZ3[2][A, B] = XYZ3[2][B, A] = Dact.vector_dot(XYZ2[2])
                
        # Core Density matrix
        Dcore = ls.Tensor.chain([self.tensors['Cocc'], self.tensors['Cocc']], [False, True])                
        Dcore[...] *= 2.0

        Xcore = Dcore.vector_dot(XYZ[0])
        Ycore = Dcore.vector_dot(XYZ[1])
        Zcore = Dcore.vector_dot(XYZ[2])

        # Get nuclear dipole
        for A, atomA in enumerate(self.qm_molecule.atoms):
            Xcore -= atomA.Z * (atomA.x - x0)
            Ycore -= atomA.Z * (atomA.y - y0)
            Zcore -= atomA.Z * (atomA.z - z0)

        for A in range(nstate):
            XYZ3[0][A, A] += Xcore
            XYZ3[1][A, A] += Ycore
            XYZ3[2][A, A] += Zcore
    
        return XYZ3                
    
    def compute_oscillator_strengths(
        self,
        S,
        XYZ,
        ):
        
        """ Calculates oscillator strengths for transitions in spin block S

        Users should call the 'oscillator_strengths' property above.

        Params:
            S (int) - spin block index
            XYZ ((nstates, nstates) ls.Tensor.array) - transition dipole matrix

        Returns:
            O ((nstates, nstates) ls.Tensor.array) - oscillator strengths

        """

        O = ls.Tensor.zeros_like(XYZ[0])
        for A, E_A in enumerate(self.evals[S]):
            for B, E_B in enumerate(self.evals[S]):
                if A >= B: continue
                dE = E_B - E_A
                T = XYZ[0][A, B]**2.0 + XYZ[1][A][B]**2.0 + XYZ[2][A][B]**2.0
                osc = (2.0/3.0) * dE * T
                O[A,B] = O[B,A] = osc

        return O

    @property
    def transition_string(self):
        """ Return a string detailing the state energies, transition energies,
            dipoles, transition dipoles, and oscillator strengths.

        Users should not call this.
        """

        s = ''

        H2eV = 27.211396132 # TODO: Fix this

        S_inds = self.options['S_inds'] 
        XYZs = self.dipoles
        Os = self.oscillator_strengths

        Es = [(S_inds[i], j, y) for i, x in enumerate(self.evals) for j, y in enumerate(self.evals[x])]
        Es = list(sorted(Es, key=lambda x : x[2]))

        s +=  'CASCI States:\n'
        s += '\n'
        s +=  '%2s %2s %2s %24s %10s %10s %10s %10s %10s\n' % ('N', 'S', 'I', 'E [au]', 'dE [eV]', 'Dx', 'Dy', 'Dz', '|D|')
        for A, E2 in enumerate(Es):
            dE = E2[2] - Es[0][2]
            S = E2[0]
            S_ind = E2[1]
            Dx = XYZs[S][0][S_ind, S_ind]
            Dy = XYZs[S][1][S_ind, S_ind]
            Dz = XYZs[S][2][S_ind, S_ind]
            D = np.sqrt(Dx**2 + Dy**2 + Dz**2)
            s +=  '%2d %2d %2d %24.16E %10.6f %10.6f %10.6f %10.6f %10.6f\n' % (A, S, S_ind, E2[2], dE * H2eV, Dx, Dy, Dz, D)
        s +=  '\n'

        for S in S_inds:
            nstates = Os[S].shape[0]
            s += 'CASCI S=%d Transitions:\n' % S
            s += '\n'
            s += '%12s %10s %10s %10s %10s %10s %10s\n' % ('Transition', 'dE [eV]', 'Tx', 'Ty', 'Tz', '|T|', 'Osc.')
            for S1 in range(nstates):
                for S2 in range(S1, nstates):
                    if S1 == S2: continue
                    dE = (self.evals[S][S2] - self.evals[S][S1])
                    Tx = XYZs[S][0][S1, S2]
                    Ty = XYZs[S][1][S1, S2]
                    Tz = XYZs[S][2][S1, S2]
                    T = np.sqrt(Tx**2 + Ty**2 + Tz**2)
                    O = Os[S][S1, S2]
                    s +=  '  %2d %2s %-2d   %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f\n' % (S1, '->', S2, dE * H2eV, Tx, Ty, Tz, T, O)
            s +=  '\n'

        return s

