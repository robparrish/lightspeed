import lightspeed as ls
import numpy as np
from . import options
from . import geometry
from . import rhf
import collections
# import time

class CIS(object):

    @staticmethod
    def default_options():
        if hasattr(CIS, '_default_options'): return CIS._default_options.copy()
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
            required=True,
            allowed_types=[rhf.RHF],
            doc='Reference RHF wavefunction (provides geometry, orbitals, orbital eigenvalues and structure of S0)')

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
            doc="List of spin indices (e.g., [0, 2] for singlet, triplet - the only allowed options in restricted CIS)")
        opt.add_option(
            key='S_nstates',
            required=True,
            allowed_types=[list],
            doc="List of number of states requested in each spin index (e.g., [2, 2])")

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

        CIS._default_options = opt
        return CIS._default_options.copy()

    def __init__(
        self,
        options,
        ):
        
        self.options = options

        # Cache some useful attributes
        self.print_level = self.options['print_level']
        self.reference = self.options['reference']
        self.geometry = self.reference.geometry
        self.resources = self.geometry.resources
        self.qm_molecule = self.geometry.qm_molecule
        self.basis = self.geometry.basis
        self.pairlist = self.geometry.pairlist
    
        # Some useful registers
        self.sizes = {}
        self.scalars = {}
        self.tensors = {}

    @staticmethod
    def from_options(**kwargs):
        """ Return an instance of this class with default options updated from values in kwargs. """
        return CIS(CIS.default_options().set_values(kwargs))

    def compute_energy(self):

        # => Title Bars <= #

        if self.print_level:
            print('==> CIS <==\n')

        # => Validity Checks <= #

        if self.reference.is_dft: raise ValueError('TDA is not yet coded')
        if self.reference.is_fomo: raise ValueError('CIS assumes the Brilluoin condition: cannot do FOMO')

        # => SCF Quantities <= #

        self.tensors['C'] = self.reference.tensors['C']
        self.tensors['Cocc'] = self.reference.tensors['Cocc']
        self.tensors['Cvir'] = self.reference.tensors['Cvir']
        self.tensors['eps'] = self.reference.tensors['eps']
        self.tensors['eps_occ'] = self.reference.tensors['eps_occ']
        self.tensors['eps_vir'] = self.reference.tensors['eps_vir']
    
        self.sizes['natom'] = self.qm_molecule.natom
        self.sizes['nao'] = self.basis.nao
        self.sizes['nmo'] = self.tensors['C'].shape[1]
        self.sizes['nocc'] = self.tensors['Cocc'].shape[1]
        self.sizes['nvir'] = self.tensors['Cvir'].shape[1]

        self.scalars['Enuc'] = self.reference.scalars['Enuc']
        self.scalars['Eext'] = self.reference.scalars['Eext']
        self.scalars['Escf'] = self.reference.scalars['Escf']

        # => Integral Thresholds <= #

        if self.print_level > 1:
            print('Integral Thresholds:')
            print('  Threshold DP = %11.3E' % self.options['thre_dp'])
            print('  Threshold SP = %11.3E' % self.options['thre_sp'])
            print('')

        # => External/Reference Energies <= #

        if self.print_level:
            print('External Environment/Reference:')
            print('  Enuc = %24.16E' % self.scalars['Enuc'])
            print('  Eext = %24.16E' % self.scalars['Eext'])
            print('  Escf = %24.16E' % self.scalars['Escf'])
            print('')

        # => State Taskings <= #

        if self.print_level > 1:
            print('CIS Requested States:')
            print('%2s %6s' % ('S', 'Nstate'))
            for S, nS in zip(self.options['S_inds'], self.options['S_nstates']):
                print('%2d %6d' % (
                    S, 
                    nS))
            print('')

        # => Diagonalization <= #

        self.converged = {}
        self.evals = {}
        self.evecs = {}
        for S_ind, S_nstate in zip(*[self.options[x] for x in ['S_inds', 'S_nstates']]):
                self.compute_states(S_ind, S_nstate)

        # => Print Amplitude Summary <= #

        if self.print_level > 1:
            print('CIS Amplitudes:\n')
            for S_ind in self.options['S_inds']:
                for S_ind2, state in enumerate(self.evecs[S_ind]):
                    print('State S=%d I=%d:' % (S_ind, S_ind2))
                    print(self.amplitude_string(
                        S_ind,  
                        S_ind2,
                        0.0001,
                        60))

        # => Print Transition Info <= #

        if self.print_level > 1:
            print(self.transition_string, end=' ')

        # => Trailer Bars <= #

        if self.print_level > 1:
            print('"Someone has to save our skins. Into the garbage chute, fly boy!"')
            print('    --Princess Leia\n')

        if self.print_level:
            print('==> End CIS <==\n')
        
    def compute_states(
        self,
        Sindex,
        nstate,
        ):

        # => Title <= #

        title = 'S=%d' % Sindex
        if self.print_level:
            print('=> %s States <=\n' % title)
        if Sindex not in [0, 2]: raise RuntimeError('Invalid Sindex for CIS: %d' % Sindex)

        # => Preconditioner <= #

        F = ls.Tensor((self.sizes['nocc'], self.sizes['nvir']))
        F.np[...] += np.einsum('i,a->ia',  np.ones(self.sizes['nocc']), self.tensors['eps_vir'])
        F.np[...] -= np.einsum('i,a->ia',  self.tensors['eps_occ'], np.ones(self.sizes['nvir']))

        # => Guess Vectors <= #

        nstate_new = nstate - 1 if Sindex == 0 else nstate        
        nguess = min(self.sizes['nocc'] * self.sizes['nvir'], nstate_new * self.options['nguess_per_root'])
        if self.print_level > 1:
            print('Guess Details:') 
            print('  Nguess: %11d' % nguess)
        Forder = sorted([(x, xind[0], xind[1]) for xind, x in np.ndenumerate(F)])
        Cs = []
        for k in range(nguess):
            C = ls.Tensor((self.sizes['nocc'], self.sizes['nvir']))
            r = Forder[k][1]
            c = Forder[k][2]
            if self.print_level > 1:
                print('  %3d: %5d -> %-5d' % (k,r,c+self.sizes['nocc']))
            C.np[r,c] = +1.0
            Cs.append(C)
        if self.print_level > 1:
            print('')

        # => Convergence Info <= #

        if self.print_level > 1:
            print('Convergence:')
            print('  Maxiter: %11d' % self.options['maxiter'])
            print('')

        # => Davidson Object <= #

        dav = ls.Davidson(
            nstate_new,
            self.options['nmax'],
            self.options['r_convergence'],
            self.options['norm_cutoff'],
            )
        if self.print_level > 1:
            print(dav)

        # ==> Davidson Iterations <== # 

        if self.print_level:
            print('%s CIS Iterations:\n' % title)
            print('%4s: %11s' % ('Iter', '|R|'))
        converged = False
        for iter in range(self.options['maxiter']):

            # Sigma Vector Builds
            Ss = [ls.Tensor.clone(C) for C in Cs]
            for C, S in zip(Cs, Ss):
                # One-Particle Term
                S.np[...] *= F 
                # Two-Particle Term
                D = ls.Tensor.chain([self.tensors['Cocc'], C, self.tensors['Cvir']], [False, False, True])
                if Sindex == 0:
                    J = ls.IntBox.coulomb(
                        self.resources,
                        ls.Ewald.coulomb(),
                        self.pairlist,
                        self.pairlist,
                        D,
                        self.options['thre_sp'],    
                        self.options['thre_dp'],    
                        )
                    J2 = ls.Tensor.chain([self.tensors['Cocc'], J, self.tensors['Cvir']], [True, False, False])
                K = ls.IntBox.exchange(
                    self.resources,
                    ls.Ewald.coulomb(),
                    self.pairlist,
                    self.pairlist,
                    D,
                    False,
                    self.options['thre_sp'],    
                    self.options['thre_dp'],    
                    )
                K2 = ls.Tensor.chain([self.tensors['Cocc'], K, self.tensors['Cvir']], [True, False, False])
                if Sindex == 0:
                    S.np[...] += 2. * J2.np - K2.np
                else:
                    S.np[...] -= K2.np
        
            # Add vectors/diagonalize Davidson subspace
            Rs, Es = dav.add_vectors(Cs,Ss,use_disk=self.options['use_disk'])

            # Print Iteration Trace
            if self.print_level:
                print('%4d: %11.3E' % (iter, dav.max_rnorm))

            # Check for convergence
            if dav.is_converged:
                converged = True
                break

            # Precondition the desired residuals
            for R, E in zip(Rs, Es):
                R.np[...] /= -(F.np[...] - E)
    
            # Get new trial vectors
            Cs = dav.add_preconditioned(Rs,use_disk=self.options['use_disk'])

        # => Output Quantities <= #
        
        # Print Convergence
        if self.print_level:
            print('')
            if converged:
                print('%s CIS Converged\n' % title)
            else:
                print('%s CIS Failed\n' % title)

        # Cache quantities
        self.converged[Sindex] = converged
        self.evals[Sindex] = []
        self.evecs[Sindex] = []
        # Make allowance for the reference state
        if Sindex == 0:
            self.evals[Sindex] += [self.scalars['Escf']]
            self.evecs[Sindex] += [None]
        self.evals[Sindex] += [x + self.scalars['Escf'] for x in dav.evals]
        self.evecs[Sindex] += [ls.Storage.from_storage(x, Ss[0]) for x in dav.evecs]

        # Print residuals
        if self.print_level:
            print('CIS %s Residuals:\n' % title)
            print('%4s: %11s' % ('I', '|R|'))
            for rind, r in enumerate(dav.rnorms):
                Soffset = 1 if Sindex == 0 else 0 # offset for the ground state
                print('%4d: %11.3E %s' % (rind + Soffset, r, 'Converged' if r < self.options['r_convergence'] else 'Not Converged'))
            print('')
    
        # Print energies
        if self.print_level:
            print('CIS %s Energies:\n' % title)
            print('%4s: %24s' % ('I', 'Total E'))
            for Eind, E in enumerate(self.evals[Sindex]):
                print('%4d: %24.16E' % (Eind, E))
            print('')

        if self.print_level:
            print('=> End %s States <=\n' % title)

    def amplitude_string(
        self,
        S,
        index,
        thre,
        max_dets,
        ):

        if S == 0 and index == 0:
            return '%18.14f XXX\n' % 1.0

        s = ''
        C = self.evecs[S][index]
        Corder = sorted([(-x, xind[0], xind[1]) for xind, x in np.ndenumerate(C[...]**2)])
        Cnp = C.np
        R2 = 1.0
        for ind in range(min(max_dets, len(Corder))):
            if R2 < thre: break
            r = Corder[ind][1]
            c = Corder[ind][2]
            Cval = Cnp[r,c]
            s += '%18.14f %5d -> %5d\n' % (Cval, r, c + self.sizes['nocc'])
            R2 -= Cval**2
        return s

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
        """ A dict mapping from S index to O oscillator strength Tensor """
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

        evecs = self.evecs[S]
        nstate = len(evecs)
        Cocc = self.tensors['Cocc']
        Cvir = self.tensors['Cvir']

        XYZ3 = [ls.Tensor((nstate, nstate)) for x in range(3)] 
        for A, evecA in enumerate(evecs):
            for B, evecB in enumerate(evecs):
                if A > B: continue
                if evecA is None and evecB is None: continue # <0|D|0>
                if evecA is None: # <0|D|0>
                    # D_ia = + sqrt(2) C_ia
                    Dao = ls.Tensor.chain([Cocc, evecB, Cvir], [False, False, True])
                    Dao[...] *= np.sqrt(2.0)
                else:
                    # D_ab = + C_ia C_ib
                    Dao = ls.Tensor.chain([Cvir, evecA, evecB, Cvir], [False, True, False, True])
                    # D_ij = - C_ja C_ia
                    Dao[...] -= ls.Tensor.chain([Cocc, evecB, evecA, Cocc], [False, False, True, True])
                
                # Dot with AB-basis
                XYZ3[0][A, B] = XYZ3[0][B, A] = Dao.vector_dot(XYZ[0])
                XYZ3[1][A, B] = XYZ3[1][B, A] = Dao.vector_dot(XYZ[1])
                XYZ3[2][A, B] = XYZ3[2][B, A] = Dao.vector_dot(XYZ[2])
                
        # Reference density matrix
        Dcore = ls.Tensor.chain([Cocc, Cocc], [False, True])                
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

        Params:
            S (int) - spin block index
            XYZ ((nstates, nstates) ls.Tensor.array) - transition dipole matrix

        Returns:
            O ((nstates, nstates) ls.Tensor.array) - oscillator strengths

        """

        O = ls.Tensor.zeros_like(XYZ[S])
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
            dipoles, transition dipoles, and oscillator strengths """

        s = ''

        H2eV = 27.211396132 # TODO: Fix this

        S_inds = self.options['S_inds'] 
        XYZs = self.dipoles
        Os = self.oscillator_strengths

        Es = [(S_inds[i], j, y) for i, x in enumerate(self.evals) for j, y in enumerate(self.evals[x])]
        Es = list(sorted(Es, key=lambda x : x[2]))

        s +=  'CIS States:\n'
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
            s += 'CIS S=%d Transitions:\n' % S
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

    # => Gradient Code <= #

    def compute_gradient(
        self,
        S,
        index,
        ):

        # Handle the reference gradient separately
        if S == 0 and index == 0: return self.reference.compute_gradient()

        # Sizes
        nao = self.sizes['nao']
        nmo = self.sizes['nmo']
        nocc = self.sizes['nocc']
        nvir = self.sizes['nvir']
       
        # CI vector and orbitals
        Cvec = self.evecs[S][index]
        C = self.tensors['C']
        Cocc = self.tensors['Cocc']
        Cvir = self.tensors['Cvir']
    
        # Difference OPDM in AO basis
        D1dao = ls.Tensor.chain([Cvir, Cvec, Cvec, Cvir], [False, True, False, True])
        D1dao[...] -= ls.Tensor.chain([Cocc, Cvec, Cvec, Cocc], [False, False, True, True])

        # Total OPDM in AO basis
        D1ao = ls.Tensor.chain([Cocc, Cocc], [False, True])
        D1ao[...] *= 2.0
        D1ao[...] += D1dao

        # CI Vector in AO basis
        C1ao = ls.Tensor.chain([Cocc, Cvec, Cvir], [False, False, True])
        
        # => J/K contributions <= #

        JD1 = ls.IntBox.coulomb(
            self.resources,
            ls.Ewald.coulomb(),
            self.pairlist,
            self.pairlist,
            D1dao,
            self.options['thre_sp'],    
            self.options['thre_dp'],    
            )
        KD1 = ls.IntBox.exchange(
            self.resources,
            ls.Ewald.coulomb(),
            self.pairlist,
            self.pairlist,
            D1dao,
            True,
            self.options['thre_sp'],    
            self.options['thre_dp'],    
            )
        JC1 = ls.IntBox.coulomb(
            self.resources,
            ls.Ewald.coulomb(),
            self.pairlist,
            self.pairlist,
            C1ao,
            self.options['thre_sp'],    
            self.options['thre_dp'],    
            )
        KC1 = ls.IntBox.exchange(
            self.resources,
            ls.Ewald.coulomb(),
            self.pairlist,
            self.pairlist,
            C1ao,
            False,
            self.options['thre_sp'],    
            self.options['thre_dp'],    
            )

        # TODO: The following manipulations can be *much* faster if one exploits block sparsity

        # => Fock Matrices (Long Way) <= #

        FO2 = ls.Tensor.chain([C, self.reference.tensors['F'], C], [True, False, False])
        FD2 = ls.Tensor.chain([C, JD1, C], [True, False, False])
        FD2.np[...] -= 0.5 * ls.Tensor.chain([C, KD1, C], [True, False, False]).np
        FA2 = ls.Tensor.array(FO2)
        FA2.np[...] += FD2

        # => MO-Basis OPDMs (Long Way) <= #

        D2 = ls.Tensor((nmo,)*2)
        D2[nocc:,nocc:] += ls.Tensor.chain([Cvec, Cvec], [True, False])
        D2[:nocc,:nocc] -= ls.Tensor.chain([Cvec, Cvec], [False, True])
        A2 = ls.Tensor.array(np.diag(self.reference.tensors['n']))
        A2[...] *= 2.0
        A2[...] += D2

        # => MO-Basis Amplitudes (Long Way) <= #

        C2 = ls.Tensor((nmo,)*2)        
        C2[:nocc,nocc:] = Cvec

        # => Lagrangian (Long Way) <= #

        X = ls.Tensor([self.sizes['nmo']] * 2)
        X.np[...] += ls.Tensor.chain([A2, FA2],[False,False])
        X.np[...] -= ls.Tensor.chain([D2, FD2],[False,False])
        if S == 0:
            X.np[...] += 2. * ls.Tensor.chain([C2,C,JC1,C],[False,True,True,False]).np
            X.np[...] += 2. * ls.Tensor.chain([C2,C,JC1,C],[True,True,False,False]).np
        X.np[...] -= 1. * ls.Tensor.chain([C2,C,KC1,C],[False,True,True,False]).np
        X.np[...] -= 1. * ls.Tensor.chain([C2,C,KC1,C],[True,True,False,False]).np
    
        X[...] *= 2.0 # Spin-summed convention
        X = X.transpose() # Ed's convention: X_pq = 2 h_pr D1_qr + 4 (pr|st) D2_qrst

        # => TPDM Gradient <= #

        grads = collections.OrderedDict()
        # Coulomb integral gradient 1
        grads['J1'] = ls.IntBox.coulombGrad(
            self.resources,
            ls.Ewald.coulomb(),
            self.pairlist,
            D1ao,
            D1ao,
            self.options['thre_sp'],
            self.options['thre_dp'],
            )
        grads['J1'].np[...] *= +0.5
        # Coulomb integral gradient 2
        grads['J2'] = ls.IntBox.coulombGrad(
            self.resources,
            ls.Ewald.coulomb(),
            self.pairlist,
            D1dao,
            D1dao,
            self.options['thre_sp'],
            self.options['thre_dp'],
            )
        grads['J2'].np[...] *= -0.5
        # Exchange integral gradient 1
        grads['K1'] = ls.IntBox.exchangeGrad(
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
        grads['K1'].np[...] *= -0.25
        # Exchange integral gradient 2
        grads['K2'] = ls.IntBox.exchangeGrad(
            self.resources,
            ls.Ewald.coulomb(),
            self.pairlist,
            D1dao,
            D1dao,
            True,
            True,
            True,
            self.options['thre_sp'],
            self.options['thre_dp'],
            )
        grads['K2'].np[...] *= +0.25
        # Coulomb integral gradient 3
        grads['J3'] = ls.IntBox.coulombGrad(
            self.resources,
            ls.Ewald.coulomb(),
            self.pairlist,
            C1ao,
            C1ao,
            self.options['thre_sp'],
            self.options['thre_dp'],
            )
        grads['J3'].np[...] *= +2.0 if S == 0 else 0.0
        # Exchange integral gradient 2
        grads['K3'] = ls.IntBox.exchangeGrad(
            self.resources,
            ls.Ewald.coulomb(),
            self.pairlist,
            C1ao,
            C1ao,
            False,
            False,
            True,
            self.options['thre_sp'],
            self.options['thre_dp'],
            )
        grads['K3'].np[...] *= -1.0

        grad2e = ls.Tensor.zeros_like(grads['J1'])
        for grad in list(grads.values()):
            grad2e[...] += grad
        
        # for key, grad in grads.iteritems():
        #     grad.name = key
        #     print grad

        return self.reference.compute_hf_based_gradient(
            D1ao,
            X,
            grad2e) 

    # => Derivative Coupling Code <= #

    def compute_coupling(
        self,
        S, 
        indexA,
        indexB,
        doETF=False, # Do Joe Subotnik ETF?
        ):

        if indexA == indexB: raise ValueError('indexA == indexB: you want a gradient?')

        # Sizes
        nao = self.sizes['nao']
        nmo = self.sizes['nmo']
        nocc = self.sizes['nocc']
        nvir = self.sizes['nvir']

        # Energy difference between states
        dE = self.evals[S][indexB] - self.evals[S][indexA]
        CA = self.evecs[S][indexA]
        CB = self.evecs[S][indexB]
        Cocc = self.tensors['Cocc']
        Cvir = self.tensors['Cvir']

        # Difference transition OPDM in AO basis
        D1dao = ls.Tensor((nao,)*2)
        if CA is None:
            # <0|B>
            # T_jb = + sqrt(2) CB_jb
            D1dao[...] += np.sqrt(2.0) * ls.Tensor.chain([Cocc, CB, Cvir], [False, False, True])
        elif CB is None:
            # <A|0>
            # T_ai = + sqrt(2) CA_ia
            D1dao[...] += np.sqrt(2.0) * ls.Tensor.chain([Cvir, CA, Cocc], [False, True, True])
        else:
            # T_ab <- + CA_ia CB_ib
            D1dao[...] = ls.Tensor.chain([Cvir, CA, CB, Cvir], [False, True, False, True])
            # T_ji <- - CA_ia CB_ja (watch the tanspose!)
            D1dao[...] -= ls.Tensor.chain([Cocc, CB, CA, Cocc], [False, False, True, True])

        raise ValueError("Not implemented")

    @staticmethod
    def compute_overlap(
        cisA,
        cisB,
        S,
        ):
        
        """ Compute the overlap between all states in spin block S in two CIS object.

        Params:
            cisA (CIS) - the bra-side CIS object
            cisB (CIS) - the ket-side CIS object
            S (int) - the spin index
        Returns:
            O (ls.Tensor, shape (nstateA, nstateB)) - the overlap matrix elements
        """

        # TODO: Benchmark these overlaps

        # Build the overlap integrals (p|q') in the AO basis
        pairlist = ls.PairList.build_schwarz(cisA.basis, cisB.basis, False, cisA.pairlist.thre)
        Sao = ls.IntBox.overlap(cisA.resources, pairlist)

        # Build the orbital overlap integrals in the occupied and active blocks of the MO basis
        Sij = ls.Tensor.chain([cisA.tensors['Cocc'], Sao, cisB.tensors['Cocc']], [True, False, False])
        Sib = ls.Tensor.chain([cisA.tensors['Cocc'], Sao, cisB.tensors['Cvir']], [True, False, False])
        Saj = ls.Tensor.chain([cisA.tensors['Cvir'], Sao, cisB.tensors['Cocc']], [True, False, False])
        Sab = ls.Tensor.chain([cisA.tensors['Cvir'], Sao, cisB.tensors['Cvir']], [True, False, False])

        # Compute the overlap inverse and determinant in the core space
        detC = Sij.invert_lu()
        
        # Form modified metric integrals
        Mij = Sij
        Mia = ls.Tensor((cisA.sizes['nocc'], cisA.sizes['nvir']))
        Mjb = ls.Tensor((cisB.sizes['nocc'], cisB.sizes['nvir']))
        Mab = ls.Tensor.array(Sab)
        if detC != 0.0:
            # Mai[...] = ls.Tensor.chain([Saj, Mij], [False, False])
            Mia[...] = ls.Tensor.chain([Mij, Saj], [True, True]) # Transposed from the notes for easier dot
            Mjb[...] = ls.Tensor.chain([Mij, Sib], [False, False])
            Mab[...] -= ls.Tensor.chain([Saj, Mjb], [False, False])

        # Compute the intermediates for connected overlap term
        A2s = []
        for A, evecA in enumerate(cisA.evecs[S]):
            if S == 0 and A == 0:
                A2s.append(None)
                continue
            A2s.append(ls.Tensor.chain([evecA, Mab], [False, False]))
        B2s = []
        for B, evecB in enumerate(cisB.evecs[S]):
            if S == 0 and B == 0:
                B2s.append(None)
                continue
            B2s.append(ls.Tensor.chain([Mij, evecB], [True, False]))

        # Compute the overlaps
        O = ls.Tensor((len(cisA.evecs[S]), len(cisB.evecs[S])))
        for A, evecA in enumerate(cisA.evecs[S]):
            for B, evecB in enumerate(cisB.evecs[S]):
                if S == 0 and A == 0 and B == 0: 
                    # <0|0>
                    O[A,B] = 1.0
                    continue
                if S == 0 and A == 0:
                    # <0|B>
                    O[A,B] = np.sqrt(2.0) * evecB.vector_dot(Mjb)
                    continue
                if S == 0 and B == 0:
                    # <A|0>
                    O[A,B] = np.sqrt(2.0) * evecA.vector_dot(Mia)
                    continue
                else:
                    # <A|B>
                    O[A,B] = A2s[A].vector_dot(B2s[B])
                    if S == 0:
                        O[A,B] += 2.0 * evecA.vector_dot(Mia) * evecB.vector_dot(Mjb)
                
        # Account for the core overlap
        O[...] *= detC**2

        return O


# => Test Routines <= #

def test_h2o():

    resources = ls.ResourceList.build()
    
    molecule = ls.Molecule.from_xyz_str("""
        O   0.000000000000  -0.143225816552   0.000000000000
        H   1.638036840407   1.136548822547  -0.000000000000
        H  -1.638036840407   1.136548822547  -0.000000000000""",
        name='h2o',
        scale=1.0)

    geom = geometry.Geometry.build(
        resources=resources,
        molecule=molecule,
        # basisname='6-31g',
        basisname='sto-3g',
        )

    ref = rhf.RHF(rhf.RHF.default_options().set_values({
        'geometry' : geom,
        'g_convergence' : 1.0E-9,
        }))
    ref.compute_energy()

    cis = CIS(CIS.default_options().set_values({
        'reference' : ref,
        'S_inds' : [0, 2],
        'S_nstates' : [3, 3],
        })) 
    cis.compute_energy()

    # print cis.compute_gradient(0, 0)
    print('Singlet Gradient:')
    print(cis.compute_gradient(0, 1))
    print('Triplet Gradient:')
    print(cis.compute_gradient(2, 0))

    print(CIS.compute_overlap(
        cis,
        cis,
        0))
    print(CIS.compute_overlap(
        cis,
        cis,
        2))
    
if __name__ == '__main__':

    test_h2o()

