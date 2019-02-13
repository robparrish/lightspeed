import lightspeed as ls
import numpy as np
from . import options

class IBO(object):

    @staticmethod
    def default_options():
        if hasattr(IBO, '_default_options'): return IBO._default_options.copy()
        opt = options.Options() 

        # > Problem Geometry < #

        opt.add_option(
            key='resources',
            required=True,
            allowed_types=[ls.ResourceList],
            doc='ResourceList to use for this EST problem')
        opt.add_option(
            key="molecule",
            required=True, 
            allowed_types=[ls.Molecule],
            doc="QM Molecule")
        opt.add_option(
            key="basis",
            required=True,
            allowed_types=[ls.Basis],
            doc="Primary AO basis set for QM molecule")
        opt.add_option(
            key="minbasis",
            required=True,
            allowed_types=[ls.Basis],
            doc="Minimal AO basis set for QM molecule")
        opt.add_option(
            key="pairlist",
            required=True,
            allowed_types=[ls.PairList],
            doc="PairList corresponding to the primary AO basis")
        opt.add_option(
            key="Cref",
            required=True,
            allowed_types=[ls.Tensor],
            doc="Reference orbitals in primary basis (nao,nref)")
    
        # > Numerical Thresholds < #

        opt.add_option(
            key='thre_condition',
            value=1.0E-12,
            required=True,
            allowed_types=[float],
            doc='Condition number for pseudoinversion in metrics')

        # > Convergence Options < #

        opt.add_option(
            key='power',
            value=4,
            required=True,
            allowed_values=[2,4],
            allowed_types=[int],
            doc='Power in localization metric (4-th power recommended)')
        opt.add_option(
            key='maxiter',
            value=50,
            required=True,
            allowed_types=[int],
            doc='Maximum number of localization iterations before failure')
        opt.add_option(
            key='g_convergence',
            value=1.0E-12,
            required=True,
            allowed_types=[float],
            doc='Maximum allowed element in the orbital gradient at convergence')

        IBO._default_options = opt
        return IBO._default_options.copy()

    def __init__(
        self,
        options,
        ):
        
        self.options = options

        # Validity checks
        if self.molecule.natom != self.basis.natom:
            raise ValueError('Molecule and basis natom must agree')
        if self.molecule.natom != self.minbasis.natom:
            raise ValueError('Molecule and minbasis natom must agree')
    
        # Initialize IAOs/Overlap
        self.A, self.S = self.build_iaos()

    @staticmethod
    def from_options(**kwargs):
        """ Return an instance of this class with default options updated from values in kwargs. """
        return IBO(IBO.default_options().set_values(kwargs))

    @property
    def resources(self):
        return self.options['resources']

    @property
    def molecule(self):
        return self.options['molecule']

    @property
    def basis(self):
        return self.options['basis']

    @property
    def minbasis(self):
        return self.options['minbasis']

    @property
    def pairlist(self):
        return self.options['pairlist']

    @property
    def Cref(self):
        return self.options['Cref']

    def build_iaos(self):

        """ Compute the IAO matrix A (and the overlap matrix S). Users should
            not call this - self.A and self.S are initialized to these values
            in the constructor.

        Returns:
            A (ls.Tensor of shape (nao, nmin)) - coefficients of IAOs in primary basis.
            S (ls.Tensor of shape (nao, nao)) - overlap matrix in primary basis.
        """

        pairlist11 = self.pairlist
        pairlist12 = ls.PairList.build_schwarz(self.basis,self.minbasis,False,self.pairlist.thre)
        pairlist22 = ls.PairList.build_schwarz(self.minbasis,self.minbasis,True,self.pairlist.thre)
        S11 = ls.IntBox.overlap(self.resources,pairlist11)
        S12 = ls.IntBox.overlap(self.resources,pairlist12)
        S22 = ls.IntBox.overlap(self.resources,pairlist22)

        S11m12 = ls.Tensor.power(S11,-1.0/2.0,self.options['thre_condition'])
        S22m12 = ls.Tensor.power(S22,-1.0/2.0,self.options['thre_condition'])

        C = self.Cref
        T1 = ls.Tensor.chain([S22m12,S12],[False,True])
        T2 = ls.Tensor.chain([S11m12,T1,T1,C],[False,True,False,False])
        T3 = ls.Tensor.chain([T2,T2],[True,False])
        T3 = ls.Tensor.power(T3,-1.0/2.0,self.options['thre_condition'])
        Ctilde = ls.Tensor.chain([S11m12,T2,T3],[False,False,False])

        D = ls.Tensor.chain([C,C],[False,True])
        Dtilde = ls.Tensor.chain([Ctilde,Ctilde],[False,True])

        DSDtilde = ls.Tensor.chain([D,S11,Dtilde],[False,False,False])
        DSDtilde[...] *= 2.0
        
        L = ls.Tensor.chain([S11m12,S11m12],[False,False]) # S11^-1 possibly unstable
        L[...] += DSDtilde
        L[...] -= D
        L[...] -= Dtilde

        AN = ls.Tensor.chain([L,S12],[False,False])
    
        V = ls.Tensor.chain([AN,S11,AN],[True,False,False])
        V = ls.Tensor.power(V,-1.0/2.0,self.options['thre_condition'])
        
        A = ls.Tensor.chain([AN,V],[False,False])
    
        return A, S11

    @property
    def minao_inds(self):
        """ list of list of indices of MinAOs for each atom """
        inds = [[] for A in range(self.minbasis.natom)]        
        for shell in self.minbasis.shells:
            inds[shell.atomIdx] += list(range(shell.aoIdx,shell.aoIdx+shell.nao))
        return inds

    def atomic_charges(
        self,
        D,
        ):

        """ Compute the atomic charges Q_A for an OPDM D_mn.

        Only the electronic contribution is considered.

        Params:
            D (ls.Tensor of shape (nao, nao)) - density matrix
        Returns:
            Q (ls.Tensor of shape (natom,)) - atomic charges
        """

        V = ls.Tensor.chain([self.A, self.S, D, self.S, self.A], [True, False, False, False, False])
        V2 = np.diag(V)
        Q = ls.Tensor((self.minbasis.natom,), 'Q IBO')
        for A, inds2 in enumerate(self.minao_inds):
            Q[A] = np.sum(V2[inds2])
        return Q

    def orbital_atomic_charges(
        self,
        C, 
        ):

        """ Compute the orbital atomic charges Q_A^i for a set of orbitals C_mi

        Only the electronic contribution is considered.

        Params:
            C (ls.Tensor of shape (nao, ni)) - Orbital coefficients
        Returns:
            Q (ls.Tensor of shape (natom, ni)) - Orbital atomic charges
        """

        L = ls.Tensor.chain([C,self.S,self.A],[True,False,False])
        Q = ls.Tensor((self.minbasis.natom,C.shape[1]), 'Q IBO')
        for A, inds2 in enumerate(self.minao_inds):
            Q[A,:] = np.sum(L[:,inds2]**2,1)
        return Q

    def localize(
        self,
        C,
        F, 
        print_level=0,
        ):

        """ Localize orbitals.

        Params:
            C (ls.Tensor of shape (nao, ni)) - Orbitals to localize - these
                must live inside the span of Cref.
            F (ls.Tensor of shape, (ni, ni)) - Fock matrix in orbital basis.
                Used to sort localized orbitals to ascending orbital energies.
            print_level (int) - 0 - no printing, 1 - print iterative history
                and converence info.
        Returns:
            U (ls.Tensor of shape (ni, ni)) - Rotation from original orbitals
                (rows) to localized orbitals (cols). 
            L (ls.Tensor of shape (nao, ni)) - Localized orbital coefficients. 
            F (ls.Tensor of shape (ni, ni)) - Fock matrix in local orbital basis.
            converged (bool) - Did the localization procedure converge (True)
                or not (False).

        Definitions:
            L = C * U
            F2 = L' * F * L

        Localized orbitals are sorted to have ascending orbital energies.
        """
        
        if print_level:
            print('==> IBO Localization <==\n')
            print('IBO Convergence Options:')
            print('  Power         = %11d'   % self.options['power'])
            print('  Maxiter       = %11d'   % self.options['maxiter'])
            print('  G Convergence = %11.3E' % self.options['g_convergence'])
            print('')

        L = ls.Tensor.chain([C,self.S,self.A],[True,False,False])
        U = ls.Tensor((C.shape[1],)*2)
        
        ret = ls.Local.localize(
            self.options['power'],
            self.options['maxiter'],
            self.options['g_convergence'],
            self.minbasis,
            L,
            U,
            )
        converged = ret[-1,1] < self.options['g_convergence']

        if print_level:
            print('IBO %4s: %24s %11s %11s' % ('Iter', 'Metric', 'Delta', 'Gradient')) 
            retnp = ret.np
            Eold = 0.0
            for ind in range(ret.shape[0]):
                E = retnp[ind,0]
                G = retnp[ind,1]
                print('IBO %4d: %24.16E %11.3E %11.3E' % (
                    ind+1,
                    E,
                    E-Eold,
                    G,
                    ))
                Eold = E
            print('')
            if converged:
                print('IBO converged\n')
            else:
                print('IBO failed\n')

        # Energy ordering
        eps = np.diag(ls.Tensor.chain([U,F,U],[True,False,False]))
        U = ls.Tensor.array(U[:,np.argsort(eps)]) 
                        
        # Localized orbitals
        C2 = ls.Tensor.chain([C,U],[False,False])        

        # Localized Fock matrix
        F = ls.Tensor.chain([U,F,U],[True,False,False])

        if print_level:
            print('==> End IBO Localization <==\n')

        return U, C2, F, converged

