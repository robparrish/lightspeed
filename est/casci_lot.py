from . import options
from . import rhf as rhf_mod
from . import casci as casci_mod

class CASCI_LOT(object):

    """ CASCI LOT dresses an RHF/CASCI level of theory up to act as
        standardized level of theory.

    """

    @staticmethod
    def default_options():
        if hasattr(CASCI_LOT, '_default_options'): return CASCI_LOT._default_options.copy()
        opt = options.Options() 

        # > Print Control < #

        opt.add_option(
            key='print_level',
            value=1,
            allowed_types=[int],
            doc='Level of detail to print (0 - nothing, 1 - minimal [dynamics], 2 - standard [SP]). This overrides print_level in underling RHF/CASCI objects.')
    
        # > Problem Geometry < #

        opt.add_option(
            key='casci',
            required=True,
            allowed_types=[casci_mod.CASCI],
            doc='CASCI Object (must be based on reference RHF object)')

        # > Convergence/Guess Keys < #

        opt.add_option(
            key='rhf_mom',
            value=False,
            required=True,
            allowed_types=[bool],
            doc='Use MOM to force orbital coincidence in RHF?')
        opt.add_option(
            key='rhf_guess',
            value=False,
            required=True,
            allowed_types=[bool],
            doc='Use projected converged density matrix as guess in RHF?')

        # > Special Keys for Energy/Gradients/Overlaps < #
        
        opt.add_option(
            key='doETF',
            value=False,
            required=True,
            allowed_types=[bool],
            doc='Use Joe Subotnik ETF treatment in CASCI.compute_coupling?')
        opt.add_option(
            key='orbital_coincidence',
            value='none',
            required=True,
            allowed_types=[str],
            doc='Enforce orbital coincidence in CASCI.compute_overlap?')
        opt.add_option(
            key='state_coincidence',
            value='none',
            required=True,
            allowed_types=[str],
            doc='Enforce state coincidence in CASCI.compute_overlap?')
            
        CASCI_LOT._default_options = opt
        return CASCI_LOT._default_options.copy()

    def __init__(
        self,
        options,
        ):

        """ Constructor """
        self.options = options
        if not self.casci.options['reference']: raise RuntimeError("casci reference must exist")

    @staticmethod
    def from_options(**kwargs):
        """ Return an instance of this class with default options updated from values in kwargs. """
        return CASCI_LOT(CASCI_LOT.default_options().set_values(kwargs))

    def __str__(self):
        s = 'CASCI_LOT:\n'
        for key in ['rhf_mom', 'rhf_guess', 'doETF', 'orbital_coincidence', 'state_coincidence']:
            s += '  %-20s = %r\n' % (key, self.options[key])
        return s

    @property
    def xyz(self):
        return self.casci.geometry.molecule.xyz
    
    @property
    def casci(self):
        """ The underlying CASCI object, for more-specialized access. """
        return self.options['casci']

    @property
    def geometry(self):
        return self.casci.geometry

    @property
    def resources(self):
        return self.casci.geometry.resources

    @property
    def molecule(self):
        return self.casci.geometry.molecule
    
    @property
    def qm_molecule(self):
        return self.casci.geometry.qm_molecule

    def update_xyz(
        self,
        xyz,
        guess=True, # Set to False to override guess/MOM options
        ):

        """ Return a new CASCI_LOT with updated geometry. """

        geometry = self.casci.geometry.update_xyz(xyz)

        reference = rhf_mod.RHF(self.casci.reference.options.copy().set_values({
            'geometry' : geometry,
            'print_level' : self.options['print_level'],
            }))
        reference.initialize()
        # Guess from old RHF if desired
        if guess and (self.options['rhf_guess'] or self.options['rhf_mom']):
            Dguess, Cocc_mom, Cact_mom = rhf_mod.RHF.guess_density(self.casci.reference, reference)
        # Converge new RHF
        reference.compute_energy(
            Dguess=Dguess if guess and self.options['rhf_guess'] else None,
            Cocc_mom=Cocc_mom if guess and self.options['rhf_mom'] else None,
            Cact_mom=Cact_mom if guess and self.options['rhf_mom'] else None,
            )
        # TODO: rescue convergence
        casci = casci_mod.CASCI(self.casci.options.copy().set_values({
            'reference' : reference,
            'print_level' : self.options['print_level'],
            }))
        casci.compute_energy() 

        return CASCI_LOT(self.options.copy().set_values({
            'casci' : casci,
            }))

    def compute_energy(
        self,
        S,
        index,
        ):

        return self.casci.evals[S][index]

    def compute_gradient(
        self,
        S,
        index,
        ):

        return self.casci.compute_gradient(
            S=S, 
            index=index)

    def compute_coupling(
        self,
        S,
        indexA,
        indexB,
        ):

        return self.casci.compute_coupling(
            S=S, 
            indexA=indexA, 
            indexB=indexB, 
            doETF=self.options['doETF'],
            )

    def compute_overlap(
        self,
        lotA,
        lotB,
        S,
        indsA=None,
        indsB=None, 
        ):

        return self.casci.compute_overlap(
            casciA=lotA.casci,
            casciB=lotB.casci,
            S=S,
            indsA=indsA,
            indsB=indsB,
            orbital_coincidence=self.options['orbital_coincidence'],
            state_coincidence=self.options['state_coincidence'],
            )
    
    def compute_energies(
        self,
        ):

        """ A dictionary of spin index S -> list of E for all energies available in this CASCI """
        return self.casci.evals

    def compute_dipoles(
        self,
        ):
        return self.casci.dipoles

    def compute_opdm_ao(
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

        return self.casci.opdm_ao(S, indexA, indexB, spin_sum)

        

    
