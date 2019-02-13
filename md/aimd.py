import lightspeed as ls
import est
import numpy as np

class AIMD(object):

    @staticmethod
    def default_options():
        if hasattr(AIMD, '_default_options'): return AIMD._default_options.copy()
        opt = est.Options() 

        # > Print Control < #

        opt.add_option(
            key='print_level',
            value=1,
            allowed_types=[int],
            doc='Level of detail to print (0 - nothing, 1 - minimal [dynamics], 2 - standard [SP])')

        # > Problem Geometry < #

        opt.add_option(
            key='lot',
            required=True,
            doc='LOT object')
        opt.add_option(
            key='integrator',
            required=True,
            doc='Integrator object')
        opt.add_option(
            key='target_S',
            required=True,
            allowed_types=[int], 
            doc='Target electronic state spin number [0 - singlet, 1 - doublet, etc]')
        opt.add_option(
            key='target_index',
            required=True,
            allowed_types=[int], 
            doc='Target electronic state [0 - based within each spin block]')
        opt.add_option(
            key='masses',
            required=True,
            allowed_types=[ls.Tensor],
            doc="Atomic masses in atomic units")
        opt.add_option(
            key='momenta',
            required=True,
            allowed_types=[ls.Tensor],
            doc="Initial momenta in atomic units")

        # > AIMD Recipes < #

        opt.add_option(
            key='reporters',
            value=[],
            required=True,
            doc='List of Reporter objects')
        opt.add_option(
            key='state_tracking',
            value='adiabatic',
            required=True,
            allowed_values=['adiabatic', 'diabatic'],
            doc='State tracking mode')

        AIMD._default_options = opt
        return AIMD._default_options.copy()

    def __init__(
        self,
        options,
        ):
        
        self.options = options

    @staticmethod
    def from_options(**kwargs):
        """ Return an instance of this class with default options updated from values in kwargs. """
        return AIMD(AIMD.default_options().set_values(kwargs))

    @property
    def print_level(self): 
        return self.options['print_level']

    @property
    def M(self):
        """ Masses in atomic units """
        return self.integrator.M

    @property
    def X(self):
        """ Positions in atomic units """
        return self.integrator.X

    @property
    def P(self):
        """ Momenta in atomic units """
        return self.integrator.P

    @property
    def F(self):
        """ Forces in atomic units """
        return self.integrator.F

    @property
    def V(self):
        """ Potential energy in atomic units """
        return self.integrator.V

    @property
    def K(self):
        """ Kinetic energy in atomic units """
        return self.integrator.K

    @property
    def E(self):
        """ Total energy in atomic units """
        return self.integrator.E

    @property
    def T(self):
        """ Instantaneous temperature in atomic units """
        return self.integrator.T

    @property
    def dt(self):
        """ Timestep in atomic units """
        return self.integrator.dt

    @property
    def md_header(self):
        """ A header string for md_line """
        return '@AIMD: %6s %24s %24s %24s %24s %24s' % (
            'Step',
            't',
            'V',
            'K',
            'E',
            'T',
            )

    @property
    def md_line(self):
        """ A string summary of the current step """
        return '@AIMD: %6d %24.16E %24.16E %24.16E %24.16E %24.16E' % (
            self.step,
            self.t,
            self.V,
            self.K,
            self.E,
            self.T,
            )

    @property
    def reporters(self):
        return self.options['reporters']
        
    def initialize(
        self,
        ):

        """ Initialize the AIMD object. """

        if self.print_level:
            print('==> AIMD <==\n')

        # Set integrator/lot/old_lot properties
        self.integrator = self.options['integrator']
        self.lot = self.options['lot']
        self.old_lot = self.lot

        # Print composition
        if self.print_level:
            print(self.integrator)
            print(self.lot)

        # Initialize step and time
        self.step = 0
        self.t = 0.0
        
        # Initial adiabatic target adiabatic state
        self.S = self.options['target_S'] 
        self.index = self.options['target_index']
        # Inital overlap between current/previous wavefunctions
        self.O = 1.0

        # Print state tracking details
        if self.print_level:
            print('State Tracking:')
            print('  Initial S     = %d' % self.S)
            print('  Initial index = %d' % self.index)
            print('  State tracking = %s' % self.options['state_tracking'])
            print('')

        # Initial state data
        V = self.lot.compute_energy(S=self.S, index=self.index)
        F = self.lot.compute_gradient(S=self.S, index=self.index)
        F[...] *= -1.0

        # Initialize integrator
        self.integrator.initialize(
            M=self.options['masses'],
            X=self.lot.xyz, 
            P=self.options['momenta'],
            F=F,
            V=V,
            )

        # Print MD line
        if self.print_level:
            print(self.md_header)
        if self.print_level:
            print(self.md_line)

        # Initialization and inital report for reporters
        for reporter in self.reporters:
            reporter.initialize()
            reporter.report(self)

    def run_step(self):

        """ Run AIMD for one step
    
        Result:
            the state of the system is updated by running dynamics. Reporter
            objects are called every step to report the details of the dynamics.
        """

        # Update the LOT geometry
        self.old_lot = self.lot
        self.lot = self.lot.update_xyz(self.integrator.Xnew)

        # Compute state overlap for metrics/state tracking
        O = self.lot.compute_overlap(
            self.old_lot,
            self.lot,
            self.S,
            )
        # Switch states as needed in diabatic state tracking
        indexA = self.index
        indexB = np.argmax(np.abs(O[self.index,:]))
        if indexA != indexB and self.options['state_tracking'] == 'diabatic':
            self.index = indexB
            if self.print_level:
                print('@AIMD: NOTE: State tracking switched from old state %d to new state %d' % (
                    indexA,
                    indexB,
                    ))
        # Else warn the user if adiabatic state tracking
        elif indexA != indexB and self.print_level:
            print('@AIMD: WARNING: Old state %1d overlaps maximally with new state %1d: %6.4f' % (
                indexA,
                indexB,
                np.abs(O[indexA,indexB]),
                ))
        # Either way, save and print the overlap metric
        self.O = np.abs(np.abs(O[indexA, self.index]))
        if self.print_level:
            print('@AIMD: State Overlap: %6.4f' % (self.O))
        
        # Compute the potential/force at the new geometry
        V = self.lot.compute_energy(S=self.S, index=self.index)
        F = self.lot.compute_gradient(S=self.S, index=self.index)
        F[...] *= -1.0

        # Integrate
        self.integrator.step(
            V=V,
            F=F,
            )  

        # Update steps
        self.step += 1
        self.t += self.dt

        # Print MD line
        if self.print_level:
            print(self.md_line)

        # Initialization and inital report for reporters
        for reporter in self.reporters:
            reporter.report(self)

    def run(
        self,
        nstep=0,
        ):

        """ Run AIMD for a number of steps.
    
        Params:
            nstep (int) - the number of steps to run.
        Result:
            the state of the system is updated by running dynamics. Reporter
            objects are called every step to report the details of the dynamics.
        """

        for step in range(nstep):
            self.run_step()

    def finalize(self):

        for reporter in self.reporters:
            reporter.finalize()

        # => Trailer Bars <= #

        if self.print_level:
            print('"We do not have time for your damned hobby, sir!"')
            print('    --Capt. Jack Aubrey\n')
            
        if self.print_level:
            print('==> End AIMD <==\n')

