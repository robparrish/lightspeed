import numpy as np
import lightspeed as ls # for Tensor
import est # for Options

# => VV Propagator <= #

class VV(object):

    """ class VV implements Velocity Verlet with Langevin damping.
    
    VV is used to implement dynamics in a model where the potential/force
    evaluation (referred to herein as 'compute_potential') and the time propagation
    are on the same level, and are called in an external loop.
    
    An example usage of VV dynamics is:
    
        # => Initialization <= #
    
        nstep # Pre-determined total simulation nstep
        dt    # Pre-determined timestep
        M     # Pre-determined masses
        X     # Pre-determined initial positions
        P     # Pre-determined initial momenta
        V, F = compute_potential(X)  # User-provided compute_potential
    
        prop = VV.from_options(
            dt=dt,
            )
        prop.initialize(
            M=M,
            X=X,
            P=P,
            F=F,
            V=V,
            )
    
        # => Dynamics Loop <= #
    
        for I in xrange(nstep):
    
            # Compute potential/force at prop.Xnew
            V, F = prop.compute_potential(prop.Xnew)
            # Advance VV time
            prop.step(V,F)
    
    VV also implements a Langevin thermostat. The algorithm is discussed in:
    
    E. Vanden-Eijnden and G. Ciccotti, Second-order integrators for Langevin equations
    with holonomic constraints, Chem. Phys. Lett. 429, 310316 (2006).
    
    To provide reproducibility between runs, the VV thermostat uses a
    np.RandomState object which can be initialized with a user-specified
    'random_seed' option (an int between 0 and 2^32-1 - see np documentation for
    more details). If no 'random_seed' option is provided, no seed is given to
    np.RandomState, and the default np.RandomState behavior is obtained (defaults
    to the OS urandom value for the seed).
    """

    @staticmethod
    def default_options():
        if hasattr(VV, '_default_options'): return VV._default_options.copy()
        opt = est.Options() 

        # > Problem Geometry < #

        opt.add_option(
            key='dt',
            required=True,
            allowed_types=[float],
            doc='Timestep in atomic units')
        opt.add_option(
            key='Ltemp',
            value=0.0,
            allowed_types=[float],
            doc='Langevin thermostat temperature')
        opt.add_option(
            key='Lgamma',
            value=0.0,
            allowed_types=[float],
            doc='Langevin thermostat gamma parameter')
        opt.add_option(
            key='random_seed',
            allowed_types=[int],
            doc='Random number seed for numpy.RandomState')

        VV._default_options = opt
        return VV._default_options.copy()

    def __init__(
        self,
        options,
        ):

        """ Constructor, also sets up random_state """

        self.options = options
        self.random_state = np.random.RandomState(self.options['random_seed'])

    @staticmethod
    def from_options(**kwargs):
        """ Return an instance of this class with default options updated from values in kwargs. """
        return VV(VV.default_options().set_values(kwargs))

    def __str__(self):
        s = 'VV:\n'
        s += '  dt          = %18.12f\n' % self.dt
        s += '  Ltemp       = %18.12f\n' % self.Ltemp
        s += '  Lgamma      = %18.12f\n' % self.Lgamma
        s += '  random seed = %18r\n' % self.options['random_seed']
        return s

    def initialize(
        self,
        M,
        X,
        P,
        F,
        V,
        ):

        """ Initialize the VV object. This must be called before calling step.

        Params:
            M (ls.Tensor) - masses
            X (ls.Tensor) - initial positions 
            P (ls.Tensor) - initial momenta
            F (ls.Tensor) - initial forces 
            V (float) - initial potential energy

        Object Attributes:
            dt (float) - timestep
            M (ls.Tensor) - masses
            X (ls.Tensor) - current position
            P (ls.Tensor) - current momenta
            F (ls.Tensor) - current force
            V (float) - current potential energy
            K (float) - current kinetic energy
            E (float) - current total energy
            Xnew (ls.Tensor) - proposed new position - force and potential should
                be computed here and passed to step to move forward in time.

        The ls.Tensor objects in VV can be of arbitrary shape (e.g., arbitrary
            problem dimensionality), but must all be the same shape.

        The input M/X/P/F arrays are deep-copied - modifications to the
        originals will not affect VV.
        """

        self.M = ls.Tensor.array(M)
        self.X = ls.Tensor.array(X)
        self.P = ls.Tensor.array(P)
        self.F = ls.Tensor.array(F)
        self.V = float(V)

        self.Lzeta = ls.Tensor.array(self.random_state.randn(*self.X.shape))
        self.Ltheta = ls.Tensor.array(self.random_state.randn(*self.X.shape))
        self.LC = ls.Tensor.array(0.5 * (self.F[...] / self.M[...] - self.Lgamma * self.P[...] / self.M[...]) * self.dt**2 \
            + self.Lsigma[...] * self.dt**(1.5) * (0.5 * self.Lzeta[...] + 0.5 / np.sqrt(3.0) * self.Ltheta[...]))

        # Positions for next force
        self.Xnew = ls.Tensor.array(self.X[...] + (self.P[...] / self.M[...]) * self.dt + self.LC[...])

    def step(
        self,
        V, 
        F, 
        ):

        """ Step forward in time using the VV propagator.

        Params:
            V (float) - the potential energy computed at self.Xnew
            F (ls.Tensor) - the force computed at self.Xnew
        
        The function changes the internal state of the VV object to advance one frame in time.

        The input F array is deep-copied - modifications to the original will not affect VV.
        """
        
        # Update current frame
        self.X = self.Xnew
        self.P = ls.Tensor.array(self.P[...] + 0.5 * (F[...] + self.F[...]) * self.dt \
            - self.dt * self.Lgamma * (self.P[...] / self.M[...]) \
            + np.sqrt(self.dt) * self.Lsigma[...] * self.Lzeta[...] \
            - self.Lgamma * self.LC[...])
        #self.P = self.P + self.F * self.dt  # 1st-Order Euler
        self.F = ls.Tensor.array(F)
        self.V = float(V)

        self.Lzeta = ls.Tensor.array(self.random_state.randn(*self.X.shape))
        self.Ltheta = ls.Tensor.array(self.random_state.randn(*self.X.shape))
        self.LC = ls.Tensor.array(0.5 * (self.F[...] / self.M[...] - self.Lgamma * self.P[...] / self.M[...]) * self.dt**2 \
            + self.Lsigma[...] * self.dt**(1.5) * (0.5 * self.Lzeta[...] + 0.5 / np.sqrt(3.0) * self.Ltheta[...]))

        # Positions for next force
        self.Xnew = ls.Tensor.array(self.X[...] + (self.P[...] / self.M[...]) * self.dt + self.LC[...])

    @property
    def dt(self):
        """ The timestep """
        return self.options['dt']

    @property
    def Ltemp(self):
        """" The Langevin thermostat temperature """
        return self.options['Ltemp']

    @property
    def Lgamma(self):
        """ The Langevin coupling gamma parameter """
        return self.options['Lgamma']

    @property
    def Lsigma(self):
        """ The Langevin sigma vector """
        return ls.Tensor.array(np.sqrt(2.0 * self.Lgamma * self.Ltemp / self.M[...]))

    @property
    def DOF(self):
        """ The number of degrees of freedom """
        return self.X.size
     
    @property
    def T(self):
        """ The instantaneous temperature """
        return 2.0 * self.K / self.DOF

    @property
    def K(self):
        """ The kinetic energy """
        return 0.5 * np.sum(self.P[...]**2 / self.M[...])

    @property
    def E(self):
        """ The total energy """
        return self.K + self.V
    
