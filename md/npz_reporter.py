import est
import numpy as np
    
class NPZReporter(object):

    """ Class NPZReporter reports a .npz archive of a number of scalar
        quantities obtained along an AIMD trajectory. These include:
            step - MD step
            t - time in atomic units
            V - potential energy in atomic units
            K - kinetic energy in atomic units
            E - total energy in atomic units
            T - temperature in atomic units
            S - spin index
            index - adiabatic state index
            O - overlap between current/previous step electronic wavefunctions.
    """
    
    @staticmethod
    def default_options():
        if hasattr(NPZReporter, '_default_options'): return NPZReporter._default_options.copy()
        opt = est.Options() 

        opt.add_option(
            key='interval',
            value=1,
            required=True,
            allowed_types=[int],
            doc='Interval between reporting events')
        opt.add_option(
            key='filename',
            required=True,
            allowed_types=[str],
            doc='Filename (should end in .npz)')
        opt.add_option(
            key='state_energies',
            value=False,
            required=True,
            allowed_types=[bool],
            doc='Record state energies?',
            )
    
        NPZReporter._default_options = opt
        return NPZReporter._default_options.copy()

    def __init__(
        self,
        options,
        ):

        """ Constructor """
        self.options = options

    @staticmethod
    def from_options(**kwargs):
        """ Return an instance of this class with default options updated from values in kwargs. """
        return NPZReporter(NPZReporter.default_options().set_values(kwargs))

    def initialize(self):
        """ Initialize reporting. """
        self.arrays = {
            'step' : [],
            't' : [],
            'V' : [],
            'K' : [],
            'E' : [],
            'T' : [],
            'S' : [],
            'index' : [],
            'O' : [],
        }

    def report(
        self,
        aimd,
        ): 

        if aimd.step % self.options['interval'] != 0: return

        self.arrays['step'].append(aimd.step)
        self.arrays['t'].append(aimd.t)
        self.arrays['V'].append(aimd.V)
        self.arrays['K'].append(aimd.K) 
        self.arrays['E'].append(aimd.E) 
        self.arrays['T'].append(aimd.T) 
        self.arrays['S'].append(aimd.S) 
        self.arrays['index'].append(aimd.index) 
        self.arrays['O'].append(aimd.O) 

        if self.options['state_energies']:
            Es = aimd.lot.compute_energies()
            for S, E2s in Es.items():
                key='E-%d' % S
                self.arrays.setdefault(key,[]).append(E2s)

    def finalize(self):
        """ Finalize - write the npz file """
        np.savez(
            self.options['filename'],
            **{ k : np.array(v) for k,v in self.arrays.items() }
            )
        


            
