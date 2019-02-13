import est
    
class XYZReporter(object):
    
    @staticmethod
    def default_options():
        if hasattr(XYZReporter, '_default_options'): return XYZReporter._default_options.copy()
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
            # allowed_types=[str, file],
            allowed_types=[str],
            doc='Filename or file handle')
        opt.add_option(
            key='mode',
            value='w',
            required=True,
            allowed_types=[str],
            doc='Mode to use in open if filename provided.')
        opt.add_option(
            key='molecule',
            value='molecule',
            required=True,
            allowed_values=['molecule', 'qm_molecule'],
            doc='Which molecular geometry to save?')
        opt.add_option(
            key='label',
            value='label',
            required=True,
            allowed_values=['label', 'symbol'],
            doc='Mode of atomic label/symbol to use.')
        opt.add_option(
            key='label_str',
            value='%-12s',
            required=True,
            allowed_types=[str],
            doc='Format string for atom symbol')
        opt.add_option(
            key='coord_str',
            value='%18.12f',
            required=True,
            allowed_types=[str],
            doc='Format string for coordinates')
    
        XYZReporter._default_options = opt
        return XYZReporter._default_options.copy()

    def __init__(
        self,
        options,
        ):

        """ Constructor """
        self.options = options

    @staticmethod
    def from_options(**kwargs):
        """ Return an instance of this class with default options updated from values in kwargs. """
        return XYZReporter(XYZReporter.default_options().set_values(kwargs))

    def initialize(
        self,
        ):

        # TODO: check if filename is an existing file handle
        self._fh = open(self.options['filename'], self.options['mode'])

    def report(
        self,
        aimd,
        ): 

        if aimd.step % self.options['interval'] != 0: return

        molecule = aimd.lot.molecule if self.options['molecule'] == 'molecule' else aimd.lot.qm_molecule
        molecule.save_xyz_file(
            self._fh,
            label=self.options['label'],    
            comment='Step: %11d t: %24.16E V: %24.16E S: %1d index: %2d' % (
                aimd.step,
                aimd.t,
                aimd.V,
                aimd.S,
                aimd.index,
                ),
            label_str=self.options['label_str'],
            coord_str=self.options['coord_str'],
            )
            

    def finalize(self):
        self._fh.close()
