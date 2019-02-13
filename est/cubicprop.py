import lightspeed as ls
from . import options

class CubicProp(object):

    @staticmethod
    def default_options():
        if hasattr(CubicProp, '_default_options'): return CubicProp._default_options.copy()
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
            key="pairlist",
            required=True,
            allowed_types=[ls.PairList],
            doc="PairList corresponding to the primary AO basis")
        opt.add_option(
            key="grid",
            required=True,
            allowed_types=[ls.CubicGrid],
            doc="Cubic grid for problem")
    
        # > Numerical Thresholds < #

        opt.add_option(
            key='thre_dens',
            value=1.0E-14,
            required=True,
            allowed_types=[float],
            doc='Cutoff threshold for density collocation')
        opt.add_option(
            key='thre_orbs',
            value=1.0E-7,
            required=True,
            allowed_types=[float],
            doc='Cutoff threshold for orbital collocation')
        opt.add_option(
            key='grid_hash_R',
            value=5.0,
            required=True,
            allowed_types=[float],
            doc='Box size for HashedGrid')

        CubicProp._default_options = opt
        return CubicProp._default_options.copy()

    def __init__(
        self,
        options,
        ):
        
        self.options = options

        # HashedGrid
        self.hashed_grid = ls.HashedGrid(
            self.grid.xyz,
            self.options['grid_hash_R'])

    @staticmethod
    def from_options(**kwargs):
        """ Return an instance of this class with default options updated from values in kwargs. """
        return CubicProp(CubicProp.default_options().set_values(kwargs))

    @staticmethod
    def build(
        molecule,
        grid=None,
        grid_overage=(5.0, 5.0, 5.0),
        grid_spacing=(0.4, 0.4, 0.4),
        **kwargs):

        """ Build a CubicProp, including determination of CubicGrid size

        Params:
            molecule (ls.Molecule) - the QM molecule
            grid (ls.CubicGrid or None) - the explicit cubic grid to build, or
                None (1st priority)
            grid_overage (tuple of len 3) - the overage beyond the QM molecule
                in au
            grid_spacing (tuple of len 3) - the spacing between grid points in
                au
            kwargs - other options, passed to from_options
        Returns:
            The completed CubicProp, with CubicGrid constructed from
                overage/spacing if needed.
        """
        
        if not grid:
            grid = ls.CubicGrid.build_next_cube(
                molecule.xyz,
                grid_overage,
                grid_spacing,
                )
        return CubicProp.from_options(
            molecule=molecule,
            grid=grid,
            **kwargs)
        
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
    def pairlist(self):
        return self.options['pairlist']

    @property
    def grid(self):
        return self.options['grid']

    def save_density_cube(
        self,
        filename,
        D,
        propertyname='density',
        ):

        """ Save a .cube file with the density collocated from density matrix D

        Params:
            filename (str) - the cube file name, including .cube extension
            D (ls.Tensor of shape (nao, nao)) - the total density matrix
            propertyname (str) - the name of the property in the comment field
                of the cube file.
        Result:
            A cube file with the collocated density is written to filename
        """
    
        # Density Collocation
        rho = ls.GridBox.ldaDensity(
            self.resources,
            self.pairlist,
            D,
            self.hashed_grid,
            self.options['thre_dens'],
            ) 
        # Save Cube File
        self.grid.save_cube_file(
            filename,
            propertyname,
            self.molecule,
            rho,
            )

    def save_esp_cube(
        self,
        filename,
        D,
        xyzZ=None,
        propertyname='esp',
        ):

        """ Save a .cube file with the electrostatic potential generated from
            density matrix D and point charges xyzZ

        Params:
            filename (str) - the cube file name, including .cube extension
            D (ls.Tensor of shape (nao, nao)) - the total density matrix
            xyzZ (ls.Tensor of shape (npoint, 4)) - the x, y, z, and Z charges of
                any desired point charges. Ignored if None.
            propertyname (str) - the name of the property in the comment field
                of the cube file.
        Result:
            A cube file with the collocated ESP is written to filename
        """
        # Electronic ESP (-)
        w = ls.IntBox.esp(
            self.resources,
            ls.Ewald.coulomb(),
            self.pairlist,
            D,
            self.grid.xyz,
            )
        w.scale(-1.0)
        # Nuclear ESP (+)
        if xyzZ:
            w = ls.IntBox.chargePotential(
                self.resources,
                ls.Ewald.coulomb(),
                self.grid.xyz,
                xyzZ,
                w,
                ) 
        # Save Cube File
        self.grid.save_cube_file(
            filename,
            propertyname,
            self.molecule,
            w,
            )

    def save_orbital_cubes(
        self,
        filepath,
        C,
        propertyname='orbital',
        ):

        """ Save a series of cube files for orbitals collocated from orbital
            coefficients C.

        Params:
            filepath (str) - the root of the orbital cube file names, without
                orbital index or .cube extension.
            C (ls.Tensor of shape (nao, norb)) - the orbital coefficients
            propertyname (str) - the name of the property in the comment field
                of the cube file.
        Result:
            A cube file for each orbital is written to '%s_%d.cube' %
                (filepath, index) where index is the 0-based orbital index.
        """
         
        # Orbital collocation
        psi = ls.GridBox.orbitals(
            self.resources,
            self.basis,
            self.grid.xyz,
            C,
            self.options['thre_orbs'],
            ) 
        # Save Cube Files
        for i in range(psi.shape[1]):
            self.grid.save_cube_file(
                '%s_%d.cube' % (filepath, i),
                '%s_%d' % (propertyname, i),
                self.molecule,
                ls.Tensor.array(psi[:,i]),
                )
            
        
