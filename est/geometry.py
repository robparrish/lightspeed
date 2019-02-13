import lightspeed as ls
from . import options
from . import qmmm
from . import dftd3
    
class Geometry(object):

    """ Class Geometry represents the geometry of an electronic structure
        problem in vacuum boundary conditions. Provision is made for simple
        gas-phase EST, QM/MM, and possible further developments (e.g.,
        pseudopotentials and PCM).

        Geometry principally represents the structure of the external
        environment (nuclear point charges, QM/MM, etc) and the various basis
        sets involved in the EST problem. 

    """
    @staticmethod
    def default_options():
        """ Return a new copy of the declared/default Options for Geometry. """
        if hasattr(Geometry, "_default_options"): return Geometry._default_options.copy()
        opt = options.Options()
        opt.add_option(
            key='resources',
            required=True,
            allowed_types=[ls.ResourceList],
            doc='ResourceList to use for this EST problem')
        opt.add_option(
            key="molecule",
            required=False, 
            allowed_types=[ls.Molecule],
            doc="Molecule for standard EST computations (e.g., no QM/MM)")
        opt.add_option(
            key='ewald',
            value=ls.Ewald.coulomb(),
            required=True,
            allowed_types=[ls.Ewald],
            doc='Ewald operator to use in J and K builds.')
        opt.add_option(
            key="qmmm", 
            required=False, 
            allowed_types=[qmmm.QMMM], 
            doc="QM/MM object, if QM/MM is to be used")
        opt.add_option(
            key="dftd3",
            required=False,
            allowed_types=[dftd3.DFTD3Base],
            doc="DFTD3 Empirical Dispersion Correction")
        opt.add_option(
            key="basis",
            required=True,
            allowed_types=[ls.Basis],
            doc="Primary AO basis set for QM molecule")
        opt.add_option(
            key="minbasis",
            required=True,
            allowed_types=[ls.Basis],
            doc="Minimal AO basis set for QM molecule (for SAD guess and localization)")
        opt.add_option(
            key="pairlist",
            required=True,
            allowed_types=[ls.PairList],
            doc="PairList corresponding to the primary AO basis")
        opt.add_option(
            key="ecp",
            required=False,
            allowed_types=[ls.ECPBasis],
            doc="ECPBasis defining ECPs for QM molecule.")
        opt.add_option(
            key="threecp",
            value=1.0E-14,
            required=True,
            allowed_types=[float],
            doc="Threshold for ECP integrals")
        Geometry._default_options = opt
        return Geometry._default_options.copy()

    def __init__(
        self,
        options,
        ):

        """ Verbatim constructor """
        self.options = options

        # Validity checks
        if self.options['molecule'] and self.options['qmmm']:
            raise RuntimeError('Cannot set both molecule and qmmm')

    @staticmethod
    def from_options(**kwargs):
        """ Return an instance of this class with default options updated from values in kwargs. """
        return Geometry(Geometry.default_options().set_values(kwargs))

    @property
    def resources(self):
        """ The ls.ResourceList for this EST problem """
        return self.options['resources']

    @property
    def molecule(self):
        """ The ls.Molecule representing the full molecular geometry """
        if self.qmmm: return self.qmmm.molecule
        else: return self.options['molecule']

    @property
    def qm_molecule(self):
        """ The ls.Molecule representing the QM molecular geometry """
        if self.qmmm: return self.qmmm.qm_molecule
        else: return self.options['molecule']

    @property
    def qm_molecule_ecp(self):
        """ The ls.Molecule representing the QM molecular geometry with ECP
            charge changes accounted for (Z fields updated). This field should
            be used in place of qm_molecule for all calls involving self or
            cross-electrostatic interactions between nuclei of the qm molecule
            and other charges."""
        if not self.ecp: return self.qm_molecule
        else: return self.qm_molecule.update_Z(ls.Tensor.array(self.qm_molecule.Z[...] - ls.Tensor.array(self.ecp.nelecs)))

    @property
    def basis(self):
        """ The primary ls.Basis for the QM molecular geometry """
        return self.options['basis']

    @property
    def minbasis(self):
        """ The MinAO ls.Basis for the QM molecular geometry """
        return self.options['minbasis']

    @property
    def ecp(self):
        """ The ls.ECPBasis for the QM molecular geometry or None if no ECPs """
        return self.options['ecp']

    @property
    def pairlist(self):
        """ The ls.PairList object for the primary basis """
        return self.options['pairlist']

    @property
    def ewald(self):
        """ The ls.Ewald object for the Ewald operator to use """
        return self.options['ewald']

    @property
    def qmmm(self):
        """ The QMMM object or None if no QM/MM """
        return self.options['qmmm']

    @property
    def dftd3(self):
        """ The DFTD3 object or None if no DFTD3 """
        return self.options['dftd3']

    def __str__(self):
        """ A String representation summarizing the Geometry """
        s = 'Geometry:\n'
        s += '  QMMM = %r\n' % (True if self.qmmm else False) 
        s += '  -D3  = %r\n' % (True if self.dftd3 else False) 
        s += '  ECP  = %r\n' % (True if self.ecp else False) 
        s += '\n'
        if self.qmmm:
            s += 'QM/MM Details:\n'
            s += '  Total atoms   = %d\n' % self.qmmm.molecule.natom
            s += '  QM atoms      = %d\n' % self.qmmm.qm_molecule.natom
            s += '  Link H atoms  = %d\n' % len(self.qmmm.options['link_inds'])
            s += '\n'
        if self.qmmm:
            s += str(self.molecule)
            s += '\n'
        s += str(self.qm_molecule)
        s += '\n'
        s += str(self.basis)
        s += '\n'
        s += str(self.minbasis)
        if self.ecp:
            s += '\n'
            s += str(self.ecp)
        if self.dftd3:
            s += '\n'
            s += str(self.dftd3)
        return s 

    # => Energy/gradient methods <= #

    def compute_nuclear_repulsion_energy(self):
        """ Return the nuclear repulsion energy of the QM molecule, often used
            as a hash for the molecular geometry. 
        """
        return ls.IntBox.chargeEnergySelf(
                    self.resources,
                    self.ewald,
                    self.qm_molecule_ecp.xyzZ,
                    )

    def compute_external_energy(self):
        """ Return the complete external energy of the Geometry, including the
            nuclear repulsion energy of the QM molecule, and any self energy
            terms involving other structures in the Geometry. """
        return sum(self.compute_detailed_external_energy().values()) 

    def compute_detailed_external_energy(self):
        """ Return a detailed map of external energy terms """
        Es = {}
        if self.qmmm:
            Es['Emm'] = self.qmmm.mm_energy 
            Es['Emm_nuc'] = self.qmmm.compute_mm_nuclear_energy(
                self.resources,
                self.ewald, # NOTE: could replace with smeared charges
                self.qm_molecule_ecp,
                )
        if self.dftd3:
            Es['Edftd3'] = self.dftd3.compute_energy([atom.N for atom in self.qm_molecule.atoms], self.qm_molecule.xyz)
        Es['Enuc'] = self.compute_nuclear_repulsion_energy()
        return Es

    def compute_external_potential(
        self, 
        pairlist,
        ):
        """ Return the complete external potential of this Geometry, including the
            nuclear attraction potential of the QM molecule, and any external
            potential terms involving other structures in the Geometry. """
        V = ls.Tensor((pairlist.basis1.nao, pairlist.basis2.nao))
        ls.IntBox.potential(
            self.resources,
            self.ewald,
            pairlist,
            self.qm_molecule_ecp.xyzZ,
            V)
        # QM/MM
        if self.qmmm:
            V[...] += self.qmmm.compute_mm_potential(
                self.resources,
                self.ewald,
                pairlist)
        # ECP
        if self.ecp:
            V[...] += ls.IntBox.ecp(
                self.resources,
                pairlist,
                self.ecp,
                self.options['threecp'],
                )
        return V

    def compute_gradient(
        self,
        pairlist,
        Dtot,
        qm_grad,
        include_self=True,
        ):
    
        """ Compute/Assemble the gradient of the full molecular system. This
        includes the self energy gradient of the Geometry (including QM nuclear
        repulsion gradient and any other self energy terms) and the interaction
        contributions between the Geometry and the electrons (including QM
        nuclear-electron attraction and any other interaction terms)

        Params:
            pairlist (ls.PairList) - the PairList on which the electronic density is built
            Dtot (ls.Tensor) - the total OPDM of the QM electrons
            qm_grad (ls.Tensor) - the electronic part of the gradient for the
                QM system (including overlap, kinetic, and ERI terms). This
                should *not* include any nuclear-electron, nuclear-nuclear, or
                other external gradient computations.
            include_self (bool) - include self-energy terms in gradient (this
                is False for coupling computations)
        Returns:
            grad (ls.Tensor) - the total gradient of the Geometry

        """

        # Total QM gradient
        Gqm_tot = ls.Tensor.array(qm_grad)
        # Nuclear repulsion self energy gradient (QM Molecule)
        if include_self:
            Gself = ls.IntBox.chargeGradSelf(
                self.resources,
                self.ewald,
                self.qm_molecule_ecp.xyzZ)
            Gqm_tot[...] += Gself
        # -D3 energy gradient (QM Molecule)
        if self.dftd3 and include_self:
            Gdftd3 = self.dftd3.compute_gradient(
                [atom.N for atom in self.qm_molecule.atoms],
                self.qm_molecule.xyz)
            Gqm_tot[...] += Gdftd3
        # Nuclear attraction energy gradient (QM Molecule) 
        Gext = ls.IntBox.potentialGrad(
            self.resources,
            self.ewald,
            pairlist,
            Dtot,
            self.qm_molecule_ecp.xyzZ)
        Gqm_tot[...] += Gext
        # ECP
        if self.ecp:
            Gecp = ls.IntBox.ecpGrad(
                self.resources,
                pairlist,
                Dtot,
                self.ecp,
                self.options['threecp'],
                )
            Gqm_tot[...] += Gecp

        # Complete at this point if not QM/MM
        if not self.qmmm:
            return Gqm_tot

        # External MM self-energy gradient (MM Molecule)
        if include_self:
            Gmm = self.qmmm.mm_gradient
            Gmm_tot = ls.Tensor.array(Gmm)
        else:
            Gmm_tot = ls.Tensor.zeros((self.molecule.natom, 3))
        # MM charge - QM nuclear charge interaction energy
        if include_self:
            Gqm_int, Gmm_int = self.qmmm.compute_mm_nuclear_gradient(
                self.resources,
                self.ewald,
                self.qm_molecule_ecp,
                )
            Gqm_tot[...] += Gqm_int
            Gmm_tot[...] += Gmm_int
        # MM charge - QM electronic potential gradient
        Gqm_pot, Gmm_pot = self.qmmm.compute_mm_potential_gradient(
            self.resources,
            self.ewald,
            pairlist,
            Dtot,
            )
        Gqm_tot[...] += Gqm_pot
        Gmm_tot[...] += Gmm_pot
        
        return self.qmmm.distribute_gradient(qm_grad=Gqm_tot, mm_grad=Gmm_tot)

    # => Polarizable Embedding (e.g., PCM) <= #

    # TODO

    @property
    def is_polarizable(self):
        """ Does this Geometry object respond to the EST charge density? """
        return False

    @property
    def state_vector(self):
        """ The current value of the state vector of a polarizable Geometry """
        return None

    @property
    def error_vector(self):
        """ The current value of the error vector of a polarizable Geometry """
        return None

    def update_density(
        self,
        Dtot,
        ):

        pass

    # => Update/Build Methods <= #
    
    def update_xyz(
        self, 
        xyz,
        ):

        """ Return a new Geometry with updated XYZ coordinates.

        """
    
        options = self.options.copy()
        if self.qmmm:
            options['qmmm'] = self.qmmm.update_xyz(xyz)
        else:
            options['molecule'] = self.molecule.update_xyz(xyz)
        qm_xyz = options['qmmm'].qm_molecule.xyz if self.qmmm else xyz
        options['basis'] = self.basis.update_xyz(qm_xyz)
        options['minbasis'] = self.minbasis.update_xyz(qm_xyz)
        if self.ecp:
            options['ecp'] = self.ecp.update_xyz(qm_xyz)
        options['pairlist'] = ls.PairList.build_schwarz(
            options['basis'],
            options['basis'],
            True,
            self.pairlist.thre)
        return Geometry(options=options)

    @staticmethod
    def build(
        resources=None,
        molecule=None,
        qmmm=None,
        dftd3obj=None,
        dftd3name=None,
        dftd3flavor=None,
        basis=None,
        minbasis=None,
        basisname='cc-pvdz',
        basisspherical=None,
        minbasisname='cc-pvdz-minao',
        minbasisspherical=None,
        ecp=None,
        ecpname=None,
        ewald=ls.Ewald.coulomb(),
        pairlist=None,
        thre_pq=1.0E-14,
        ): 

        """ A factory constructor to manage reading Basis objects from GBS files """

        if basis is None:
            basis = ls.Basis.from_gbs_file(
                qmmm.qm_molecule if qmmm else molecule,
                basisname,
                spherical=basisspherical)
    
        if minbasis is None:
            minbasis = ls.Basis.from_gbs_file(
                qmmm.qm_molecule if qmmm else molecule,
                minbasisname,
                spherical=minbasisspherical)

        if ecp is None:
            if ecpname:
                ecp = ls.ECPBasis.from_ecp_file(
                    qmmm.qm_molecule if qmmm else molecule,
                    ecpname,
                    )

        if dftd3obj is None:
            if dftd3name:
                dftd3obj = dftd3.DFTD3Base.build(name=dftd3name, flavor=dftd3flavor)

        if pairlist is None:
            pairlist = ls.PairList.build_schwarz(
                basis,
                basis,
                True,
                thre_pq)
    
        geometry = Geometry(Geometry.default_options().set_values({
            'resources' : resources,
            'molecule' : molecule,
            'qmmm' : qmmm,
            'dftd3' : dftd3obj,
            'basis' : basis,
            'minbasis' : minbasis,
            'pairlist' : pairlist,
            'ecp' : ecp,
            'ewald' : ewald,
            }))
        return geometry

if __name__ == '__main__':

    resources = ls.ResourceList.build()

    molecule = ls.Molecule.from_xyz_file("test/h2o.xyz")

    geometry = Geometry.build(
        resources=resources,
        molecule=molecule,
        basisname='cc-pvdz',
        )
        
    geometry2 = geometry.update_xyz(molecule.xyz)

        

         
