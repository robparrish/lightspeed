import lightspeed as ls
import numpy as np
from . import options
import collections

try:
    import simtk.unit as openmm_units
    import simtk.openmm.app as openmm_app
    import simtk.openmm as openmm
    
    class QMMM(object):
    
        """ Class QMMM represents all relevant parts of a QM/MM computation in LS,
            including an OpenMM Simulation object (modified to provide QM/MM), a
            list of qm_inds, a list of link_inds, and ls.Molecule objects for the
            total system and for the QM molecule (including linking H atoms).
    
            We use "Scheme 3" of the mechanical + electrostatic embedding of the
            2008 AMBER QM/MM Paper: R.C. Walker, M.F. Crowley, and D.A. Case, J.
            Comp. Chem. 29, 1018 (2008).
    
            In this scheme we start with an MM treatment of the full system (in
            vacuum BCs only). We remove any bond, angle, torsion, or VDW term which
            has all atoms entirely within the QM region. We add link H atoms a
            user-specified distance between QM and MM atoms in links (typically 1.0
            A). We zero all MM charges on QM atoms or MM link atoms, and evenly
            redistribute the lost charge to all other MM atoms. We then solve the
            Schrodinger equation for the QM molecule (including link H atoms)
            within the point charge field of the non-QM and non-link MM atoms.
            Forces on H link atoms are distributed to their owning QM and MM atoms
            by the lever rule.
        """
    
        @staticmethod
        def default_options():
            """ Return a new copy of the declared/default Options for QMMM. """
            if hasattr(QMMM, "_default_options"): return QMMM._default_options.copy()
            opt = options.Options()
            opt.add_option(
                key="simulation", 
                required=True, 
                allowed_types=[openmm_app.Simulation], 
                doc="OpenMM Simulation object") 
            opt.add_option(
                key="qm_inds", 
                required=True, 
                allowed_types=[list], 
                doc="List of indices (int) of QM atoms (0-based)")
            opt.add_option(
                key="link_inds", 
                required=True, 
                allowed_types=[list], 
                doc="List of tuples of (indQM, indMM) (int) showing QM/MM atom link indices (0-based)")
            opt.add_option(
                key="molecule",
                required=True,
                allowed_types=[ls.Molecule],
                doc="ls.Molecule representation of complete system")
            opt.add_option(
                key="qm_molecule",
                required=True,
                allowed_types=[ls.Molecule],
                doc="ls.Molecule representation of QM molecule, including link H atoms")
            opt.add_option(
                key="link_RXH",
                value=1.09 * ls.units['bohr_per_ang'],
                required=True,
                allowed_types=[float],
                doc="X-H bond distance for H link atoms (au). 1.09 Angstrom is default in TeraChem")
            QMMM._default_options = opt
            return QMMM._default_options.copy()
    
        def __init__(
            self,
            options,
            ):
    
            """ Verbatim constructor """
            self.options = options
    
        @staticmethod
        def from_options(**kwargs):
            """ Return an instance of this class with default options updated from values in kwargs. """
            return QMMM(QMMM.default_options().set_values(kwargs))
    
        @property
        def simulation(self):
            """ OpenMM Simulation object """
            return self.options['simulation']
    
        @property
        def qm_inds(self):
            """ List of indices (int) of QM atoms (0-based) """
            return self.options['qm_inds']
    
        @property
        def link_inds(self):
            """ List of tuples of (indQM, indMM) (int) showing QM/MM atom link indices (0-based) """
            return self.options['link_inds']
    
        @property
        def molecule(self):
            """ ls.Molecule representation of the complete system. """
            return self.options['molecule']
    
        @property
        def qm_molecule(self):
            """ ls.Molecule representation of the QM molecule, including link H atoms """
            return self.options['qm_molecule']
    
        def update_xyz(
            self,
            xyz,
            ):
    
            """ Return a new QMMM object which is a copy of self with positions updated to xyz.
    
            Params:
                xyz - ls.Tensor (molecule.natom, 3) with new coordinates in au
            Returns:
                new QMMM object with updated coordinates in molecule and
                qm_molecule (note that simulation is a context-heavy OpenMM object,
                and is shallow copied.)
            """
    
            molecule = self.molecule.update_xyz(xyz)
            
            xyz2 = ls.Tensor((self.qm_molecule.natom, 3))
            xyznp = xyz.np
            xyz2np = xyz2.np
            link_RXH = self.options['link_RXH']
            for A2, A in enumerate(self.qm_inds):
                xyz2np[A2,:] = xyznp[A,:]
            for A2, bond in enumerate(self.link_inds):
                atomQM = molecule.atoms[bond[0]]
                atomMM = molecule.atoms[bond[1]]
                NAB = np.array([
                    atomMM.x - atomQM.x, 
                    atomMM.y - atomQM.y, 
                    atomMM.z - atomQM.z,
                    ])
                NAB /= np.sqrt(np.sum(NAB**2))
                xyz2np[A2 + len(self.qm_inds), :] = np.array([
                    atomQM.x + link_RXH * NAB[0],
                    atomQM.y + link_RXH * NAB[1],
                    atomQM.z + link_RXH * NAB[2],
                    ])
            qm_molecule = self.qm_molecule.update_xyz(xyz2) 
    
            qmmm = QMMM(options=self.options.copy().set_values({
                'simulation' : self.simulation,
                'molecule' : molecule,
                'qm_molecule' : qm_molecule,
                }))
            return qmmm
    
        @property
        def mm_energy(self):
            """ Return the external MM energy in au """
            if not hasattr(self, "_mm_energy"):
                self.compute_mm_energy_and_gradient()
            return self._mm_energy
    
        @property
        def mm_gradient(self):
            """ Return the external MM gradient in au """
            if not hasattr(self, "_mm_gradient"):
                self.compute_mm_energy_and_gradient()
            return self._mm_gradient
    
        def compute_mm_energy_and_gradient(self): 
            """ Explicitly compute the mm energy and gradient (memoization function) """ 
    
            # Update coordinates of simulation (shallow-copied object)
            xyz_nm = 0.1 * ls.units['ang_per_bohr'] * self.molecule.xyz[...]
            self.simulation.context.setPositions(xyz_nm)
    
            # Compute the energy and force
            state = self.simulation.context.getState(
                getEnergy=True, 
                getForces=True,
                )
            E = state.getPotentialEnergy()
            F = state.getForces()
    
            # Convert the energy and force to au and memoize
            self._mm_energy = E.value_in_unit(openmm_units.kilocalories / openmm_units.moles)
            self._mm_energy *= ls.units['H_per_kcal']
            self._mm_gradient = ls.Tensor.array(F.value_in_unit(openmm_units.kilocalories / openmm_units.moles / openmm_units.angstroms))
            self._mm_gradient[...] *= -1.0 * ls.units['H_per_kcal'] * ls.units['ang_per_bohr']
    
        def compute_mm_energy_and_gradient_detail(
            self,
            ):
    
            """ Compute the specific components of the MM energy and return in H energy units. 
    
            The keys of the energy are the int forceGroup index. Details of the
            corresponding force can be obtained by, e.g.:
                self.simulation.getSystem().getForce(index)
    
            Returns:
                E (ordered dict of int -> float) - Energies in H
                G (ordered dict of int -> ls.Tensor) - Gradients in H / bohr.
            """
    
            # Update coordinates of simulation (shallow-copied object)
            xyz_nm = 0.1 * ls.units['ang_per_bohr'] * self.molecule.xyz[...]
            self.simulation.context.setPositions(xyz_nm)
        
            # Now compute the forces
            Es = collections.OrderedDict()
            Gs = collections.OrderedDict()
            system = self.simulation.context.getSystem()
            for i in range(system.getNumForces()):
                state = self.simulation.context.getState(
                    getEnergy=True, 
                    getForces=True,
                    groups=2**i)
                E = state.getPotentialEnergy()
                F = state.getForces()
                # Convert the energy and force to au 
                E = E.value_in_unit(openmm_units.kilocalories / openmm_units.moles)
                E *= ls.units['H_per_kcal']
                G = ls.Tensor.array(F.value_in_unit(openmm_units.kilocalories / openmm_units.moles / openmm_units.angstroms))
                G[...] *= -1.0 * ls.units['H_per_kcal'] * ls.units['ang_per_bohr']
    
                Es[i] = E
                Gs[i] = G
    
            return Es, Gs
    
        def compute_mm_nuclear_energy(
            self,
            resources,
            ewald,
            qm_molecule_ecp,
            ):
    
            """ Compute the MM--QM point-charge--nuclear interaction energy.
    
            Params:
                resources (ls.ResourceList) - Resources for the potential matrix build.
                ewald (ls.Ewald) - Ewald operator for the potential matrix build
                    (allows one to blur MM point charges out).
                qm_molecule_ecp (ls.Molecule) - Molecule for QM region, with any
                    ECP nuclear-charge considerations built in.
            Returns:
                Eint (float) - the MM--QM interaction energy
            """
    
            # Get modified MM charges
            charges = None
            system = self.simulation.system
            for find in range(system.getNumForces()):
                force = system.getForce(find)
                if isinstance(force, openmm.NonbondedForce):
                    charges = [force.getParticleParameters(x)[0] for x in range(force.getNumParticles())]
            if charges is None:
                raise RuntimeError('QMMM: Cannot find charges')
            charges = ls.Tensor.array([x.value_in_unit(openmm_units.elementary_charges) for x in charges])
            
            # Get positions of MM charges
            xyz = self.molecule.xyz
    
            # Get joint position/charge vector
            xyzc = ls.Tensor((xyz.shape[0], 4))
            xyzc[:,:3] = xyz
            xyzc[:,3] = charges
        
            # Compute and return interaction energy
            return ls.IntBox.chargeEnergyOther(
                resources,
                ewald,
                qm_molecule_ecp.xyzZ,
                xyzc, 
                )
    
        def compute_mm_nuclear_gradient(
            self,
            resources,
            ewald,
            qm_molecule_ecp,
            ):
    
            """ Compute the MM--QM point-charge--nuclear interaction energy gradient.
    
            Params:
                resources (ls.ResourceList) - Resources for the potential matrix build.
                ewald (ls.Ewald) - Ewald operator for the potential matrix build
                    (allows one to blur MM point charges out).
                qm_molecule_ecp (ls.Molecule) - Molecule for QM region, with any
                    ECP nuclear-charge considerations built in.
            Returns:
                Gqm (ls.Tensor of shape (natom_qm, 3)) - gradient on QM molecule
                Gmm (ls.Tensor of shape (natom_mm, 3)) - gradient on MM point charges
            """
    
            # Get modified MM charges
            charges = None
            system = self.simulation.system
            for find in range(system.getNumForces()):
                force = system.getForce(find)
                if isinstance(force, openmm.NonbondedForce):
                    charges = [force.getParticleParameters(x)[0] for x in range(force.getNumParticles())]
            if charges is None:
                raise RuntimeError('QMMM: Cannot find charges')
            charges = ls.Tensor.array([x.value_in_unit(openmm_units.elementary_charges) for x in charges])
            
            # Get positions of MM charges
            xyz = self.molecule.xyz
    
            # Get joint position/charge vector
            xyzc = ls.Tensor((xyz.shape[0], 4))
            xyzc[:,:3] = xyz
            xyzc[:,3] = charges
        
            # Compute and return interaction energy
            return ls.IntBox.chargeGradOther(
                resources,
                ewald,
                qm_molecule_ecp.xyzZ,
                xyzc, 
                )
    
        def compute_mm_potential(
            self,
            resources,
            ewald,
            pairlist,
            ):
    
            """ Compute the AO-basis potential matrix of the MM point charges.
    
            Params:
                resources (ls.ResourceList) - Resources for the potential matrix build.
                ewald (ls.Ewald) - Ewald operator for the potential matrix build
                    (allows one to blur MM point charges out).
                pairlist (ls.PairList) - A PairList to compute the potential matrix in.
            Returns:
                V (ls.Tensor) - Tensor of size (nao1, nao2) with potential of MM
                    point charges only (no QM charges involved).
            """
    
            # Get modified MM charges
            charges = None
            system = self.simulation.system
            for find in range(system.getNumForces()):
                force = system.getForce(find)
                if isinstance(force, openmm.NonbondedForce):
                    charges = [force.getParticleParameters(x)[0] for x in range(force.getNumParticles())]
            if charges is None:
                raise RuntimeError('QMMM: Cannot find charges')
            charges = ls.Tensor.array([x.value_in_unit(openmm_units.elementary_charges) for x in charges])
            
            # Get positions of MM charges
            xyz = self.molecule.xyz
    
            # Get joint position/charge vector
            xyzc = ls.Tensor((xyz.shape[0], 4))
            xyzc[:,:3] = xyz
            xyzc[:,3] = charges
    
            # Compute and return potential
            return ls.IntBox.potential(
                resources,
                ewald,
                pairlist,
                xyzc, 
                )
    
        def compute_mm_potential_gradient(
            self,
            resources,
            ewald,
            pairlist,
            Dtot,
            ):
    
            """ Compute the gradient contributions from the MM potential.
    
            Params:
                resources (ls.ResourceList) - Resources for the gradient build.
                ewald (ls.Ewald) - Ewald operator for the gradient build
                    (allows one to blur MM point charges out).
                pairlist (ls.PairList) - A PairList to compute the gradient build in.
                Dtot (ls.
            Returns:
                Gqm (ls.Tensor of shape (natom_qm, 3)) - gradient on QM molecule
                Gmm (ls.Tensor of shape (natom_mm, 3)) - gradient on MM point charges
            """
    
            # Get modified MM charges
            charges = None
            system = self.simulation.system
            for find in range(system.getNumForces()):
                force = system.getForce(find)
                if isinstance(force, openmm.NonbondedForce):
                    charges = [force.getParticleParameters(x)[0] for x in range(force.getNumParticles())]
            if charges is None:
                raise RuntimeError('QMMM: Cannot find charges')
            charges = ls.Tensor.array([x.value_in_unit(openmm_units.elementary_charges) for x in charges])
            
            # Get positions of MM charges
            xyz = self.molecule.xyz
    
            # Get joint position/charge vector
            xyzc = ls.Tensor((xyz.shape[0], 4))
            xyzc[:,:3] = xyz
            xyzc[:,3] = charges
    
            # Compute and return potential
            return ls.IntBox.potentialGradAdv2(
                resources,
                ewald,
                pairlist,
                Dtot,
                xyzc, 
                )
    
        def distribute_gradient(
            self,
            qm_grad,
            mm_grad,
            ):
    
            """ Distribute gradient contributions from QM atoms and MM atoms
                (actually all atoms) to total gradient contributions. 
        
            QM atom contributions are added to MM atom contributions from
            corresponding atoms. QM link H atom contributions are added to the two
            spanning link atoms according to the level rule.
        
            Params:
                qm_grad (ls.Tensor of shape (qm_molecule.natom, 3)) - Contributions
                    on QM molecule, including link H contributions. 
                mm_grad (ls.Tensor of shape (molecule.natom, 3)) - Contributions
                    on MM molecule (actually full molecule).
            Returns
                grad (ls.Tensor of shape (molecule.natom, 3)) - Total gradient.
            """
    
            # Validity checks
            qm_grad.shape_error((len(self.qm_inds)+len(self.link_inds), 3))
            mm_grad.shape_error((self.molecule.natom, 3))
    
            # Molecular geometry
            mm_xyz = self.molecule.xyz
            # qm_xyz = self.qm_molecule.xyz
    
            # Pure MM contributions
            grad = ls.Tensor.array(mm_grad)
            # Pure QM contributions
            for A, A2 in enumerate(self.qm_inds): 
                grad[A2,:] += qm_grad[A,:]
            # Link H contributions (lever rule)
            for A3, link in enumerate(self.link_inds):
                G_lh = qm_grad[A3 + len(self.qm_inds),:] # H atom gradient
                A_qm = link[0]
                A_mm = link[1]
                r_qm = mm_xyz[A_qm,:]
                r_mm = mm_xyz[A_mm,:]
                d_lh_qm = self.options['link_RXH']
                d_mm_qm = np.sqrt(np.sum((r_qm - r_mm)**2))
                # First term in Walker 2008 eq 32
                f_mm = d_lh_qm / d_mm_qm
                f_qm = (1.0 - f_mm)
                G_qm = f_qm * G_lh
                G_mm = f_mm * G_lh
                # Second term in Walker 2008 eq 32
                delta_mm_qm = r_mm - r_qm
                Q2 = (r_mm - r_qm) * d_lh_qm / d_mm_qm**3 * np.sum(G_lh * (r_mm - r_qm)) 
                G_qm += Q2
                G_mm -= Q2
                grad[A_qm,:] += G_qm
                grad[A_mm,:] += G_mm
                
            return grad
            
        @staticmethod
        def from_prmtop(
            prmtopfile,
            inpcrdfile,
            qmindsfile,
            charge=0.0,
            multiplicity=1.0,
            link_RXH=1.09 * ls.units['bohr_per_ang'],
            ):
    
            """ Build a QMMM object from an AMBER prmtop file, an AMBER inpcrd or
                rst file, and a QM index file.
    
                This uses the following non-default arguments in
                AmberPrmtopFile.createSystem, yielding a QM/MM-appropriate
                simulation object: 
                    rigidWater=False, 
                    removeCMMotion=False,
    
                Params:
                    prmtopfile (str) - path to an AMBER PRMTOP file (gas phase)
                    inpcrdfile (str) - path to an AMBER INPCRD or RST file
                    qmindsfile (str) - path to a TC-style QMINDS file (0-based ordering)
                    charge (float) - total charge of QM molecule
                    multiplicity (float) - multiplicity of QM molecule
                    link_RXH (float) - X-H distance in H link atoms (in au). The
                        default of 1.09 Angstrom is used in TeraChem.
                Returns:
                    A fully initialized gas-phase QMMM object 
            """
    
            # Create and initialize System object from prmtop/inpcrd
            prmtop = openmm_app.AmberPrmtopFile(prmtopfile)
            inpcrd = openmm_app.AmberInpcrdFile(inpcrdfile)
            system = prmtop.createSystem(
                rigidWater=False, 
                removeCMMotion=False,
                )
    
            # Read qm_inds
            with open(qmindsfile) as fh:
                qm_inds_lines = fh.readlines()
            qm_inds = [int(x) for x in ' '.join(qm_inds_lines).split()]
    
            # Determine link_inds from topology
            link_inds = []
            for bond in prmtop.topology.bonds():
                Aind = bond.atom1.index
                Bind = bond.atom2.index
                if Aind in qm_inds and Bind not in qm_inds:
                    link_inds.append((Aind, Bind))
                if Bind in qm_inds and Aind not in qm_inds:
                    link_inds.append((Bind, Aind))
    
            # Modify forces to do QM/MM
            for find in range(system.getNumForces()):
                force = system.getForce(find)
                if isinstance(force, openmm.HarmonicBondForce):
                    # Bonds to 0 if all involved atoms are QM
                    for index in range(force.getNumBonds()):
                        bforce = force.getBondParameters(index)
                        if bforce[0] in qm_inds and bforce[1] in qm_inds:
                            force.setBondParameters(index, *bforce[:-1], k=0.0)
                elif isinstance(force, openmm.HarmonicAngleForce):
                    # Angles to 0 if all involved atoms are QM
                    for index in range(force.getNumAngles()):
                        aforce = force.getAngleParameters(index)
                        if aforce[0] in qm_inds and aforce[1] in qm_inds and aforce[2] in qm_inds:
                            force.setAngleParameters(index, *aforce[:-1], k=0.0)
                elif isinstance(force, openmm.PeriodicTorsionForce):
                    # Torsions to 0 if all involved atoms are QM
                    for index in range(force.getNumTorsions()):
                        tforce = force.getTorsionParameters(index)
                        if tforce[0] in qm_inds and tforce[1] in qm_inds and tforce[2] in qm_inds and tforce[3] in qm_inds:
                            force.setTorsionParameters(index, *tforce[:-1], k=0.0)
                elif isinstance(force, openmm.NonbondedForce):
                    # Zero charges on QM atoms + Link MM atoms, redistribute missing charges to all other MM atoms
                    qm_plus_link_inds = qm_inds + [x[1] for x in link_inds]
        
                    # Find the total charge on QM + Link MM atoms, minus the total charge of the qm_molecule
                    Q = 0.0
                    for aind in qm_plus_link_inds:
                        Q += force.getParticleParameters(aind)[0].value_in_unit(openmm_units.elementary_charges)
                    Q -= charge
                    den = force.getNumParticles() - len(qm_inds) - len(link_inds)
        
                    # Charge correction on non QM + Link MM atoms
                    dQ = 0.0 if den == 0 else Q / den
                    dQ = dQ * openmm_units.elementary_charges
                    d0 = 0.0 * openmm_units.elementary_charges
        
                    # Retain original charges to figure out screening values in exceptions
                    charges0 = [force.getParticleParameters(x)[0] for x in range(force.getNumParticles())]
    
                    # Apply charge correction on non QM + Link MM atoms, zero QM + Link MM atom charges
                    for aind in range(force.getNumParticles()):
                        params = force.getParticleParameters(aind)
                        if aind in qm_plus_link_inds:
                            force.setParticleParameters(aind, d0, *params[1:])
                        else:
                            force.setParticleParameters(aind, params[0] + dQ, *params[1:])
    
                    # Figure out new charges to adjust exceptions
                    charges = [force.getParticleParameters(x)[0] for x in range(force.getNumParticles())]
    
                    # Make sure existing exceptions (e.g., 1-2, 1-3, 1-4 interactions) respect altered charges
                    # But also make sure that 1-2, 1-3, 1-4 screening (scee) is respected
                    for eind in range(force.getNumExceptions()):
                        params = force.getExceptionParameters(eind) 
                        # Infer the scee value for this pair
                        Q12 = params[2].value_in_unit(openmm_units.elementary_charges**2)
                        Q1 = charges0[params[0]].value_in_unit(openmm_units.elementary_charges)
                        Q2 = charges0[params[1]].value_in_unit(openmm_units.elementary_charges)
                        scee = Q12 / (Q1 * Q2) if Q1 * Q2 != 0.0 else 0.0
                        # Update the charge product with the new charges but old scee
                        params[2] = charges[params[0]] * charges[params[1]] * scee
                        force.setExceptionParameters(eind, *params)
    
                    # Elst and VDW to 0 if all involved atoms are QM (override existing exceptions)
                    for qm_ind1 in qm_inds:
                        for qm_ind2 in qm_inds:
                            if qm_ind1 >= qm_ind2: continue
                            force.addException(qm_ind1, qm_ind2, 0.0, 1.0, 0.0, True)
                else:
                    pass # Let the user apply custom forces without complaint
                # Make sure to set the force group index
                force.setForceGroup(find)
    
            # Get positions in atomic inits
            xyz = np.array(inpcrd.positions.value_in_unit(openmm_units.angstroms))
            xyz *= ls.units['bohr_per_ang'] 
    
            # Build ls.Molecule molecule from system
            atoms = []
            for A, atom in enumerate(prmtop.topology.atoms()): 
                atoms.append(ls.Atom(
                    atom.name,
                    atom.element.symbol,
                    atom.element.atomic_number,
                    xyz[A,0],
                    xyz[A,1],
                    xyz[A,2],
                    float(atom.element.atomic_number),
                    A)) 
            molecule = ls.Molecule("QM/MM Full Molecule", atoms, 0.0, 1.0)
    
            # Build ls.Molecule qm_molecule
            qm_atoms = []
            for A2, A in enumerate(qm_inds):
                atom = molecule.atoms[A]
                qm_atoms.append(ls.Atom(
                    atom.label,
                    atom.symbol,
                    atom.N,
                    atom.x,
                    atom.y,
                    atom.z,
                    atom.Z,
                    A2))
            for A2, bond in enumerate(link_inds):
                atomQM = molecule.atoms[bond[0]]
                atomMM = molecule.atoms[bond[1]]
                NAB = np.array([
                    atomMM.x - atomQM.x, 
                    atomMM.y - atomQM.y, 
                    atomMM.z - atomQM.z,
                    ])
                NAB /= np.sqrt(np.sum(NAB**2))
                qm_atoms.append(ls.Atom(
                    "H",
                    "H",
                    1,
                    atomQM.x + link_RXH * NAB[0],
                    atomQM.y + link_RXH * NAB[1],
                    atomQM.z + link_RXH * NAB[2],
                    1.0,
                    A2 + len(qm_inds)))
            qm_molecule = ls.Molecule("QM/MM QM Molecule", qm_atoms, charge, multiplicity)
    
            # Build an OpenMM Simulation object
            # Integrator will never be used (Simulation requires one)
            integrator = openmm.VerletIntegrator(1.0)
            simulation = openmm_app.Simulation(
                prmtop.topology,
                system,
                integrator,
                )
            simulation.context.setPositions(inpcrd.positions)
            # RMP: I don't know that these will ever be used, but it cannot hurt
            if inpcrd.velocities is not None:
                simulation.context.setVelocities(inpcrd.velocities)
            if inpcrd.boxVectors is not None:
                raise RuntimeError('QMMM: must be gas-phase (vacuum BCs)')
    
            # Build and return QMMM object
            qmmm = QMMM(options=QMMM.default_options().set_values({
                'simulation' : simulation,
                'qm_inds' : qm_inds,
                'link_inds' : link_inds,    
                'molecule' : molecule,
                'qm_molecule' : qm_molecule,
                'link_RXH' : link_RXH,
                }))
            return qmmm

except ImportError:
    
    """ Dummy version if no OpenMM """

    class QMMM(object):

        @staticmethod
        def default_options():
    
            raise RuntimeError('QMMM: No OpenMM')

        def __init__(
            self,
            options,
            ):

            raise RuntimeError('QMMM: No OpenMM')
    
        @staticmethod
        def from_options(**kwargs):
    
            raise RuntimeError('QMMM: No OpenMM')

        @staticmethod
        def from_prmtop(
            prmtopfile,
            inpcrdfile,
            qmindsfile,
            charge=0.0,
            multiplicity=1.0,
            link_RXH=1.09 * ls.units['bohr_per_ang'],
            ):

            raise RuntimeError('QMMM: No OpenMM')
    


if __name__ == '__main__':

    resources = ls.ResourceList.build()

    qmmm = QMMM.from_prmtop(
        prmtopfile='test/system.prmtop',
        inpcrdfile='test/system.rst',
        qmindsfile='test/system.qm',
        charge=-1.0,
        )
    # print QMMM.default_options
    # print qmmm.options
    # print qmmm.options.get_option('molecule')
        
    import time
    start = time.time()
    qmmm2 = qmmm.update_xyz(qmmm.molecule.xyz)
    print('%11.3E' % (time.time() - start))

    print(qmmm2.mm_energy)
    print(qmmm2.mm_gradient)

    print(ls.Tensor.array(qmmm2.mm_gradient[qmmm2.qm_inds,:]))

    basis = ls.Basis.from_gbs_file(qmmm2.qm_molecule, '6-31g')
    pairlist = ls.PairList.build_schwarz(
        basis, basis, True, 1.0E-14)

    start = time.time()
    print(qmmm2.compute_mm_potential(
        resources=resources,
        ewald=ls.Ewald.coulomb(),
        pairlist=pairlist,
        ).shape)
    print('%11.3E' % (time.time() - start))

