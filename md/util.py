import lightspeed as ls
import numpy as np
from . import masses

def compute_masses(
    molecule,
    ):

    """ Look up and return standard masses in atomic units for the atoms
        denoted by atomic number atom.N in molecule.

    Params:
        molecule (ls.Molecule) - molecule to look masses up for. The atom.N
            field is used for each atom in molecule.atoms.
    Returns:
        M (ls.Tensor of shape (molecule.natom, 3)) - Tensor of masses in au.
        These are taken from the most-abundant isotopic mass table in
        masses.py.
   """ 

    M = ls.Tensor.array([
        masses.mass_table[masses.atom_symbol_table[atom.N]] for atom in molecule.atoms])
    M = ls.Tensor.array(np.outer(M, np.array([1.0, 1.0, 1.0])))
    return M

def compute_boltzmann_momenta(
    kT,
    masses, 
    remove_vcom=True,
    seed=None,
    ):
    
    """ Compute random Boltzmann momenta.

    Params:
        kT (float) - temperature in atomic units.
        masses (ls.Tensor of shape (natom,3)) - masses in atomic units
        remove_vcom (bool) - remove center of mass velocity?
        seed (None or int) - seed for np.random.RandomState random number
            generator
    Returns:
        P (ls.Tensor of shape masses.shape) - momenta in atomic units.
    """

    # TODO: Implement
    
    raise NotImplemented

def momenta_from_xyz_file(
    filename,
    masses,
    units='v_amber',
    ):

    """ Read velocities from XYZ file and convert to moments. By convention,
        velocities in AMBER units are stored in XYZ files.

    Params:
        filename (str) - file to read from. Will be read in by
            ls.Molecule.from_xyz_file and then converted to Tensor of atomic
            units of momenta.
        masses (ls.Tensor of shape (natom,3)) - masses in atomic units.
        units (str) - Unit convention:
            'v_amber' : velocities in AMBER units.
            'p_au' : momenta in atomic units.
    Returns:
        P (ls.Tensor of shape masses.shape) - momenta in atomic units.
    """

    mol = ls.Molecule.from_xyz_file(filename, scale=1.0)
    
    if units == 'v_amber':
        # Amber units per au of momentum
        amb_per_au = ls.units['ang_per_bohr'] * ls.units['au_per_fs'] * 1.E3 / 20.455
        au_per_amb = 1.0 / amb_per_au
        P = ls.Tensor.array(masses[...] * mol.xyz[...])
        P[...] *= au_per_amb
    elif units == 'p_au':
        P = mol.xyz
    else:
        raise ValueError("Unknown units: %s" % units)

    return P

    

    
