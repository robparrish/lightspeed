from . import lightspeed as pls
import numpy as np

SAD = pls.SAD

@staticmethod
def _sad_orbitals(
    resources,
    molecule,
    basis,
    minbasis,
    Qocc = None,   # Specific occupations in electron pairs (1st priority)
    Qatom = None,  # Specific atomic electronic charges in electron pairs (2nd priority)
    Qtotal = None, # Specific total charge in electron pairs (3rd priority)
    necps = None,  # Number of orbitals to zero occupations for each atom, to account for ECPs
    ):

    if Qocc:
        Qocc2 = Qocc
    elif Qatom:
        Qocc2 = pls.SAD.sad_nocc(molecule,Qatom)
    elif Qtotal:
        if not isinstance(Qtotal,float):
            raise TypeError("SAD.orbitals: Qtotal must be float")
        Qatom2 = pls.Tensor((molecule.natom,))
        for A, atom in enumerate(molecule.atoms):
            Qatom2[A] = 0.5 * atom.Z 
        Qatom2[...] *= Qtotal / np.sum(Qatom2) 
        Qocc2 = pls.SAD.sad_nocc(molecule,Qatom2)
    else:
        Qatom2 = pls.Tensor((molecule.natom,))
        for A, atom in enumerate(molecule.atoms):
            Qatom2[A] = 0.5 * atom.Z 
        Qocc2 = pls.SAD.sad_nocc(molecule,Qatom2)

    # Adjust occupations for ECPs
    if necps is None:
        necps = [0 for k in range(basis.natom)]
    
    return pls.SAD.sad_orbitals(
        resources,
        Qocc2,
        minbasis,
        basis,
        necps)

@staticmethod
def _sad_atomic_electronic_charges(
    molecule,
    Qatom, # Specific total atomic charges (e.g., -1.0 for O^-)
    ):

    """ Convert between total atomic charges (e.g., -1.0 for O^-) and
        SAD-compatible electronic atomic charges in electron pairs (e.g., -4.5 for O^-1)
        
    Args:
        molecule (Molecule) - the molecule to obtain nuclear charges from (taken from atoms[...].Z fields)
        Qatom (Tensor, [natom,]) - the atomic total charges in electrons
    
    Returns:
        Qatom2 (Tensor, [natom,]) - the atomic electronic charges in electron pairs

    """
        
    if molecule.natom != len(Qatom):
        raise ValueError('natom does not match')
    
    Qatom2 = pls.Tensor.array(Qatom)
    Qatom2[...] *= -1.0
    Qatom2[...] += molecule.Z
    Qatom2[...] *= 0.5
    return Qatom2
    
SAD.orbitals = _sad_orbitals
SAD.atomic_electronic_charges = _sad_atomic_electronic_charges
