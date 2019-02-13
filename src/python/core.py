from . import lightspeed as pls
from . import util
import re
import os
import math

Atom = pls.Atom
Molecule = pls.Molecule
AngularMomentum = pls.AngularMomentum
Primitive = pls.Primitive
Shell = pls.Shell
Basis = pls.Basis
ECPShell = pls.ECPShell
ECPBasis = pls.ECPBasis
Pair = pls.Pair
PairListL = pls.PairListL
PairList = pls.PairList
Ewald = pls.Ewald
PureTransform = pls.PureTransform
Rys = pls.Rys
Boys = pls.Boys
GH = pls.GH
Molden = pls.Molden
Local = pls.Local

# => Atom Decorators <= #

# Map from uppercase symbol to N
_atomic_numbers_ = {
'X'   :   0,
'H'   :   1, 'HE'  :   2,
'LI'  :   3, 'BE'  :   4, 'B'   :   5, 'C'   :   6, 'N'   :   7, 'O'   :   8, 'F'   :   9, 'NE'  :  10,
'NA'  :  11, 'MG'  :  12, 'AL'  :  13, 'SI'  :  14, 'P'   :  15, 'S'   :  16, 'CL'  :  17, 'AR'  :  18,
'K'   :  19, 'CA'  :  20, 'SC'  :  21, 'TI'  :  22, 'V'   :  23, 'CR'  :  24, 'MN'  :  25, 'FE'  :  26, 'CO'  :  27, 'NI'  :  28, 'CU'  :  29, 'ZN'  :  30, 'GA'  :  31, 'GE'  :  32, 'AS'  :  33, 'SE'  :  34, 'BR'  :  35, 'KR'  :  36,
'RB'  :  37, 'SR'  :  38, 'Y'   :  39, 'ZR'  :  40, 'NB'  :  41, 'MO'  :  42, 'TC'  :  43, 'RU'  :  44, 'RH'  :  45, 'PD'  :  46, 'AG'  :  47, 'CD'  :  48, 'IN'  :  49, 'SN'  :  50, 'SB'  :  51, 'TE'  :  52, 'I'   :  53, 'XE'  :  54,
'CS'  :  55, 'BA'  :  56, 'LA'  :  57, 'CE'  :  58, 'PR'  :  59, 'ND'  :  60, 'PM'  :  61, 'SM'  :  62, 'EU'  :  63, 'GD'  :  64, 'TB'  :  65, 'DY'  :  66, 'HO'  :  67, 'ER'  :  68, 'TM'  :  69, 'YB'  :  70, 'LU'  :  71, 'HF'  :  72, 'TA'  :  73, 'W'   :  74, 'RE'  :  75, 'OS'  :  76, 'IR'  :  77, 'PT'  :  78, 'AU'  :  79, 'HG'  :  80, 'TL'  :  81, 'PB'  :  82, 'BI'  :  83, 'PO'  :  84, 'AT'  :  85, 'RN'  :  86,
'FR'  :  87, 'RA'  :  88, 'AC'  :  89, 'TH'  :  90, 'PA'  :  91, 'U'   :  92, 'NP'  :  93, 'PU'  :  94, 'AM'  :  95, 'CM'  :  96, 'BK'  :  97, 'CF'  :  98, 'ES'  :  99, 'FM'  : 100, 'MD'  : 101, 'NO'  : 102, 'LR'  : 103, 'RF'  : 104, 'DB'  : 105, 'SG'  : 106, 'BH'  : 107, 'HS'  : 108, 'MT'  : 109, 'DS'  : 110, 'RG'  : 111, 'CP'  : 112, 'UUT' : 113, 'UUQ' : 114, 'UUP' : 115, 'UUH' : 116, 'UUS' : 117, 'UUO' : 118,
}

# True Atom
_atom_re1_ = re.compile(r'^\s*([a-z]+)(\d*)\s+(\S+)\s+(\S+)\s+(\S+)\s*$', re.IGNORECASE)
# Ghost Atom
_atom_re2_ = re.compile(r'^\s*(Gh)\(([a-z]+)(\d*)\)\s+(\S+)\s+(\S+)\s+(\S+)\s*$', re.IGNORECASE)

@staticmethod
def _atom_from_xyz_line(
    line, 
    scale = util.units['bohr_per_ang'],
    ):
    
    mobj = re.match(_atom_re1_,line)
    if mobj:
        symbol = mobj.group(1).upper()
        label  = mobj.group(1) + mobj.group(2)
        try:
            x = float(mobj.group(3)) * scale
            y = float(mobj.group(4)) * scale
            z = float(mobj.group(5)) * scale
        except:
            raise ValueError("XYZ Line has malformed x,y,z coordinates:\n%s" % line)
        if not symbol in _atomic_numbers_:
            raise ValueError("XYZ Line has unrecognized atomic symbol:\n%s" % line)
        N = _atomic_numbers_[symbol]
        Z = float(N)
        return Atom(label, symbol, N, x, y, z, Z, 0)
    
    mobj = re.match(_atom_re2_,line)
    if mobj:
        symbol = mobj.group(2).upper()
        label  = mobj.group(1) + "(" + mobj.group(2) + mobj.group(3) + ")"
        try:
            x = float(mobj.group(4)) * scale
            y = float(mobj.group(5)) * scale
            z = float(mobj.group(6)) * scale
        except:
            raise ValueError("XYZ Line has malformed x,y,z coordinates:\n%s" % line)
        if not symbol in _atomic_numbers_:
            raise ValueError("XYZ Line has unrecognized atomic symbol:\n%s" % line)
        N = _atomic_numbers_[symbol]
        Z = 0.0
        return Atom(label, symbol, N, x, y, z, Z, 0)
            
    raise ValueError("XYZ Line is malformed:\n%s" % line) 

def _atom_str(self):
    return '%-12s %18.12f %18.12f %18.12f' % (
        self.label,
        self.x,
        self.y,
        self.z,
        )
        
Atom = pls.Atom
Atom.from_xyz_line = _atom_from_xyz_line
Atom.__str__ = _atom_str # TODO: Move this up to C++

# => Molecule Decorator s<= #

@staticmethod
def _molecule_from_xyz_str(
    strval,
    scale = util.units['bohr_per_ang'],
    name = 'mol',
    ):

    lines = re.split('\n', strval)

    # Remove all blank lines (even in XYZ format - doesn't matter)
    lines2 = []
    for line in lines:
        if len(line) and not line.isspace():
            lines2.append(line)
    lines = lines2

    # Check that there are actually atoms
    if len(lines) == 0:
        raise ValueError("No lines in molecule: %s" % name)

    # Sometimes the first line is blank - forgive this
    if len(lines[0]) == 0:
        lines = lines[1:]

    # Sometimes the user puts the number of atoms as the first line - forgive this
    mobj = re.match(r'^\s*(\d+)\s*$', lines[0])
    if mobj:
        lines = lines[1:]

    # Sometimes the user puts the charge/multiplicity as the second line - grab this
    Q = 0.0  # Default
    M = 1.0  # Default
    mobj = re.match(r'^\s*(-?\d+)\s+(\d+)\s*$', lines[0])
    if mobj:
        Q = float(mobj.group(1))
        M = float(mobj.group(2))
        lines = lines[1:]
    # Sometimes the user puts a comment on the second line (defined here as not a valid atom)
    else:
        mobj1 = re.match(_atom_re1_, lines[0])
        mobj2 = re.match(_atom_re2_, lines[0])
        if not mobj1 and not mobj2:
            lines = lines[1:]

    # finally we can parse the atoms and their coordinates
    atoms = [Atom.from_xyz_line(line,scale) for line in lines]
    atoms = [Atom(x.label,x.symbol,x.N,x.x,x.y,x.z,x.Z,xind) for xind, x in enumerate(atoms)]
    mol = Molecule(name, atoms, Q, M)
    return mol

@staticmethod
def _molecule_from_xyz_file(
    filename, 
    scale = util.units['bohr_per_ang'],
    ):

    mobj = re.match(r'^(\S+?)(\.xyz)$', os.path.basename(os.path.normpath(filename)))
    if mobj is None:
        raise ValueError("XYZ File must be of type (PATH/)name.xyz: %s" % filename)
    name = mobj.group(1)
    with open(filename) as fh:
        strval = fh.read()
    return Molecule.from_xyz_str(strval=strval,name=name,scale=scale)

def _molecule_save_xyz_file(
    self,
    filename,
    append=False,
    scale=util.units['ang_per_bohr'],
    label='label', # label or symbol
    comment=None,
    label_str='%-12s',
    coord_str='%18.12f',
    ):

    if label not in ['label', 'symbol']:
        raise ValueError('Invalid label argument')

    if isinstance(filename, file):
        fh = filename
    else:
        fh = open(filename, "a" if append else "w")
    fh.write('%d\n' % (self.natom))
    if comment:
        fh.write('%s\n' % comment)
    else:
        if round(self.charge) == self.charge and round(self.multiplicity) == self.multiplicity:
            fh.write('%d %d\n' % (int(self.charge),int(self.multiplicity)))
        else:
            fh.write('%24.16E %24.16E\n' % (self.charge,self.multiplicity))

    format_str = '%s %s %s %s\n' % (label_str, coord_str, coord_str, coord_str)
    for atom in self.atoms:
        fh.write(format_str % (
            atom.label if label == 'label' else atom.symbol,
            scale*atom.x,
            scale*atom.y,
            scale*atom.z))
    fh.flush()
    
    if not isinstance(filename, file):
        fh.close()

Molecule.from_xyz_str = _molecule_from_xyz_str
Molecule.from_xyz_file = _molecule_from_xyz_file
Molecule.save_xyz_file = _molecule_save_xyz_file

def _molecule_nfrozen_core(
    self, 
    scheme='orca',
    ):

    """ Compute the number of frozen core orbitals (doubly occupied) in this molecule. 

    for each atom in molecule, the atom.N field is used to determine the number
    of frozen core orbitals *unless* the atom.Z field is 0.0 (indicating a
    ghost atom).

    Params:
        scheme (str) - scheme of frozen core orbitals to use.
            'orca' - scheme used in ORCA https://sites.google.com/site/orcainputlibrary/frozen-core-calculations
        
    Returns:
        nfrc (int) - number of doubly-occupied frozen core orbitals in this molecule
    """
    
    schemes = ['orca',]
    if scheme not in schemes: raise RuntimeError("Scheme: %s not in schemes: %r" % (scheme, schemes))

    if scheme == 'orca':
        def nfrzc_fun(N):
            if N <= 0:    return 0  # Gh
            elif N <= 2:  return 0  # H-He
            elif N <= 4:  return 0  # Li-Be
            elif N <= 10: return 1  # B-Ne
            elif N <= 12: return 1  # Na-Mg
            elif N <= 18: return 5  # Al-Ar
            elif N <= 20: return 5  # K-Ca
            elif N <= 30: return 5  # Sc-Zn
            elif N <= 36: return 9  # Ga-Kr
            elif N <= 38: return 9  # Rb-Sr
            elif N <= 48: return 14 # Y-Cd
            elif M <= 54: return 18 # In-Xe
            
    nfrzc = 0
    for atom in self.atoms:
        if atom.Z == 0.0: continue # Frozen
        nfrzc += nfrzc_fun(atom.N)
    return nfrzc

Molecule.nfrozen_core = _molecule_nfrozen_core

# => Angular Momentum <= #

def _angular_momentum_str(self):
    s = 'AngularMomentum: L = %d\n' % self.L
    s += '  Ncartesian = %2d\n' % self.ncart
    s += '  Nspherical = %2d\n' % self.npure
    return s
    
AngularMomentum = pls.AngularMomentum
AngularMomentum.__str__ = _angular_momentum_str # TODO: Move this up to C++

# => Primitive <= #

@staticmethod
def _primitives_from_gbs(
    is_spherical,     # Spherical or Cartesian?
    am,               # Angular momentum
    N,                # Overall normalization
    ws,               # Initial primitive contraction coefficents
    es,               # Primitive exponents
    normalize = True, # Normalize, or just use ws?
    ):

    if normalize:

        pi32 = math.pow(math.pi,1.5)
        twoL = math.pow(2,am)
        dfact = 1
        for l in range(1,am+1):
            dfact *= 2*l-1

        K = len(ws)
        cs = [x for x in ws]
        for k in range(K):
            cs[k] *= math.sqrt(twoL * math.pow(es[k] + es[k],am + 1.5) / (pi32 * dfact))

        V = 0.0
        for k1 in range(K):  
            for k2 in range(K):
                V += math.pow(math.sqrt(4*es[k1]*es[k2])/(es[k1]+es[k2]),am+1.5) * ws[k1] * ws[k2]
        V = math.sqrt(N) * math.pow(V,-1.0/2.0)
        cs = [V * x for x in cs]

    else:
        
        cs = [x for x in ws]

    prims = []
    prim_idx = 0
    for c, e, w in zip(cs, es, ws):
        prims.append(Primitive(
            c, 
            e, 
            0.0, 
            0.0,
            0.0,
            w,
            am,
            is_spherical,
            0,
            0,
            prim_idx,
            0,
            0))
        prim_idx += 1

    return prims

Primitive.from_gbs = _primitives_from_gbs

# => Basis <= #

# Shell name to L
_shell_data_ = {
'S' : 0,
'P' : 1,
'D' : 2,
'F' : 3,
'G' : 4,
'H' : 5,
'I' : 6,
'K' : 7,
'L' : 8,
'M' : 9,
'N' : 10,
'O' : 11,
'Q' : 12,
'R' : 13,
'T' : 14,
'U' : 15,
'V' : 16,
'W' : 17,
'X' : 18,
'Y' : 19,
'Z' : 20,
}

@staticmethod
def _basis_from_gbs_lines(
    molecule, 
    lines,
    name = 'bas', 
    normalize = True, # Normalize?
    spherical = None,
    ):

    # Validity check: molecule must be dense
    for A2, atom in enumerate(molecule.atoms):
        if A2 != atom.atomIdx:
            raise RuntimeError('Basis::from_gbs_lines: cannot parse basis for sparse molecule')

    # Remove comment lines
    lines2 = []
    for line in lines:
        if re.match(r'^\s*$', line):
            continue
        if re.match(r'^\s*!', line):
            continue
        lines2.append(line) 

    # Determine spherical/cartesian from first line in file, unless the user passed in the "spherical" flag
    if (re.match(r'^\s*cartesian\s*$', lines2[0], re.IGNORECASE)):
        spherical2 = False
    elif (re.match(r'^\s*spherical\s*$', lines2[0], re.IGNORECASE)):
        spherical2 = True
    else:
        raise ValueError("GBS File: Where is the cartesian/spherical line?: %s" % name)
    lines2 = lines2[1:]
    if spherical is None:
        spherical = spherical2

    # Find "star lines" with the form "****" separating atom entries
    star_inds = [0]
    for ind in range(len(lines2)):
        line = lines2[ind]
        if re.match('^\s*\*\*\*\*\s*$', line):
            star_inds.append(ind)
    star_inds.append(len(lines2))

    # Extract the lines pertaining to each atom symbol
    atom_gbs = {}
    for k in range(len(star_inds) - 1):
        ind1 = star_inds[k] + 1
        ind2 = star_inds[k+1]
        if (ind2 - ind1) <= 0:
            continue
        mobj = re.match(r'^\s*(\S+)\s+(\d+)\s*$', lines2[ind1])
        if mobj == None:
            raise Exception("Where is the ID V line?")
        if mobj.group(2) != '0':
            continue
        atom_gbs[mobj.group(1).upper()] = lines2[ind1+1:ind2]

    # Extract a list of list of primitives (a list of contracted shells) for each atom symbol
    atom_shells = {}
    for key, atom in atom_gbs.items():
        atom_stuff = []
        
        ind = 0;
        while ind < len(atom):
            mobj = re.match(r'^\s*(\S+)\s+(\d+)\s+(\S+)\s*$', atom[ind])
            if mobj == None:
                raise Exception("Where is the L K N line?")
            ID = mobj.group(1).upper()
            K = int(mobj.group(2))
            N = float(mobj.group(3))
            ind=ind+1
            if ID == 'SPD':
                E  = []
                W0 = []
                W1 = []
                W2 = []
                for k in range(K):
                    mobj = re.match('^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$', re.sub(r'[Dd]', 'E', atom[ind + k]))
                    if mobj == None:
                        raise Exception("Where is the E W0 W1 W2 line?")
                    E.append(float(mobj.group(1)))
                    W0.append(float(mobj.group(2)))
                    W1.append(float(mobj.group(3)))
                    W2.append(float(mobj.group(4)))
                atom_stuff.append(Primitive.from_gbs(spherical,0,N,W0,E,normalize))
                atom_stuff.append(Primitive.from_gbs(spherical,1,N,W1,E,normalize))
                atom_stuff.append(Primitive.from_gbs(spherical,2,N,W2,E,normalize))
                ind=ind+K
            elif ID == 'SP':
                E  = []
                W0 = []
                W1 = []
                for k in range(K):
                    mobj = re.match('^\s*(\S+)\s+(\S+)\s+(\S+)\s*$', re.sub(r'[Dd]', 'E', atom[ind + k]))
                    if mobj == None:
                        raise Exception("Where is the E W0 W1 line?")
                    E.append(float(mobj.group(1)))
                    W0.append(float(mobj.group(2)))
                    W1.append(float(mobj.group(3)))
                atom_stuff.append(Primitive.from_gbs(spherical,0,N,W0,E,normalize))
                atom_stuff.append(Primitive.from_gbs(spherical,1,N,W1,E,normalize))
                ind=ind+K
            else:
                L = _shell_data_[ID]
                E  = []
                W0 = []
                for k in range(K):
                    mobj = re.match('^\s*(\S+)\s+(\S+)\s*$', re.sub(r'[Dd]', 'E', atom[ind + k]))
                    if mobj == None:
                        raise Exception("Where is the E W line?")
                    E.append(float(mobj.group(1)))
                    W0.append(float(mobj.group(2)))
                atom_stuff.append(Primitive.from_gbs(spherical,L,N,W0,E,normalize))
                ind=ind+K
                
        atom_shells[key] = atom_stuff

    # Populate the specifics of this molecule
    prims = []
    nao = 0
    ncart = 0
    nprim = 0
    nshell = 0
    natom = 0
    for A, atom in enumerate(molecule.atoms):
        shells = atom_shells[atom.symbol]
        for P, shell in enumerate(shells):
            Pnao = shell[0].nao
            Pncart = shell[0].ncart
            for K, prim in enumerate(shell):
                prims.append(Primitive(
                    prim.c,
                    prim.e,
                    atom.x,
                    atom.y,
                    atom.z,
                    prim.c0,
                    prim.L,
                    prim.is_pure,
                    nao,
                    ncart,
                    nprim,
                    nshell,
                    natom)) 
                nprim += 1
            nao += Pnao
            ncart += Pncart
            nshell += 1
        natom += 1

    bas = Basis(
        name, 
        prims)

    return bas

@staticmethod
def _basis_from_gbs_str(
    molecule,
    strval,
    **kwargs):

    return Basis.from_gbs_lines(molecule=molecule, lines=strval.split('\n'), **kwargs)

@staticmethod
def _basis_from_gbs_file(
    molecule, 
    filename,
    normalize = True,
    spherical = None, # If None, reads from GBS file, else forces True/False
    ):

    # The following is done to make things easy with the filename:
    # (1) if filename does not end in ".gbs", this is appended
    # (2) if filename (including any path) is a valid file, 
    #     this file is used (local search path)
    # (3) if filename (including any path) is not a valid file,
    #     lightspeed.basisdir + '/' + filename is used
    #     (default search path)
    # (4) if the file produced in (3) is not found, a ValueError is 
    #     raised

    if len(filename) >= 4 and filename[-4:] != '.gbs':
        filename += '.gbs'

    if not os.path.isfile(filename):
        filename = Basis.basisdir + '/' + filename

    if not os.path.isfile(filename):
        raise ValueError("Could not find GBS file: %s" % filename)

    with open(filename) as fh:
        lines = fh.readlines()

    mobj = re.match(r'^(\S+?)(\.gbs)$', os.path.basename(os.path.normpath(filename)))
    name = mobj.group(1)

    return Basis.from_gbs_lines(molecule=molecule, name=name, lines=lines, normalize=normalize, spherical=spherical)
    
Basis.from_gbs_lines = _basis_from_gbs_lines
Basis.from_gbs_str = _basis_from_gbs_str
Basis.from_gbs_file  = _basis_from_gbs_file

# => ECPShell <= #

def _ecp_shell_str(self):
    s = 'ECPShell: L = %d\n' % self.am
    s += '  Highest L  = %r\n' % self.is_max_am
    s += '  Nprimitive = %3d\n' % self.nprimitive
    return s

ECPShell.__str__ = _ecp_shell_str # TODO: Move this up to C++

# => ECPBasis <= #

@staticmethod
def _ecp_basis_set_from_ecp_lines(
    molecule, 
    lines,
    name = 'bas', 
    ):

    # Remove comments and **** lines
    lines2 = []
    for line in lines:
        if re.match(r'^\s*$', line):
            continue
        if re.match(r'^\s*!', line):
            continue
        if re.match(r'^\s*\*\*\*\*', line):
            continue
        lines2.append(line) 

    # Parse all shells/nelecs in file
    atoms_nelecs = {}
    atoms_shells = {}
    starts = [x for x,y in enumerate(lines2) if re.match(r'^\s*(\S+)-ECP\s+(\d+)\s+(\d+)', y, re.IGNORECASE)]
    starts.append(len(lines2))  
    for sind in range(len(starts)-1):
        start  = starts[sind]
        stop   = starts[sind+1]
        lines3 = lines2[start:stop]
        mobj = re.match(r'^\s*(\S+)-ECP\s+(\d+)\s+(\d+)', lines3[0], re.IGNORECASE)
        if mobj is None:
            raise ValueError('ECP: where is the X-ECP Lmax Nelec line? ' + name)
        symbol = mobj.group(1).upper()
        Lmax = int(mobj.group(2))
        nelec = int(mobj.group(3))
        atoms_nelecs[symbol] = nelec
        atoms_shells[symbol] = []
        starts2 = [x for x,y in enumerate(lines3) if re.match(r'^\s*(\S)-?\S* potential', y, re.IGNORECASE)] 
        starts2.append(len(lines3))
        for sind2 in range(len(starts2)-1):
            start2  = starts2[sind2]
            stop2   = starts2[sind2+1]
            pot_line = lines3[start2]
            nprim_line = lines3[start2+1]
            lines4 = lines3[start2+2:stop2]
            mobj = re.match(r'^\s*(\S)-?\S* potential', pot_line, re.IGNORECASE)
            if mobj is None:
                raise ValueError('ECP: Where is the L-ul potential line? ' + name)
            L = _shell_data_[mobj.group(1).upper()]
            mobj = re.match(r'^\s*(\d+)', nprim_line)
            if mobj is None:
                raise ValueError('ECP: Where is the nprimitive line? ' + name)
            nprim = int(mobj.group(1))
            ns = []
            cs = []
            es = []
            for k in range(nprim):
                mobj = re.match(r'^\s*(\d+)\s+(\S+)\s+(\S+)\s*$', lines4[k])
                if mobj is None:
                    raise ValueError('ECP: Where is the N exp coef line? ' + name)
                ns.append(int(mobj.group(1))-2) # -2: See Eq. 2 of McD JCC, 44, 289 (1981).
                es.append(float(mobj.group(2)))
                cs.append(float(mobj.group(3)))
            atoms_shells[symbol].append(ECPShell(
                0.0,
                0.0,
                0.0,
                L,
                (L==Lmax),
                ns,
                cs,
                es,
                0,
                0,
                ))

    shell_vecs = []
    nelec_vecs = []
    nshell2 = 0
    for A in range(molecule.natom):
        atom = molecule.atoms[A]
        nelecs = atoms_nelecs.get(atom.symbol, 0)
        shells = atoms_shells.get(atom.symbol, [])
        for shell in shells:
            gs = ECPShell(
                atom.x,
                atom.y,
                atom.z,
                shell.L,
                shell.is_max_L,
                shell.ns,
                shell.cs,
                shell.es,
                A,
                nshell2,
                )
            nshell2 += 1
            shell_vecs.append(gs)
        nelec_vecs.append(nelecs)
    bas = ECPBasis(name, shell_vecs, nelec_vecs)

    return bas

@staticmethod
def _ecp_basis_set_from_ecp_file(
    molecule, 
    filename,
    ):

    # The following is done to make things easy with the filename:
    # (1) if filename does not end in ".ecp", this is appended
    # (2) if filename (including any path) is a valid file, 
    #     this file is used (local search path)
    # (3) if filename (including any path) is not a valid file,
    #     lightspeed.basisdir + '/' + filename is used
    #     (default search path)
    # (4) if the file produced in (3) is not found, a ValueError is 
    #     raised

    if len(filename) >= 4 and filename[-4:] != '.ecp':
        filename += '.ecp'

    if not os.path.isfile(filename):
        filename = Basis.basisdir + '/' + filename

    if not os.path.isfile(filename):
        raise ValueError("Could not find ECP file: %s" % filename)

    lines = open(filename).readlines();

    mobj = re.match(r'^(\S+?)(\.ecp)$', os.path.basename(os.path.normpath(filename)))
    name = mobj.group(1)

    return ECPBasis.from_ecp_lines(molecule=molecule, name=name, lines=lines)
    
ECPBasis.from_ecp_lines = _ecp_basis_set_from_ecp_lines
ECPBasis.from_ecp_file  = _ecp_basis_set_from_ecp_file

