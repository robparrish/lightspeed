from . import lightspeed as pls
import functools

LebedevGrid = pls.LebedevGrid
RadialGrid = pls.RadialGrid
AtomGrid = pls.AtomGrid
BeckeGrid = pls.BeckeGrid

@staticmethod
def _becke_buildSG1(
    resources,
    mol,
    atomic_scheme = 'FLAT',
    ):
    
    # Standard Grid 1 (SG-1)
    #
    # Reference: P.M.W. Gill, B.G. Johnson, J.A. Pople, 
    #   Chem. Phys. Lett., 209(5-6), 506 (1993)
    #
    # Note that there is some ambiguity on the pruning scheme
    # in this paper (edge points), but the computed grids
    # match those reported in the SG-0 paper:
    #   H-He:  3752 points per atom
    #   Li-Ne: 3816 points per atom
    #   Na-Ar: 3760 points per atom
    # This removes the ambiguity about radial pruning scheme
    # and also validates the HANDY radial grids
    #
    # Note that SG-1 is defined only to Ar. For deeper atoms
    # we have elected to retain the overall structure of SG-1,
    # with 50-point HANDY grids (scaled to the usual HANDY
    # radii) and 194-point spherical grids (no pruning is used)
    #   K-Db:  9700 points per atom
    #  

    orient = pls.Tensor((3,3))
    orient.identity()
    
    radial_table = {
        'H'  : functools.partial(RadialGrid.build, "HANDY", 50, 1.0000),
        'HE' : functools.partial(RadialGrid.build, "HANDY", 50, 0.5882),
        'LI' : functools.partial(RadialGrid.build, "HANDY", 50, 3.0769),
        'BE' : functools.partial(RadialGrid.build, "HANDY", 50, 2.0513),
        'B'  : functools.partial(RadialGrid.build, "HANDY", 50, 1.5385),
        'C'  : functools.partial(RadialGrid.build, "HANDY", 50, 1.2308),
        'N'  : functools.partial(RadialGrid.build, "HANDY", 50, 1.0256),
        'O'  : functools.partial(RadialGrid.build, "HANDY", 50, 0.8791),
        'F'  : functools.partial(RadialGrid.build, "HANDY", 50, 0.7692),
        'NE' : functools.partial(RadialGrid.build, "HANDY", 50, 0.6838),
        'NA' : functools.partial(RadialGrid.build, "HANDY", 50, 4.0909),
        'MG' : functools.partial(RadialGrid.build, "HANDY", 50, 3.1579),
        'AL' : functools.partial(RadialGrid.build, "HANDY", 50, 2.5714),
        'SI' : functools.partial(RadialGrid.build, "HANDY", 50, 2.1687),
        'P'  : functools.partial(RadialGrid.build, "HANDY", 50, 1.8750),
        'S'  : functools.partial(RadialGrid.build, "HANDY", 50, 1.6514),
        'CL' : functools.partial(RadialGrid.build, "HANDY", 50, 1.4754),
        'AR' : functools.partial(RadialGrid.build, "HANDY", 50, 1.3333),
        }    

    spherical_table = {
        'H'  : [(6,16), (38,5), (86,4), (194,9), (86,16)],
        'HE' : [(6,16), (38,5), (86,4), (194,9), (86,16)],
        'LI' : [(6,14), (38,7), (86,3), (194,9), (86,17)],
        'BE' : [(6,14), (38,7), (86,3), (194,9), (86,17)],
        'B'  : [(6,14), (38,7), (86,3), (194,9), (86,17)],
        'C'  : [(6,14), (38,7), (86,3), (194,9), (86,17)],
        'N'  : [(6,14), (38,7), (86,3), (194,9), (86,17)],
        'O'  : [(6,14), (38,7), (86,3), (194,9), (86,17)],
        'F'  : [(6,14), (38,7), (86,3), (194,9), (86,17)],
        'NE' : [(6,14), (38,7), (86,3), (194,9), (86,17)],
        'NA' : [(6,12), (38,7), (86,5), (194,7), (86,19)],
        'MG' : [(6,12), (38,7), (86,5), (194,7), (86,19)],
        'AL' : [(6,12), (38,7), (86,5), (194,7), (86,19)],
        'SI' : [(6,12), (38,7), (86,5), (194,7), (86,19)],
        'P'  : [(6,12), (38,7), (86,5), (194,7), (86,19)],
        'S'  : [(6,12), (38,7), (86,5), (194,7), (86,19)],
        'CL' : [(6,12), (38,7), (86,5), (194,7), (86,19)],
        'AR' : [(6,12), (38,7), (86,5), (194,7), (86,19)],
        }
    
    radial_default = functools.partial(RadialGrid.build_by_N, "HANDY", 50)
    spherical_default = [(194,50)]

    atoms = []
    for A in range(mol.natom):
        ID = mol.atoms[A].symbol
        N = mol.atoms[A].N
        xc = mol.atoms[A].x
        yc = mol.atoms[A].y
        zc = mol.atoms[A].z
        #orient = orient.orientation(A)
        if ID in radial_table:
            rad = radial_table[ID]()
            spheres = []
            for item in spherical_table[ID]:
                spheres += [LebedevGrid.build(item[0]) for ind in range(item[1])]
        else:
            rad = radial_default(N)
            spheres = []
            for item in spherical_default:
                spheres += [LebedevGrid.build(item[0]) for ind in range(item[1])]
        atoms.append(AtomGrid(N,xc,yc,zc,orient,rad,spheres))

    becke = BeckeGrid(resources,"SG-1",atomic_scheme,atoms)
    return becke

@staticmethod
def _becke_buildSG0(
    resources,
    mol,
    atomic_scheme = 'FLAT',
    ):
    
    # Standard Grid 0 (SG-0)
    #
    # Reference: 
    #  S-H. Chien, P.M.W. Gill, J. Comput. Chem., 27, 730 (2006) 
    #
    # The pruning scheme is a bit laborious to type out, so the
    # total number of points per atom has been checked by script
    # against the above. Total points per atom:
    #   H:  1406 points per atom
    #   Li: 1406 points per atom
    #   Be: 1390 points per atom
    #   B:  1426 points per atom
    #   C:  1390 points per atom
    #   N:  1414 points per atom
    #   O:  1154 points per atom
    #   F:  1494 points per atom
    #   Na: 1328 points per atom
    #   Mg: 1468 points per atom - note that there is a mistake in
    #   Al: 1496 points per atom   Chien and Gill for Mg in the 
    #   Si: 1496 points per atom   Ntot table.
    #   P:  1496 points per atom
    #   S:  1456 points per atom
    #   Cl: 1480 points per atom
    # 
    # Note: He, Ne, and Ar are retained with their SG-1 parametrization
    #   He: 3752 points per atom
    #   Ne: 3816 points per atom
    #   Ar: 3760 points per atom
    #
    # Note: Atoms past Ar are retained with their SG-1 guess (see SG-1.grid)
    #   K-Db:  9700 points per atom
    #

    orient = pls.Tensor((3,3))
    orient.identity()
    
    radial_table = {
        'H'  : functools.partial(RadialGrid.build, "MULTIEXP", 23, 1.30),
        'LI' : functools.partial(RadialGrid.build, "MULTIEXP", 23, 1.95),
        'BE' : functools.partial(RadialGrid.build, "MULTIEXP", 23, 2.20),
        'B'  : functools.partial(RadialGrid.build, "MULTIEXP", 23, 1.45),
        'C'  : functools.partial(RadialGrid.build, "MULTIEXP", 23, 1.20),
        'N'  : functools.partial(RadialGrid.build, "MULTIEXP", 23, 1.10),
        'O'  : functools.partial(RadialGrid.build, "MULTIEXP", 23, 1.10),
        'F'  : functools.partial(RadialGrid.build, "MULTIEXP", 23, 1.20),
        'NA' : functools.partial(RadialGrid.build, "MULTIEXP", 26, 2.30), 
        'MG' : functools.partial(RadialGrid.build, "MULTIEXP", 26, 2.20),
        'AL' : functools.partial(RadialGrid.build, "MULTIEXP", 26, 2.10),
        'SI' : functools.partial(RadialGrid.build, "MULTIEXP", 26, 1.30),
        'P'  : functools.partial(RadialGrid.build, "MULTIEXP", 26, 1.30),
        'S'  : functools.partial(RadialGrid.build, "MULTIEXP", 26, 1.10),
        'CL' : functools.partial(RadialGrid.build, "MULTIEXP", 26, 1.45), 
        'HE' : functools.partial(RadialGrid.build, "HANDY", 50, 0.5882),  
        'NE' : functools.partial(RadialGrid.build, "HANDY", 50, 0.6838),  
        'AR' : functools.partial(RadialGrid.build, "HANDY", 50, 1.3333),
        }    

    spherical_table = {
        'H'  : [(6,6), ( 18,3), ( 26,1), ( 38,1), ( 74,1), ( 110,1), ( 146,6), ( 86,1), ( 50,1), ( 38,1), ( 18,1), ],
        'LI' : [(6,6), ( 18,3), ( 26,1), ( 38,1), ( 74,1), ( 110,1), ( 146,6), ( 86,1), ( 50,1), ( 38,1), ( 18,1), ],
        'BE' : [(6,4), ( 18,2), ( 26,1), ( 38,2), ( 74,1), ( 86,1), ( 110,2), ( 146,5), ( 50,1), ( 38,1), ( 18,1), ( 6,2), ],
        'B'  : [(6,4), ( 26,4), ( 38,3), ( 86,3), ( 146,6), ( 38,1), ( 6,2), ],
        'C'  : [(6,6), ( 18,2), ( 26,1), ( 38,2), ( 50,2), ( 86,1), ( 110,1), ( 146,1), ( 170,2), ( 146,2), ( 86,1), ( 38,1), ( 18,1), ],
        'N'  : [(6,6), ( 18,3), ( 26,1), ( 38,2), ( 74,2), ( 110,1), ( 170,2), ( 146,3), ( 86,1), ( 50,2), ],
        'O'  : [(6,5), ( 18,1), ( 26,2), ( 38,1), ( 50,4), ( 86,1), ( 110,5), ( 86,1), ( 50,1), ( 38,1), ( 6,1), ],
        'F'  : [(6,4), ( 38,2), ( 50,4), ( 74,2), ( 110,2), ( 146,2), ( 110,2), ( 86,3), ( 50,1), ( 6,1), ],
        'NA' : [(6,6), ( 18,2), ( 26,3), ( 38,1), ( 50,2), ( 110,8), ( 74,2), ( 6,2), ],
        'MG' : [(6,5), ( 18,2), ( 26,2), ( 38,2), ( 50,2), ( 74,1), ( 110,2), ( 146,4), ( 110,1), ( 86,1), ( 38,2), ( 18,1), ( 6,1), ],
        'AL' : [(6,6), ( 18,2), ( 26,1), ( 38,2), ( 50,2), ( 74,1), ( 86,1), ( 146,2), ( 170,2), ( 110,2), ( 86,1), ( 74,1), ( 26,1), ( 18,1), ( 6,1), ],
        'SI' : [(6,5), ( 18,4), ( 38,4), ( 50,3), ( 74,1), ( 110,2), ( 146,1), ( 170,3), ( 86,1), ( 50,1), ( 6,1), ],
        'P'  : [(6,5), ( 18,4), ( 38,4), ( 50,3), ( 74,1), ( 110,2), ( 146,1), ( 170,3), ( 86,1), ( 50,1), ( 6,1), ],
        'S'  : [(6,4), ( 18,1), ( 26,8), ( 38,2), ( 50,1), ( 74,2), ( 110,1), ( 170,3), ( 146,1), ( 110,1), ( 50,1), ( 6,1), ],
        'CL' : [(6,4), ( 18,7), ( 26,2), ( 38,2), ( 50,1), ( 74,1), ( 110,2), ( 170,3), ( 146,1), ( 110,1), ( 86,1), ( 6,1), ],
        'HE' : [(6,16), ( 38,5), ( 86,4), ( 194,9), ( 86,16), ],
        'NE' : [(6,14), ( 38,7), ( 86,3), ( 194,9), ( 86,17), ],
        'AR' : [(6,12), ( 38,7), ( 86,5), ( 194,7), ( 86,19), ],
        }
    
    radial_default = functools.partial(RadialGrid.build_by_N, "HANDY", 50)
    spherical_default = [(194,50)]

    atoms = []
    for A in range(mol.natom):
        ID = mol.atoms[A].symbol
        N = mol.atoms[A].N
        xc = mol.atoms[A].x
        yc = mol.atoms[A].y
        zc = mol.atoms[A].z
        #orient = orient.orientation(A)
        if ID in radial_table:
            rad = radial_table[ID]()
            spheres = []
            for item in spherical_table[ID]:
                spheres += [LebedevGrid.build(item[0]) for ind in range(item[1])]
        else:
            rad = radial_default(N)
            spheres = []
            for item in spherical_default:
                spheres += [LebedevGrid.build(item[0]) for ind in range(item[1])]
        atoms.append(AtomGrid(N,xc,yc,zc,orient,rad,spheres))

    becke = BeckeGrid(resources,"SG-0",atomic_scheme,atoms)
    return becke

@staticmethod
def _becke_build_simple(
    resources,
    mol,
    atomic_scheme = 'FLAT',
    radial_scheme = 'AHLRICHS',
    nradial = 99,
    nspherical = 302, 
    ):

    orient = pls.Tensor((3,3))
    orient.identity()

    atoms = []
    for A in range(mol.natom):
        ID = mol.atoms[A].symbol
        N = mol.atoms[A].N
        xc = mol.atoms[A].x
        yc = mol.atoms[A].y
        zc = mol.atoms[A].z
        #orient = orient.orientation(A)
        rad = RadialGrid.build_by_N(radial_scheme, nradial, N)
        spheres = [LebedevGrid.build(nspherical) for ind in range(nradial)]
        atoms.append(AtomGrid(N,xc,yc,zc,orient,rad,spheres))
    
    becke = BeckeGrid(resources,"Custom",atomic_scheme,atoms)
    return becke

@staticmethod
def _becke_build(
    resources,
    mol,
    options,
    ):

    if options['dft_grid_name'] == 'SG0':
        return BeckeGrid.buildSG0(
            resources,
            mol,
            options['dft_grid_atomic_scheme'],
            )
    elif options['dft_grid_name'] == 'SG1':
        return BeckeGrid.buildSG1(
            resources,
            mol,
            options['dft_grid_atomic_scheme'],
            )
    else:
        return BeckeGrid.build_simple(
            resources,
            mol,
            options['dft_grid_atomic_scheme'],
            options['dft_grid_radial_scheme'],
            options['dft_grid_nradial'],
            options['dft_grid_nspherical'],
            )

BeckeGrid.buildSG1 = _becke_buildSG1
BeckeGrid.buildSG0 = _becke_buildSG0
BeckeGrid.build_simple = _becke_build_simple
BeckeGrid.build = _becke_build
