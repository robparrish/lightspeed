import os
import re
import sys
import numpy as np
import lightspeed as ls
import psiw

rootdir = os.path.dirname(os.path.realpath(__file__))
pymoldir = rootdir + '/pymol'

# => Molden File Parser <= #

def read_molden_file(
    filename,
    full_normalization=True, # Is the Molden file in fully normalized convention (default)
    ):

    lines = open(filename).readlines()

    re_tag = re.compile(r'^\s*\[(\S+)\]')
    tags = []
    tag_inds = []
    for lind, line in enumerate(lines):
        mobj = re.match(re_tag,line)
        if mobj:
            tags.append(mobj.group(1))
            tag_inds.append(lind)
    tag_inds.append(len(lines))

    sections = {}
    for tag_index, tag in enumerate(tags):
        sections[tag] = lines[tag_inds[tag_index]:tag_inds[tag_index+1]]

    # [5D] [5D7F] [9G] Tags
    pure = False
    if any([x in sections for x in ['5D', '5D7F', '9G']]):  
        pure = True
    if any([x in sections for x in ['5D10F', '7F']]):
        raise ValueError('Cannot handle mixed cartesian/spherical')

    # [Atoms] Section to Molecule
    if 'Atoms' not in sections:
        raise ValueError('No [Atoms] Section')
    atom_lines = sections['Atoms']
    units_tag = re.match(r'^\s*\[Atoms\]\s+(\S+)', atom_lines[0]).group(1)
    if units_tag == 'AU':
        units_scale = 1.0
    elif units_tag == 'Angs':
        units_scale = ls.units['bohr_per_ang']
    else:
        raise ValueError('Unknown units')
    re_atom = re.compile(r'^\s*(\S+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$')
    atom_strs = []
    for line in atom_lines[1:]:
        mobj = re.match(re_atom, line)
        atom_strs.append('%s %s %s %s' % (
            mobj.group(1),
            mobj.group(4),
            mobj.group(5),
            mobj.group(6),
        ))
    mol = ls.Molecule.from_xyz_str('\n'.join(atom_strs), scale=units_scale, name='molden')

    # [GTO] Section to Basis
    if 'GTO' not in sections:
        raise ValueError('No [GTO] Section')
    basis_lines = sections['GTO'][1:]
    basis_lines = [x for x in basis_lines if len(x.strip())]
    am_map = { 's' : 0, 'p' : 1, 'd' : 2, 'f' : 3, 'g' : 4 }
    re_basis_atom = re.compile(r'^\s*(\d+)\s+0\s*$')
    re_shell = re.compile(r'^\s*(\S+)\s+(\d+)\s+1\.00\s*$')
    re_prim = re.compile(r'^\s*(\S+)\s+(\S+)\s*$')
    basis_atoms = []
    basis_atom_inds = []
    for lind, line in enumerate(basis_lines):
        mobj = re.match(re_basis_atom, line)
        if mobj:
            basis_atoms.append(int(mobj.group(1)))
            basis_atom_inds.append(lind)
    basis_atom_inds.append(len(basis_lines))
    if basis_atoms != list(range(1,len(atom_strs)+1)):
        raise ValueError('GTO section does not have all atoms present')
    prims = []
    nao = 0 
    ncart = 0
    nprim = 0   
    nshell = 0
    for atom_ind, atom in enumerate(mol.atoms):
        atom_lines = basis_lines[basis_atom_inds[atom_ind]+1:basis_atom_inds[atom_ind+1]] 
        shell_ams = []
        shell_nprims = []
        shell_inds = [] 
        for lind, line in enumerate(atom_lines):
            mobj = re.match(re_shell, line)
            if mobj:
                shell_ams.append(am_map[mobj.group(1)])
                shell_nprims.append(int(mobj.group(2)))
                shell_inds.append(lind)
        shell_inds.append(len(atom_lines))
        for shell_ind, shell_am in enumerate(shell_ams):
            shell_lines = atom_lines[shell_inds[shell_ind]+1:shell_inds[shell_ind+1]]
            if len(shell_lines) != shell_nprims[shell_ind]:
                raise ValueError('Number of primitive lines does not equal promised number')
            es = []
            ws = []
            for prim_ind, prim_line in enumerate(shell_lines):
                mobj = re.match(re_prim, prim_line)
                e = float(mobj.group(1))
                w = float(mobj.group(2))
                es.append(e)
                ws.append(w)
            prim_template = ls.Primitive.from_gbs(
                pure,
                shell_am,
                1.0,
                ws,
                es,
                )
            for prim_ind, prim in enumerate(prim_template):
                prims.append(ls.Primitive(
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
                    atom_ind,
                    ))
                nprim += 1
            nao += prim_template[0].nao
            ncart += prim_template[0].ncart
            nshell += 1
    basis = ls.Basis('molden', prims) 

    # [MO] Section to C, n, eps
    if 'MO' not in sections:
        raise ValueError('No [MO] Section')
    mo_lines = sections['MO']
    
    eps = []
    re_eps = re.compile(r'^\s*Ene=\s*(\S+)\s*$')
    for line in mo_lines:
        mobj = re.match(re_eps, line)
        if mobj:
            eps.append(float(mobj.group(1)))
    occ = []
    re_occ = re.compile(r'^\s*Occup=\s*(\S+)\s*$')
    for line in mo_lines:
        mobj = re.match(re_occ, line)
        if mobj:
            occ.append(float(mobj.group(1)))
    spin = []
    re_spin = re.compile(r'^\s*Spin=\s*(\S+)\s*$')
    for line in mo_lines:
        mobj = re.match(re_spin, line)
        if mobj:
            spin.append(mobj.group(1))
    if len(eps) != len(occ) or len(eps) != len(spin):
        raise ValueError('Missing fields in MO lines')

    eps = ls.Tensor.array(eps)
    occ = ls.Tensor.array(occ)

    if not all([x == 'Alpha' for x in spin]) and not all([x == 'Beta' for x in spin]):
        raise ValueError('Spin must be all alpha or all beta for now') 
            
    inds = []
    Cs = []
    re_mo = re.compile(r'^\s*(\d+)\s+(\S+)\s*$') 
    for line in mo_lines:
        mobj = re.match(re_mo,line)
        if mobj:
            inds.append(int(mobj.group(1)))
            Cs.append(float(mobj.group(2)))
    nbf = max(inds)
    nmo = len(inds) // nbf    
    if nbf * nmo != len(inds):
        raise ValueError('Inconsistent number of MO entries')
    if inds != list(range(1,nbf+1)) * nmo:
        raise ValueError('MO entries are not well sorted')
    C1 = np.reshape(np.array(Cs), (nmo, nbf)).T

    if pure:
        # Only p shells need to be reordered
        C2 = np.array(C1)
        for shell in basis.shells:
            off = shell.aoIdx
            if shell.L == 1:
                C2[off+0,:] = C1[off+2,:] # z
                C2[off+1,:] = C1[off+0,:] # x
                C2[off+2,:] = C1[off+1,:] # y
    else:
        # Only d, f, g shells need to be reordered
        C2 = np.array(C1)
        for shell in basis.shells:
            off = shell.aoIdx
            if shell.L == 2:
                C2[off+0,:] = C1[off+0,:] # xx
                C2[off+1,:] = C1[off+3,:] # xy
                C2[off+2,:] = C1[off+4,:] # xz
                C2[off+3,:] = C1[off+1,:] # yy
                C2[off+4,:] = C1[off+5,:] # yz
                C2[off+5,:] = C1[off+2,:] # zz
            if shell.L == 3:
                C2[off+0,:] = C1[off+0,:] # xxx
                C2[off+1,:] = C1[off+4,:] # xxy
                C2[off+2,:] = C1[off+5,:] # xxz
                C2[off+3,:] = C1[off+3,:] # xyy
                C2[off+4,:] = C1[off+9,:] # xyz
                C2[off+5,:] = C1[off+6,:] # xzz
                C2[off+6,:] = C1[off+1,:] # yyy
                C2[off+7,:] = C1[off+8,:] # yyz
                C2[off+8,:] = C1[off+7,:] # yzz
                C2[off+9,:] = C1[off+2,:] # zzz
            if shell.L == 3:
                C2[off+ 0,:] = C1[off+ 0,:] # xxxx
                C2[off+ 1,:] = C1[off+ 3,:] # xxxy
                C2[off+ 2,:] = C1[off+ 4,:] # xxxz
                C2[off+ 3,:] = C1[off+ 9,:] # xxyy
                C2[off+ 4,:] = C1[off+12,:] # xxyz
                C2[off+ 5,:] = C1[off+10,:] # xxzz
                C2[off+ 6,:] = C1[off+ 5,:] # xyyy
                C2[off+ 7,:] = C1[off+13,:] # xyyz
                C2[off+ 8,:] = C1[off+14,:] # xyzz
                C2[off+ 9,:] = C1[off+ 7,:] # xzzz
                C2[off+10,:] = C1[off+ 1,:] # yyyy
                C2[off+11,:] = C1[off+ 6,:] # yyyz
                C2[off+12,:] = C1[off+11,:] # yyzz
                C2[off+13,:] = C1[off+ 8,:] # yzzz
                C2[off+14,:] = C1[off+ 2,:] # zzzz

    # Convert to LS CCA Cartesian normalization (bilinear d functions not normalized, etc)
    if full_normalization and not pure:
        for shell in basis.shells:
            off = shell.aoIdx
            if shell.L == 2:
                # C2[off+0,:] *= np.sqrt(1.0)
                C2[off+1,:] *= np.sqrt(3.0)
                C2[off+2,:] *= np.sqrt(3.0)
                # C2[off+3,:] *= np.sqrt(1.0)
                C2[off+4,:] *= np.sqrt(3.0)
                # C2[off+5,:] *= np.sqrt(1.0)
            if shell.L == 3:
                # C2[off+0,:] *= np.sqrt(1.0)
                C2[off+1,:] *= np.sqrt(5.0)
                C2[off+2,:] *= np.sqrt(5.0)
                C2[off+3,:] *= np.sqrt(5.0)
                C2[off+4,:] *= np.srt(15.0)
                C2[off+5,:] *= np.sqrt(5.0)
                # C2[off+6,:] *= np.sqrt(1.0)
                C2[off+7,:] *= np.sqrt(5.0)
                C2[off+8,:] *= np.sqrt(5.0)
                # C2[off+9,:] *= np.sqrt(1.0)
            if shell.L == 4:
                # C2[off+ 0,:] *= np.sqrt(1.0)
                C2[off+ 1,:] *= np.sqrt(7.0)
                C2[off+ 2,:] *= np.sqrt(7.0)
                C2[off+ 3,:] *= np.sqrt(35.0/3.0)
                C2[off+ 4,:] *= np.sqrt(35.0)
                C2[off+ 5,:] *= np.sqrt(35.0/3.0)
                C2[off+ 6,:] *= np.sqrt(7.0)
                C2[off+ 7,:] *= np.sqrt(35.0)
                C2[off+ 8,:] *= np.sqrt(35.0)
                C2[off+ 9,:] *= np.sqrt(7.0)
                # C2[off+10,:] *= np.sqrt(1.0)
                C2[off+11,:] *= np.sqrt(7.0)
                C2[off+12,:] *= np.sqrt(35.0/3.0)
                C2[off+13,:] *= np.sqrt(7.0)
                # C2[off+14,:] *= np.sqrt(1.0)
    
    C = ls.Tensor.array(C2)

    # OPDM
    CT = ls.Tensor.array(C)
    CT[...] *= np.outer(np.ones((C.shape[0],)), occ)
    D = ls.Tensor.chain([C,CT],[False,True])
    D.symmetrize()
            
    return mol, basis, C, eps, occ, D

# => Utility to generate cubes from molden files for common tasks <= #

def density_cube(
    moldenfile,
    cubefile,
    pymol_root=pymoldir, # Location of pymol template files
    grid_spacing=(0.3,0.3,0.3),
    grid_overage=(5.0,5.0,5.0),
    ):

    print('=> Density Cube <=\n')
    print('  Molden File    = %s' % moldenfile)
    print('  Cube File Root = %s' % cubefile)
    print('')

    mol, basis, C, eps, occ, D = read_molden_file(moldenfile)

    resources = ls.ResourceList.build_cpu()
    pairlist = ls.PairList.build_schwarz(basis,basis,True,1.0E-14)

    S = ls.IntBox.overlap(
        resources,
        pairlist)

    print('Density in this molden file: %14.6f\n' % D.vector_dot(S))
    
    # Build a CubicProp helper object
    cube = psiw.CubicProp.build(
        resources=resources,
        molecule=mol,
        basis=basis,
        pairlist=pairlist,
        grid_spacing=grid_spacing,
        grid_overage=grid_overage,
        )
    print(cube.grid)
    # Save a density cube corresponding to Dtot in rho.cube
    cube.save_density_cube(
        cubefile + '.cube', 
        D,
        )

    # Pymol files
    os.system('cp %s/vis.pymol .' % pymol_root)
    os.system('cp %s/orient.pymol .' % pymol_root)
    lines = open('%s/D.pymol' % pymol_root).readlines()
    lines2 = [re.sub('DTAG', '%s' % (cubefile), x) for x in lines]
    open('%s.pymol' % (cubefile),'w').writelines(lines2)

    # Geom.xyz
    mol.save_xyz_file('geom.xyz')

def difference_density_cube(
    moldenfile1,
    moldenfile2,
    cubefile,
    pymol_root=pymoldir, # Location of pymol template files
    grid_spacing=(0.3,0.3,0.3),
    grid_overage=(5.0,5.0,5.0),
    ):

    print('=> Difference Density Cube <=\n')
    print('  Molden File 1  = %s'    % moldenfile1)
    print('  Molden File 2  = %s'    % moldenfile2)
    print('  Cube File Root = %s' % cubefile)
    print('')

    mol1, basis1, C1, eps1, occ1, D1 = read_molden_file(moldenfile1)
    mol2, basis2, C2, eps2, occ2, D2 = read_molden_file(moldenfile2)
    Dd = ls.Tensor.array(D2[...] - D1[...])

    resources = ls.ResourceList.build_cpu()
    pairlist1 = ls.PairList.build_schwarz(basis1,basis1,True,1.0E-14)

    S = ls.IntBox.overlap(
        resources,
        pairlist1)

    print('Density in molden file 1: %14.6f' % D1.vector_dot(S))
    print('Density in molden file 2: %14.6f' % D2.vector_dot(S))
    print('Density in difference:    %14.6f' % Dd.vector_dot(S))
    print('')
    
    # Build a CubicProp helper object
    cube1 = psiw.CubicProp.build(
        resources=resources,
        molecule=mol1,
        basis=basis1,
        pairlist=pairlist1,
        grid_spacing=grid_spacing,
        grid_overage=grid_overage,
        )
    print(cube1.grid)
    # Save a density cube corresponding to Dtot in rho.cube
    cube1.save_density_cube(
        cubefile + '.cube',
        Dd,
        )

    # Pymol files
    os.system('cp %s/vis.pymol .' % pymol_root)
    os.system('cp %s/orient.pymol .' % pymol_root)
    lines = open('%s/D.pymol' % pymol_root).readlines()
    lines2 = [re.sub('DTAG', '%s' % (cubefile), x) for x in lines]
    open('%s.pymol' % (cubefile),'w').writelines(lines2)

    # Geom.xyz
    mol1.save_xyz_file('geom.xyz')

def orbital_cubes(
    moldenfile,
    cubefile,
    inds=[],
    pymol_root=pymoldir, # Location of pymol template files
    grid_spacing=(0.3,0.3,0.3),
    grid_overage=(5.0,5.0,5.0),
    ):

    print('=> Orbital Cubes <=\n')
    print('  Molden File =    %s' % moldenfile)
    print('  Cube File Root = %s' % cubefile)
    print('')

    mol, basis, C, eps, occ, D = read_molden_file(moldenfile)

    print('%5s %5s %14s %14s' % (
        'N',
        'Ind',
        'Occ',
        'Eps',
        ))
    for ind2, ind in enumerate(inds):
        print('%5d %5d %14.6f %14.6f' % (
            ind2,
            ind,
            occ[ind],
            eps[ind],
            ))
    print('')

    resources = ls.ResourceList.build_cpu()
    pairlist = ls.PairList.build_schwarz(basis,basis,True,1.0E-14)

    S = ls.IntBox.overlap(
        resources,
        pairlist)

    print('Density in this molden file: %14.6f\n' % D.vector_dot(S))
    
    cube = psiw.CubicProp.build(
        resources=resources,
        molecule=mol,
        basis=basis,
        pairlist=pairlist,
        grid_spacing=grid_spacing,
        grid_overage=grid_overage,
        )
    print(cube.grid)
    C2 = ls.Tensor.array(C[:,inds])
    cube.save_orbital_cubes(
        cubefile,
        C2,
        )

    # Pymol files
    os.system('cp %s/vis.pymol .' % pymol_root)
    os.system('cp %s/orient.pymol .' % pymol_root)
    lines = open('%s/C.pymol' % pymol_root).readlines()
    for ind in range(len(inds)):
        lines2 = [re.sub('CTAG', '%s_%d' % (cubefile, ind), x) for x in lines]
        open('%s_%d.pymol' % (cubefile, ind),'w').writelines(lines2)
    open('%s.pymol' % cubefile, 'w').writelines(['@%s_%d.pymol\n' % (cubefile, ind) for ind in range(len(inds))])

    # Geom.xyz
    mol.save_xyz_file('geom.xyz')
    
if __name__ == '__main__':

    run = sys.argv[1]
    if run == '--density':
        density_cube(sys.argv[2],sys.argv[3])
    elif run == '--diff':
        difference_density_cube(sys.argv[2],sys.argv[3],sys.argv[4])
    elif run == '--orbitals':
        orbital_cubes(sys.argv[2],sys.argv[3],eval(sys.argv[4]))
