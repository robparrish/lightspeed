#!/usr/bin/env python
import lightspeed as ls
import numpy as np

mol = ls.Molecule.from_xyz_str("""
    O 0.000000000000  0.000000000000 -0.129476890157
    H 0.000000000000 -1.494186750504  1.027446102928
    H 0.000000000000  1.494186750504  1.027446102928""", 
    name='h2o',
    scale=1.0)

bas = ls.Basis.from_gbs_file(mol, 'cc-pvdz')

minao = ls.Basis.from_gbs_file(mol, 'cc-pvdz-minao')

res = ls.ResourceList.build_cpu()

nocc = ls.SAD.sad_nocc_neutral(mol)
print(nocc)

Q = ls.Tensor((3,))
Q[0] = 4.5
Q[1] = 0.5
Q[2] = 0.5

nocc = ls.SAD.sad_nocc(mol,Q)
print(nocc)

print(ls.SAD.sad_nocc_atoms())

S = ls.IntBox.overlap(
    res,
    ls.PairList.build_schwarz(bas,bas,True,1.0E-14))
print(S)

C = ls.SAD.sad_orbitals(
    res,
    nocc,
    minao,
    bas)
print(C)

D = ls.Tensor.chain([C,C],[False,True])
print(D)

print(np.sum(S[...] * D[...]))

C = ls.SAD.orbitals(
    res,
    mol,
    bas,
    minao)
print(C)

D = ls.Tensor.chain([C,C],[False,True])
print(D)

print(np.sum(S[...] * D[...]))

C = ls.SAD.orbitals(
    res,
    mol,
    bas,
    minao,
    Qtotal=11.0/2)
print(C)

D = ls.Tensor.chain([C,C],[False,True])
print(D)

print(np.sum(S[...] * D[...]))

C = ls.SAD.orbitals(
    res,
    mol,
    bas,
    minao,
    Qatom=ls.Tensor.array([4.2,0.4,0.4]))
print(C)

D = ls.Tensor.chain([C,C],[False,True])
print(D)

print(np.sum(S[...] * D[...]))

C = ls.SAD.orbitals(
    res,
    mol,
    bas,
    minao,
    Qocc=ls.Tensor.array([1.0,0.75,0.75,0.75,0.75,0.5,0.5]))
print(C)

D = ls.Tensor.chain([C,C],[False,True])
print(D)

print(np.sum(S[...] * D[...]))



