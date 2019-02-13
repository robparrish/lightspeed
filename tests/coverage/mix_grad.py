#!/usr/bin/env python
import unittest
import collections
import numpy as np
import lightspeed as ls

# => unittest class <= #

class MixGradTest(unittest.TestCase):

    def test_mix_grad(self):

        print("\n=> Mixed Basis Gradient Test <=\n")
        self.assertTrue(run_mix_grad_2('coverage/data/'))

# => Dimer test <= #

def run_mix_grad_2(run_dir):

    tol = 1.0E-09

    res = ls.ResourceList.build(1024**2,1024**2)
    ewald = ls.Ewald([1.0], [-1.0])

    xyz_list = ['h2o', 'nh3']
    bas_list = ['sto-3g', '6-31gs', 'cc-pvdz']

    # molecule list

    mol = []
    for i, xyz_i in enumerate(xyz_list):
        mol.append(ls.Molecule.from_xyz_file(run_dir + xyz_i + '.xyz'))

    molC = ls.Molecule.concatenate(mol,0,1)
    xyzc = molC.xyzZ

    print("  Task: V, J and K gradient")
    print("\n  %-15s%-15s%15s" % \
            ("Mol1", "Mol2", "G_Diff"))
    print("  %s" % ('-'*45))

    # reference npz file

    npzfile = run_dir + "mix"
    for xyzname in xyz_list:
        npzfile += "_" + xyzname
    npzfile += ".npz"

    refs = np.load(npzfile)

    # basis list

    for a, bas_a in enumerate(bas_list):
        bas1 = ls.Basis.from_gbs_file(mol[0], bas_a)
        N1 = bas1.nao
        natom1 = mol[0].natom
        xyzc1 = mol[0].xyzZ

        for b, bas_b in enumerate(bas_list):
            if (b == a): continue
            bas2 = ls.Basis.from_gbs_file(mol[1], bas_b)
            basC = ls.Basis.concatenate([bas1,bas2])
            N2 = bas2.nao
            natom2 = mol[1].natom
            xyzc2 = mol[1].xyzZ

            xyzcC = ls.Molecule.concatenate(mol,0,1).xyzZ

            pairs_11 = ls.PairList.build_schwarz(bas1,bas1,True,1.0E-14)
            pairs_22 = ls.PairList.build_schwarz(bas2,bas2,True,1.0E-14)
            pairs_12 = ls.PairList.build_schwarz(bas1,bas2,False,1.0E-14)
            pairs = ls.PairList.build_schwarz(basC,basC,True,1.0E-14)

            suffix = "_" + bas_a.upper() + "_" + bas_b.upper()

            # read in density matrices of monomers

            D22 = ls.Tensor.zeros((N2,N2))
            D22.np[...] = refs['D1'+suffix][N1:,N1:]

            D11 = ls.Tensor.zeros((N1,N1))
            D11.np[...] = refs['D2'+suffix][:N1,:N1]

            # idiot check to make sure that the D11 and D22 are not zero
            assert(np.max(D22.np) > 0.1)
            assert(np.max(D11.np) > 0.1)

            # D == D1 (+) D2

            D = ls.Tensor.zeros((N1+N2,N1+N2))
            D1 = ls.Tensor.zeros((N1+N2,N1+N2))
            D2 = ls.Tensor.zeros((N1+N2,N1+N2))

            D.np[:N1,:N1] = D11.np[...]
            D.np[N1:,N1:] = D22.np[...]
            D1.np[:N1,:N1] = D11.np[...]
            D2.np[N1:,N1:] = D22.np[...]

            # reference V gradient

            Vgr = ls.IntBox.potentialGrad(
                    res, ewald, pairs, D, xyzcC
                    )

            # reference J and K gradient: (G(D1+D2) - G(D1) - G(D2)) / 2

            Jgr = ls.IntBox.coulombGrad(
                    res, ewald, pairs, D, D, 1.0E-14, 1.0E-14
                    )

            Jgr.np[...] -= ls.IntBox.coulombGrad(
                    res, ewald, pairs, D2, D2, 1.0E-14, 1.0E-14
                    ).np

            Jgr.np[...] -= ls.IntBox.coulombGrad(
                    res, ewald, pairs, D1, D1, 1.0E-14, 1.0E-14
                    ).np

            Jgr.scale(0.5)

            Kgr = ls.IntBox.exchangeGrad(
                    res, ewald, pairs, D, D,
                    True, True, True, 1.0E-14, 1.0E-14
                    )

            Kgr.np[...] -= ls.IntBox.exchangeGrad(
                    res, ewald, pairs, D2, D2,
                    True, True, True, 1.0E-14, 1.0E-14
                    ).np

            Kgr.np[...] -= ls.IntBox.exchangeGrad(
                    res, ewald, pairs, D1, D1,
                    True, True, True, 1.0E-14, 1.0E-14
                    ).np

            Kgr.scale(0.5)

            # calculated V gradient from advanced routine

            vec_v1 = ls.IntBox.potentialGradAdv2(
                    res, ewald, pairs_11, D11, xyzcC,
                    )

            vec_v2 = ls.IntBox.potentialGradAdv2(
                    res, ewald, pairs_22, D22, xyzcC,
                    )

            v12 = ls.Tensor.zeros((natom1+natom2,3))
            v12.np[:natom1,:] += vec_v1[0].np
            v12.np[natom1:,:] += vec_v2[0].np
            v12.np[:,:] += vec_v1[1].np
            v12.np[:,:] += vec_v2[1].np

            # calculated J and K gradient from advanced routine

            vec_j12 = ls.IntBox.coulombGradAdv(
                    res, ewald, pairs_11, pairs_22, D11, D22,
                    1.0E-14, 1.0E-14
                    )

            vec_k12 = ls.IntBox.exchangeGradAdv(
                    res, ewald, pairs_12, pairs_12, D11, D22,
                    True, True, False, 1.0E-14, 1.0E-14
                    )

            j12 = ls.Tensor.zeros((natom1+natom2,3))
            j12.np[:natom1,:] += vec_j12[0].np
            j12.np[:natom1,:] += vec_j12[1].np
            j12.np[natom1:,:] += vec_j12[2].np
            j12.np[natom1:,:] += vec_j12[3].np

            k12 = ls.Tensor.zeros((natom1+natom2,3))
            k12.np[:natom1,:] += vec_k12[0].np
            k12.np[natom1:,:] += vec_k12[1].np
            k12.np[:natom1,:] += vec_k12[2].np
            k12.np[natom1:,:] += vec_k12[3].np

            # compare results

            diff_Vgr = np.max(np.abs(v12.np - Vgr.np))
            diff_Jgr = np.max(np.abs(j12.np - Jgr.np))
            diff_Kgr = np.max(np.abs(k12.np - Kgr.np))
            print("  %-15s%-15s%15.6e" % \
                    (xyz_list[0].upper()+'/'+bas_a, 
                     xyz_list[1].upper()+'/'+bas_b, 
                     max(diff_Vgr, diff_Jgr, diff_Kgr)))

            if (diff_Vgr >= tol):
                print("  %-6s diff=%15.6e\n" % ("Vgr", diff_Vgr))
                return False

            if (diff_Jgr >= tol):
                print("  %-6s diff=%15.6e\n" % ("Jgr", diff_Jgr))
                return False

            if (diff_Kgr >= tol):
                print("  %-6s diff=%15.6e\n" % ("Kgr", diff_Kgr))
                return False

    return True

# => main function <= #

if __name__ == "__main__":
    print("\n=> Mixed Basis Gradient Test <=\n")
    assert(run_mix_grad_2('data/'))
