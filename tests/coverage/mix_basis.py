#!/usr/bin/env python
import unittest
import collections
import numpy as np
import lightspeed as ls

# => unittest class <= #

class MixBasisTest(unittest.TestCase):

    def test_mix_basis(self):

        print("\n=> Mixed Basis Integral Test <=\n")
        self.assertTrue(run_mix_basis_2('coverage/data/'))
        self.assertTrue(run_mix_basis_4('coverage/data/'))

# => Dimer test <= #

def run_mix_basis_2(run_dir):

    tol = 1.0E-12

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

    print("  Task: S,T,V,J,K")
    print("\n  %-15s%-15s%15s" % \
            ("Mol1", "Mol2", "Max_Diff"))
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

        for b, bas_b in enumerate(bas_list):
            if (b == a): continue
            bas2 = ls.Basis.from_gbs_file(mol[1], bas_b)
            basC = ls.Basis.concatenate([bas1,bas2])
            N2 = bas2.nao

            pairs_11 = ls.PairList.build_schwarz(bas1,bas1,True,1.0E-14)
            pairs_12 = ls.PairList.build_schwarz(bas1,bas2,False,1.0E-14)
            pairs_21 = ls.PairList.build_schwarz(bas2,bas1,False,1.0E-14)
            pairs_22 = ls.PairList.build_schwarz(bas2,bas2,True,1.0E-14)

            suffix = "_" + bas_a.upper() + "_" + bas_b.upper()

            # calculate integrals in mixed basis set

            vals = collections.OrderedDict()

            # overlap
            vals['S'+suffix] = ls.Tensor.zeros((N1+N2,N1+N2))
            vals['S'+suffix].np[:N1,:N1] = ls.IntBox.overlap(res, pairs_11).np[:,:]
            vals['S'+suffix].np[:N1,N1:] = ls.IntBox.overlap(res, pairs_12).np[:,:]
            vals['S'+suffix].np[N1:,:N1] = ls.IntBox.overlap(res, pairs_21).np[:,:]
            vals['S'+suffix].np[N1:,N1:] = ls.IntBox.overlap(res, pairs_22).np[:,:]

            # kinetic
            vals['T'+suffix] = ls.Tensor.zeros((N1+N2,N1+N2))
            vals['T'+suffix].np[:N1,:N1] = ls.IntBox.kinetic(res, pairs_11).np[:,:]
            vals['T'+suffix].np[:N1,N1:] = ls.IntBox.kinetic(res, pairs_12).np[:,:]
            vals['T'+suffix].np[N1:,:N1] = ls.IntBox.kinetic(res, pairs_21).np[:,:]
            vals['T'+suffix].np[N1:,N1:] = ls.IntBox.kinetic(res, pairs_22).np[:,:]

            # nuclear potential
            vals['V'+suffix] = ls.Tensor.zeros((N1+N2,N1+N2))
            vals['V'+suffix].np[:N1,:N1] = ls.IntBox.potential(res, ewald, pairs_11, xyzc).np[:,:]
            vals['V'+suffix].np[:N1,N1:] = ls.IntBox.potential(res, ewald, pairs_12, xyzc).np[:,:]
            vals['V'+suffix].np[N1:,:N1] = ls.IntBox.potential(res, ewald, pairs_21, xyzc).np[:,:]
            vals['V'+suffix].np[N1:,N1:] = ls.IntBox.potential(res, ewald, pairs_22, xyzc).np[:,:]

            D12 = ls.Tensor.zeros((N1,N2))
            D12.np[...] = refs['D0'+suffix][:N1,N1:]

            # J0: (11|12), (12|12), (21|12), (22|12)
            vals['J0'+suffix] = ls.Tensor.zeros((N1+N2,N1+N2))
            vals['J0'+suffix].np[:N1,:N1] = ls.IntBox.coulomb(
                    res, ewald, pairs_11, pairs_12, D12, 1.0E-14, 1.0E-14
                    ).np[:,:]
            vals['J0'+suffix].np[:N1,N1:] = ls.IntBox.coulomb(
                    res, ewald, pairs_12, pairs_12, D12, 1.0E-14, 1.0E-14
                    ).np[:,:]
            vals['J0'+suffix].np[N1:,:N1] = ls.IntBox.coulomb(
                    res, ewald, pairs_21, pairs_12, D12, 1.0E-14, 1.0E-14
                    ).np[:,:]
            vals['J0'+suffix].np[N1:,N1:] = ls.IntBox.coulomb(
                    res, ewald, pairs_22, pairs_12, D12, 1.0E-14, 1.0E-14
                    ).np[:,:]

            # K0: (11|12), (11|22), (21|12), (21|22)
            vals['K0'+suffix] = ls.Tensor.zeros((N1+N2,N1+N2))
            vals['K0'+suffix].np[:N1,:N1] = ls.IntBox.exchange(
                    res, ewald, pairs_11, pairs_12, D12, False, 1.0E-14, 1.0E-14
                    ).np[:,:]
            vals['K0'+suffix].np[:N1,N1:] = ls.IntBox.exchange(
                    res, ewald, pairs_11, pairs_22, D12, False, 1.0E-14, 1.0E-14
                    ).np[:,:]
            vals['K0'+suffix].np[N1:,:N1] = ls.IntBox.exchange(
                    res, ewald, pairs_21, pairs_12, D12, False, 1.0E-14, 1.0E-14
                    ).np[:,:]
            vals['K0'+suffix].np[N1:,N1:] = ls.IntBox.exchange(
                    res, ewald, pairs_21, pairs_22, D12, False, 1.0E-14, 1.0E-14
                    ).np[:,:]

            D22 = ls.Tensor.zeros((N2,N2))
            D22.np[...] = refs['D1'+suffix][N1:,N1:]

            # J1: (11|22), (12|22), (21|22), (22|22)
            vals['J1'+suffix] = ls.Tensor.zeros((N1+N2,N1+N2))
            vals['J1'+suffix].np[:N1,:N1] = ls.IntBox.coulomb(
                    res, ewald, pairs_11, pairs_22, D22, 1.0E-14, 1.0E-14
                    ).np[:,:]
            vals['J1'+suffix].np[:N1,N1:] = ls.IntBox.coulomb(
                    res, ewald, pairs_12, pairs_22, D22, 1.0E-14, 1.0E-14
                    ).np[:,:]
            vals['J1'+suffix].np[N1:,:N1] = ls.IntBox.coulomb(
                    res, ewald, pairs_21, pairs_22, D22, 1.0E-14, 1.0E-14
                    ).np[:,:]
            vals['J1'+suffix].np[N1:,N1:] = ls.IntBox.coulomb(
                    res, ewald, pairs_22, pairs_22, D22, 1.0E-14, 1.0E-14
                    ).np[:,:]

            # K1: (12|12), (12|22), (22|12), (22|22)
            vals['K1'+suffix] = ls.Tensor.zeros((N1+N2,N1+N2))
            vals['K1'+suffix].np[:N1,:N1] = ls.IntBox.exchange(
                    res, ewald, pairs_12, pairs_12, D22, True, 1.0E-14, 1.0E-14
                    ).np[:,:]
            vals['K1'+suffix].np[:N1,N1:] = ls.IntBox.exchange(
                    res, ewald, pairs_12, pairs_22, D22, True, 1.0E-14, 1.0E-14
                    ).np[:,:]
            vals['K1'+suffix].np[N1:,:N1] = ls.IntBox.exchange(
                    res, ewald, pairs_22, pairs_12, D22, True, 1.0E-14, 1.0E-14
                    ).np[:,:]
            vals['K1'+suffix].np[N1:,N1:] = ls.IntBox.exchange(
                    res, ewald, pairs_22, pairs_22, D22, True, 1.0E-14, 1.0E-14
                    ).np[:,:]

            D11 = ls.Tensor.zeros((N1,N1))
            D11.np[...] = refs['D2'+suffix][:N1,:N1]

            # J2: (11|11), (12|11), (21|11), (22|11)
            vals['J2'+suffix] = ls.Tensor.zeros((N1+N2,N1+N2))
            vals['J2'+suffix].np[:N1,:N1] = ls.IntBox.coulomb(
                    res, ewald, pairs_11, pairs_11, D11, 1.0E-14, 1.0E-14
                    ).np[:,:]
            vals['J2'+suffix].np[:N1,N1:] = ls.IntBox.coulomb(
                    res, ewald, pairs_12, pairs_11, D11, 1.0E-14, 1.0E-14
                    ).np[:,:]
            vals['J2'+suffix].np[N1:,:N1] = ls.IntBox.coulomb(
                    res, ewald, pairs_21, pairs_11, D11, 1.0E-14, 1.0E-14
                    ).np[:,:]
            vals['J2'+suffix].np[N1:,N1:] = ls.IntBox.coulomb(
                    res, ewald, pairs_22, pairs_11, D11, 1.0E-14, 1.0E-14
                    ).np[:,:]

            # K2: (11|11), (11|21), (21|11), (21|21)
            vals['K2'+suffix] = ls.Tensor.zeros((N1+N2,N1+N2))
            vals['K2'+suffix].np[:N1,:N1] = ls.IntBox.exchange(
                    res, ewald, pairs_11, pairs_11, D11, True, 1.0E-14, 1.0E-14
                    ).np[:,:]
            vals['K2'+suffix].np[:N1,N1:] = ls.IntBox.exchange(
                    res, ewald, pairs_11, pairs_21, D11, True, 1.0E-14, 1.0E-14
                    ).np[:,:]
            vals['K2'+suffix].np[N1:,:N1] = ls.IntBox.exchange(
                    res, ewald, pairs_21, pairs_11, D11, True, 1.0E-14, 1.0E-14
                    ).np[:,:]
            vals['K2'+suffix].np[N1:,N1:] = ls.IntBox.exchange(
                    res, ewald, pairs_21, pairs_21, D11, True, 1.0E-14, 1.0E-14
                    ).np[:,:]

            D21 = ls.Tensor.zeros((N2,N1))
            D21.np[...] = refs['D3'+suffix][N1:,:N1]

            # J3: (11|21), (12|21), (21|21), (22|21)
            vals['J3'+suffix] = ls.Tensor.zeros((N1+N2,N1+N2))
            vals['J3'+suffix].np[:N1,:N1] = ls.IntBox.coulomb(
                    res, ewald, pairs_11, pairs_21, D21, 1.0E-14, 1.0E-14
                    ).np[:,:]
            vals['J3'+suffix].np[:N1,N1:] = ls.IntBox.coulomb(
                    res, ewald, pairs_12, pairs_21, D21, 1.0E-14, 1.0E-14
                    ).np[:,:]
            vals['J3'+suffix].np[N1:,:N1] = ls.IntBox.coulomb(
                    res, ewald, pairs_21, pairs_21, D21, 1.0E-14, 1.0E-14
                    ).np[:,:]
            vals['J3'+suffix].np[N1:,N1:] = ls.IntBox.coulomb(
                    res, ewald, pairs_22, pairs_21, D21, 1.0E-14, 1.0E-14
                    ).np[:,:]

            # K3: (12|11), (12|21), (22|11), (22|21)
            vals['K3'+suffix] = ls.Tensor.zeros((N1+N2,N1+N2))
            vals['K3'+suffix].np[:N1,:N1] = ls.IntBox.exchange(
                    res, ewald, pairs_12, pairs_11, D21, False, 1.0E-14, 1.0E-14
                    ).np[:,:]
            vals['K3'+suffix].np[:N1,N1:] = ls.IntBox.exchange(
                    res, ewald, pairs_12, pairs_21, D21, False, 1.0E-14, 1.0E-14
                    ).np[:,:]
            vals['K3'+suffix].np[N1:,:N1] = ls.IntBox.exchange(
                    res, ewald, pairs_22, pairs_11, D21, False, 1.0E-14, 1.0E-14
                    ).np[:,:]
            vals['K3'+suffix].np[N1:,N1:] = ls.IntBox.exchange(
                    res, ewald, pairs_22, pairs_21, D21, False, 1.0E-14, 1.0E-14
                    ).np[:,:]

            # compare with reference

            max_diff = 0.0
            error_str = "\n"
            for key, val in list(vals.items()):
                ref = ls.Tensor.array(refs[key])
                diff = np.max(np.abs(val.np - ref.np))
                if (diff > max_diff):
                    max_diff = diff
                if (diff >= tol):
                    error_str += "  %-6s diff=%15.6e\n" % (key, diff)

            print("  %-15s%-15s%15.6e" % \
                    (xyz_list[0].upper()+'/'+bas_a, 
                     xyz_list[1].upper()+'/'+bas_b, 
                     max_diff))

            if (max_diff >= tol): 
                print(error_str)
                return False

    return True

# => Tetramer test <= #

def run_mix_basis_4(run_dir):

    tol = 1.0E-12

    res = ls.ResourceList.build(1024**2,1024**2)
    ewald = ls.Ewald([1.0], [-1.0])

    xyz_list = ['h2o', 'nh3', 'ch4', 'c2h4']
    bas_list = ['sto-3g', '6-31gs', 'cc-pvdz', '3-21g']

    # molecule list

    mol = []
    for i, xyz_i in enumerate(xyz_list):
        mol.append(ls.Molecule.from_xyz_file(run_dir + xyz_i + '.xyz'))

    molC = ls.Molecule.concatenate(mol,0,1)
    xyzc = molC.xyzZ

    print("")
    print("  Task: J,K")
    print("\n  %-12s%-12s%-12s%-12s%15s" % \
            ("Mol1", "Mol2", "Mol3", "Mol4", "Max_Diff"))
    print("  %s" % ('-'*63))

    # reference npz file

    npzfile = run_dir + "mix"
    for xyzname in xyz_list:
        npzfile += "_" + xyzname
    npzfile += ".npz"

    refs = np.load(npzfile)

    # basis list

    bas_list_2 = bas_list + bas_list

    for ind in range(len(bas_list)):

        new_bas_list = bas_list_2[ind:ind+len(xyz_list)]

        bas = []
        N = []
        for i, bas_i in enumerate(new_bas_list):
            bas.append(ls.Basis.from_gbs_file(mol[i], bas_i))
            N.append(bas[i].nao)

        N1 = N[0]
        N2 = N[0] + N[1]
        N3 = N[0] + N[1] + N[2]
        N4 = N[0] + N[1] + N[2] + N[3]

        basC = ls.Basis.concatenate(bas)

        pairs_11 = ls.PairList.build_schwarz(bas[0],bas[0],True,1.0E-14)
        pairs_22 = ls.PairList.build_schwarz(bas[1],bas[1],True,1.0E-14)
        pairs_33 = ls.PairList.build_schwarz(bas[2],bas[2],True,1.0E-14)

        pairs_12 = ls.PairList.build_schwarz(bas[0],bas[1],False,1.0E-14)
        pairs_21 = ls.PairList.build_schwarz(bas[1],bas[0],False,1.0E-14)
        pairs_13 = ls.PairList.build_schwarz(bas[0],bas[2],False,1.0E-14)
        pairs_31 = ls.PairList.build_schwarz(bas[2],bas[0],False,1.0E-14)
        pairs_23 = ls.PairList.build_schwarz(bas[1],bas[2],False,1.0E-14)
        pairs_32 = ls.PairList.build_schwarz(bas[2],bas[1],False,1.0E-14)
        pairs_34 = ls.PairList.build_schwarz(bas[2],bas[3],False,1.0E-14)
        pairs_42 = ls.PairList.build_schwarz(bas[3],bas[1],False,1.0E-14)

        suffix = ""
        for basname in new_bas_list:
            suffix += "_" + basname.upper()

        # calculate integrals in mixed basis set

        vals = collections.OrderedDict()

        D11 = ls.Tensor.array(refs['D11'+suffix])
        D12 = ls.Tensor.array(refs['D12'+suffix])

        # D11 (11|pq) => Jpq
        vals['J1111'+suffix] = ls.IntBox.coulomb(
                res, ewald, pairs_11, pairs_11, D11, 1.0E-14, 1.0E-14
                )
        vals['J1112'+suffix] = ls.IntBox.coulomb(
                res, ewald, pairs_12, pairs_11, D11, 1.0E-14, 1.0E-14
                )
        vals['J1122'+suffix] = ls.IntBox.coulomb(
                res, ewald, pairs_22, pairs_11, D11, 1.0E-14, 1.0E-14
                )
        vals['J1123'+suffix] = ls.IntBox.coulomb(
                res, ewald, pairs_23, pairs_11, D11, 1.0E-14, 1.0E-14
                )

        # D12 (12|pq) => Jpq
        vals['J1234'+suffix] = ls.IntBox.coulomb(
                res, ewald, pairs_34, pairs_12, D12, 1.0E-14, 1.0E-14
                )
        vals['J1233'+suffix] = ls.IntBox.coulomb(
                res, ewald, pairs_33, pairs_12, D12, 1.0E-14, 1.0E-14
                )
        vals['J1223'+suffix] = ls.IntBox.coulomb(
                res, ewald, pairs_23, pairs_12, D12, 1.0E-14, 1.0E-14
                )
        vals['J1222'+suffix] = ls.IntBox.coulomb(
                res, ewald, pairs_22, pairs_12, D12, 1.0E-14, 1.0E-14
                )
        vals['J1213'+suffix] = ls.IntBox.coulomb(
                res, ewald, pairs_13, pairs_12, D12, 1.0E-14, 1.0E-14
                )
        vals['J1212'+suffix] = ls.IntBox.coulomb(
                res, ewald, pairs_12, pairs_12, D12, 1.0E-14, 1.0E-14
                )

        # D11 (1p|1q) => Kpq
        vals['K1111'+suffix] = ls.IntBox.exchange(
                res, ewald, pairs_11, pairs_11, D11, True, 1.0E-14, 1.0E-14
                )
        vals['K1112'+suffix] = ls.IntBox.exchange(
                res, ewald, pairs_11, pairs_21, D11, True, 1.0E-14, 1.0E-14
                )
        vals['K1122'+suffix] = ls.IntBox.exchange(
                res, ewald, pairs_21, pairs_21, D11, True, 1.0E-14, 1.0E-14
                )
        vals['K1123'+suffix] = ls.IntBox.exchange(
                res, ewald, pairs_21, pairs_31, D11, True, 1.0E-14, 1.0E-14
                )

        # D12 (1p|2q) => Kpq
        vals['K1234'+suffix] = ls.IntBox.exchange(
                res, ewald, pairs_31, pairs_42, D12, False, 1.0E-14, 1.0E-14
                )
        vals['K1233'+suffix] = ls.IntBox.exchange(
                res, ewald, pairs_31, pairs_32, D12, False, 1.0E-14, 1.0E-14
                )
        vals['K1223'+suffix] = ls.IntBox.exchange(
                res, ewald, pairs_21, pairs_32, D12, False, 1.0E-14, 1.0E-14
                )
        vals['K1222'+suffix] = ls.IntBox.exchange(
                res, ewald, pairs_21, pairs_22, D12, False, 1.0E-14, 1.0E-14
                )
        vals['K1213'+suffix] = ls.IntBox.exchange(
                res, ewald, pairs_11, pairs_32, D12, False, 1.0E-14, 1.0E-14
                )
        vals['K1212'+suffix] = ls.IntBox.exchange(
                res, ewald, pairs_11, pairs_22, D12, False, 1.0E-14, 1.0E-14
                )

        # compare with reference

        max_diff = 0.0
        error_str = "\n"
        for key, val in list(vals.items()):
            ref = ls.Tensor.array(refs[key])
            diff = np.max(np.abs(val.np - ref.np))
            if (diff > max_diff):
                max_diff = diff
            if (diff >= tol):
                error_str += "  %-6s diff=%15.6e\n" % (key, diff)

        print("  %-12s%-12s%-12s%-12s%15.6e" % \
                (xyz_list[0].upper() + '/' + new_bas_list[0],
                 xyz_list[1].upper() + '/' + new_bas_list[1],
                 xyz_list[2].upper() + '/' + new_bas_list[2],
                 xyz_list[3].upper() + '/' + new_bas_list[3], 
                 max_diff))

        if (max_diff >= tol): 
            print(error_str)
            return False

    return True

# => main function <= #

if __name__ == "__main__":
    print("\n=> Mixed Basis Integral Test <=\n")
    assert(run_mix_basis_2('data/'))
    assert(run_mix_basis_4('data/'))
