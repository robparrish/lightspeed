#!/usr/bin/env python
import unittest
import collections
import numpy as np
import lightspeed as ls

# => unittest class <= #

class EwaldIntTest(unittest.TestCase):

    def test_ewald_ints(self):

        print_title()
        self.assertTrue(run_ewald_int('coverage/data/', 'h2o', 'sto-3g'))
        self.assertTrue(run_ewald_int('coverage/data/', 'h2o', '6-31gs'))
        self.assertTrue(run_ewald_int('coverage/data/', 'h2o', 'cc-pvdz'))

# => compare calculated and reference values <= #

def run_ewald_int(run_dir, run_xyz, run_basis):

    tol = 1.0E-12

    xyzfile = run_dir + run_xyz + ".xyz"

    mol = ls.Molecule.from_xyz_file(xyzfile)
    bas = ls.Basis.from_gbs_file(mol, run_basis)
    res = ls.ResourceList.build(1024**2,1024**2)
    xyzc = mol.xyzZ

    # long-range, short-range, and mixed ewald operators

    ewald_ops = [
            ls.Ewald([1.0],[0.3]),                    # lr
            ls.Ewald([0.7,0.7,0.7],[0.2, 0.1, 0.05]), # lr2
            ls.Ewald([-0.7,0.7],[0.3,-1.0]),          # sr
            ls.Ewald([-0.7,0.7],[-1.0,-1.0]),         # sr2
            ls.Ewald([0.8,0.2],[0.2, -1.0])           # mix
            ]

    ewald_names = ["lr", "lr2", "sr", "sr2", "mix"]

    # reference npz file

    npzfile = run_dir + "ewald_" + run_xyz + "_" + run_basis + ".npz"
    refs = np.load(npzfile)

    max_diff = 0.0
    error_str = "\n"

    # Test both symmetric and nonsymmetric pairlists

    for run in range(2):

        pairlist_sym = (run == 0)
        pairs = ls.PairList.build_schwarz(bas,bas,pairlist_sym,1.0E-14)

        # Test ewald operators

        for ind, ewald in enumerate(ewald_ops):

            suffix = '_' + ewald_names[ind].upper()

            # calculate ewald-related integrals (V,J,K,ESP)

            vals = collections.OrderedDict()

            vals['V' + suffix] = ls.IntBox.potential(res, ewald, pairs, xyzc)

            D = ls.Tensor.array(refs['D' + suffix])
            D2 = ls.Tensor.array(refs['D2' + suffix])

            vals['J' + suffix] = ls.IntBox.coulomb(
                    res, ewald, pairs, pairs, D, 1.0E-14, 1.0E-14
                    )

            vals['J2' + suffix] = ls.IntBox.coulomb(
                    res, ewald, pairs, pairs, D2, 1.0E-14, 1.0E-14
                    )

            vals['K' + suffix] = ls.IntBox.exchange(
                    res, ewald, pairs, pairs, D, True, 1.0E-14, 1.0E-14
                    )

            vals['K2' + suffix] = ls.IntBox.exchange(
                    res, ewald, pairs, pairs, D2, False, 1.0E-14, 1.0E-14
                    )

            XYZ = ls.Tensor.array(refs['XYZ' + suffix])

            vals['ESP' + suffix] = ls.IntBox.esp(res, ewald, pairs, D, XYZ)

            vals['ESP2' + suffix] = ls.IntBox.esp(res, ewald, pairs, D2, XYZ)

            # compare results

            for key, val in list(vals.items()):
                ref = ls.Tensor.array(refs[key])
                diff = np.max(np.abs(val.np - ref.np))
                if (diff > max_diff):
                    max_diff = diff
                if (diff >= tol):
                    error_str += "  %-6s diff=%15.6e\n" % (key, diff)

        symm_str = "yes" if pairlist_sym else "no"
        print("  %-12s %-6s %15.6e" % (run_xyz+'/'+run_basis, symm_str, max_diff))

        if (max_diff >= tol): 
            print(error_str)
            return False

    return True

def print_title():

    print("\n=> Ewald Integral Test <=\n")
    print("  Task: V,J,K,ESP")
    print("  Ewald: LR,SR,MIX")
    print("\n  %-12s %-6s %15s" % ("Mol", "Symm", "Max_Diff"))
    print("  %s" % ('-'*35))

# => main function <= #

if __name__ == "__main__":
    print_title()
    assert(run_ewald_int('data/', 'h2o', 'sto-3g'))
    assert(run_ewald_int('data/', 'h2o', '6-31gs'))
    assert(run_ewald_int('data/', 'h2o', 'cc-pvdz'))
