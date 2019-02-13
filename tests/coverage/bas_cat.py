#!/usr/bin/env python
import lightspeed as ls
import unittest

class BasCatTest(unittest.TestCase):

    def test_bas_cat(self):

        mol = ls.Molecule.from_xyz_str("""
            O 0.000000000000  0.000000000000 -0.129476890157
            H 0.000000000000 -1.494186750504  1.027446102928
            H 0.000000000000  1.494186750504  1.027446102928""", 
            name='h2o',
            scale=1.0)

        bas = ls.Basis.from_gbs_file(mol, 'cc-pvdz')

        basC = ls.Basis.concatenate([bas,bas,bas])

        bas2 = basC.subset([0,1,2])
        bas3 = basC.subset([3,4,5])
        bas4 = basC.subset([6,7,8])

        print("\n=> Bas_cat_test <=")

        self.assertTrue(ls.Basis.equivalent(bas,bas2))
        self.assertTrue(ls.Basis.equivalent(bas,bas3))
        self.assertTrue(ls.Basis.equivalent(bas,bas4))

        print("   passed!")
