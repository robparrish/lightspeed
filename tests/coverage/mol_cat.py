#!/usr/bin/env python
import lightspeed as ls
import unittest

class MolCatTest(unittest.TestCase):

    def test_mol_cat(self):

        mol = ls.Molecule.from_xyz_str("""
            O 0.000000000000  0.000000000000 -0.129476890157
            H 0.000000000000 -1.494186750504  1.027446102928
            H 0.000000000000  1.494186750504  1.027446102928""", 
            name='h2o',
            scale=1.0)

        molC = ls.Molecule.concatenate([mol,mol,mol],0,1)

        mol2 = molC.subset([0,1,2],0,1)
        mol3 = molC.subset([3,4,5],0,1)
        mol4 = molC.subset([6,7,8],0,1)

        print("\n=> Mol_cat_test <=")

        self.assertTrue(ls.Molecule.equivalent(mol,mol2))
        self.assertTrue(ls.Molecule.equivalent(mol,mol3))
        self.assertTrue(ls.Molecule.equivalent(mol,mol4))

        print("   passed!")

    def test_mol_xyz(self):
        # test for the reading in of xyz files with frame labels

        mol1 = ls.Molecule.from_xyz_str("""
            O 0.000000000000  0.000000000000 -0.129476890157
            H 0.000000000000 -1.494186750504  1.027446102928
            H 0.000000000000  1.494186750504  1.027446102928""",
            name='h2o',
            scale=1.0)

        mol2 = ls.Molecule.from_xyz_str("""
            3
            FRame 96
            O 0.000000000000  0.000000000000 -0.129476890157
            H 0.000000000000 -1.494186750504  1.027446102928
            H 0.000000000000  1.494186750504  1.027446102928""",
            name='h2o',
            scale=1.0)

        self.assertTrue(ls.Molecule.equivalent(mol1, mol2))

        print("   passed!")
