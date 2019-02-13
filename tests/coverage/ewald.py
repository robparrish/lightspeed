#!/usr/bin/env python
import unittest
import lightspeed as ls

class EwaldTest(unittest.TestCase):

    def test_bad_constructor(self):
            
        with self.assertRaises(RuntimeError):
            ewald = ls.Ewald([],[])
        with self.assertRaises(RuntimeError):
            ewald = ls.Ewald([1.0],[1.0,1.0])

    def test_coulomb(self):
        ewald = ls.Ewald([1.0],[-1.0])
        self.assertTrue(len(ewald.scales) == 1)
        self.assertTrue(len(ewald.omegas) == 1)
        self.assertTrue(ewald.scales[0] == 1.0)
        self.assertTrue(ewald.omegas[0] == -1.0)
        self.assertTrue(ewald.is_coulomb)
        self.assertFalse(ewald.is_sr)
        self.assertFalse(ewald.is_lr)

        with self.assertRaises(IndexError):
            ewald.scales[1]
        with self.assertRaises(IndexError):
            ewald.omegas[1]

        with self.assertRaises(RuntimeError):
            ewald.sr_scale
        with self.assertRaises(RuntimeError):
            ewald.sr_omega
        
        with self.assertRaises(RuntimeError):
            ewald.lr_scale
        with self.assertRaises(RuntimeError):
            ewald.lr_omega
        
    def test_lr(self):
        ewald = ls.Ewald([0.7],[0.3])
        self.assertTrue(len(ewald.scales) == 1)
        self.assertTrue(len(ewald.omegas) == 1)
        self.assertTrue(ewald.scales[0] == 0.7)
        self.assertTrue(ewald.omegas[0] == 0.3)
        self.assertFalse(ewald.is_coulomb)
        self.assertFalse(ewald.is_sr)
        self.assertTrue(ewald.is_lr)

        with self.assertRaises(IndexError):
            ewald.scales[1]
        with self.assertRaises(IndexError):
            ewald.omegas[1]

        with self.assertRaises(RuntimeError):
            ewald.sr_scale
        with self.assertRaises(RuntimeError):
            ewald.sr_omega
        
        self.assertTrue(ewald.lr_scale == 0.7)
        self.assertTrue(ewald.lr_omega == 0.3)
        
    def test_sr(self):
        ewald = ls.Ewald([-0.7,0.7],[0.3,-1.0])
        self.assertTrue(len(ewald.scales) == 2)
        self.assertTrue(len(ewald.omegas) == 2)
        self.assertTrue(ewald.scales[0] == -0.7)
        self.assertTrue(ewald.scales[1] == 0.7)
        self.assertTrue(ewald.omegas[0] == 0.3)
        self.assertTrue(ewald.omegas[1] == -1.0)
        self.assertFalse(ewald.is_coulomb)
        self.assertTrue(ewald.is_sr)
        self.assertFalse(ewald.is_lr)

        with self.assertRaises(IndexError):
            ewald.scales[2]
        with self.assertRaises(IndexError):
            ewald.omegas[2]
        
        self.assertTrue(ewald.sr_scale == 0.7)
        self.assertTrue(ewald.sr_omega == 0.3)

        with self.assertRaises(RuntimeError):
            ewald.lr_scale
        with self.assertRaises(RuntimeError):
            ewald.lr_omega
        
    def test_fake_sr(self):
        ewald = ls.Ewald([-0.7,0.7],[-1.0,-1.0])
        self.assertTrue(len(ewald.scales) == 2)
        self.assertTrue(len(ewald.omegas) == 2)
        self.assertTrue(ewald.scales[0] == -0.7)
        self.assertTrue(ewald.scales[1] == 0.7)
        self.assertTrue(ewald.omegas[0] == -1.0)
        self.assertTrue(ewald.omegas[1] == -1.0)
        self.assertFalse(ewald.is_coulomb)
        self.assertFalse(ewald.is_sr)
        self.assertFalse(ewald.is_lr)

        with self.assertRaises(IndexError):
            ewald.scales[2]
        with self.assertRaises(IndexError):
            ewald.omegas[2]
        
        with self.assertRaises(RuntimeError):
            ewald.sr_scale
        with self.assertRaises(RuntimeError):
            ewald.sr_omega

        with self.assertRaises(RuntimeError):
            ewald.lr_scale
        with self.assertRaises(RuntimeError):
            ewald.lr_omega
        
    def test_gen(self):
        ewald = ls.Ewald([0.7,0.7,0.7],[0.2, 0.1, 0.05])
        self.assertTrue(len(ewald.scales) == 3)
        self.assertTrue(len(ewald.omegas) == 3)
        self.assertTrue(ewald.scales[0] == 0.7)
        self.assertTrue(ewald.scales[1] == 0.7)
        self.assertTrue(ewald.scales[2] == 0.7)
        self.assertTrue(ewald.omegas[0] == 0.2)
        self.assertTrue(ewald.omegas[1] == 0.1)
        self.assertTrue(ewald.omegas[2] == 0.05)
        self.assertFalse(ewald.is_coulomb)
        self.assertFalse(ewald.is_sr)
        self.assertFalse(ewald.is_lr)

        with self.assertRaises(IndexError):
            ewald.scales[3]
        with self.assertRaises(IndexError):
            ewald.omegas[3]
        
        with self.assertRaises(RuntimeError):
            ewald.sr_scale
        with self.assertRaises(RuntimeError):
            ewald.sr_omega

        with self.assertRaises(RuntimeError):
            ewald.lr_scale
        with self.assertRaises(RuntimeError):
            ewald.lr_omega
        
    def test_other(self):
        ewald = ls.Ewald.coulomb()
        self.assertTrue(ewald.is_coulomb)    
