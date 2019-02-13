#!/usr/bin/env python
import unittest
import numpy as np
import lightspeed as ls

# => unittest class <= #

class ChargeTest(unittest.TestCase):

    def test_charge(self):

        print_title()
        self.assertTrue(run_charge('coverage/data/'))

# => compare calculated and reference values <= #

def run_charge(run_dir):

    energy_tol = 1.0E-11
    grad_tol = 1.0E-07

    res = ls.ResourceList.build_cpu()
    ewald = ls.Ewald([1.0], [-1.0])
    ewald_lr = ls.Ewald([0.8], [0.3])
    ewald_sr = ls.Ewald([-0.8, 0.8], [0.3, -1.0])

    xyz_list = ['h2o', 'nh3', 'ch4']

    ref_energy = [
            [8.002367030356384, 4.035198827718412, 2.3666947965666942], 
            [20.55322877327911, 15.492218243839702, 0.9503647747835862], 
            [18.525177050762682, 14.28124517557294, 0.5388964650372082], 
            [11.880867186027105, 5.743556199066088, 3.7611375497555968], 
            [11.715537761726901, 9.367396015926191, 0.005034193455331343], 
            [13.686514602165177, 7.002374264251431, 3.9468374174807144]
            ]

    print("\n  %-7s%-7s%15s%15s" % ("Mol1", "Mol2", "E_Diff", "G_Diff"))
    print("  %s" % ('-'*44))

    calc_energy = ls.Tensor.zeros((len(xyz_list)*(len(xyz_list)+1)//2, 3))

    ind = 0

    for i in range(len(xyz_list)):
        mol_1 = ls.Molecule.from_xyz_file(run_dir + xyz_list[i] + ".xyz")
        xyzc_1 = mol_1.xyzZ

        for j in range(i,len(xyz_list)):
            mol_2 = ls.Molecule.from_xyz_file(run_dir + xyz_list[j] + ".xyz")
            xyzc_2 = mol_2.xyzZ

            # Calculate energies
            if ls.Molecule.equivalent(mol_1, mol_2):
                calc_energy.np[ind,:] = [
                        ls.IntBox.chargeEnergySelf(res, ewald,    xyzc_1),
                        ls.IntBox.chargeEnergySelf(res, ewald_lr, xyzc_1),
                        ls.IntBox.chargeEnergySelf(res, ewald_sr, xyzc_1)
                        ]
            else:
                calc_energy.np[ind,:] = [
                        ls.IntBox.chargeEnergyOther(res, ewald,    xyzc_1, xyzc_2),
                        ls.IntBox.chargeEnergyOther(res, ewald_lr, xyzc_1, xyzc_2),
                        ls.IntBox.chargeEnergyOther(res, ewald_sr, xyzc_1, xyzc_2)
                        ]

            diff_energy = np.max(np.abs(ref_energy[ind][:] - calc_energy.np[ind,:]))
            if (diff_energy >= energy_tol):
                print("\n  %s  %s  diff_energy %16.6e\n" % (xyz_list[i], xyz_list[j], diff_energy))
                return False

            # Calculate gradients
            # Each gradient is calculated twice to test the accumulation
            if (ls.Molecule.equivalent(mol_1, mol_2)):
                for my_ewald in [ewald, ewald_sr, ewald_lr]:
                    ana_grad = ls.IntBox.chargeGradSelf(res, my_ewald, xyzc_1)
                    ana_grad = ls.IntBox.chargeGradSelf(res, my_ewald, xyzc_1, ana_grad)
                    num_grad = run_num_grad_1(res, my_ewald, xyzc_1)
                    # Need the prefactor of 0.5 since each gradient is calculated twice
                    diff_grad = np.max(np.abs(ana_grad.np * 0.5 - num_grad.np))
                    if (diff_grad >= grad_tol):
                        print("\n  %s  diff_grad %16.6e\n" % (xyz_list[i], diff_grad))
                        return False
            else:
                for my_ewald in [ewald, ewald_sr, ewald_lr]:
                    ana_grad = ls.IntBox.chargeGradOther(res, my_ewald, xyzc_1, xyzc_2)
                    ana_grad = ls.IntBox.chargeGradOther(res, my_ewald, xyzc_1, xyzc_2, ana_grad)
                    num_grad = [
                            run_num_grad_2(res, my_ewald, xyzc_1, xyzc_2),
                            run_num_grad_2(res, my_ewald, xyzc_2, xyzc_1)
                            ]
                    # Need the prefactor of 0.5 since each gradient is calculated twice
                    diff_grad = max(
                            np.max(np.abs(ana_grad[0].np * 0.5 - num_grad[0].np)),
                            np.max(np.abs(ana_grad[1].np * 0.5 - num_grad[1].np))
                            )
                    if (diff_grad >= grad_tol):
                        print("\n  %s  %s  diff_grad %16.6e\n" % (xyz_list[i], xyz_list[j], diff_grad))
                        return False

            print("  %-7s%-7s%15.6e%15.6e" % (
                    xyz_list[i].upper(), xyz_list[j].upper(), diff_energy, diff_grad
                    ))

            ind += 1

    return True

def run_num_grad_1(res, ewald, xyzc_1):

    dr = 0.0001

    natom_1 = xyzc_1.shape[0]
    num_grad_1 = ls.Tensor.zeros((natom_1, 3))

    for i in range(natom_1):

        xyzc_1.np[i,0] += dr
        ep = ls.IntBox.chargeEnergySelf(res, ewald, xyzc_1)
        xyzc_1.np[i,0] -= dr * 2.0
        em = ls.IntBox.chargeEnergySelf(res, ewald, xyzc_1)
        num_grad_1.np[i,0] = (ep - em) / (2.0 * dr)
        xyzc_1.np[i,0] += dr

        xyzc_1.np[i,1] += dr
        ep = ls.IntBox.chargeEnergySelf(res, ewald, xyzc_1)
        xyzc_1.np[i,1] -= dr * 2.0
        em = ls.IntBox.chargeEnergySelf(res, ewald, xyzc_1)
        num_grad_1.np[i,1] = (ep - em) / (2.0 * dr)
        xyzc_1.np[i,1] += dr

        xyzc_1.np[i,2] += dr
        ep = ls.IntBox.chargeEnergySelf(res, ewald, xyzc_1)
        xyzc_1.np[i,2] -= dr * 2.0
        em = ls.IntBox.chargeEnergySelf(res, ewald, xyzc_1)
        num_grad_1.np[i,2] = (ep - em) / (2.0 * dr)
        xyzc_1.np[i,2] += dr

    return num_grad_1

def run_num_grad_2(res, ewald, xyzc_1, xyzc_2):

    dr = 0.0001

    natom_1 = xyzc_1.shape[0]
    num_grad_1 = ls.Tensor.zeros((natom_1, 3))

    for i in range(natom_1):

        xyzc_1.np[i,0] += dr
        ep = ls.IntBox.chargeEnergyOther(res, ewald, xyzc_1, xyzc_2)
        xyzc_1.np[i,0] -= dr * 2.0
        em = ls.IntBox.chargeEnergyOther(res, ewald, xyzc_1, xyzc_2)
        num_grad_1.np[i,0] = (ep - em) / (2.0 * dr)
        xyzc_1.np[i,0] += dr

        xyzc_1.np[i,1] += dr
        ep = ls.IntBox.chargeEnergyOther(res, ewald, xyzc_1, xyzc_2)
        xyzc_1.np[i,1] -= dr * 2.0
        em = ls.IntBox.chargeEnergyOther(res, ewald, xyzc_1, xyzc_2)
        num_grad_1.np[i,1] = (ep - em) / (2.0 * dr)
        xyzc_1.np[i,1] += dr

        xyzc_1.np[i,2] += dr
        ep = ls.IntBox.chargeEnergyOther(res, ewald, xyzc_1, xyzc_2)
        xyzc_1.np[i,2] -= dr * 2.0
        em = ls.IntBox.chargeEnergyOther(res, ewald, xyzc_1, xyzc_2)
        num_grad_1.np[i,2] = (ep - em) / (2.0 * dr)
        xyzc_1.np[i,2] += dr

    return num_grad_1

def print_title():

    print("\n=> Charge Test <=\n")
    print("  Task: Energy and Gradient")
    print("  Ewald: FR,LR,SR")

# => main function <= #

if __name__ == "__main__":
    print_title()
    assert(run_charge('data/'))
