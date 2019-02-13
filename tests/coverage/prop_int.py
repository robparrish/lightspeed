#!/usr/bin/env python
import time
import unittest
import collections
import numpy as np
import lightspeed as ls

# => unittest class <= #

class PropIntTest(unittest.TestCase):

    def test_runs(self):

        print("\n=> Property Integral Test <=")
        self.assertTrue(run_prop_int('coverage/data/', 'h2o', 'sto-3g'))
        self.assertTrue(run_prop_int('coverage/data/', 'h2o', '6-31gs'))
        self.assertTrue(run_prop_int('coverage/data/', 'h2o', 'cc-pvdz'))

# => compare calculated and reference values <= #

def run_prop_int(run_dir, run_xyz, run_basis):

    tol = 1.0E-9

    xyzfile = run_dir + run_xyz + ".xyz"
    npzfile = run_dir + "prop_" + run_xyz + "_" + run_basis + ".npz"

    mol = ls.Molecule.from_xyz_file(xyzfile)
    bas = ls.Basis.from_gbs_file(mol, run_basis)
    res = ls.ResourceList.build(1024**2,1024**2)
    #res = ls.ResourceList.build_cpu()
    ewald = ls.Ewald([1.0], [-1.0])
    xyzc = mol.xyzZ
    xyz = mol.xyz

    print(res)

    vals = []
    timing = []

    for run in range(2):

        pairlist_sym = (run == 0)
        pairs = ls.PairList.build_schwarz(bas,bas,pairlist_sym,1.0E-14)

        # Calculate integrals and gradients
        # Each integral/gradient is calculated twice to test the accumulation

        vals.append(collections.OrderedDict())
        timing.append(collections.OrderedDict())

        t0 = time.time()
        vals[run]['S'] = ls.IntBox.overlap(res, pairs)
        timing[run]['S'] = time.time() - t0
        vals[run]['S'] = ls.IntBox.overlap(res, pairs, vals[run]['S'])

        t0 = time.time()
        vals[run]['T'] = ls.IntBox.kinetic(res, pairs)
        timing[run]['T'] = time.time() - t0
        vals[run]['T'] = ls.IntBox.kinetic(res, pairs, vals[run]['T'])

        t0 = time.time()
        vals[run]['V'] = ls.IntBox.potential(res, ewald, pairs, xyzc)
        timing[run]['V'] = time.time() - t0
        vals[run]['V'] = ls.IntBox.potential(res, ewald, pairs, xyzc, vals[run]['V'])


        D = ls.Tensor.array(np.load(npzfile)['D'])
        D2 = ls.Tensor.array(np.load(npzfile)['D2'])

        t0 = time.time()
        vals[run]['EF'] = ls.IntBox.field(res, ewald, pairs, D, xyz)
        vals[run]['EF'] = ls.IntBox.field(res, ewald, pairs, D, xyz, vals[run]['EF'])

        t0 = time.time()
        vals[run]['J'] = ls.IntBox.coulomb(
                res, ewald, pairs, pairs, D, 1.0E-14, 1.0E-14
                )
        timing[run]['J'] = time.time() - t0
        vals[run]['J'] = ls.IntBox.coulomb(
                res, ewald, pairs, pairs, D, 1.0E-14, 1.0E-14, vals[run]['J']
                )

        t0 = time.time()
        vals[run]['J2'] = ls.IntBox.coulomb(
                res, ewald, pairs, pairs, D2, 1.0E-14, 1.0E-14
                )
        timing[run]['J2'] = time.time() - t0
        vals[run]['J2'] = ls.IntBox.coulomb(
                res, ewald, pairs, pairs, D2, 1.0E-14, 1.0E-14, vals[run]['J2']
                )

        t0 = time.time()
        vals[run]['K'] = ls.IntBox.exchange(
                res, ewald, pairs, pairs, D, True, 1.0E-14, 1.0E-14
                )
        timing[run]['K'] = time.time() - t0
        vals[run]['K'] = ls.IntBox.exchange(
                res, ewald, pairs, pairs, D, True, 1.0E-14, 1.0E-14, vals[run]['K']
                )

        t0 = time.time()
        vals[run]['K2'] = ls.IntBox.exchange(
                res, ewald, pairs, pairs, D2, False, 1.0E-14, 1.0E-14
                )
        timing[run]['K2'] = time.time() - t0
        vals[run]['K2'] = ls.IntBox.exchange(
                res, ewald, pairs, pairs, D2, False, 1.0E-14, 1.0E-14, vals[run]['K2']
                )

        W = ls.Tensor.array(np.load(npzfile)['W'])
        W2 = ls.Tensor.array(np.load(npzfile)['W2'])

        if (pairlist_sym):

            """
            t0 = time.time()
            vals[run]['Vgr'] = ls.IntBox.potentialGrad(res, ewald, pairs, D, xyzc)
            timing[run]['Vgr'] = time.time() - t0
            vals[run]['Vgr'] = ls.IntBox.potentialGrad(res, ewald, pairs, D, xyzc, vals[run]['Vgr'])

            t0 = time.time()
            vals[run]['Vgr2'] = ls.IntBox.potentialGrad(res, ewald, pairs, D2, xyzc)
            timing[run]['Vgr2'] = time.time() - t0
            vals[run]['Vgr2'] = ls.IntBox.potentialGrad(res, ewald, pairs, D2, xyzc, vals[run]['Vgr2'])
            """

            t0 = time.time()
            Vgr = ls.IntBox.potentialGradAdv2(res, ewald, pairs, D, xyzc)
            timing[run]['Vgr'] = time.time() - t0
            Vgr = ls.IntBox.potentialGradAdv2(res, ewald, pairs, D, xyzc, Vgr)
            vals[run]['Vgr'] = Vgr[0].clone()
            vals[run]['Vgr'].np[...] += Vgr[1].np

            t0 = time.time()
            Vgr2 = ls.IntBox.potentialGradAdv2(res, ewald, pairs, D2, xyzc)
            timing[run]['Vgr2'] = time.time() - t0
            Vgr2 = ls.IntBox.potentialGradAdv2(res, ewald, pairs, D2, xyzc, Vgr2)
            vals[run]['Vgr2'] = Vgr2[0].clone()
            vals[run]['Vgr2'].np[...] += Vgr2[1].np

            t0 = time.time()
            vals[run]['Jgr'] = ls.IntBox.coulombGrad(
                    res, ewald, pairs, D, D, 1.0E-14, 1.0E-14
                    )
            timing[run]['Jgr'] = time.time() - t0
            vals[run]['Jgr'] = ls.IntBox.coulombGrad(
                    res, ewald, pairs, D, D, 1.0E-14, 1.0E-14, vals[run]['Jgr']
                    )

            t0 = time.time()
            vals[run]['Jgr2'] = ls.IntBox.coulombGrad(
                    res, ewald, pairs, D2, D2, 1.0E-14, 1.0E-14
                    )
            timing[run]['Jgr2'] = time.time() - t0
            vals[run]['Jgr2'] = ls.IntBox.coulombGrad(
                    res, ewald, pairs, D2, D2, 1.0E-14, 1.0E-14, vals[run]['Jgr2']
                    )

            t0 = time.time()
            vals[run]['Jgr3'] = ls.IntBox.coulombGrad(
                    res, ewald, pairs, D, D2, 1.0E-14, 1.0E-14
                    )
            timing[run]['Jgr3'] = time.time() - t0
            vals[run]['Jgr3'] = ls.IntBox.coulombGrad(
                    res, ewald, pairs, D, D2, 1.0E-14, 1.0E-14, vals[run]['Jgr3']
                    )

            t0 = time.time()
            vals[run]['Jgr4'] = ls.IntBox.coulombGrad(
                    res, ewald, pairs, D2, D, 1.0E-14, 1.0E-14
                    )
            timing[run]['Jgr4'] = time.time() - t0
            vals[run]['Jgr4'] = ls.IntBox.coulombGrad(
                    res, ewald, pairs, D2, D, 1.0E-14, 1.0E-14, vals[run]['Jgr4']
                    )

            t0 = time.time()
            vals[run]['Kgr'] = ls.IntBox.exchangeGrad(
                    res, ewald, pairs, D, D, True, True, True, 1.0E-14, 1.0E-14
                    )
            timing[run]['Kgr'] = time.time() - t0
            vals[run]['Kgr'] = ls.IntBox.exchangeGrad(
                    res, ewald, pairs, D, D, True, True, True, 1.0E-14, 1.0E-14, vals[run]['Kgr']
                    )

            t0 = time.time()
            vals[run]['Kgr2'] = ls.IntBox.exchangeGrad(
                    res, ewald, pairs, D2, D2, False, False, True, 1.0E-14, 1.0E-14
                    )
            timing[run]['Kgr2'] = time.time() - t0
            vals[run]['Kgr2'] = ls.IntBox.exchangeGrad(
                    res, ewald, pairs, D2, D2, False, False, True, 1.0E-14, 1.0E-14, vals[run]['Kgr2']
                    )

            t0 = time.time()
            vals[run]['Kgr3'] = ls.IntBox.exchangeGrad(
                    res, ewald, pairs, D, D2, True, False, False, 1.0E-14, 1.0E-14
                    )
            timing[run]['Kgr3'] = time.time() - t0
            vals[run]['Kgr3'] = ls.IntBox.exchangeGrad(
                    res, ewald, pairs, D, D2, True, False, False, 1.0E-14, 1.0E-14, vals[run]['Kgr3']
                    )

            t0 = time.time()
            vals[run]['Kgr4'] = ls.IntBox.exchangeGrad(
                    res, ewald, pairs, D2, D, False, True, False, 1.0E-14, 1.0E-14
                    )
            timing[run]['Kgr4'] = time.time() - t0
            vals[run]['Kgr4'] = ls.IntBox.exchangeGrad(
                    res, ewald, pairs, D2, D, False, True, False, 1.0E-14, 1.0E-14, vals[run]['Kgr4']
                    )

            t0 = time.time()
            vals[run]['Sgr'] = ls.IntBox.overlapGrad(res, pairs, W)
            timing[run]['Sgr'] = time.time() - t0
            vals[run]['Sgr'] = ls.IntBox.overlapGrad(res, pairs, W, vals[run]['Sgr'])

            t0 = time.time()
            vals[run]['Sgr2'] = ls.IntBox.overlapGrad(res, pairs, W2)
            timing[run]['Sgr2'] = time.time() - t0
            vals[run]['Sgr2'] = ls.IntBox.overlapGrad(res, pairs, W2, vals[run]['Sgr2'])

            t0 = time.time()
            vals[run]['Tgr'] = ls.IntBox.kineticGrad(res, pairs, D)
            timing[run]['Tgr'] = time.time() - t0
            vals[run]['Tgr'] = ls.IntBox.kineticGrad(res, pairs, D, vals[run]['Tgr'])

            t0 = time.time()
            vals[run]['Tgr2'] = ls.IntBox.kineticGrad(res, pairs, D2)
            timing[run]['Tgr2'] = time.time() - t0
            vals[run]['Tgr2'] = ls.IntBox.kineticGrad(res, pairs, D2, vals[run]['Tgr2'])

            t0 = time.time()
            Dipgrad = ls.IntBox.dipoleGrad(res, pairs, 0.0, 0.0 ,0.0, D)
            Dipgrad = ls.IntBox.dipoleGrad(res, pairs, 0.0, 0.0, 0.0, D, Dipgrad)
            vals[run]['DXgr'] = Dipgrad[0] 
            vals[run]['DYgr'] = Dipgrad[1] 
            vals[run]['DZgr'] = Dipgrad[2] 

        else:

            t0 = time.time()
            Vgr = ls.IntBox.potentialGradAdv(res, ewald, pairs, D, xyzc)
            timing[run]['Vgr'] = time.time() - t0
            Vgr = ls.IntBox.potentialGradAdv(res, ewald, pairs, D, xyzc, Vgr)
            vals[run]['Vgr'] = Vgr[0].clone()
            vals[run]['Vgr'].np[...] += Vgr[1].np
            vals[run]['Vgr'].np[...] += Vgr[2].np

            t0 = time.time()
            Vgr2 = ls.IntBox.potentialGradAdv(res, ewald, pairs, D2, xyzc)
            timing[run]['Vgr2'] = time.time() - t0
            Vgr2 = ls.IntBox.potentialGradAdv(res, ewald, pairs, D2, xyzc, Vgr2)
            vals[run]['Vgr2'] = Vgr2[0].clone()
            vals[run]['Vgr2'].np[...] += Vgr2[1].np
            vals[run]['Vgr2'].np[...] += Vgr2[2].np

            # TODO: Coulomb gradient with asymmetric pairlist
            vals[run]['Jgr'] = vals[0]['Jgr'].clone()
            vals[run]['Jgr2'] = vals[0]['Jgr2'].clone()
            vals[run]['Jgr3'] = vals[0]['Jgr3'].clone()
            vals[run]['Jgr4'] = vals[0]['Jgr4'].clone()

            t0 = time.time()
            Kgr = ls.IntBox.exchangeGradAdv(
                    res, ewald, pairs, pairs, D, D, True, True, True, 1.0E-14, 1.0E-14
                    )
            timing[run]['Kgr'] = time.time() - t0
            Kgr = ls.IntBox.exchangeGradAdv(
                    res, ewald, pairs, pairs, D, D, True, True, True, 1.0E-14, 1.0E-14, Kgr
                    )
            vals[run]['Kgr'] = Kgr[0].clone()
            vals[run]['Kgr'].np[...] += Kgr[1].np
            vals[run]['Kgr'].np[...] += Kgr[2].np
            vals[run]['Kgr'].np[...] += Kgr[3].np

            t0 = time.time()
            Kgr2 = ls.IntBox.exchangeGradAdv(
                    res, ewald, pairs, pairs, D2, D2, False, False, True, 1.0E-14, 1.0E-14
                    )
            timing[run]['Kgr2'] = time.time() - t0
            Kgr2 = ls.IntBox.exchangeGradAdv(
                    res, ewald, pairs, pairs, D2, D2, False, False, True, 1.0E-14, 1.0E-14, Kgr2
                    )
            vals[run]['Kgr2'] = Kgr2[0].clone()
            vals[run]['Kgr2'].np[...] += Kgr2[1].np
            vals[run]['Kgr2'].np[...] += Kgr2[2].np
            vals[run]['Kgr2'].np[...] += Kgr2[3].np

            t0 = time.time()
            Kgr3 = ls.IntBox.exchangeGradAdv(
                    res, ewald, pairs, pairs, D, D2, True, False, False, 1.0E-14, 1.0E-14
                    )
            timing[run]['Kgr3'] = time.time() - t0
            Kgr3 = ls.IntBox.exchangeGradAdv(
                    res, ewald, pairs, pairs, D, D2, True, False, False, 1.0E-14, 1.0E-14, Kgr3
                    )
            vals[run]['Kgr3'] = Kgr3[0].clone()
            vals[run]['Kgr3'].np[...] += Kgr3[1].np
            vals[run]['Kgr3'].np[...] += Kgr3[2].np
            vals[run]['Kgr3'].np[...] += Kgr3[3].np

            t0 = time.time()
            Kgr4 = ls.IntBox.exchangeGradAdv(
                    res, ewald, pairs, pairs, D2, D, False, True, False, 1.0E-14, 1.0E-14
                    )
            timing[run]['Kgr4'] = time.time() - t0
            Kgr4 = ls.IntBox.exchangeGradAdv(
                    res, ewald, pairs, pairs, D2, D, False, True, False, 1.0E-14, 1.0E-14, Kgr4
                    )
            vals[run]['Kgr4'] = Kgr4[0].clone()
            vals[run]['Kgr4'].np[...] += Kgr4[1].np
            vals[run]['Kgr4'].np[...] += Kgr4[2].np
            vals[run]['Kgr4'].np[...] += Kgr4[3].np

            t0 = time.time()
            Sgr = ls.IntBox.overlapGradAdv(res, pairs, W)
            timing[run]['Sgr'] = time.time() - t0
            Sgr = ls.IntBox.overlapGradAdv(res, pairs, W, Sgr)
            vals[run]['Sgr'] = Sgr[0].clone()
            vals[run]['Sgr'].np[...] += Sgr[1].np

            t0 = time.time()
            Sgr2 = ls.IntBox.overlapGradAdv(res, pairs, W2)
            timing[run]['Sgr2'] = time.time() - t0
            Sgr2 = ls.IntBox.overlapGradAdv(res, pairs, W2, Sgr2)
            vals[run]['Sgr2'] = Sgr2[0].clone()
            vals[run]['Sgr2'].np[...] += Sgr2[1].np

            t0 = time.time()
            Tgr = ls.IntBox.kineticGradAdv(res, pairs, D)
            timing[run]['Tgr'] = time.time() - t0
            Tgr = ls.IntBox.kineticGradAdv(res, pairs, D, Tgr)
            vals[run]['Tgr'] = Tgr[0].clone()
            vals[run]['Tgr'].np[...] += Tgr[1].np

            t0 = time.time()
            Tgr2 = ls.IntBox.kineticGradAdv(res, pairs, D2)
            timing[run]['Tgr2'] = time.time() - t0
            Tgr2 = ls.IntBox.kineticGradAdv(res, pairs, D2, Tgr2)
            vals[run]['Tgr2'] = Tgr2[0].clone()
            vals[run]['Tgr2'].np[...] += Tgr2[1].np

            t0 = time.time()
            Dipgrad = ls.IntBox.dipoleGradAdv(res, pairs, 0.0, 0.0 ,0.0, D)
            Dipgrad = ls.IntBox.dipoleGradAdv(res, pairs, 0.0, 0.0, 0.0, D, Dipgrad)
            vals[run]['DXgr'] = Dipgrad[0][0].clone()
            vals[run]['DXgr'].np[...] += Dipgrad[0][1]
            vals[run]['DYgr'] = Dipgrad[1][0].clone()
            vals[run]['DYgr'].np[...] += Dipgrad[1][1]
            vals[run]['DZgr'] = Dipgrad[2][0].clone()
            vals[run]['DZgr'].np[...] += Dipgrad[2][1]

        # anti-symmetric overlap gradient
        ASgr = ls.IntBox.overlapGradAdv(res, pairs, W)
        ASgr = ls.IntBox.overlapGradAdv(res, pairs, W, ASgr)
        vals[run]['ASgr'] = ASgr[1]

        ASgr2 = ls.IntBox.overlapGradAdv(res, pairs, W2)
        ASgr2 = ls.IntBox.overlapGradAdv(res, pairs, W2, ASgr2)
        vals[run]['ASgr2'] = ASgr2[1]

        # read in a number of grid points and calculate ESP
        XYZ = ls.Tensor.array(np.load(npzfile)['XYZ'])

        t0 = time.time()
        vals[run]['ESP'] = ls.IntBox.esp(res, ewald, pairs, D, XYZ)
        timing[run]['ESP'] = time.time() - t0
        vals[run]['ESP'] = ls.IntBox.esp(res, ewald, pairs, D, XYZ, vals[run]['ESP'])

        t0 = time.time()
        vals[run]['ESP2'] = ls.IntBox.esp(res, ewald, pairs, D2, XYZ)
        timing[run]['ESP2'] = time.time() - t0
        vals[run]['ESP2'] = ls.IntBox.esp(res, ewald, pairs, D2, XYZ, vals[run]['ESP2'])

        Dipole = ls.IntBox.dipole(res, pairs, 0., 0., 0.)
        Dipole = ls.IntBox.dipole(res, pairs, 0., 0., 0., Dipole)
        vals[run]['DX'] = Dipole[0]
        vals[run]['DY'] = Dipole[1]
        vals[run]['DZ'] = Dipole[2]

        Quadrupole = ls.IntBox.quadrupole(res, pairs, 0., 0., 0.)
        Quadrupole = ls.IntBox.quadrupole(res, pairs, 0., 0., 0., Quadrupole)
        vals[run]['QXX'] = Quadrupole[0]
        vals[run]['QXY'] = Quadrupole[1]
        vals[run]['QXZ'] = Quadrupole[2]
        vals[run]['QYY'] = Quadrupole[3]
        vals[run]['QYZ'] = Quadrupole[4]
        vals[run]['QZZ'] = Quadrupole[5]

        Nabla = ls.IntBox.nabla(res, pairs)
        Nabla = ls.IntBox.nabla(res, pairs, Nabla)
        vals[run]['PX'] = Nabla[0]
        vals[run]['PY'] = Nabla[1]
        vals[run]['PZ'] = Nabla[2]

        AngMom = ls.IntBox.angularMomentum(res, pairs, 0., 0., 0.)
        AngMom = ls.IntBox.angularMomentum(res, pairs, 0., 0., 0., AngMom)
        vals[run]['LX'] = AngMom[0]
        vals[run]['LY'] = AngMom[1]
        vals[run]['LZ'] = AngMom[2]

    # compare results

    print("%-11s %-27s %-25s" % (run_basis, "sym", "non-sym"))
    max_diff = 0.0
    max_diff2 = 0.0
    for key, val in list(vals[0].items()):

        ref = ls.Tensor.array(np.load(npzfile)[key])
        ref[...] *= 2.0
        val2 = vals[1][key]

        # Need the prefactor of 0.5 since each integral/gradient is calculated twice
        diff = np.max(np.abs(val.np - ref.np))
        diff2 = np.max(np.abs(val2.np - ref.np))

        time_str = ""
        if key in timing[0]:
            time_str += "%.0f ms" % (timing[0][key]*1000)
        time_str2 = ""
        if key in timing[1]:
            time_str2 += "%.0f ms" % (timing[1][key]*1000)

        print("  %-6s %15.6e%12s %15.6e%12s" % (key, diff, time_str, diff2, time_str2))
        if (diff > max_diff):
            max_diff = diff
        if (diff2 > max_diff2):
            max_diff2 = diff2
        if (diff >= tol):
            print("")
            ref.name = key + "(ref)"
            val.name = key + "(calc)"
            print(ref)
            print(val)
            print(ls.Tensor.array(ref[...] - val[...]))
        if (diff2 >= tol):
            print("")
            ref.name = key + "(ref)"
            val2.name = key + "(calc)"
            print(ref)
            print(val2)
    print("  %s" % ("-"*50))
    print("  %-6s%16.6e%28.6e" % ("MAX", max_diff, max_diff2))

    return (max_diff < tol and max_diff2 < tol)

# => main function <= #

if __name__ == "__main__":
    print("\n=> Property Integral Test <=")
    assert(run_prop_int('data/', 'h2o', 'sto-3g'))
    assert(run_prop_int('data/', 'h2o', '6-31gs'))
    assert(run_prop_int('data/', 'h2o', 'cc-pvdz'))
