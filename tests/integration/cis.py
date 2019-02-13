import lightspeed as ls
import est
import numpy as np
import collections
import unittest

# => unittest class <= #

class CISTest(unittest.TestCase):

    def test_cis(self):

        self.assertTrue(run_cis('integration/data/', 'hbdi.xyz', '6-31gss', 'cis.data.npz'))

# => CIS <= #

def run_cis(run_dir, run_xyz, run_basis, ref_npz):

    start_time = ls.timings_header()

    resources = ls.ResourceList.build()

    molecule = ls.Molecule.from_xyz_file(run_dir + run_xyz)

    geom = est.Geometry.build(
        resources=resources,
        molecule=molecule,
        basisname=run_basis,
        )

    ref = est.RHF.from_options(
        geometry=geom,
        g_convergence=1.0E-6,
        cphf_r_convergence=1.0E-6,
        )
    ref.compute_energy()

    cis = est.CIS.from_options(
        reference=ref,
        S_inds=[0, 2],
        S_nstates=[3, 3],
        )
    cis.compute_energy()

    ES = ls.Tensor.array(cis.evals[0])
    ET = ls.Tensor.array(cis.evals[2])

    print('S0 Gradient:\n')
    GS0 = cis.compute_gradient(0, 0)
    GS0.name = 'S0 Gradient'
    print(GS0)

    print('S1 Gradient:\n')
    GS1 = cis.compute_gradient(0, 1)
    GS1.name = 'S1 Gradient'
    print(GS1)

    print('T1 Gradient:\n')
    GT1 = cis.compute_gradient(2, 1)
    GT1.name = 'T1 Gradient'
    print(GT1)

    overlap_time = ls.timings_header("Overlap")
    MSS = est.CIS.compute_overlap(
        cis,
        cis,
        0)
    print(MSS)
    MTT = est.CIS.compute_overlap(
        cis,
        cis,
        2)
    print(MTT)
    ls.timings_footer(overlap_time, "Overlap")

    ls.timings_footer(start_time)

    # Save reference data
    if False:
        np.savez(
            run_dir + ref_npz,
            Enuc=ref.scalars['Enuc'],
            Eext=ref.scalars['Eext'],
            Escf=ref.scalars['Escf'],
            ES=ES,
            ET=ET,
            OS=cis.oscillator_strengths[0],
            OT=cis.oscillator_strengths[2],
            D=ref.tensors['D'],
            F=ref.tensors['F'],
            # C=ref.tensors['C'], # Need phase rules
            eps=ref.tensors['eps'],
            n=ref.tensors['n'],
            GS0=GS0,
            GS1=GS1,
            GT1=GT1,
            )
        
    # Check reference data
    data = np.load(run_dir + ref_npz)
    checks = collections.OrderedDict()
    checks['Enuc'] = [(x, x < 1.0E-10) for x in [np.max(np.abs(data['Enuc'] - ref.scalars['Enuc']))]][0]
    checks['Eext'] = [(x, x < 1.0E-10) for x in [np.max(np.abs(data['Eext'] - ref.scalars['Eext']))]][0]
    checks['Escf'] = [(x, x < 1.0E-9) for x in [np.max(np.abs(data['Escf'] - ref.scalars['Escf']))]][0]
    checks['ES'] = [(x, x < 1.0E-9) for x in [np.max(np.abs(data['ES'] - ES[...]))]][0]
    checks['ET'] = [(x, x < 1.0E-9) for x in [np.max(np.abs(data['ET'] - ET[...]))]][0]
    checks['OS'] = [(x, x < 1.0E-8) for x in [np.max(np.abs(data['OS'] - cis.oscillator_strengths[0][...]))]][0]
    checks['OT'] = [(x, x < 1.0E-8) for x in [np.max(np.abs(data['OT'] - cis.oscillator_strengths[2][...]))]][0]
    checks['D'] = [(x, x < 1.0E-8) for x in [np.max(np.abs(data['D'] - ref.tensors['D'][...]))]][0]
    checks['F'] = [(x, x < 1.0E-8) for x in [np.max(np.abs(data['F'] - ref.tensors['F'][...]))]][0]
    # checks['C'] = [(x, x < 1.0E-8) for x in [np.max(np.abs(data['C'] - ref.tensors['C'][...]))]][0]
    checks['eps'] = [(x, x < 1.0E-6) for x in [np.max(np.abs(data['eps'] - ref.tensors['eps'][...]))]][0]
    checks['n'] = [(x, x < 1.0E-8) for x in [np.max(np.abs(data['n'] - ref.tensors['n'][...]))]][0]
    checks['GS0'] = [(x, x < 1.0E-8) for x in [np.max(np.abs(data['GS0'] - GS0[...]))]][0]
    checks['GS1'] = [(x, x < 1.0E-8) for x in [np.max(np.abs(data['GS1'] - GS1[...]))]][0]
    checks['GT1'] = [(x, x < 1.0E-8) for x in [np.max(np.abs(data['GT1'] - GT1[...]))]][0]

    print('Test Results:')
    for key, value in checks.items():
        print('%-20s: %11.3E %s' % (
            key,
            value[0],
            'OK' if value[1] else 'BAD',
            ))
    return all(x[1] for x in list(checks.values()))

# => main function <= #

if __name__ == "__main__":
    assert(run_cis('data/', 'hbdi.xyz', '6-31gss', 'cis.data.npz'))
