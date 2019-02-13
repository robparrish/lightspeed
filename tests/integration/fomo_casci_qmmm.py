import lightspeed as ls
import est
import numpy as np
import collections
import unittest

# => unittest class <= #

class FOMO_CASCI_Test(unittest.TestCase):

    def test_fomo_casci(self):

        self.assertTrue(run_fomo_casci('integration/data/', 'pyp', '6-31g', 'fomo_casci_qmmm.data.npz'))

# => FOMO-CASCI <= #

def run_fomo_casci(run_dir, run_xyz, run_basis, ref_npz):

    start_time = ls.timings_header()

    resources = ls.ResourceList.build()

    qmmm = est.QMMM.from_prmtop(
        prmtopfile='%s/%s.prmtop' % (run_dir, run_xyz),
        inpcrdfile='%s/%s.rst' % (run_dir, run_xyz),
        qmindsfile='%s/%s.qm' % (run_dir, run_xyz),
        charge=-1.0,
        )

    geom = est.Geometry.build(
        resources=resources,
        qmmm=qmmm,
        basisname=run_basis,
        )

    ref = est.RHF.from_options(
        geometry=geom,
        g_convergence=1.0E-6,
        fomo=True,
        fomo_method='gaussian',
        fomo_temp=0.2,
        fomo_nocc=107,
        fomo_nact=3,
        )
    ref.compute_energy()

    cas = est.CASCI.from_options(
        reference=ref,
        nocc=107,
        nact=3,
        nalpha=2,
        nbeta=2,
        S_inds=[0, 2],
        S_nstates=[3, 3],
        )
    cas.compute_energy()

    ES = ls.Tensor.array(cas.evals[0])
    ET = ls.Tensor.array(cas.evals[2])

    print('S0 Gradient:\n')
    GS0 = cas.compute_gradient(0, 0)
    GS0.name = 'S0 Gradient'
    print(GS0)

    print('S1 Gradient:\n')
    GS1 = cas.compute_gradient(0, 1)
    GS1.name = 'S1 Gradient'
    print(GS1)

    print('T1 Gradient:\n')
    GT1 = cas.compute_gradient(2, 1)
    GT1.name = 'T1 Gradient'
    print(GT1)

    print('S0-S1 Coupling:\n')
    CS0S1 = cas.compute_coupling(0, 0, 1)
    CS0S1.name = 'S0-S1 Coupling'
    print(CS0S1)

    print('T0-T1 Coupling:\n')
    CT0T1 = cas.compute_coupling(0, 0, 1)
    CT0T1.name = 'T0-T1 Coupling'
    print(CT0T1)

    overlap_time = ls.timings_header("Overlap")
    MSS = est.CASCI.compute_overlap(
        cas,
        cas,
        0)
    print(MSS)
    MTT = est.CASCI.compute_overlap(
        cas,
        cas,
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
            TSfomo=ref.scalars['TSfomo'],
            Emu=ref.scalars['Emu'],
            ES=ES,
            ET=ET,
            OS=cas.oscillator_strengths[0],
            OT=cas.oscillator_strengths[2],
            D=ref.tensors['D'],
            F=ref.tensors['F'],
            # C=ref.tensors['C'], # Need phase rules
            eps=ref.tensors['eps'],
            n=ref.tensors['n'],
            GS0=GS0,
            GS1=GS1,
            GT1=GT1,
            CS0S1=CS0S1,
            CT0T1=CT0T1,
            )
        
    # Check reference data
    data = np.load(run_dir + ref_npz)
    checks = collections.OrderedDict()
    checks['Enuc'] = [(x, x < 1.0E-10) for x in [np.max(np.abs(data['Enuc'] - ref.scalars['Enuc']))]][0]
    checks['Eext'] = [(x, x < 1.0E-10) for x in [np.max(np.abs(data['Eext'] - ref.scalars['Eext']))]][0]
    checks['Escf'] = [(x, x < 1.0E-9) for x in [np.max(np.abs(data['Escf'] - ref.scalars['Escf']))]][0]
    checks['TSfomo'] = [(x, x < 1.0E-9) for x in [np.max(np.abs(data['TSfomo'] - ref.scalars['TSfomo']))]][0]
    checks['Emu'] = [(x, x < 1.0E-9) for x in [np.max(np.abs(data['Emu'] - ref.scalars['Emu']))]][0]
    checks['ES'] = [(x, x < 1.0E-9) for x in [np.max(np.abs(data['ES'] - ES[...]))]][0]
    checks['ET'] = [(x, x < 1.0E-9) for x in [np.max(np.abs(data['ET'] - ET[...]))]][0]
    checks['OS'] = [(x, x < 1.0E-8) for x in [np.max(np.abs(data['OS'] - cas.oscillator_strengths[0][...]))]][0]
    checks['OT'] = [(x, x < 1.0E-8) for x in [np.max(np.abs(data['OT'] - cas.oscillator_strengths[2][...]))]][0]
    checks['D'] = [(x, x < 1.0E-8) for x in [np.max(np.abs(data['D'] - ref.tensors['D'][...]))]][0]
    checks['F'] = [(x, x < 1.0E-8) for x in [np.max(np.abs(data['F'] - ref.tensors['F'][...]))]][0]
    # checks['C'] = [(x, x < 1.0E-8) for x in [np.max(np.abs(data['C'] - ref.tensors['C'][...]))]][0]
    checks['eps'] = [(x, x < 1.0E-6) for x in [np.max(np.abs(data['eps'] - ref.tensors['eps'][...]))]][0]
    checks['n'] = [(x, x < 1.0E-8) for x in [np.max(np.abs(data['n'] - ref.tensors['n'][...]))]][0]
    checks['GS0'] = [(x, x < 1.0E-8) for x in [np.max(np.abs(data['GS0'] - GS0[...]))]][0]
    checks['GS1'] = [(x, x < 1.0E-8) for x in [np.max(np.abs(data['GS1'] - GS1[...]))]][0]
    checks['GT1'] = [(x, x < 1.0E-8) for x in [np.max(np.abs(data['GT1'] - GT1[...]))]][0]
    checks['CS0S1'] = [(x, x < 1.0E-7) for x in [np.max(np.abs(data['CS0S1'] - np.copysign(CS0S1[...],data['CS0S1'])))]][0]
    checks['CT0T1'] = [(x, x < 1.0E-7) for x in [np.max(np.abs(data['CT0T1'] - np.copysign(CT0T1[...],data['CT0T1'])))]][0]

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
    assert(run_fomo_casci('data/', 'pyp', '6-31g', 'fomo_casci_qmmm.data.npz'))
