import lightspeed as ls
import est
import numpy as np
import collections
import unittest

# => unittest class <= #

class RHFTest(unittest.TestCase):

    def test_rhf(self):

        self.assertTrue(run_rhf('integration/data/', 'hbdi.xyz', '6-31gss', 'rhf.data.npz'))

# => RHF <= #

def run_rhf(run_dir, run_xyz, run_basis, ref_npz):

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
        )
    ref.compute_energy()
    GS0 = ref.compute_gradient()

    GS0.name = 'SCF Gradient'
    print(GS0)

    ls.timings_footer(start_time)

    # Save reference data
    if False:
        np.savez(
            run_dir + ref_npz,
            Enuc=ref.scalars['Enuc'],
            Eext=ref.scalars['Eext'],
            Escf=ref.scalars['Escf'],
            D=ref.tensors['D'],
            F=ref.tensors['F'],
            # C=ref.tensors['C'], # Need phase rules
            eps=ref.tensors['eps'],
            n=ref.tensors['n'],
            GS0=GS0,
            )
        
    # Check reference data
    data = np.load(run_dir + ref_npz)
    checks = collections.OrderedDict()
    checks['Enuc'] = [(x, x < 1.0E-10) for x in [np.max(np.abs(data['Enuc'] - ref.scalars['Enuc']))]][0]
    checks['Eext'] = [(x, x < 1.0E-10) for x in [np.max(np.abs(data['Eext'] - ref.scalars['Eext']))]][0]
    checks['Escf'] = [(x, x < 1.0E-9) for x in [np.max(np.abs(data['Escf'] - ref.scalars['Escf']))]][0]
    checks['D'] = [(x, x < 1.0E-8) for x in [np.max(np.abs(data['D'] - ref.tensors['D'][...]))]][0]
    checks['F'] = [(x, x < 1.0E-8) for x in [np.max(np.abs(data['F'] - ref.tensors['F'][...]))]][0]
    # checks['C'] = [(x, x < 1.0E-8) for x in [np.max(np.abs(data['C'] - ref.tensors['C'][...]))]][0]
    checks['eps'] = [(x, x < 1.0E-6) for x in [np.max(np.abs(data['eps'] - ref.tensors['eps'][...]))]][0]
    checks['n'] = [(x, x < 1.0E-8) for x in [np.max(np.abs(data['n'] - ref.tensors['n'][...]))]][0]
    checks['GS0'] = [(x, x < 1.0E-8) for x in [np.max(np.abs(data['GS0'] - GS0[...]))]][0]

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
    assert(run_rhf('data/', 'hbdi.xyz', '6-31gss', 'rhf.data.npz'))
