import lightspeed as ls
import est
import numpy as np
import collections
import unittest

# => unittest class <= #

class GuessTest(unittest.TestCase):

    def test_guess(self):

        self.assertTrue(run_guess('integration/data/', 'h2o.xyz', '6-31gs', '6-31gss'))

# => Guess <= #

def run_guess(run_dir, run_xyz, run_basis, run_basis_2):

    # Validity checks: { key : (delta, OK) }
    checks = collections.OrderedDict()

    start_time = ls.timings_header()
    
    resources = ls.ResourceList.build()
    
    molecule = ls.Molecule.from_xyz_file(run_dir + run_xyz)
    
    geom = est.Geometry.build(
        resources=resources,
        molecule=molecule,
        basisname=run_basis,
        )

    geom2 = est.Geometry.build(
        resources=resources,
        molecule=molecule,
        basisname=run_basis_2,
        )
        
    # ==> RHF <== #
    
    ref = est.RHF.from_options(
        geometry=geom,
        )
    ref.compute_energy()
    
    # => Identity Guess: should be the same <= #
    
    ref2 = est.RHF(ref.options.copy().set_values({
        'geometry' : geom,
        }))
    ref2.initialize()
    Dguess, Cocc_mom, Cocc_act = est.RHF.guess_density(ref, ref2)
    ref2.compute_energy(Dguess)
    
    dD = np.max(np.abs(Dguess[...] - ref.tensors['D']))
    checks['dD-RHF'] = (dD, dD < 1.0E-13) 

    # => Cast-Up Guess <= #

    ref2 = est.RHF(ref.options.copy().set_values({
        'geometry' : geom2,
        }))
    ref2.initialize()
    Dguess, Cocc_mom, Cact_mom = est.RHF.guess_density(ref, ref2)
    ref2.compute_energy(Dguess)

    # Check if NO structure makes sense
    n1s = ref.tensors['n']
    n2s = ls.Tensor.eigh(ls.Tensor.chain([ref2.tensors['X'], ref2.tensors['S'], Dguess, ref2.tensors['S'], ref2.tensors['X']], [True, False, False, False, False]))[0]
    dn = np.max([abs(n1 - n2) for n1, n2 in zip(sorted(-np.abs(n1s)), sorted(-np.abs(n2s)))])
    checks['dn-RHF'] = (dn, dn < 1.0E-12)

    # ==> FOMO-RHF <== #
    
    ref = est.RHF.from_options(
        geometry=geom,
        fomo=True,
        fomo_method='gaussian',
        fomo_temp=0.3,
        fomo_nocc=2,
        fomo_nact=5,
        )
    ref.compute_energy()

    # => Identity Guess: should be the same <= #
    
    ref2 = est.RHF(ref.options.copy().set_values({
        'geometry' : geom,
        }))
    ref2.initialize()
    Dguess, Cocc_mom, Cact_mom = est.RHF.guess_density(ref, ref2)
    ref2.compute_energy(Dguess)
    checks['dD-FOMO-RHF'] = (dD, dD < 1.0E-13) 

    # => Cast-Up Guess <= #
    
    ref2 = est.RHF(ref.options.copy().set_values({
        'geometry' : geom2,
        }))
    ref2.initialize()
    Dguess, Cocc_mom, Cact_mom = est.RHF.guess_density(ref, ref2)
    ref2.compute_energy(Dguess)

    # Check if NO structure makes sense
    n1s = ref.tensors['n']
    n2s = ls.Tensor.eigh(ls.Tensor.chain([ref2.tensors['X'], ref2.tensors['S'], Dguess, ref2.tensors['S'], ref2.tensors['X']], [True, False, False, False, False]))[0]
    dn = np.max([abs(n1 - n2) for n1, n2 in zip(sorted(-np.abs(n1s)), sorted(-np.abs(n2s)))])
    checks['dn-FOMO-RHF'] = (dn, dn < 1.0E-12)

    ls.timings_footer(start_time)

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
    assert(run_guess('data/', 'h2o.xyz', '6-31gs', '6-31gss'))
