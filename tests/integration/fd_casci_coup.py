import lightspeed as ls
import est
import numpy as np
import unittest

# => unittest class <= #

class FD_CASCI_Coup_Test(unittest.TestCase):

    def test_fd_casci_coup(self):

        self.assertTrue(run_fd_casci_coup('integration/data/', 'h2o.xyz', '6-31gs'))

# => Manager for CASCI Finite Differences <= #

class CASCIOverlapFDManager(object):

    def __init__(
        self,
        casci,
        overlap_S=0,
        overlap_indexA=0,
        overlap_indexB=1,
        ):

        self.casci = casci
        self.overlap_S = overlap_S
        self.overlap_indexA = overlap_indexA
        self.overlap_indexB = overlap_indexB

    @property
    def xyz(self):
        return self.casci.geometry.molecule.xyz

    def compute_overlap(self, xyz):

        geom2 = self.casci.geometry.update_xyz(xyz)
        rhf2 = est.RHF(self.casci.reference.options.copy().set_values({
            'geometry' : geom2,
            }))
        rhf2.compute_energy(Dguess=self.casci.reference.tensors['D'])
        casci2 = est.CASCI(self.casci.options.copy().set_values({
            'reference' : rhf2,
            }))
        casci2.compute_energy()
        O = est.CASCI.compute_overlap(self.casci, casci2, self.overlap_S)
        # Phase the |J'> states to match the <I| states
        for k in range(O.shape[1]):
            if O[k,k] < 0.0:
                O[:,k] *= -1.0
        return O[self.overlap_indexA, self.overlap_indexB]

# => CASCI Coupling Finite Difference <= #

def run_fd_casci_coup(run_dir, run_xyz, run_basis):

    start_time = ls.timings_header()
    
    # => RHF Single Point <= #
    
    resources = ls.ResourceList.build()
    
    molecule = ls.Molecule.from_xyz_file(run_dir + run_xyz)
    
    geom = est.Geometry.build(
        resources=resources,
        molecule=molecule,
        basisname=run_basis,
        thre_pq=1.0E-15,
        )
    
    ref = est.RHF.from_options(
        geometry=geom,
        g_convergence=1.0E-13,
        cphf_r_convergence=1.0E-13,
        fomo=True,
        fomo_method='gaussian',
        fomo_temp=0.3,
        fomo_nocc=2,
        fomo_nact=5,
        print_level=2,
        thre_dp=1.0E-16,
        cphf_thre_dp=1.0E-16,
        grad_thre_dp=1.0E-16,
        )
    ref.compute_energy()

    cas = est.CASCI.from_options(
        reference=ref,
        nocc=2,
        nact=5,
        nalpha=3,
        nbeta=3,
        S_inds=[0, 2],
        S_nstates=[3, 3],
        print_level=2,
        thre_dp=1.0E-16,
        grad_thre_dp=1.0E-16,
        )
    cas.compute_energy()

    # Lower the print level for FD displacements
    ref.options['print_level'] = 0
    cas.options['print_level'] = 0
    
    # => CASCI Overlap Finite Difference Manager <= #

    S = 0
    indexA = 0
    indexB = 1
    
    cas_man = CASCIOverlapFDManager(cas, S, indexA, indexB)

    # => Finite Difference Couplings <= #
    
    G = est.fd_gradient(
        cas_man.xyz,
        cas_man.compute_overlap,
        h=0.001, 
        npoint=7,
        print_level=1,
        )
    G.name = 'Finite Difference CASCI Coupling'
    print(G)
    
    G2 = cas.compute_coupling(S, indexA, indexB)
    G2.name = 'Analytical CASCI Coupling'
    print(G2)
    
    dG = np.max(np.abs(G[...] - G2[...]))
    print('Max Difference: %11.3E\n' % dG)

    ls.timings_footer(start_time)
    return (dG < 1.0E-9)

# => main function <= #

if __name__ == "__main__":
    assert(run_fd_casci_coup('data/', 'h2o.xyz', '6-31gs'))
