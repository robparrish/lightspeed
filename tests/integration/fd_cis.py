import lightspeed as ls
import est
import numpy as np
import unittest

# => unittest class <= #

class FD_CIS_Test(unittest.TestCase):

    def test_fd_cis(self):

        self.assertTrue(run_fd_cis('integration/data/', 'h2o.xyz', '6-31gs'))

# => Manager for CIS Finite Differences <= #

class CISFDManager(object):

    def __init__(
        self,
        cisci,
        energy_S=0,
        energy_index=0,
        gradient_S=0,
        gradient_index=0,
        ):

        self.cisci = cisci
        self.energy_S = energy_S
        self.energy_index = energy_index
        self.gradient_S = gradient_S
        self.gradient_index = gradient_index

    @property
    def xyz(self):
        return self.cisci.geometry.molecule.xyz

    def compute_energy(self, xyz):

        geom2 = self.cisci.geometry.update_xyz(xyz)
        rhf2 = est.RHF(self.cisci.reference.options.copy().set_values({
            'geometry' : geom2,
            }))
        rhf2.compute_energy(Dguess=self.cisci.reference.tensors['D'])
        cisci2 = est.CIS(self.cisci.options.copy().set_values({
            'reference' : rhf2,
            }))
        cisci2.compute_energy()
        return ls.Tensor.array(cisci2.evals[self.energy_S][self.energy_index])

    def compute_gradient(self, xyz):

        geom2 = self.cisci.geometry.update_xyz(xyz)
        rhf2 = est.RHF(self.cisci.reference.options.copy().set_values({
            'geometry' : geom2,
            }))
        rhf2.compute_energy(Dguess=self.cisci.reference.tensors['D'])
        cisci2 = est.CIS(self.cisci.options.copy().set_values({
            'reference' : rhf2,
            }))
        cisci2.compute_energy()
        return cisci2.compute_gradient(self.gradient_S, self.gradient_index)

# => CIS Finite Difference <= #

def run_fd_cis(run_dir, run_xyz, run_basis):

    start_time = ls.timings_header()
    
    # => RHF Single Point <= #
    
    resources = ls.ResourceList.build()
    
    molecule = ls.Molecule.from_xyz_file(run_dir + run_xyz)
    
    geom = est.Geometry.build(
        resources=resources,
        molecule=molecule,
        basisname=run_basis,
        )
    
    ref = est.RHF.from_options(
        geometry=geom,
        g_convergence=1.0E-10,
        cphf_r_convergence=1.0E-10,
        print_level=2,
        )
    ref.compute_energy()

    cis = est.CIS.from_options(
        reference=ref,
        S_inds=[0, 2],
        S_nstates=[3, 3],
        print_level=2,
        )
    cis.compute_energy()

    # Lower the print level for FD displacements
    ref.options['print_level'] = 0
    cis.options['print_level'] = 0
    
    # => CIS Finite Difference Manager <= #
    
    cis_man = CISFDManager(cis, energy_index=1, gradient_index=1)

    # => Finite Difference Gradients <= #
    
    G = est.fd_gradient(
        cis_man.xyz,
        cis_man.compute_energy,
        h=0.001, 
        npoint=5,
        print_level=1,
        )
    G.name = 'Finite Difference CIS Gradient'
    print(G)
    
    G2 = cis.compute_gradient(0, 1)
    G2.name = 'Analytical CIS Gradient'
    print(G2)
    
    dG = np.max(np.abs(G[...] - G2[...]))
    print('Max Difference: %11.3E\n' % dG)

    # => Finite Difference Hessian (Note: not at minimum) <= #
    
    H = est.fd_gradient(
        cis_man.xyz,
        cis_man.compute_gradient,
        h=0.001, 
        npoint=5,
        print_level=1,
        )
    H.name = 'Finite Difference CIS Hessian'
    print(H)

    non_symm_err = np.max(np.abs(H[...] - H[...].T))
    print('Max Non-symmetric error: %11.3E\n' % non_symm_err)

    ls.timings_footer(start_time)
    return (non_symm_err < 1.0E-6)

# => main function <= #

if __name__ == "__main__":
    assert(run_fd_cis('data/', 'h2o.xyz', '6-31gs'))
