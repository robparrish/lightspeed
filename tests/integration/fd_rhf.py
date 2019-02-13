import lightspeed as ls
import est
import numpy as np
import unittest

# => unittest class <= #

class FD_RHF_Test(unittest.TestCase):

    def test_fd_rhf(self):

        self.assertTrue(run_fd_rhf('integration/data/', 'h2o.xyz', 'sto-3g'))

# => Manager for RHF Finite Differences <= #

class RHFFDManager(object):

    def __init__(
        self,
        rhf, # A completed RHF object at the reference geometry
        ):

        self.rhf = rhf

    @property
    def xyz(self):
        return self.rhf.geometry.molecule.xyz

    def compute_energy(self, xyz):

        geom2 = self.rhf.geometry.update_xyz(xyz)
        rhf2 = est.RHF(self.rhf.options.copy().set_values({
            'geometry' : geom2,
            }))
        rhf2.compute_energy(Dguess=self.rhf.tensors['D'])
        return ls.Tensor.array(rhf2.scalars['Escf'])

    def compute_gradient(self, xyz):

        geom2 = self.rhf.geometry.update_xyz(xyz)
        rhf2 = est.RHF(self.rhf.options.copy().set_values({
            'geometry' : geom2,
            }))
        rhf2.compute_energy(Dguess=self.rhf.tensors['D'])
        return rhf2.compute_gradient()

# => RHF Finite Difference <= #

def run_fd_rhf(run_dir, run_xyz, run_basis):

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
        g_convergence=1.0E-9,
        print_level=0,
        )
    ref.compute_energy()
    
    # => RHF Finite Difference Manager <= #
    
    ref_man = RHFFDManager(ref)

    # => Finite Difference Gradients <= #
    
    G = est.fd_gradient(
        ref_man.xyz,
        ref_man.compute_energy,
        h=0.001, 
        npoint=5,
        print_level=1,
        )
    G.name = 'Finite Difference RHF Gradient'
    print(G)
    
    G2 = ref.compute_gradient()
    G2.name = 'Analytical RHF Gradient'
    print(G2)
    
    dG = np.max(np.abs(G[...] - G2[...]))
    print('Max Difference: %11.3E\n' % dG)

    # => Finite Difference Hessian (Note: not at minimum) <= #

    H = est.fd_gradient(
        ref_man.xyz,
        ref_man.compute_gradient,
        h=0.001, 
        npoint=5,
        print_level=1,
        )
    H.name = 'Finite Difference RHF Hessian'
    print(H)

    non_symm_err = np.max(np.abs(H[...] - H[...].T))
    print('Max Non-symmetric error: %11.3E\n' % non_symm_err)

    ls.timings_footer(start_time)
    return (non_symm_err < 1.0E-6)

# => main function <= #

if __name__ == "__main__":
    assert(run_fd_rhf('data/', 'h2o.xyz', 'sto-3g'))
