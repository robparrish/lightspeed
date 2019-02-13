import lightspeed as ls
import psiw
import aiscat as ai
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':

    resources = ls.ResourceList.build()

    molecule = ls.Molecule.from_xyz_file('geom.xyz')

    geom = psiw.Geometry.build(
        resources=resources,
        molecule=molecule,
        basisname='6-31gs',
        )

    ref = psiw.RHF.from_options(
        geometry=geom,
        g_convergence=1.0E-6,
        fomo=True,
        fomo_method='gaussian',
        fomo_temp=0.3,
        fomo_nocc=7,
        fomo_nact=2,
        )
    ref.compute_energy()

    cas = psiw.CASCI.from_options(
        reference=ref,
        nocc=7,
        nact=2,
        nalpha=1,
        nbeta=1,
        S_inds=[0],
        S_nstates=[2],
        )
    cas.compute_energy()

    lot = psiw.CASCI_LOT.from_options(
        casci=cas,
        )

    # L = 0.003 # 3.1 MeV e-
    # mode='ued'    
    L = 1.3 # 9.53 KeV x-ray
    mode='xray'    
    s = ls.Tensor.array(np.linspace(0.01, 9.0, 100))
    eta = ls.Tensor.array(np.linspace(0.0, 2.0*np.pi, 120))
    anisotropy = 'perpendicular'
    algorithm = 'gpu_f_d' # FP32/FP64 is ideal for rotationally-averaged scattering
    cmap = plt.get_cmap('coolwarm')

    grid = ls.BeckeGrid.buildSG0(lot.resources, lot.qm_molecule)
    print grid

    # Perpendicular scattering
    aiscat = ai.AISCAT.from_options(
        lot=lot,
        grid=grid,
        mode=mode,
        L=L,
        s=s,
        eta=eta,
        anisotropy=anisotropy,
        algorithm=algorithm,
        )
    
    # Ab initio scattering (S, indexA, indexB)
    I0 = aiscat.diffraction_pattern(0, 0, 0)
    I1 = aiscat.diffraction_pattern(0, 1, 1)
    I01 = ls.Tensor.array((I1[...] - I0[...]) / I0[...])
    ai.AISCAT.pattern_plot(s=s,eta=eta,I=I01, filename='%s-I01.png' % mode, cmap=cmap, twosided=True, colorbar=True)
    
    # IAM scattering
    IA = aiscat.diffraction_pattern_iam()
    I0A = ls.Tensor.array((I0[...] - IA[...]) / IA[...])
    ai.AISCAT.pattern_plot(s=s,eta=eta,I=I0A, filename='%s-I0A.png' % mode, cmap=cmap, twosided=True, colorbar=True)

    # Moments (easier to store)
    M0 = aiscat.diffraction_moments(0, 0, 0)
    M1 = aiscat.diffraction_moments(0, 1, 1)
    MA = aiscat.diffraction_moments_iam()

    # Aligned diffraction
    cmap = plt.get_cmap('viridis')
    aiscat = ai.AISCAT.from_options(
        lot=lot,
        grid=grid,
        mode=mode,
        L=L,
        s=s,
        eta=eta,
        anisotropy='aligned',
        algorithm='gpu_d_d', # Have to use FP64/FP64 for aligned scattering
        )
    I0 = aiscat.diffraction_pattern(0, 0, 0)
    I1 = aiscat.diffraction_pattern(0, 1, 1)
    IA = aiscat.diffraction_pattern_iam()
    ai.AISCAT.pattern_plot(s=s,eta=eta,I=I0, filename='%s-al-I0.png' % mode, cmap=cmap, twosided=False, colorbar=True)
    ai.AISCAT.pattern_plot(s=s,eta=eta,I=I1, filename='%s-al-I1.png' % mode, cmap=cmap, twosided=False, colorbar=True)
    ai.AISCAT.pattern_plot(s=s,eta=eta,I=IA, filename='%s-al-IA.png' % mode, cmap=cmap, twosided=False, colorbar=True)
    
        
        

