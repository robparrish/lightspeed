import lightspeed as ls # The lightspeed module
import est # The "psidewinder" lightweight electronic structure module
# CPU/GPU Resources for problem
resources = ls.ResourceList.build()
# OpenMM-based QM/MM (Mechanical + Coulomb Embedding w/ link H atoms)
qmmm = est.QMMM.from_prmtop(
    prmtopfile='system.prmtop',
    inpcrdfile='system.rst7',
    qmindsfile='system.qm',
    charge=-1.0)
# Geometry manages all external environment considerations
geometry = est.Geometry.build(
    resources=resources,
    qmmm=qmmm,
    basisname='cc-pvdz')
# FON-RHF (4 active electrons in 3 active orbitals) 
reference = est.RHF.from_options(
    geometry=geometry,
    g_convergence=1.0E-6,
    fomo=True,
    fomo_method='gaussian',
    fomo_temp=0.2,
    fomo_nocc=107,
    fomo_nact=3)
ref.compute_energy()
# FOMO-CASCI (3 singlets, 3 triplets)
casci = est.CASCI.from_options(
    reference=reference,
    nocc=107,
    nact=3,
    nalpha=2,
    nbeta=2,
    S_inds=[0, 2],
    S_nstates=[3, 3])
casci.compute_energy()
# Gradient of 1th state in spin block 0 (S1)
print(casci.compute_gradient(0,1))
# Coupling between the 1th and 2th state in spin block 0 (S1/S2)
print(casci.compute_coupling(0,1,2))
# Many-body electronic overlap integrals in spin block 0 (all states)
print(est.CASCI.compute_overlap(casci, casci, 0))