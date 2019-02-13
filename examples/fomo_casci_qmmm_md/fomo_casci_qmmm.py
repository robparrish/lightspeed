import lightspeed as ls
import psiw
import bs

resources = ls.ResourceList.build()

qmmm = psiw.QMMM.from_prmtop(
    prmtopfile='pyp.prmtop',
    inpcrdfile='pyp.rst',
    qmindsfile='pyp.qm',
    charge=-1.0,
    )

geom = psiw.Geometry.build(
    resources=resources,
    qmmm=qmmm,
    basisname='6-31g',
    )

ref = psiw.RHF.from_options(
    geometry=geom,
    g_convergence=1.0E-6,
    fomo=True,
    fomo_method='gaussian',
    fomo_temp=0.2,
    fomo_nocc=107,
    fomo_nact=3,
    )
ref.compute_energy()

casci = psiw.CASCI.from_options(
    reference=ref,
    nocc=107,
    nact=3,
    nalpha=2,
    nbeta=2,
    S_inds=[0, 2],
    S_nstates=[3, 3],
    )
casci.compute_energy()

lot = psiw.CASCI_LOT.from_options(
    casci=casci,
    print_level=0,
    rhf_guess=True,
    rhf_mom=True,
    orbital_coincidence='none',
    state_coincidence='full',
    )

vv = bs.VV.from_options(
    dt=20.0,
    )

masses = bs.compute_masses(qmmm.molecule)

# momenta = bs.momenta_from_xyz_file(
#     filename='v.xyz',
#     masses=masses,
#     )
momenta = ls.Tensor.zeros_like(qmmm.molecule.xyz)

xyz_reporter = bs.XYZReporter.from_options(
    interval=1,
    filename='adiabatic.xyz',
    )

npz_reporter = bs.NPZReporter.from_options(
    interval=1,
    filename='adiabatic.npz',
    state_energies=True,
    )

aimd = bs.AIMD.from_options(
    lot=lot,
    integrator=vv,
    target_S=0,
    target_index=1,
    masses=masses,
    momenta=momenta,
    state_tracking='adiabatic',
    print_level=2,
    reporters=[xyz_reporter, npz_reporter],
    )
aimd.initialize()
aimd.run(200)
aimd.finalize()    
