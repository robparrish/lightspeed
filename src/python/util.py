def _build_units():
    units = {}
    units['ang_per_bohr'] = 0.52917720859 # PSI4
    # units['ang_per_bohr'] = 0.52917724924 # TC
    units['ev_per_H'] = 27.21138
    units['kcal_per_H'] = 627.5095
    units['au_per_amu'] = 1.8228884855409500E+03
    units['au_per_fs'] = 1.0 / 2.418884326505E-2
    # Inverse conversions to machine precision
    for key in units.copy().keys():
        A, B = key.split('_per_')
        units['%s_per_%s' % (B,A)] = 1.0 / units['%s_per_%s' % (A,B)]
    return units

units = _build_units()
