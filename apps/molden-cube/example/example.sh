setenv MOLDEN_CUBE ..
python $MOLDEN_CUBE/molden-cube.py --density Singlet.1.molden S1D
python $MOLDEN_CUBE/molden-cube.py --diff Singlet.1.molden Singlet.2.molden S12D
python $MOLDEN_CUBE/molden-cube.py --orbitals Singlet.1.molden S1C '[42,43]'
