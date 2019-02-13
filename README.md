# Lightspeed

Lightspeed (LS) is a C++/Python library to facilitate the rapid development and
deployment of new ideas in Gaussian basis electronic structure theory (EST).
The main precept of LS is that most EST methods consist of two fundamentally
different operations:

1. A few common numerical kernels, which are encountered in many places.
Examples include Gaussian integrals, evaluation of basis function properties on
grid points, construction of J/K matrices, construction of low-rank
factorizations of interaction integral tensors, contractions of density
matrices with integrals and integral derivatives, etc. These operations are
often exceedingly rate-limiting in terms of FLOPs, memory, and disk storage.
Moreover, there may exist several different algorithms to do any one of these
operations.

2. An enormous number of combinations of these numerical kernels to produce the
myriad methods of EST. One notable example is how HF/DFT, CIS/TD-HF/TDA/TD-DFT,
CPHF, and limited active-space CASCI/CASSCF methods can all be formulated in
terms of rate-limiting J/K builds, plus large amounts of wrapper code that is
not rate limiting (eigensolves, guesses, iteration control and stabilization,
etc).

**Lightspeed does not do electronic structure theory -- Lightspeed helps you
do electronic structure theory.** LS accomplishes this by providing a clean
interface to fast algorithms for evaluation of the heavy numerical kernels in
Point 1, leaving more time and flexibility for the user to focus on the
layering of these kernels to produce useful EST methods in Point 2.

The part of this repository which constitutes LS is in src/*

# EST

EST (Electronic Structure Theory) is a simple Python library which implements
production-scale electronic structure methods in a flexible, extensible manner.
One of the key design goals with EST is that there is no "main" method and that
EST methods should be built as objects for each problem encountered. These
objects can solve for the electronic wavefunction, compute extra desired
properties on demand (gradients, couplings, overlaps, observables), and allow
fine-grained access to the state variables computed during wavefunction
optimization. 

The part of this repository which constitutes EST in in est/*

# Source Directory Structure

The major LS source directories are:

* src - the main C++/Python source tree constituting the heart of LS
* est - a simple python-based electronic structure code stack
* data - ancillary data needed to use LS, such as basis sets
* tests - unit/production tests for the LS library

More minor LS source directories are:

* md - a simple adiabatic / overlap following molecular dynamics code.
* apps - small plugin applications using LS and EST to accomplish custom
    theoretical chemistry workflows.

# Key Directories For New Users

* src/lightspeed - C++ library API (wrapped ~100% to Python)
* src/python - C++ -> Python wrapping and Python extensions
* est - Python EST code stack

More detailed documentation, installation instructions, tutorials, and unit
tests are forthcoming.

# Authors

* Robert Parrish (robparrish@gmail.com)
* Xin Li - Integrals/Gradients/Infrastructure
* Jason Ford - BlurBox Routines/Infrastructure
* Ruben Guerrero - Davidson/Potential Gradients

RMP is pleased to acknowledge many ideas from Justin Turney, Francesco
Evangelista, Daniel Smith, Lori Burns, David Sherrill, Nathan Luehr, Ed
Hohenstein, and Todd Martinez
