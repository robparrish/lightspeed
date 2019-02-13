#ifndef LS_ATOMIC_RADII_HPP
#define LS_ATOMIC_RADII_HPP

#include <stdexcept>

namespace lightspeed {

//    Table of Bragg-Slater Atomic Radii (Angstroms)
//
//    Bragg-Slater radii: J.C. Slater, Symmetry and Energy Bands in Crystals,
//                        Dover, N.Y. 1972, page 55.
//                        The radii of noble gas atoms are set to be equal 
//                        to the radii of the corresponding halogen atoms.
//                        The radius of At is set to be equal to the radius of
//                        Po. 
//                        The radius of Fr is set to be equal to the radius of
//                        Cs. 
const double bragg_slater_radii_[106] =  {
1.00,// Dummy
0.35,                                                                                0.35, // He 
1.45,1.05,                                                  0.85,0.70,0.65,0.60,0.50,0.50, // Ne 
1.80,1.50,                                                  1.25,1.10,1.00,1.00,1.00,1.00, // Ar 
2.20,1.80,1.60,1.40,1.35,1.40,1.40,1.40,1.35,1.35,1.35,1.35,1.30,1.25,1.15,1.15,1.15,1.15, // Kr 
2.35,2.00,1.80,1.55,1.45,1.45,1.35,1.30,1.35,1.40,1.60,1.55,1.55,1.45,1.45,1.40,1.40,1.40, // Xe 
2.60,2.15,
          1.95,1.85,1.85,1.85,1.85,1.85,1.85,1.80,1.75,1.75,1.75,1.75,1.75,1.75,           // Lanthanide Series
          1.75,1.55,1.45,1.35,1.35,1.30,1.35,1.35,1.35,1.50,1.90,1.80,1.60,1.90,1.90,1.90, // Rn 
2.60,2.15,
          1.95,1.80,1.80,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,           // Actinide Series
          1.75,1.55,1.55};                                                                 // Last Element is Db = 105

//    Table of Ahlrichs Atomic Radii (a.u.)
//
//    O. Treutler and R. Ahlrichs, J. Chem. Phys., 102, 346, (1995), Table I
//    Optimized for the Ahlrichs radial grid for up to Kr (Z = 36)
const double ahlrichs_radii_[37] = {
1.00,// Dummy
0.80,                                                                                0.90, // He 
1.80,1.40,                                                  1.30,1.10,0.90,0.90,0.90,0.90, // Ne 
1.40,1.30,                                                  1.30,1.20,1.10,1.00,1.00,1.00, // Ar 
1.50,1.40,1.30,1.20,1.20,1.20,1.20,1.20,1.20,1.10,1.10,1.10,1.10,1.00,0.90,0.90,0.90,0.90  // Kr 
}; // Last Element is Kr = 36

//    Table of Multi-Exp Atomic Radii (a.u.)
//
//    S.H. Chien, P.M.W. Gill, J. Comp. Chem., 27, 730 (2006)
//    Optimized for the MultiExp radial grid (in SG-0) for up to Kr (Z = 17), 
//    He and Ne taken from H and F
const double multiexp_radii_[37] = {
1.00,// Dummy
1.30,                                                                                1.30, // He 
1.95,2.20,                                                  1.45,1.20,1.10,1.10,1.20,1.20, // Ne 
2.30,2.20,                                                  2.10,1.30,1.30,1.10,1.45,1.45, // Ar 
}; // Last Element is Ar = 18

class AtomicRadii {

public:

// => Bragg-Slater Radii <= ///

/** 
 * The Bragg-Slater radii are defined as above 
 **/
static double bragg_slater_radius(int Z) {
    if (Z > 105) throw std::runtime_error("AtomicRadii: Bragg-Slater Radii only defined to Z = 105");        
    return bragg_slater_radii_[Z] / 0.52917720859;
}

// => Radial Quadrature Scale Parameters <= //

/**
 * The scale parameters for the Becke quadrature
 * are half of the the Bragg-Slater radii, with
 * the exception of H, for which the full Bragg-
 * Slater radius is used
 **/
static double becke_radius(int Z) {
    if (Z == 1) return AtomicRadii::bragg_slater_radius(Z);
    else return 0.5 * AtomicRadii::bragg_slater_radius(Z);
}

/**
 * The scale parameters for the Handy quadrature
 * are simply the Bragg-Slater radii
 **/
static double handy_radius(int Z) {
    return AtomicRadii::bragg_slater_radius(Z);
}

/**
 * The scale parameters for the Knowles quadrature 
 * are optimized separately for each block of the 
 * periodic table, up through Ca (Z = 20). It seems
 * that the first two columns always get 7.0 a.u.
 * (except H), while everything else gets 5.0 a.u.
 * We have therefore extrapolated this trend for
 * all elements. WARNING: For Z > 20, this is a guess!
 **/
static double knowles_radius(int Z) {
    if (Z > 105) throw std::runtime_error("AtomicRadii: Z > 105?!?! Call Rob if you discovered the island of stability.");
    if (Z == 3  || Z == 4  || 
        Z == 11 || Z == 12 || 
        Z == 19 || Z == 20 || 
        Z == 37 || Z == 38 || 
        Z == 55 || Z == 56 || 
        Z == 87 || Z == 88) {
        return 7.0;
    } else {
        return 5.0;
    }
}

/**
 * The scale parameters for the Ahlrichs quadrature 
 * are optimized separately for each atom of the 
 * periodic table, up through Kr (Z = 36). It seems that
 * there is not too much variation in these radii,
 * and that all radii are approximately 1.0 a.u.
 * We have therefore used 1.0 a.u for all of the rest
 * of the elements. WARNING: For Z > 36, this is a guess!
 **/
static double ahlrichs_radius(int Z) {
    if (Z > 105) throw std::runtime_error("AtomicRadii: Z > 105?!?! Call Rob if you discovered the island of stability.");
    else if (Z > 36) return 1.0;
    else return ahlrichs_radii_[Z];
}

/**
 * The scale parameters for the MultiExp quadrature 
 * are optimized separately for each atom of the 
 * periodic table, up through Cl (Z = 17), except 
 * nobel gases (which are assigned the values of the
 * correspoinding halogen). It seems that
 * there is some variation in these radii,
 * but that these mostly fall between 1.0 and 2.0
 * We have therefore used 1.5 a.u for all of the rest
 * of the elements. WARNING: For Z > 18, this is a guess!
 **/
static double multiexp_radius(int Z) {
    if (Z > 105) throw std::runtime_error("AtomicRadii: Z > 105?!?! Call Rob if you discovered the island of stability.");
    else if (Z > 18) return 1.5;
    else return multiexp_radii_[Z];
}

};

} // namespace lightspeed

#endif
