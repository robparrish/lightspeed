#ifndef LS_CUBIC_HPP
#define LS_CUBIC_HPP

#include <memory>
#include <vector>

namespace lightspeed {

class Tensor;
class Molecule;

/**
 * Class CubicGrid encapsulates an orthorhombic Cartesian grid. Such grids are
 * often used in producing Gaussian Cube files for visualization, or as Fourier
 * grids for orthorhombic problems.
 *
 * CubicGrid allows for an arbitrary origin, grid spacing, and grid sizing in
 * x, y, and z.
 *
 * The grid points start at the origin (the lower left corner of the grid), and
 * increment upward by "ij"-type indexing, i.e., so that z changes first, y
 * second, and x last (standard C-style ordering).
 **/
class CubicGrid {

public:

/**
 * Constructor, takes user-specified origin, spacing, and sizing parameters and
 * constructs the grid coordinates xyz according to the usual C-style ordering.
 * @param origin a vector of size 3 with the {x0,y0,z0} coordinates of the origin
 * @param spacing a vector of size 3 with the {dx,dy,dz} grid spacing
 * @param sizing a vector of size 3 with the {nx,ny,nz} grid point sizes
 * @result the xyz coordinates of the grid are defined by:
 *  x_i = x0 + ix dx
 *  y_i = y0 + iy dy
 *  z_i = z0 + iz dz
 * where ix ranges from 0 to nx-1, and (ix,iy,iz) increment lexically.
 **/
CubicGrid(
    const std::vector<double>& origin,
    const std::vector<double>& spacing,
    const std::vector<size_t>& sizing);

/**
 * Build a CubicGrid suitable for use as a Fourier grid for a given unit cell
 * @param origin a vector of size 3 with the {x0,y0,z0} coordinates of the origin
 * @param extents the dimensions of the unit cell {Lx,Ly,Lz}
 * @param sizing the number of grid points
 * @result a standard Fourier grid will be constructed, which starts at the
 * origin and goes up to a maximum displacement of Lx * (nx-1) / (nx). The point
 * at Lx is assumed to lie in the next periodic grid image. The grid spacing is
 * derived to be Lx / (nx).
 **/
static
std::shared_ptr<CubicGrid> build_fourier(
    const std::vector<double>& origin,
    const std::vector<double>& extents,
    const std::vector<size_t>& sizing); 

/**
 * Build the best possible CubicGrid suitable for use as a Fourier grid for a
 * given unit cell. In this utility function, the user specifies a desired
 * spacing, and the next-smallest spacing value consistent with a discrete grid
 * is used to build the CubicGrid.
 * @param origin a vector of size 3 with the {x0,y0,z0} coordinates of the origin
 * @param extents the exact dimensions of the unit cell {Lx,Ly,Lz}
 * @param spacing the desired spacing of the grid. 
 * @result a Fourier grid will be constructed according to build_fourier. The
 * sizing parameter will be computed as ceil(Lx / dx). The resultant grid
 * spacing is dx2 = Lx / (ceil(Lx / dx)) <= dx.
 **/
static 
std::shared_ptr<CubicGrid> build_next_fourier(
    const std::vector<double>& origin,
    const std::vector<double>& extents,
    const std::vector<double>& spacing);

/**
 * Build the best possible CubicGrid suitable for use as a Gaussian Cube File
 * for a given molecular geometry. In this utility function, the user specifies
 * a set of critical molecular geometry points (such as the coordinates of the
 * atoms), and a desired overage in each dimension. The method finds the
 * maximum extents of the xyz points, and adds +/- overage to this in each
 * dimension. The overage is expanded slightly to accomodate an integer number
 * of points with exactly specified spacing.
 * @param xyz a (nP,3) Tensor with coordinates of interesting points to build a
 *  CubicGrid around
 * @param overage the minimum allowed buffer around any xyz point to the edge
 *  of the CubicGrid. The true overage used will generally be slightly larger to
 *  accomodate an integral number of points with spacing specified below.
 * @param spacing the exact spacing expecting in the grid.
 * @result a CubicGrid will be constructed according to the rules above.
 **/   
static 
std::shared_ptr<CubicGrid> build_next_cube(
    const std::shared_ptr<Tensor>& xyz,
    const std::vector<double>& overage,
    const std::vector<double>& spacing);

// => Accessors <= //
      
/// The total number of points in the grid (nx*ny*nz)
size_t size() const { return sizing_[0] * sizing_[1] * sizing_[2]; }

/// The {x0,y0,z0} origin (lower left corner of the grid)
const std::vector<double>& origin() const { return origin_; }
/// The {dx,dy,dz} spacing (displacement between points in each dimension)
const std::vector<double>& spacing() const { return spacing_; }
/// The {nx,ny,nz} sizing (number of points in each dimension)
const std::vector<size_t>& sizing() const { return sizing_; }

/// The x,y,z coordinates of the grid ordered lexically in C-order, a (size,3) Tensor
std::shared_ptr<Tensor> xyz() const { return xyz_; }

/// The uniform quadrature weight of each grid point (dx*dy*dz)
double w() const { return spacing_[0] * spacing_[1] * spacing_[2]; }

/// A handy string representation of this object
std::string string() const;

// => Gaussian Cube File Utility <= //

/**
 * Write a Gaussian Cube File (.cube) to disk.
 * @param filename the filepath to write to (opened with "w" flag).
 * @param property the descriptive name of the property being written
 * @param mol the molecular geometry
 * @param v the scalar field data to write, a (size,) Tensor
 * @result a .cube file containing the grid geometry, molecular geometry, and
 * scalar field v is written to filename.
 **/
void save_cube_file(
    const std::string& filename,
    const std::string& property,
    const std::shared_ptr<Molecule>& mol,
    const std::shared_ptr<Tensor>& v) const;

private:

std::vector<double> origin_;
std::vector<double> spacing_;   
std::vector<size_t> sizing_;

std::shared_ptr<Tensor> xyz_;

};



} // namespace lightspeed

#endif
