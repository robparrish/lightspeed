#include <lightspeed/cubic.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/molecule.hpp>
#include "../util/string.hpp"
#include <stdexcept>
#include <cstdio>
#include <cmath>

namespace lightspeed {
    
CubicGrid::CubicGrid(
    const std::vector<double>& origin,
    const std::vector<double>& spacing,
    const std::vector<size_t>& sizing) :
    origin_(origin),
    spacing_(spacing),
    sizing_(sizing)
{
    if (origin_.size() != 3) throw std::runtime_error("CubicGrid: origin is wrong size");
    if (spacing_.size() != 3) throw std::runtime_error("CubicGrid: spacing is wrong size");
    if (sizing_.size() != 3) throw std::runtime_error("CubicGrid: sizing is wrong size");
    
    std::vector<size_t> dim;
    dim.push_back(size());
    dim.push_back(3);
    xyz_ = std::shared_ptr<Tensor>(new Tensor(dim));
    double* xyzp = xyz_->data().data();

    size_t index = 0;
    for (size_t ix = 0; ix < sizing_[0]; ix++) {
    for (size_t iy = 0; iy < sizing_[1]; iy++) {
    for (size_t iz = 0; iz < sizing_[2]; iz++) {
        xyzp[3*index+0] = origin_[0] + ix * spacing_[0];
        xyzp[3*index+1] = origin_[1] + iy * spacing_[1];
        xyzp[3*index+2] = origin_[2] + iz * spacing_[2];
        index++;
    }}}
}
std::shared_ptr<CubicGrid> CubicGrid::build_fourier(
    const std::vector<double>& origin,
    const std::vector<double>& extents,
    const std::vector<size_t>& sizing)
{
    if (origin.size() != 3) throw std::runtime_error("CubicGrid: origin is wrong size");
    if (extents.size() != 3) throw std::runtime_error("CubicGrid: extents is wrong size");
    if (sizing.size() != 3) throw std::runtime_error("CubicGrid: sizing is wrong size");

    std::vector<double> spacing;
    spacing.push_back(extents[0] / (sizing[0]));
    spacing.push_back(extents[1] / (sizing[1]));
    spacing.push_back(extents[2] / (sizing[2]));
    return std::shared_ptr<CubicGrid>(new CubicGrid(
        origin,
        spacing,
        sizing));
}
std::shared_ptr<CubicGrid> CubicGrid::build_next_fourier(
    const std::vector<double>& origin,
    const std::vector<double>& extents,
    const std::vector<double>& spacing)
{
    if (origin.size() != 3) throw std::runtime_error("CubicGrid: origin is wrong size");
    if (extents.size() != 3) throw std::runtime_error("CubicGrid: extents is wrong size");
    if (spacing.size() != 3) throw std::runtime_error("CubicGrid: spacing is wrong size");
    std::vector<size_t> sizing;
    sizing.push_back(ceil(extents[0] / spacing[0]));
    sizing.push_back(ceil(extents[1] / spacing[1]));
    sizing.push_back(ceil(extents[2] / spacing[2]));

    return CubicGrid::build_fourier(
        origin,
        extents,
        sizing);
}
std::shared_ptr<CubicGrid> CubicGrid::build_next_cube(
    const std::shared_ptr<Tensor>& xyz,
    const std::vector<double>& overage,
    const std::vector<double>& spacing)
{
    xyz->ndim_error(2);
    std::vector<size_t> dim;
    dim.push_back(xyz->shape()[0]);
    dim.push_back(3);
    xyz->shape_error(dim);
    if (overage.size() != 3) throw std::runtime_error("CubicGrid: overage is wrong size");
    if (spacing.size() != 3) throw std::runtime_error("CubicGrid: spacing is wrong size");

    size_t nA = xyz->shape()[0];
    if (nA == 0) throw std::runtime_error("CubicGrid: no xyz points provided to build_next_cube");
    const double* xyzp = xyz->data().data();
    
    std::vector<double> xmin;
    xmin.push_back(xyzp[0]);
    xmin.push_back(xyzp[1]);
    xmin.push_back(xyzp[2]);
    std::vector<double> xmax;
    xmax.push_back(xyzp[0]);
    xmax.push_back(xyzp[1]);
    xmax.push_back(xyzp[2]);
    for (size_t A = 0; A < nA; A++) {
        xmin[0] = std::min(xmin[0],xyzp[3*A+0]); 
        xmin[1] = std::min(xmin[1],xyzp[3*A+1]); 
        xmin[2] = std::min(xmin[2],xyzp[3*A+2]); 
        xmax[0] = std::max(xmax[0],xyzp[3*A+0]); 
        xmax[1] = std::max(xmax[1],xyzp[3*A+1]); 
        xmax[2] = std::max(xmax[2],xyzp[3*A+2]); 
    } 

    xmin[0] -= overage[0];
    xmin[1] -= overage[1];
    xmin[2] -= overage[2];
    xmax[0] += overage[0];
    xmax[1] += overage[1];
    xmax[2] += overage[2];

    std::vector<size_t> ns;
    ns.push_back((size_t) ((spacing[0] == 0.0 ? 0. : ceil((xmax[0] - xmin[0]) / spacing[0])) + 1.));
    ns.push_back((size_t) ((spacing[1] == 0.0 ? 0. : ceil((xmax[1] - xmin[1]) / spacing[1])) + 1.));
    ns.push_back((size_t) ((spacing[2] == 0.0 ? 0. : ceil((xmax[2] - xmin[2]) / spacing[2])) + 1.));
    
    std::vector<double> os;
    os.push_back(0.5 * (xmin[0] + xmax[0]) - 0.5 * (ns[0] - 1) * spacing[0]);
    os.push_back(0.5 * (xmin[1] + xmax[1]) - 0.5 * (ns[1] - 1) * spacing[1]);
    os.push_back(0.5 * (xmin[2] + xmax[2]) - 0.5 * (ns[2] - 1) * spacing[2]);
    
    return std::shared_ptr<CubicGrid>(new CubicGrid(os,spacing,ns));
}
std::string CubicGrid::string() const
{
    std::string str = "";
    str += sprintf2("CubicGrid:\n");
    str += sprintf2("  Total Points: %16zu\n", size());
    str += sprintf2("  X Points:     %16zu\n", sizing_[0]);
    str += sprintf2("  Y Points:     %16zu\n", sizing_[1]);
    str += sprintf2("  Z Points:     %16zu\n", sizing_[2]);
    str += sprintf2("  X Spacing:    %16.8E\n", spacing_[0]);
    str += sprintf2("  Y Spacing:    %16.8E\n", spacing_[1]);
    str += sprintf2("  Z Spacing:    %16.8E\n", spacing_[2]);
    str += sprintf2("  X Minimum:    %16.8E\n", origin_[0]);
    str += sprintf2("  Y Minimum:    %16.8E\n", origin_[1]);
    str += sprintf2("  Z Minimum:    %16.8E\n", origin_[2]);
    str += sprintf2("  X Maximum:    %16.8E\n", origin_[0] + (sizing_[0] - 1.0) * spacing_[0]);
    str += sprintf2("  Y Maximum:    %16.8E\n", origin_[1] + (sizing_[1] - 1.0) * spacing_[1]);
    str += sprintf2("  Z Maximum:    %16.8E\n", origin_[2] + (sizing_[2] - 1.0) * spacing_[2]);
    return str;
}
void CubicGrid::save_cube_file(
    const std::string& filename,
    const std::string& property,
    const std::shared_ptr<Molecule>& mol,
    const std::shared_ptr<Tensor>& v) const
{
    std::vector<size_t> dim;
    dim.push_back(size());
    v->shape_error(dim);

    FILE* fh = fopen(filename.c_str(), "w");
    if (fh == NULL) throw std::runtime_error("CubicGird::save_cube_file; cannot open file: " + filename);

    // Two comment lines
    fprintf(fh, "Lightspeed Gaussian Cube File.\n");
    fprintf(fh, "Property: %s\n", property.c_str());
   
    // Number of atoms plus origin of data 
    fprintf(fh, "%6zu %10.6f %10.6f %10.6f\n", 
        mol->atoms().size(), 
        origin()[0], 
        origin()[1], 
        origin()[2]);
    
    // Number of points along axis, displacement along x,y,z
    fprintf(fh, "%6zu %10.6f %10.6f %10.6f\n", sizing()[0], spacing()[0], 0.0, 0.0);
    fprintf(fh, "%6zu %10.6f %10.6f %10.6f\n", sizing()[1], 0.0, spacing()[1], 0.0);
    fprintf(fh, "%6zu %10.6f %10.6f %10.6f\n", sizing()[2], 0.0, 0.0, spacing()[2]);

    // Atoms of molecule (Z, Q?, x, y, z)
    for (int A = 0; A < mol->atoms().size(); A++) {
        fprintf(fh, "%3d %10.6f %10.6f %10.6f %10.6f\n", 
            (int) mol->atoms()[A].Z(),
            0.0, 
            mol->atoms()[A].x(),
            mol->atoms()[A].y(),
            mol->atoms()[A].z());
    }

    // Data, striped (x, y, z)
    const double* vp = v->data().data();
    for (size_t ind = 0; ind < size(); ind++) {
        fprintf(fh, "%12.5E ", vp[ind]);
        if (ind % 6 == 5) fprintf(fh,"\n");
    }

    fclose(fh);   
}


} // namespace lightspeed
