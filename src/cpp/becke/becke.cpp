#include <lightspeed/becke.hpp>
#include <lightspeed/tensor.hpp>
#include "../util/string.hpp"
#include "atomic_radii.hpp"
#include <cmath>
#include <cstdio>

namespace lightspeed {

BeckeGrid::BeckeGrid(
    const std::shared_ptr<ResourceList>& resources,
    const std::string& name,
    const std::string& atomic_scheme,
    const std::vector<std::shared_ptr<AtomGrid> >& atomic):
    name_(name),
    atomic_scheme_(atomic_scheme),
    atomic_(atomic)
{
    size_ = 0L;
    atomic_starts_.clear();
    atomic_sizes_.clear();
    radial_sizes_.clear();
    spherical_sizes_.clear();
    size_t nA = natom();
    for (int A = 0; A < nA; A++) {
        size_t Asize = atomic_[A]->size();
        atomic_starts_.push_back(size_);
        atomic_sizes_.push_back(Asize);
        radial_sizes_.push_back(atomic_[A]->radial_size());
        spherical_sizes_.push_back(atomic_[A]->spherical_sizes());
        size_ += Asize;
    }

    atomic_inds_.resize(size_);
    for (int A = 0; A < nA; A++) {
        size_t start = atomic_starts_[A];
        size_t size = atomic_sizes_[A];
        for (size_t P = start; P < start + size; P++) {
            atomic_inds_[P] = A;
        }
    } 

    xyzw_ = compute_xyzw(resources);
}

size_t BeckeGrid::total_index(
    size_t atomic_index,
    size_t radial_index,
    size_t spherical_index) const
{
    size_t Astart = atomic_starts_[atomic_index];
    size_t Vstart = atomic_[atomic_index]->atomic_index(radial_index,spherical_index);    
    return Astart + Vstart;
}

size_t BeckeGrid::atomic_index(
    size_t total_index) const 
{
    for (size_t ind = 0L; ind < atomic_starts_.size(); ind++) {
        if (atomic_starts_[ind] + atomic_sizes_[ind] > total_index)    
            return ind;
    }    
    throw std::runtime_error("AtomGrid: atomic index is too large");
}

size_t BeckeGrid::radial_index(
    size_t total_index) const
{
    size_t aind = atomic_index(total_index);
    size_t dind = total_index - atomic_starts_[aind];
    return atomic_[aind]->radial_index(dind);
}

size_t BeckeGrid::spherical_index(
    size_t total_index) const
{
    size_t aind = atomic_index(total_index);
    size_t dind = total_index - atomic_starts_[aind];
    return atomic_[aind]->spherical_index(dind);
}

size_t BeckeGrid::max_spherical_size() const 
{
    size_t maxP = 0L;
    for (size_t aind = 0L; aind < atomic_.size(); aind++) {
        maxP = std::max(atomic_[aind]->max_spherical_size(),maxP);
    }
    return maxP;
}

size_t BeckeGrid::max_radial_size() const 
{
    size_t maxP = 0L;
    for (size_t aind = 0L; aind < atomic_.size(); aind++) {
        maxP = std::max(atomic_[aind]->radial_size(),maxP);
    }
    return maxP;
}

size_t BeckeGrid::max_atomic_size() const 
{
    size_t maxP = 0L;
    for (size_t aind = 0L; aind < atomic_.size(); aind++) {
        maxP = std::max(atomic_[aind]->size(),maxP);
    }
    return maxP;
}

bool BeckeGrid::is_pruned() const 
{
    bool pruned = false;
    for (size_t aind = 0L; aind < atomic_.size(); aind++) {
        if (atomic_[aind]->is_pruned()) {
            pruned = true;
        }
    }
    return pruned;
}

std::shared_ptr<Tensor> BeckeGrid::xyzw_raw() const 
{
    size_t nA = natom();
    size_t nP = size_;
    std::vector<size_t> dimt;
    dimt.push_back(nP);
    dimt.push_back(4);
    std::shared_ptr<Tensor> xyzw(new Tensor(dimt));
    double * xyzwp = xyzw->data().data();
    size_t offset = 0;
    for (int A = 0; A < nA; A++) {
        std::shared_ptr<Tensor> xyzw2 = atomic_[A]->xyzw();
        const double * xyzw2p = xyzw2->data().data();
        size_t nP = xyzw2->shape()[0];
        for (size_t P = 0; P < nP; P++) {
            xyzwp[4*P + 0 + offset] = xyzw2p[4*P + 0];
            xyzwp[4*P + 1 + offset] = xyzw2p[4*P + 1];
            xyzwp[4*P + 2 + offset] = xyzw2p[4*P + 2];
            xyzwp[4*P + 3 + offset] = xyzw2p[4*P + 3];
        }
        offset += 4*nP;
    }
    return xyzw;
}

std::shared_ptr<Tensor> BeckeGrid::xyz() const
{
    std::vector<size_t> dim;
    dim.push_back(size());
    dim.push_back(3);
    std::shared_ptr<Tensor> ret(new Tensor(dim));
    double* retp = ret->data().data();
    
    std::shared_ptr<Tensor> ret2 = xyzw();
    const double* ret2p = ret2->data().data();
    size_t npoint = size();
    for (size_t P = 0; P < npoint; P++) {
        retp[3*P + 0] = ret2p[4*P+0];
        retp[3*P + 1] = ret2p[4*P+1];
        retp[3*P + 2] = ret2p[4*P+2];
    }
    return ret;
}

std::string BeckeGrid::string() const
{
    std::string str = "";
    str += sprintf2("BeckeGrid:\n");
    str += sprintf2("  Name               = %11s\n", name_.c_str());
    str += sprintf2("  Atomic Scheme      = %11s\n", atomic_scheme_.c_str());
    str += sprintf2("  Natom              = %11zu\n", natom());
    str += sprintf2("  Total Size         = %11zu\n", size());
    str += sprintf2("  Max Atomic Size    = %11zu\n", max_atomic_size());
    str += sprintf2("  Max Radial Size    = %11zu\n", max_radial_size());
    str += sprintf2("  Max Spherical Size = %11zu\n", max_spherical_size());
    str += sprintf2("  Pruned             = %11s\n", (is_pruned() ? "Yes" : "No")); 
    return str;
}

std::shared_ptr<Tensor> BeckeGrid::compute_a() const
{
    // Atomic size adjustment parameters
    size_t nA = natom();
    std::vector<size_t> dima;
    dima.push_back(nA);
    dima.push_back(nA);
    std::shared_ptr<Tensor> ah(new Tensor(dima)); 
    double* ap = ah->data().data();
    for (int A = 0; A < nA; A++) {
        for (int B = 0; B < nA; B++) {
            double chi;
            if (atomic_scheme_ == "FLAT") {
                chi = 1.0;
            } else if (atomic_scheme_ == "BECKE") {
                chi = AtomicRadii::bragg_slater_radius(atomic_[A]->N()) /
                      AtomicRadii::bragg_slater_radius(atomic_[B]->N());
            } else if (atomic_scheme_ == "AHLRICHS") {
                chi = sqrt(AtomicRadii::bragg_slater_radius(atomic_[A]->N()) /
                           AtomicRadii::bragg_slater_radius(atomic_[B]->N()));
            } else {
                throw std::runtime_error("BeckeGrid: invalid atomic weighting scheme");
            }
            double u = (chi - 1.0) / (chi + 1.0);
            double a = u / (u * u - 1.0);
            a = (a > 0.5 ? 0.5 : a);
            a = (a < -0.5 ? -0.5 : a);
            ap[A*nA + B] = a;
        }
    }
    return ah;
}
std::shared_ptr<Tensor> BeckeGrid::compute_rinv() const
{
    size_t nA = natom();
    std::vector<size_t> dima;
    dima.push_back(nA);
    dima.push_back(nA);
    std::shared_ptr<Tensor> rinv(new Tensor(dima));
    double* rinvp = rinv->data().data();
    for (int A = 0; A < nA; A++) {
        for (int B = 0; B < nA; B++) {
            if (A != B) {
                double dx = atomic_[A]->x() - atomic_[B]->x();
                double dy = atomic_[A]->y() - atomic_[B]->y();
                double dz = atomic_[A]->z() - atomic_[B]->z();
                double r = sqrt(dx * dx + dy * dy + dz *dz);
                rinvp[A*nA + B] = 1.0 / r;
            }
        }
    }
    return rinv;
}
std::shared_ptr<Tensor> BeckeGrid::compute_xyz_atoms() const
{
    size_t nA = natom();
    std::vector<size_t> dimG;
    dimG.push_back(nA);
    dimG.push_back(3);
    std::shared_ptr<Tensor> xyz(new Tensor(dimG));
    double* xyzp = xyz->data().data();
    for (int A = 0; A < nA; A++) {
        xyzp[3*A + 0] = atomic_[A]->x();
        xyzp[3*A + 1] = atomic_[A]->y();
        xyzp[3*A + 2] = atomic_[A]->z();
    }
    return xyz;
}
std::shared_ptr<Tensor> BeckeGrid::compute_xyzw(
    const std::shared_ptr<ResourceList>& resources) const
{
    // Compute weights
    size_t nA = natom();
    size_t nP = size();
       
    std::shared_ptr<Tensor> a = compute_a();
    const double* ap = a->data().data();

    std::shared_ptr<Tensor> rinv = compute_rinv();
    const double* rinvp = rinv->data().data(); 

    std::shared_ptr<Tensor> xyzw = xyzw_raw();
    double* xyzwp = xyzw->data().data();

    std::shared_ptr<Tensor> xyz = compute_xyz_atoms();
    const double* xyzp = xyz->data().data();

    std::vector<double> r(nA);
    for (int P = 0; P < nP; P++) {
        double xP = xyzwp[4*P + 0];
        double yP = xyzwp[4*P + 1];
        double zP = xyzwp[4*P + 2];
        for (int A = 0; A < nA; A++) {
            double dx = xyzp[3*A + 0] - xP;
            double dy = xyzp[3*A + 1] - yP;
            double dz = xyzp[3*A + 2] - zP;
            r[A] = sqrt(dx * dx + dy * dy + dz * dz);
        }
        double num = 0.0;
        double den = 0.0;
        for (int A = 0; A < nA; A++) {
            double Pval = 1.0;
            for (int B = 0; B < nA; B++) {
                if (A == B) continue;
                double mu = (r[A] - r[B]) * rinvp[A*nA + B];
                double nu = mu + ap[A*nA + B] * (1.0 - mu * mu);
                double f = nu;
                f = 1.5 * f - 0.5 * f * f * f;
                f = 1.5 * f - 0.5 * f * f * f;
                f = 1.5 * f - 0.5 * f * f * f;
                double s = 0.5 * (1.0 - f);
                Pval *= s;
            } 
            if (atomic_inds_[P] == A) num = Pval;
            den += Pval;           
        }
        xyzwp[4*P + 3] *= num / den;
    }
    
    return xyzw;
}
std::shared_ptr<Tensor> BeckeGrid::grad(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& v,
    const std::shared_ptr<Tensor>& GA
    )
{
    size_t nA = natom();
    size_t nP = size();

    std::vector<size_t> dimv;
    dimv.push_back(nP);
    v->shape_error(dimv);

    std::vector<size_t> dimG;
    dimG.push_back(nA);
    dimG.push_back(3);
    std::shared_ptr<Tensor> GA2 = GA;
    if (!GA) {
        GA2 = std::shared_ptr<Tensor>(new Tensor(dimG));
    }
    GA2->shape_error(dimG);
    double* GA2p = GA2->data().data();

    std::shared_ptr<Tensor> a = compute_a();
    const double* ap = a->data().data();

    std::shared_ptr<Tensor> rinv = compute_rinv();
    const double* rinvp = rinv->data().data(); 

    std::shared_ptr<Tensor> xyzw = xyzw_raw();
    const double* xyzwp = xyzw->data().data();

    std::shared_ptr<Tensor> xyz = compute_xyz_atoms();
    const double* xyzp = xyz->data().data();

    const double* vp = v->data().data();

    double Exc = 0.0;
    std::vector<double> Ps(nA);
    std::vector<double> r(nA);
    for (int P = 0; P < nP; P++) {
        double xP = xyzwp[4*P + 0];
        double yP = xyzwp[4*P + 1];
        double zP = xyzwp[4*P + 2];
        double eP = xyzwp[4*P + 3] * vp[P];
        double num = 0.0;
        double den = 0.0;
        for (int A = 0; A < nA; A++) {
            double dx = xyzp[3*A + 0] - xP;
            double dy = xyzp[3*A + 1] - yP;
            double dz = xyzp[3*A + 2] - zP;
            r[A] = sqrt(dx * dx + dy * dy + dz * dz);
        }
        for (int A = 0; A < nA; A++) {
            double Pval = 1.0;
            for (int B = 0; B < nA; B++) {
                if (A == B) continue;
                double mu = (r[A] - r[B]) * rinvp[A*nA + B];
                double nu = mu + ap[A*nA + B] * (1.0 - mu * mu);
                double f = nu;
                f = 1.5 * f - 0.5 * f * f * f;
                f = 1.5 * f - 0.5 * f * f * f;
                f = 1.5 * f - 0.5 * f * f * f;
                double s = 0.5 * (1.0 - f);
                Pval *= s;
            } 
            if (atomic_inds_[P] == A) num = Pval;
            den += Pval;           
            Ps[A] = Pval;
        }
        int D = atomic_inds_[P];
        for (int A = 0; A < nA; A++) {
            double E_P = - num / (den * den) * eP;
            if (atomic_inds_[P] == A) E_P += 1.0 / den * eP;
            for (int B = 0; B < nA; B++) {
                if (A == B) continue;
                double mu = (r[A] - r[B]) * rinvp[A*nA + B];
                double nu = mu + ap[A*nA + B] * (1.0 - mu * mu);
                double f = nu;
                double s_nu = 1.5 * (1.0 - nu * nu);
                f = 1.5 * f - 0.5 * f * f * f;
                s_nu *= 1.5 * (1.0 - f * f);
                f = 1.5 * f - 0.5 * f * f * f;
                s_nu *= 1.5 * (1.0 - f * f);
                f = 1.5 * f - 0.5 * f * f * f;
                double s = 0.5 * (1.0 - f);
                s_nu *= -0.5;
                if (s == 0.0) continue;
                double E_mu = E_P * Ps[A] * s_nu / s * (1.0 - 2.0 * ap[A*nA + B] * mu);

                double xA = xyzp[3*A + 0];
                double yA = xyzp[3*A + 1];
                double zA = xyzp[3*A + 2];
                double xB = xyzp[3*B + 0];
                double yB = xyzp[3*B + 1];
                double zB = xyzp[3*B + 2];

                double rAP = r[A];
                double rBP = r[B];
                double rABinv = rinvp[A*nA + B];

                GA2p[3*A + 0] += (- (rAP - rBP) * (xA - xB) * pow(rABinv,3)) * E_mu;
                GA2p[3*A + 1] += (- (rAP - rBP) * (yA - yB) * pow(rABinv,3)) * E_mu;
                GA2p[3*A + 2] += (- (rAP - rBP) * (zA - zB) * pow(rABinv,3)) * E_mu;

                GA2p[3*B + 0] += (  (rAP - rBP) * (xA - xB) * pow(rABinv,3)) * E_mu;
                GA2p[3*B + 1] += (  (rAP - rBP) * (yA - yB) * pow(rABinv,3)) * E_mu;
                GA2p[3*B + 2] += (  (rAP - rBP) * (zA - zB) * pow(rABinv,3)) * E_mu;

                GA2p[3*A + 0] += (  rABinv * (xA - xP) / rAP) * E_mu;
                GA2p[3*A + 1] += (  rABinv * (yA - yP) / rAP) * E_mu;
                GA2p[3*A + 2] += (  rABinv * (zA - zP) / rAP) * E_mu;

                GA2p[3*B + 0] += (- rABinv * (xB - xP) / rBP) * E_mu;
                GA2p[3*B + 1] += (- rABinv * (yB - yP) / rBP) * E_mu;
                GA2p[3*B + 2] += (- rABinv * (zB - zP) / rBP) * E_mu;

                GA2p[3*D + 0] -= (  rABinv * (xA - xP) / rAP) * E_mu;
                GA2p[3*D + 1] -= (  rABinv * (yA - yP) / rAP) * E_mu;
                GA2p[3*D + 2] -= (  rABinv * (zA - zP) / rAP) * E_mu;

                GA2p[3*D + 0] -= (- rABinv * (xB - xP) / rBP) * E_mu;
                GA2p[3*D + 1] -= (- rABinv * (yB - yP) / rBP) * E_mu;
                GA2p[3*D + 2] -= (- rABinv * (zB - zP) / rBP) * E_mu;
            }
        }
    }
        
    return GA2;
}

} // namespace lightspeed 
