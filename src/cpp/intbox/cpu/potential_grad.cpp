#include <lightspeed/intbox.hpp>
#include <lightspeed/am.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/resource_list.hpp>
#include <lightspeed/pair_list.hpp>
#include <lightspeed/pure_transform.hpp>
#include <lightspeed/ewald.hpp>
#include <lightspeed/hermite.hpp>
#include <lightspeed/gh.hpp>
#include <lightspeed/rys.hpp>
#include <cmath>
#include <cstring>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace lightspeed { 

std::shared_ptr<Tensor> IntBox::potentialGradCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<Tensor>& xyzc,
    const std::shared_ptr<Tensor>& G
    )
{
    // Throws if pairlist is not symmetric
    if (!pairlist->is_symmetric()) {
        throw std::runtime_error("IntBox::potentialGradCPU: Pairlist should be symmetric.");
    }

    // Throws if the number of point charges does not equal the number of atoms in the basis set
    if (pairlist->basis1()->natom() != xyzc->shape()[0]) {
        throw std::runtime_error("IntBox::potentialGradCPU: Inconsistent number of atoms!");
    }

    // Size of G
    std::vector<size_t> dim1;
    dim1.push_back(pairlist->basis1()->natom());
    dim1.push_back(3);
        
    // The working register of G
    std::shared_ptr<Tensor> GT = G;
    if (!G) {
        GT = std::shared_ptr<Tensor>(new Tensor(dim1));
    }
    GT->shape_error(dim1);
    
    // => Setup <= //
    
    // Spherical to Cartesian transform
    std::shared_ptr<Tensor> D2 = PureTransform::pureToCart2(
        pairlist->basis1(),
        pairlist->basis2(),
        D);  

    // Hermite representation of basis pairs
    std::shared_ptr<Hermite> Vhermite(new Hermite(pairlist)); 
    std::shared_ptr<Hermite> Dhermite(new Hermite(pairlist)); 

    // Transformation of D from cartesians to Hermite
    Dhermite->cart_to_herm(D2, false);

    std::vector<HermiteL>& Vherms = Vhermite->hermites();
    std::vector<HermiteL>& Dherms = Dhermite->hermites();

    // Grab Rys singleton
    std::shared_ptr<Rys> rys = Rys::instance();
    const Rys* rysp = rys.get();

    // Extract xyzc data
    size_t nA = xyzc->shape()[0];
    const double* xyzcp = xyzc->data().data();

    // Magic masks for extracting Hermite tx,ty,tz
    const int xmask = (1 << 30) - (1 << 20);
    const int ymask = (1 << 20) - (1 << 10);
    const int zmask = (1 << 10) - (1 <<  0);

    // local copies of G on each thread
    std::vector<std::shared_ptr<Tensor> > G_omp;
    for (int tid = 0; tid < resources->nthread(); tid++) {
        G_omp.push_back(std::shared_ptr<Tensor>(new Tensor(dim1)));
    }

    // => Master Loop <= //

    for (size_t ewald_ind = 0; ewald_ind < ewald->scales().size(); ewald_ind++) {
        double alpha = ewald->scales()[ewald_ind];
        double omega = ewald->omegas()[ewald_ind];
        if ((alpha == 0.0) || (omega == 0.0)) continue;

        for (size_t ind_L = 0; ind_L < Vherms.size(); ind_L++) {
            HermiteL& Vherm = Vherms[ind_L];
            HermiteL& Dherm = Dherms[ind_L];
            int L = Vherm.L();
            int V = L + 1;
            int H = Vherm.H();
            
            size_t npair = Vherm.npair();
            double* datap = Vherm.data().data();
            const double* Ddatap = Dherm.data().data();
            const double* geomp = Vherm.geom().data();
            const double* etap = Vherm.etas().data();
            const int* indp = Vherm.inds().data();

            int nrys = (L + 1)/2 + 1;
            std::vector<int> angl = HermiteUtil::angl(H);
            
            #pragma omp parallel for schedule(dynamic) num_threads(resources->nthread())
            for (size_t pair_ind = 0; pair_ind < npair; pair_ind++) {

                // raw pointers for each thread
#ifdef _OPENMP
                int tid = omp_get_thread_num();
#else
                int tid = 0;
#endif
                double* Gp = G_omp[tid]->data().data();
                
                // Geometry
                double pP = geomp[4*pair_ind + 0];
                double xP = geomp[4*pair_ind + 1];
                double yP = geomp[4*pair_ind + 2];
                double zP = geomp[4*pair_ind + 3];
                double eta1 = etap[2*pair_ind + 0];
                double eta2 = etap[2*pair_ind + 1];
                int A1 = indp[2*pair_ind + 0];
                int A2 = indp[2*pair_ind + 1];

                double pref = alpha * 2.0 * M_PI / pP;
                double d2 = 1.0;
                if (omega != -1.0){
                    d2 = omega*omega / (pP + omega*omega);
                    pref *= sqrt(d2);
                }

                double vbuf[H];
                ::memset(vbuf,'\0',sizeof(double)*H);

                double Dbuf[H];
                ::memcpy(Dbuf, Ddatap + pair_ind*H, sizeof(double)*H);

                for (size_t A = 0; A < nA; A++) {
                    double xA = xyzcp[4*A + 0];
                    double yA = xyzcp[4*A + 1];
                    double zA = xyzcp[4*A + 2];
                    double cA = xyzcp[4*A + 3];

                    double xPA = xP - xA;
                    double yPA = yP - yA;
                    double zPA = zP - zA;
                    double rPA2 = pow(xPA,2) + pow(yPA,2) + pow(zPA,2);
                    double T = pP * d2 * rPA2;

                    for (int i = 0; i < nrys; i++) {
                        double t = rysp->interpolate_t(nrys, i, T);
                        double w = rysp->interpolate_w(nrys, i, T);
                        double t2 = t*t;
                        double U = -2.0 * pP * d2 * t2;
                        double Hx[L + 2];
                        double Hy[L + 2];
                        double Hz[L + 2];
                        Hx[0] = -pref*w*cA;
                        Hy[0] = 1.0;
                        Hz[0] = 1.0;
                        Hx[1] = U * xPA * Hx[0];
                        Hy[1] = U * yPA;
                        Hz[1] = U * zPA;
                        for (int u = 1; u < L + 1; u++) {
                            Hx[u+1] = U * (xPA*Hx[u] + u*Hx[u-1]);
                            Hy[u+1] = U * (yPA*Hy[u] + u*Hy[u-1]);
                            Hz[u+1] = U * (zPA*Hz[u] + u*Hz[u-1]);
                        }
                        for (int t_ind = 0; t_ind < H; t_ind++) {
                            int t_vec = angl[t_ind];
                            int tx = (t_vec & xmask) >> 20;
                            int ty = (t_vec & ymask) >> 10;
                            int tz = (t_vec & zmask);

                            vbuf[t_ind] += Hx[tx] * Hy[ty] * Hz[tz];
                            
                            Gp[3*A1 + 0] += Dbuf[t_ind] * Hx[tx+1] * Hy[ty]   * Hz[tz]   * eta1;
                            Gp[3*A1 + 1] += Dbuf[t_ind] * Hx[tx]   * Hy[ty+1] * Hz[tz]   * eta1;
                            Gp[3*A1 + 2] += Dbuf[t_ind] * Hx[tx]   * Hy[ty]   * Hz[tz+1] * eta1;

                            Gp[3*A2 + 0] += Dbuf[t_ind] * Hx[tx+1] * Hy[ty]   * Hz[tz]   * eta2;
                            Gp[3*A2 + 1] += Dbuf[t_ind] * Hx[tx]   * Hy[ty+1] * Hz[tz]   * eta2;
                            Gp[3*A2 + 2] += Dbuf[t_ind] * Hx[tx]   * Hy[ty]   * Hz[tz+1] * eta2;

                            Gp[3*A  + 0] -= Dbuf[t_ind] * Hx[tx+1] * Hy[ty]   * Hz[tz];
                            Gp[3*A  + 1] -= Dbuf[t_ind] * Hx[tx]   * Hy[ty+1] * Hz[tz];
                            Gp[3*A  + 2] -= Dbuf[t_ind] * Hx[tx]   * Hy[ty]   * Hz[tz+1];
                        }
                    }
                }
                
                double* data2p = datap + pair_ind*H;
                for (int t_ind = 0; t_ind < H; t_ind++) {
                    (*data2p++) += vbuf[t_ind];
                }
            }
        }
    }

    // collect gradient from each thread
    for (int tid = 0; tid < resources->nthread(); tid++) {
        for (int i = 0; i < GT->size(); i++) {
            GT->data()[i] += G_omp[tid]->data()[i];
        }
    }

    // gradient contribution from Hermite coefficients
    Vhermite->grad(D2, GT, GT, false);

    return GT;
}

std::vector<std::shared_ptr<Tensor> > IntBox::potentialGradAdvCPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<Tensor>& xyzc,
    const std::vector<std::shared_ptr<Tensor> >& G12A 
    )
{
    // Throws if pairlist is symmetric
    if (pairlist->is_symmetric()) {
        throw std::runtime_error("IntBox::potentialGradAdvCPU: Pairlist should not be symmetric.");
    }

    // Number of atoms in each basis
    std::vector<size_t> natom(3);
    natom[0] = pairlist->basis1()->natom();
    natom[1] = pairlist->basis2()->natom();
    natom[2] = xyzc->shape()[0];

    // sizes of G1,G2 and GA
    std::vector<std::vector<size_t> > dim(3);
    for (int a = 0; a < 3; a++) {
        dim[a].push_back(natom[a]);
        dim[a].push_back(3);
    }

    // The working register of G1, G2 and GA
    std::vector<std::shared_ptr<Tensor> > G12AT;
    if (G12A.size() == 0) {
        // The user has not allocated G12A and we should allocate it
        for (int a = 0; a < 3; a++) {
            G12AT.push_back(std::shared_ptr<Tensor>(new Tensor(dim[a])));
        }
    } else if (G12A.size() == 3) {
        // The user has allocated G12A and we should check that this is the right size
        G12AT = G12A;
        for (int a = 0; a < 3; a++) {
            G12AT[a]->shape_error(dim[a]);
        }
    } else {
        // The user doesn't know how many perturbations there are
        throw std::runtime_error("IntBox::overlapGradAdvCPU: G12A should be size 0 or size 3");
    }
    
    // => Setup <= //
    
    // Spherical to Cartesian transform
    std::shared_ptr<Tensor> D2 = PureTransform::pureToCart2(
        pairlist->basis1(),
        pairlist->basis2(),
        D);  

    // Hermite representation of basis pairs
    std::shared_ptr<Hermite> Vhermite(new Hermite(pairlist)); 
    std::shared_ptr<Hermite> Dhermite(new Hermite(pairlist)); 
    
    // Transformation of D from cartesians to Hermite
    Dhermite->cart_to_herm(D2, false);

    std::vector<HermiteL>& Vherms = Vhermite->hermites();
    std::vector<HermiteL>& Dherms = Dhermite->hermites();

    // Grab Rys singleton
    std::shared_ptr<Rys> rys = Rys::instance();
    const Rys* rysp = rys.get();

    // Extract xyzc data
    size_t nA = xyzc->shape()[0];
    const double* xyzcp = xyzc->data().data();

    // Magic masks for extracting Hermite tx,ty,tz
    const int xmask = (1 << 30) - (1 << 20);
    const int ymask = (1 << 20) - (1 << 10);
    const int zmask = (1 << 10) - (1 <<  0);

    // local copies of G on each thread
    std::vector<std::vector<std::shared_ptr<Tensor> > > G_omp(3);
    for (int a = 0; a < 3; a++) {
        for (int tid = 0; tid < resources->nthread(); tid++) {
            G_omp[a].push_back(std::shared_ptr<Tensor>(new Tensor(dim[a])));
        }
    }

    // => Master Loop <= //
    
    for (size_t ewald_ind = 0; ewald_ind < ewald->scales().size(); ewald_ind++) {
        double alpha = ewald->scales()[ewald_ind];
        double omega = ewald->omegas()[ewald_ind];
        if ((alpha == 0.0) || (omega == 0.0)) continue;

        for (size_t ind_L = 0; ind_L < Vherms.size(); ind_L++) {
            HermiteL& Vherm = Vherms[ind_L];
            HermiteL& Dherm = Dherms[ind_L];
            int L = Vherm.L();
            int V = L + 1;
            int H = Vherm.H();
            
            size_t npair = Vherm.npair();
            double* datap = Vherm.data().data();
            const double* Ddatap = Dherm.data().data();
            const double* geomp = Vherm.geom().data();
            const double* etap = Vherm.etas().data();
            const int* indp = Vherm.inds().data();

            int nrys = (L + 1)/2 + 1;
            std::vector<int> angl = HermiteUtil::angl(H);
            
            #pragma omp parallel for schedule(dynamic) num_threads(resources->nthread())
            for (size_t pair_ind = 0; pair_ind < npair; pair_ind++) {

                // raw pointers for each thread
#ifdef _OPENMP
                int tid = omp_get_thread_num();
#else
                int tid = 0;
#endif
                double* G1p = G_omp[0][tid]->data().data();
                double* G2p = G_omp[1][tid]->data().data();
                double* GAp = G_omp[2][tid]->data().data();
                
                // Geometry
                double pP = geomp[4*pair_ind + 0];
                double xP = geomp[4*pair_ind + 1];
                double yP = geomp[4*pair_ind + 2];
                double zP = geomp[4*pair_ind + 3];
                double eta1 = etap[2*pair_ind + 0];
                double eta2 = etap[2*pair_ind + 1];
                int A1 = indp[2*pair_ind + 0];
                int A2 = indp[2*pair_ind + 1];

                double pref = alpha * 2.0 * M_PI / pP;
                double d2 = 1.0;
                if (omega != -1.0){
                    d2 = omega*omega / (pP + omega*omega);
                    pref *= sqrt(d2);
                }

                double vbuf[H];
                ::memset(vbuf,'\0',sizeof(double)*H);

                double Dbuf[H];
                ::memcpy(Dbuf, Ddatap + pair_ind*H, sizeof(double)*H);

                for (size_t A = 0; A < nA; A++) {
                    double xA = xyzcp[4*A + 0];
                    double yA = xyzcp[4*A + 1];
                    double zA = xyzcp[4*A + 2];
                    double cA = xyzcp[4*A + 3];

                    double xPA = xP - xA;
                    double yPA = yP - yA;
                    double zPA = zP - zA;
                    double rPA2 = pow(xPA,2) + pow(yPA,2) + pow(zPA,2);
                    double T = pP * d2 * rPA2;

                    for (int i = 0; i < nrys; i++) {
                        double t = rysp->interpolate_t(nrys, i, T);
                        double w = rysp->interpolate_w(nrys, i, T);
                        double t2 = t*t;
                        double U = -2.0 * pP * d2 * t2;
                        double Hx[L + 2];
                        double Hy[L + 2];
                        double Hz[L + 2];
                        Hx[0] = -pref*w*cA;
                        Hy[0] = 1.0;
                        Hz[0] = 1.0;
                        Hx[1] = U * xPA * Hx[0];
                        Hy[1] = U * yPA;
                        Hz[1] = U * zPA;
                        for (int u = 1; u < L + 1; u++) {
                            Hx[u+1] = U * (xPA*Hx[u] + u*Hx[u-1]);
                            Hy[u+1] = U * (yPA*Hy[u] + u*Hy[u-1]);
                            Hz[u+1] = U * (zPA*Hz[u] + u*Hz[u-1]);
                        }
                        for (int t_ind = 0; t_ind < H; t_ind++) {
                            int t_vec = angl[t_ind];
                            int tx = (t_vec & xmask) >> 20;
                            int ty = (t_vec & ymask) >> 10;
                            int tz = (t_vec & zmask);

                            vbuf[t_ind] += Hx[tx] * Hy[ty] * Hz[tz];
                            
                            G1p[3*A1 + 0] += Dbuf[t_ind] * Hx[tx+1] * Hy[ty]   * Hz[tz]   * eta1;
                            G1p[3*A1 + 1] += Dbuf[t_ind] * Hx[tx]   * Hy[ty+1] * Hz[tz]   * eta1;
                            G1p[3*A1 + 2] += Dbuf[t_ind] * Hx[tx]   * Hy[ty]   * Hz[tz+1] * eta1;

                            G2p[3*A2 + 0] += Dbuf[t_ind] * Hx[tx+1] * Hy[ty]   * Hz[tz]   * eta2;
                            G2p[3*A2 + 1] += Dbuf[t_ind] * Hx[tx]   * Hy[ty+1] * Hz[tz]   * eta2;
                            G2p[3*A2 + 2] += Dbuf[t_ind] * Hx[tx]   * Hy[ty]   * Hz[tz+1] * eta2;

                            GAp[3*A  + 0] -= Dbuf[t_ind] * Hx[tx+1] * Hy[ty]   * Hz[tz];
                            GAp[3*A  + 1] -= Dbuf[t_ind] * Hx[tx]   * Hy[ty+1] * Hz[tz];
                            GAp[3*A  + 2] -= Dbuf[t_ind] * Hx[tx]   * Hy[ty]   * Hz[tz+1];
                        }
                    }
                }
                
                double* data2p = datap + pair_ind*H;
                for (int t_ind = 0; t_ind < H; t_ind++) {
                    (*data2p++) += vbuf[t_ind];
                }
            }
        }
    }
    
    // collect gradient from each thread
    for (int a = 0; a < 3; a++) {
        for (int tid = 0; tid < resources->nthread(); tid++) {
            for (int i = 0; i < G12AT[a]->size(); i++) {
               G12AT[a]->data()[i] += G_omp[a][tid]->data()[i];
            }
        }
    }

    // gradient contribution from Hermite coefficients
    Vhermite->grad(D2, G12AT[0], G12AT[1], false);

    return G12AT;
}

std::vector<std::shared_ptr<Tensor> > IntBox::potentialGradAdv2CPU(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Ewald>& ewald,
    const std::shared_ptr<PairList>& pairlist,
    const std::shared_ptr<Tensor>& D,
    const std::shared_ptr<Tensor>& xyzc,
    const std::vector<std::shared_ptr<Tensor> >& G12A 
    )
{
    // Throws if pairlist is not symmetric
    if (!pairlist->is_symmetric()) {
        throw std::runtime_error("IntBox::potentialGradAdv2CPU: Pairlist should be symmetric.");
    }

    // Number of atoms in each basis
    std::vector<size_t> natom(2);
    natom[0] = pairlist->basis1()->natom();
    natom[1] = xyzc->shape()[0];

    // sizes of G12 and GA
    std::vector<std::vector<size_t> > dim(2);
    for (int a = 0; a < 2; a++) {
        dim[a].push_back(natom[a]);
        dim[a].push_back(3);
    }

    // The working register of G12 and GA
    std::vector<std::shared_ptr<Tensor> > G12AT;
    if (G12A.size() == 0) {
        // The user has not allocated G12A and we should allocate it
        for (int a = 0; a < 2; a++) {
            G12AT.push_back(std::shared_ptr<Tensor>(new Tensor(dim[a])));
        }
    } else if (G12A.size() == 2) {
        // The user has allocated G12A and we should check that this is the right size
        G12AT = G12A;
        for (int a = 0; a < 2; a++) {
            G12AT[a]->shape_error(dim[a]);
        }
    } else {
        // The user doesn't know how many perturbations there are
        throw std::runtime_error("IntBox::overlapGradAdv2CPU: G12A should be size 0 or size 2");
    }
    
    // => Setup <= //
    
    // Spherical to Cartesian transform
    std::shared_ptr<Tensor> D2 = PureTransform::pureToCart2(
        pairlist->basis1(),
        pairlist->basis2(),
        D);  

    // Hermite representation of basis pairs
    std::shared_ptr<Hermite> Vhermite(new Hermite(pairlist)); 
    std::shared_ptr<Hermite> Dhermite(new Hermite(pairlist)); 

    // Transformation of D from cartesians to Hermite
    Dhermite->cart_to_herm(D2, false);

    std::vector<HermiteL>& Vherms = Vhermite->hermites();
    std::vector<HermiteL>& Dherms = Dhermite->hermites();

    // Grab Rys singleton
    std::shared_ptr<Rys> rys = Rys::instance();
    const Rys* rysp = rys.get();

    // Extract xyzc data
    size_t nA = xyzc->shape()[0];
    const double* xyzcp = xyzc->data().data();

    // Magic masks for extracting Hermite tx,ty,tz
    const int xmask = (1 << 30) - (1 << 20);
    const int ymask = (1 << 20) - (1 << 10);
    const int zmask = (1 << 10) - (1 <<  0);

    // local copies of G on each thread
    std::vector<std::vector<std::shared_ptr<Tensor> > > G_omp(2);
    for (int a = 0; a < 2; a++) {
        for (int tid = 0; tid < resources->nthread(); tid++) {
            G_omp[a].push_back(std::shared_ptr<Tensor>(new Tensor(dim[a])));
        }
    }

    // => Master Loop <= //
    
    for (size_t ewald_ind = 0; ewald_ind < ewald->scales().size(); ewald_ind++) {
        double alpha = ewald->scales()[ewald_ind];
        double omega = ewald->omegas()[ewald_ind];
        if ((alpha == 0.0) || (omega == 0.0)) continue;

        for (size_t ind_L = 0; ind_L < Vherms.size(); ind_L++) {
            HermiteL& Vherm = Vherms[ind_L];
            HermiteL& Dherm = Dherms[ind_L];
            int L = Vherm.L();
            int V = L + 1;
            int H = Vherm.H();
            
            size_t npair = Vherm.npair();
            double* datap = Vherm.data().data();
            const double* Ddatap = Dherm.data().data();
            const double* geomp = Vherm.geom().data();
            const double* etap = Vherm.etas().data();
            const int* indp = Vherm.inds().data();

            int nrys = (L + 1)/2 + 1;
            std::vector<int> angl = HermiteUtil::angl(H);
            
            #pragma omp parallel for schedule(dynamic) num_threads(resources->nthread())
            for (size_t pair_ind = 0; pair_ind < npair; pair_ind++) {

                // raw pointers for each thread
#ifdef _OPENMP
                int tid = omp_get_thread_num();
#else
                int tid = 0;
#endif
                double* G12p = G_omp[0][tid]->data().data();
                double* GAp  = G_omp[1][tid]->data().data();
                
                // Geometry
                double pP = geomp[4*pair_ind + 0];
                double xP = geomp[4*pair_ind + 1];
                double yP = geomp[4*pair_ind + 2];
                double zP = geomp[4*pair_ind + 3];
                double eta1 = etap[2*pair_ind + 0];
                double eta2 = etap[2*pair_ind + 1];
                int A1 = indp[2*pair_ind + 0];
                int A2 = indp[2*pair_ind + 1];

                double pref = alpha * 2.0 * M_PI / pP;
                double d2 = 1.0;
                if (omega != -1.0){
                    d2 = omega*omega / (pP + omega*omega);
                    pref *= sqrt(d2);
                }

                double vbuf[H];
                ::memset(vbuf,'\0',sizeof(double)*H);

                double Dbuf[H];
                ::memcpy(Dbuf, Ddatap + pair_ind*H, sizeof(double)*H);

                for (size_t A = 0; A < nA; A++) {
                    double xA = xyzcp[4*A + 0];
                    double yA = xyzcp[4*A + 1];
                    double zA = xyzcp[4*A + 2];
                    double cA = xyzcp[4*A + 3];

                    double xPA = xP - xA;
                    double yPA = yP - yA;
                    double zPA = zP - zA;
                    double rPA2 = pow(xPA,2) + pow(yPA,2) + pow(zPA,2);
                    double T = pP * d2 * rPA2;

                    for (int i = 0; i < nrys; i++) {
                        double t = rysp->interpolate_t(nrys, i, T);
                        double w = rysp->interpolate_w(nrys, i, T);
                        double t2 = t*t;
                        double U = -2.0 * pP * d2 * t2;
                        double Hx[L + 2];
                        double Hy[L + 2];
                        double Hz[L + 2];
                        Hx[0] = -pref*w*cA;
                        Hy[0] = 1.0;
                        Hz[0] = 1.0;
                        Hx[1] = U * xPA * Hx[0];
                        Hy[1] = U * yPA;
                        Hz[1] = U * zPA;
                        for (int u = 1; u < L + 1; u++) {
                            Hx[u+1] = U * (xPA*Hx[u] + u*Hx[u-1]);
                            Hy[u+1] = U * (yPA*Hy[u] + u*Hy[u-1]);
                            Hz[u+1] = U * (zPA*Hz[u] + u*Hz[u-1]);
                        }
                        for (int t_ind = 0; t_ind < H; t_ind++) {
                            int t_vec = angl[t_ind];
                            int tx = (t_vec & xmask) >> 20;
                            int ty = (t_vec & ymask) >> 10;
                            int tz = (t_vec & zmask);

                            vbuf[t_ind] += Hx[tx] * Hy[ty] * Hz[tz];
                            
                            G12p[3*A1 + 0] += Dbuf[t_ind] * Hx[tx+1] * Hy[ty]   * Hz[tz]   * eta1;
                            G12p[3*A1 + 1] += Dbuf[t_ind] * Hx[tx]   * Hy[ty+1] * Hz[tz]   * eta1;
                            G12p[3*A1 + 2] += Dbuf[t_ind] * Hx[tx]   * Hy[ty]   * Hz[tz+1] * eta1;

                            G12p[3*A2 + 0] += Dbuf[t_ind] * Hx[tx+1] * Hy[ty]   * Hz[tz]   * eta2;
                            G12p[3*A2 + 1] += Dbuf[t_ind] * Hx[tx]   * Hy[ty+1] * Hz[tz]   * eta2;
                            G12p[3*A2 + 2] += Dbuf[t_ind] * Hx[tx]   * Hy[ty]   * Hz[tz+1] * eta2;

                            GAp[3*A  + 0] -= Dbuf[t_ind] * Hx[tx+1] * Hy[ty]   * Hz[tz];
                            GAp[3*A  + 1] -= Dbuf[t_ind] * Hx[tx]   * Hy[ty+1] * Hz[tz];
                            GAp[3*A  + 2] -= Dbuf[t_ind] * Hx[tx]   * Hy[ty]   * Hz[tz+1];
                        }
                    }
                }
                
                double* data2p = datap + pair_ind*H;
                for (int t_ind = 0; t_ind < H; t_ind++) {
                    (*data2p++) += vbuf[t_ind];
                }
            }
        }
    }
    
    // collect gradient from each thread
    for (int a = 0; a < 2; a++) {
        for (int tid = 0; tid < resources->nthread(); tid++) {
            for (int i = 0; i < G12AT[a]->size(); i++) {
               G12AT[a]->data()[i] += G_omp[a][tid]->data()[i];
            }
        }
    }

    // gradient contribution from Hermite coefficients
    Vhermite->grad(D2, G12AT[0], G12AT[0], false);

    return G12AT;
}

} // namespace lightspeed
