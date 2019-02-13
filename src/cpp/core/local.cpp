#include <lightspeed/local.hpp>
#include <lightspeed/basis.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/math.hpp>
#include <math.h>
#include <stdexcept>

namespace lightspeed {

std::shared_ptr<Tensor> Local::localize(
    int power,
    int maxiter,
    double convergence,
    const std::shared_ptr<Basis>& basis,
    const std::shared_ptr<Tensor>& L,
    const std::shared_ptr<Tensor>& U)
{
    // => Validity Checks <= //

    if (!(power == 2 || power == 4)) throw std::runtime_error("Local:localize: power must be 2 or 4");
    U->square_error();
    L->shape_error({U->shape()[0],basis->nao()});

    // => Sizes <= //

    size_t nA = basis->natom();
    size_t nmin = basis->nao();
    size_t nmo  = L->shape()[0];
    
    // => IAO Basis Orbitals <= //
    
    double* Lp = L->data().data();
    double* Up = U->data().data();
    U->identity();

    // => Active MinAOs <= // 

    std::vector<std::vector<int>> minao_inds(nA);
    for (auto shell : basis->shells()) {
        for (int p = 0; p < shell.nao(); p++) {
            minao_inds[shell.atomIdx()].push_back(p + shell.aoIdx());
        }
    }

    // => Unique ij pairs <= //

    std::vector<std::pair<int,int> > rot_inds;
    for (int i = 0; i < nmo; i++) {
        for (int j = 0; j < i; j++) {
            rot_inds.push_back(std::pair<int,int>(i,j));
        }
    }
    
    // => Master Loop <= //

    std::vector<double> metrics;
    std::vector<double> gradients;

    bool converged = false;
    for (int iter = 1; iter <= maxiter; iter++) {

        double metric = 0.0;
        for (int i = 0; i < nmo; i++) {
            for (int A = 0; A < minao_inds.size(); A++) {
                double Lval = 0.0;
                for (int m = 0; m < minao_inds[A].size(); m++) {
                    int mind = minao_inds[A][m];
                    Lval += Lp[i*nmin + mind] * Lp[i*nmin + mind];
                }
                metric += pow(Lval, power);
            } 
        }
        metric = pow(metric, 1.0 / power);
        metrics.push_back(metric);

        double gradient = 0.0;
        for (int ind = 0; ind < rot_inds.size(); ind++) {
            int i = rot_inds[ind].first;
            int j = rot_inds[ind].second;

            double Aij = 0.0;
            double Bij = 0.0;
            for (int A = 0; A < minao_inds.size(); A++) {
                double Qii = 0.0;
                double Qij = 0.0;
                double Qjj = 0.0;
                for (int m = 0; m < minao_inds[A].size(); m++) {
                    int mind = minao_inds[A][m];
                    Qii += Lp[i*nmin + mind] * Lp[i*nmin + mind];
                    Qij += Lp[i*nmin + mind] * Lp[j*nmin + mind];
                    Qjj += Lp[j*nmin + mind] * Lp[j*nmin + mind];
                }
                if (power == 2) { 
                    Aij += 4.0 * Qij * Qij - (Qii - Qjj) * (Qii - Qjj);
                    Bij += 4.0 * Qij * (Qii - Qjj);
                } else if (power == 4) {
                    Aij += (-1.0) * Qii * Qii * Qii * Qii - Qjj * Qjj * Qjj * Qjj + 6.0 * (Qii * Qii + Qjj * Qjj) * Qij * Qij + Qii * Qii * Qii * Qjj + Qii * Qjj * Qjj * Qjj; 
                    Bij += 4.0 * Qij * (Qii * Qii * Qii - Qjj * Qjj * Qjj);
                } else {
                    throw std::runtime_error("Localizer: invalid power");
                }
            }

            double phi = 0.25 * atan2(Bij, -Aij);
            double c = cos(phi);
            double s = sin(phi);

            //printf("%2d %2d: %11.3E %11.3E\n", i, j, c, s);

            C_DROT(nmin,Lp + i*nmin,1,Lp + j*nmin,1,c,s);
            C_DROT(nmo,Up + i*nmo,1,Up + j*nmo,1,c,s);

            gradient += Bij * Bij;

        }
        gradient = sqrt(gradient);
        gradients.push_back(gradient);

        if (gradient < convergence) {
            converged = true;
            break;
        }

    }

    // => U transpose <= //

    for (int i = 0; i < nmo; i++) {
        for (int j = 0; j < i; j++) {
            std::swap(Up[i*nmo+j],Up[j*nmo+i]);
        }
    }

    // => Targets <= //

    std::shared_ptr<Tensor> ret(new Tensor({gradients.size(),2}));
    double* retp = ret->data().data();
    for (size_t ind = 0; ind < gradients.size(); ind++) {
        retp[2*ind+0] = metrics[ind];
        retp[2*ind+1] = gradients[ind];
    }
    return ret; 
}

} // namespace lightspeed
