#include <lightspeed/solver.hpp>
#include <lightspeed/tensor.hpp>
#include "../util/string.hpp"
#include <stdexcept>
#include <cmath>

namespace lightspeed {
    
DIIS::DIIS(
    size_t max_vectors) :
    max_vectors_(max_vectors)
{
    std::vector<size_t> dim1;
    dim1.push_back(max_vectors);
    dim1.push_back(max_vectors);
    E_ = std::shared_ptr<Tensor>(new Tensor(dim1));
}
std::shared_ptr<Storage> DIIS::iterate(
    const std::shared_ptr<Storage>& state,
    const std::shared_ptr<Storage>& error)
{
    // => Index Replacement <= //

    size_t index;
    if (current_vectors() < max_vectors()) {
        index = current_vectors();
        state_vecs_.push_back(state);
        error_vecs_.push_back(error);
    } else {
        double maxE = 0.0;
        const double* Ep = E_->data().data();
        for (size_t ind = 0; ind < max_vectors_; ind++) {
            if (Ep[ind * max_vectors_ + ind] >= maxE) {
                maxE = Ep[ind * max_vectors_ + ind];
                index = ind;
            }
        }
        state_vecs_[index] = state;
        error_vecs_[index] = error;
    }

    // => Error Inner Product <= //

    std::vector<double> Econt(current_vectors(),0.0);
    for (size_t ind = 0; ind < current_vectors(); ind++) {
        Econt[ind] = Storage::dot(error,error_vecs_[ind]);
    }
    double* Ep = E_->data().data();
    for (size_t ind = 0; ind < current_vectors(); ind++) {
        Ep[ind * max_vectors_ + index] =
        Ep[index * max_vectors_ + ind] =
        Econt[ind];
    }

    // => Extrapolation <= //

    std::shared_ptr<Tensor> d = compute_coefs();
    const double* dp = d->data().data();
    std::shared_ptr<Storage> S = Storage::zeros_like(state); 
    for (size_t ind = 0; ind < current_vectors(); ind++) {
        Storage::axpby(state_vecs_[ind],S,dp[ind],1.0);
    } 
    return S;
}
std::shared_ptr<Tensor> DIIS::compute_coefs() const
{
    // => Raw B Matrix <= //

    size_t cur = current_vectors();
    std::vector<size_t> dim1;
    dim1.push_back(cur+1);
    std::vector<size_t> dim2;
    dim2.push_back(cur+1);
    dim2.push_back(cur+1);
    std::shared_ptr<Tensor> B = std::shared_ptr<Tensor>(new Tensor(dim2));
    double* Bp = B->data().data();
    const double* Ep = E_->data().data();
    for (size_t i = 0; i < cur; i++) {
        for (size_t j = 0; j < cur; j++) {
            Bp[i * (cur + 1) + j] = Ep[i * max_vectors() + j];
        }
        Bp[i * (cur + 1) + cur] =
        Bp[cur * (cur + 1) + i] =
        1.0;
    }
    //B->print();

    // => Target Coefficients <= //

    std::shared_ptr<Tensor> d = std::shared_ptr<Tensor>(new Tensor(dim1));
    double* dp = d->data().data();

    // => Zero/Negative Trapping <= //

    bool is_zero = false;
    for (size_t i = 0; i < cur; i++) {
        if(Bp[i * (cur + 1) + i] < 0.0) {
            throw std::runtime_error("Negative diagonal in B matrix.");
        }
        if(Bp[i * (cur + 1) + i] == 0.0) {
            dp[i] = 1.0;
            is_zero = true;
            break;
        }
    }
    if (is_zero) {
        return d;
    }

    // => Balancing Registers <= //

    std::shared_ptr<Tensor> s = std::shared_ptr<Tensor>(new Tensor(dim1));
    double* sp = s->data().data();

    // => Balancing <= //

    for (size_t i = 0; i < cur; i++) {
        sp[i] = pow(Bp[i * (cur + 1) + i],-1.0/2.0);
    }
    sp[cur] = 1.0;
    //s->print();

    for (size_t i = 0; i < cur + 1; i++) {
        for (size_t j = 0; j < cur + 1; j++) {
            Bp[i * (cur + 1) + j] *= sp[i]*sp[j];
        }
    }
    //B->print();

    // => Inversion <= //

    std::shared_ptr<Tensor> Binv = Tensor::power(B,-1.0,1E-12);
    //Binv->print();
    double* Binvp = Binv->data().data();
    
    // => Result (Last Column) <= //

    for (size_t i = 0; i < cur + 1; i++) {
        dp[i] = sp[i] * Binvp[i * (cur + 1) + cur];
    }
    //d->print();

    return d;
}
std::string DIIS::string() const 
{
    std::string str = "";
    str += sprintf2("DIIS:\n");
    str += sprintf2("  Max Vectors = %5zu\n", max_vectors());  
    return str;
}

} // namespace lightspeed
