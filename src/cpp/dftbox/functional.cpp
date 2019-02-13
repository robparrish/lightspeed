#include "functional_impl.hpp"

namespace lightspeed {
    
std::string Functional::name() const 
{
    return impl_->name();
}
std::string Functional::citation() const 
{
    return impl_->citation();
}
std::string Functional::string() const 
{
    return impl_->string();
}
int Functional::type() const 
{
    return impl_->type();
}
bool Functional::has_lsda() const
{
    return impl_->has_lsda();
}
bool Functional::has_gga() const
{
    return impl_->has_gga();
}
int Functional::deriv() const
{
    return impl_->deriv();
}
double Functional::get_param(const std::string& id) const
{
    return impl_->get_param(id);
}
void Functional::set_param(const std::string& id, double val)
{
    impl_->set_param(id,val);
}
double Functional::alpha() const
{
    return impl_->alpha();
}
double Functional::beta() const
{
    return impl_->beta();
}
double Functional::omega() const
{
    return impl_->omega();
}
void Functional::set_alpha(double alpha)
{
    return impl_->set_alpha(alpha);
}
void Functional::set_beta(double beta)
{
    return impl_->set_beta(beta);
}
void Functional::set_omega(double omega)
{
    return impl_->set_omega(omega);
}
bool Functional::is_alpha_fixed() const
{
    return impl_->is_alpha_fixed();
}
bool Functional::is_beta_fixed() const
{
    return impl_->is_beta_fixed();
}
bool Functional::is_omega_fixed() const
{
    return impl_->is_omega_fixed();
}
std::shared_ptr<Tensor> Functional::compute(
    const std::shared_ptr<Tensor>& rho,
    int deriv) const
{
    return impl_->compute(rho,deriv);
}

} // namespace lightspeed
