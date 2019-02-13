#include "libxc_functional_impl.hpp"

namespace lightspeed {
    
std::shared_ptr<Functional> Functional::build(
    const std::string& name)
{
    std::shared_ptr<Functional> fun(new Functional);
    fun->impl_ = std::shared_ptr<FunctionalImpl>(new LibXCFunctionalImpl(name));
    return fun;
}

} // namespace lightspeed
