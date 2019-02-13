#include "tc_cibox.hpp"
#include <lightspeed/tensor.hpp>
#include <lightspeed/casbox.hpp>
#include <lightspeed/resource_list.hpp>
#include <stdexcept>

namespace lightspeed {

  std::shared_ptr<Tensor> CASBox::sigma_det_gpu(const std::shared_ptr<ResourceList>& resources,
						       const std::shared_ptr<Tensor>& C) const
{
  std::shared_ptr<TC_CIBox> tc = TC_CIBox::instance(resources,M_,Na_,Nb_,H_,I_);

  std::shared_ptr<Tensor> S(new Tensor({Da_, Db_}));
  tc->computeSigma(C->data().data(),S->data().data());

  return S;
}

} // namespace lightspeed
