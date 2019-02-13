#include "tc_cibox.hpp"
#include <lightspeed/tensor.hpp>
#include <lightspeed/casbox.hpp>
#include <lightspeed/resource_list.hpp>
#include <stdexcept>

namespace lightspeed {

  std::shared_ptr<Tensor> CASBox::opdm_det_gpu(const std::shared_ptr<ResourceList>& resources,
						      const std::shared_ptr<Tensor>& A,
						      const std::shared_ptr<Tensor>& B) const
{
  std::shared_ptr<TC_CIBox> tc = TC_CIBox::instance(resources,M_,Na_,Nb_,H_,I_);

  std::vector<size_t> dimN;
  dimN.push_back(M_);
  dimN.push_back(M_);
  std::shared_ptr<Tensor> D(new Tensor(dimN));
  tc->computeOPDM(A->data().data(),B->data().data(),D->data().data());

  return D;
}

} // namespace lightspeed
