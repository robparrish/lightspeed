#include "tc_cibox.hpp"
#include <lightspeed/tensor.hpp>
#include <lightspeed/casbox.hpp>
#include <lightspeed/resource_list.hpp>
#include <stdexcept>

namespace lightspeed {

  std::shared_ptr<Tensor> CASBox::tpdm_det_gpu(const std::shared_ptr<ResourceList>& resources,
						      const std::shared_ptr<Tensor>& A,
						      const std::shared_ptr<Tensor>& B,
						      bool symmetrize) const
{
  std::shared_ptr<TC_CIBox> tc = TC_CIBox::instance(resources,M_,Na_,Nb_,H_,I_);

  std::vector<size_t> dimN;
  dimN.push_back(M_);
  dimN.push_back(M_);
  dimN.push_back(M_);
  dimN.push_back(M_);
  std::shared_ptr<Tensor> D(new Tensor(dimN));
  tc->computeTPDM(A->data().data(),B->data().data(),D->data().data());

  // => Symmetrization <= //

  if (!symmetrize) return D;

  double* Dp = D->data().data();

  std::shared_ptr<Tensor> D2(new Tensor(dimN));
  double* D2p = D2->data().data();
  for (size_t t = 0; t < M_; t++) {
    for (size_t u = 0; u < M_; u++) {
      for (size_t v = 0; v < M_; v++) {
	for (size_t w = 0; w < M_; w++) {
	  size_t index = t*M_*M_*M_ + u*M_*M_ + v*M_ + w;
	  D2p[index] = 0.25 * (Dp[t*M_*M_*M_ + u*M_*M_ + v*M_ + w] +
			       Dp[t*M_*M_*M_ + u*M_*M_ + w*M_ + v] +
			       Dp[u*M_*M_*M_ + t*M_*M_ + v*M_ + w] +
			       Dp[u*M_*M_*M_ + t*M_*M_ + w*M_ + v]);
	}
      }
    }
  }

  return D2;
}

} // namespace lightspeed
