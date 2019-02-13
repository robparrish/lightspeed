#include "tc_cibox.hpp"
#include <lightspeed/resource_list.hpp>
#include <lightspeed/gpu_context.hpp>
#include <lightspeed/tensor.hpp>
#include "../../util/string.hpp"
#include "../bits.hpp"
#include <stdexcept>
#include <cstring>
#include <cmath>

#ifdef HAVE_TERACHEM
#include <cibox/dcibox.h> // TeraChem's dcibox.h 
#endif

namespace lightspeed {

  std::shared_ptr<TC_CIBox> TC_CIBox::instance_ = std::shared_ptr<TC_CIBox>();
  size_t TC_CIBox::M_last_ = 0;
  size_t TC_CIBox::Na_last_ = 0;
  size_t TC_CIBox::Nb_last_ = 0;

  std::shared_ptr<TC_CIBox> TC_CIBox::instance(const std::shared_ptr<ResourceList>& resources,
						      const size_t M,
						      const size_t Na,
						      const size_t Nb,
						      const std::shared_ptr<Tensor>& H,
						      const std::shared_ptr<Tensor>& I)
  {
    if (!instance_) {
      instance_ = std::shared_ptr<TC_CIBox>(new TC_CIBox(resources,M,Na,Nb,H,I));
      M_last_ = M;
      Na_last_ = Na;
      Nb_last_ = Nb;
    } else if (M != M_last_ || Na != Na_last_ || Nb != Nb_last_) {
      // shutdown and startup if the orbital space has changed
      M_last_ = M;
      Na_last_ = Na;
      Nb_last_ = Nb;
      clear_instance();
      instance_ = std::shared_ptr<TC_CIBox>(new TC_CIBox(resources,M,Na,Nb,H,I));
    }
    return instance_;
}

#ifdef HAVE_TERACHEM

  TC_CIBox::TC_CIBox(const std::shared_ptr<ResourceList>& resources,
		     const size_t M,
		     const size_t Na,
		     const size_t Nb,
		     const std::shared_ptr<Tensor>& H,
		     const std::shared_ptr<Tensor>& I) :
    resources_(resources),
    M_(M),
    Na_(Na),
    Nb_(Nb),
    H_(H),
    I_(I)
  {

    if (resources_->ngpu() < 1) {
      throw std::runtime_error("No GPUs in this ResourceList object");
    }
    
    std::vector<int> dev_ids;
    for (auto gpu : resources_->gpus()) {
      dev_ids.push_back(gpu->id());
    }

    CIBoxLightSpeedStartup(resources_->ngpu(),
			   const_cast<int*>(dev_ids.data()),
			   M_,
			   Na_,
			   Nb_,
			   H->data().data(),
			   I->data().data());
  }

  TC_CIBox::~TC_CIBox()
  {
    CIBoxShutdown();
  }

  std::string TC_CIBox::string() const
  {
    std::string str = "";

    str += sprintf2( "TC_CIBox:\n");
    str += sprintf2( "  TC_CIBox is Activated\n");
    str += sprintf2( "\n");

    return str;
  }

  void TC_CIBox::computeSigma(double* C, double *sigma)
  {
    CIBoxGetSigma(C, sigma);
  }

  void TC_CIBox::computeOPDM(double* Ci, double *Cj, double *Gij)
  {
    CIBoxOPDMClosedLS(Gij,Ci,Cj);
  }

  void TC_CIBox::computeTPDM(double* Ci, double *Cj, double *Gijkl)
  {
    CIBoxTPDMClosedLS(Gijkl,Ci,Cj);
  }

#else

  TC_CIBox::TC_CIBox(const std::shared_ptr<ResourceList>& resources,
		     const size_t M,
		     const size_t Na,
		     const size_t Nb,
		     const std::shared_ptr<Tensor>& H,
		     const std::shared_ptr<Tensor>& I) :
    resources_(resources),
    M_(M),
    Na_(Na),
    Nb_(Nb),
    H_(H),
    I_(I)
  {
  }
  TC_CIBox::~TC_CIBox()
  {
  }
  std::string TC_CIBox::string() const
  {
    std::string str = "";

    str += sprintf2( "TC_CIBox:\n");
    str += sprintf2( "  TC_CIBox is Deactivated\n");
    str += sprintf2( "\n");

    return str;
  }
  void TC_CIBox::computeSigma(double *C, double *sigma)
  {
    throw std::runtime_error("TC_CIBox: Lightspeed was not compiled with TC_CIBox.");
  }
  void TC_CIBox::computeOPDM(double* Ci, double *Cj, double *Gij)
  {
    throw std::runtime_error("TC_CIBox: Lightspeed was not compiled with TC_CIBox.");
  }
  void TC_CIBox::computeTPDM(double* Ci, double *Cj, double *Gijkl)
  {
    throw std::runtime_error("TC_CIBox: Lightspeed was not compiled with TC_CIBox.");
  }

#endif

} // namespace lightspeed
