#ifndef LS_TC_CIBOX_HPP
#define LS_TC_CIBOX_HPP

#include <cstddef>
#include <vector>
#include <cstdio>
#include <memory>

namespace lightspeed {

class ResourceList;
class Tensor;

class TC_CIBox {

public:

  ~TC_CIBox();
  std::string string() const;

  std::shared_ptr<ResourceList> resources() const { return resources_; }

protected:

  TC_CIBox(const std::shared_ptr<ResourceList>& resources,
	   const size_t M,
	   const size_t Na,
	   const size_t Nb,
	   const std::shared_ptr<Tensor>& H,
	   const std::shared_ptr<Tensor>& I);

  std::shared_ptr<ResourceList> resources_;
  size_t M_;
  size_t Na_;
  size_t Nb_;
  std::shared_ptr<Tensor> H_;
  std::shared_ptr<Tensor> I_;

public:

  static std::shared_ptr<TC_CIBox> instance(const std::shared_ptr<ResourceList>& resources,
						   const size_t M,
						   const size_t Na,
						   const size_t Nb,
						   const std::shared_ptr<Tensor>& H,
						   const std::shared_ptr<Tensor>& I);

  static void clear_instance() { instance_ = nullptr; }

protected:

  static std::shared_ptr<TC_CIBox> instance_;
  static size_t M_last_;
  static size_t Na_last_;
  static size_t Nb_last_;

public:
      
  // > Direct CIBox Wrappers (See CIBox's dcibox.h) < // 

  void computeSigma(double *C, double *sigma);
  void computeOPDM(double *Ci, double *Cj, double *Gij);
  void computeTPDM(double *Ci, double *Cj, double *Gij);

};

} // namespace lightspeed

#endif
