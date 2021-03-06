#include <lightspeed/tensor.hpp>
#include <lightspeed/math.hpp>
#include "../util/string.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstring>

namespace lightspeed { 

size_t Tensor::total_memory__ = 0;

Tensor::Tensor(
    const std::vector<size_t>& shape,
    const std::string& name
    ) :
    shape_(shape),
    name_(name)
{
    strides_.resize(shape_.size());
    size_ = 1L;
    for (int i = shape_.size() - 1; i >= 0; i--) {
        strides_[i] = size_;
        size_ *= shape_[i];
    }  
    data_.resize(size_,0.0);

    // Ed's special memory thing
    total_memory__ += data_.size() * sizeof(double);
}
Tensor::~Tensor()
{
    // Ed's special memory thing
    total_memory__ -= data_.size() * sizeof(double);
}
void Tensor::ndim_error(size_t ndims) const
{
    if (!(ndim() == ndims)) {
        std::stringstream ss;
        ss << "Tensor should be " << ndims << " ndim, but is " << ndim() << " ndim.";
        throw std::runtime_error(ss.str());
    }
}
void Tensor::shape_error(const std::vector<size_t>& shape) const
{
    ndim_error(shape.size());
    for (size_t i = 0; i < ndim(); i++) {
        if (shape_[i] != shape[i]) {
            std::stringstream ss;
            ss << "Tensor should be (";
            for (size_t j = 0; j < ndim(); j++) {
                ss << shape[j];
                if (j < ndim() - 1) {
                    ss << ",";
                }
            }
            ss << ") shape, but is (";
            for (size_t j = 0; j < ndim(); j++) {
                ss << shape_[j];
                if (j < ndim() - 1) {
                    ss << ",";
                }
            }
            ss << ") shape.";
            throw std::runtime_error(ss.str());
        }
    }
}
void Tensor::square_error() const 
{
    ndim_error(2);
    if (shape_[0] != shape_[1]) {
        std::stringstream ss;
        ss << "Tensor should be square, but is ";
        ss << "(" << shape_[0] << "," << shape_[1] << ") shape.";
        throw std::runtime_error(ss.str());
    }
}
std::shared_ptr<Tensor> Tensor::clone()
{
    return std::shared_ptr<Tensor>(new Tensor(*this)); 
}
void Tensor::zero()
{
    memset(data_.data(),'\0',sizeof(double)*size_);
}
void Tensor::identity()
{
    square_error();
    zero();
    for (size_t i = 0; i < shape_[0]; i++) {
        data_[i * shape_[1] + i] = 1.0;
    }
}
void Tensor::symmetrize()
{
    square_error();
    for (size_t i = 0; i < shape_[0]; i++) {
        for (size_t j = 0; j < shape_[0]; j++) {
            data_[i * shape_[1] + j] =
            data_[j * shape_[1] + i] = 0.5 * (
            data_[i * shape_[1] + j] +
            data_[j * shape_[1] + i]);
        }
    }
}
void Tensor::antisymmetrize()
{
    square_error();
    for (size_t i = 0; i < shape_[0]; i++) {
        for (size_t j = 0; j < shape_[0]; j++) {
            double val = 0.5 * (
            data_[i * shape_[1] + j] -
            data_[j * shape_[1] + i]);
            data_[i * shape_[1] + j] = val;
            data_[j * shape_[1] + i] = - val;
        }
    }
}
void Tensor::scale(double a)
{
    C_DSCAL(size_,a,data_.data(),1);
}
void Tensor::copy(
    const std::shared_ptr<Tensor>& other
    )
{
    shape_error(other->shape());
    
    ::memcpy(data_.data(),other->data().data(),sizeof(double)*size_);
}
void Tensor::axpby(
    const std::shared_ptr<Tensor>& other,
    double a,
    double b
    )
{
    shape_error(other->shape());
    
    C_DSCAL(size_,b,data_.data(),1);
    C_DAXPY(size_,a,other->data().data(),1,data_.data(),1); 
}
double Tensor::vector_dot(
    const std::shared_ptr<Tensor>& other
    ) const
{
    shape_error(other->shape());

    return C_DDOT(size_,const_cast<double*>(data_.data()),1,other->data().data(),1);
}
std::shared_ptr<Tensor> Tensor::transpose() const
{
    ndim_error(2);
    std::shared_ptr<Tensor> T(new Tensor({shape_[1], shape_[0]}));
    double* Tp = T->data().data();
    const double* Ap = data_.data();
    for (size_t ind1 = 0; ind1 < shape_[0]; ind1++) {
    for (size_t ind2 = 0; ind2 < shape_[1]; ind2++) {
        Tp[ind2 * shape_[0] + ind1] = Ap[ind1 * shape_[1] + ind2];
    }}
    return T;
}
std::string Tensor::string(
    bool print_data, 
    int maxcols,
    const std::string& data_format,
    const std::string& header_format
    ) const
{
    std::string str = "";
    str += sprintf2( "Tensor: %s\n", name_.c_str());
    str += sprintf2( "  Ndim  = %zu\n", ndim());
    str += sprintf2( "  Size  = %zu\n", size());
    str += sprintf2( "  Shape = (");
    for (size_t dim = 0; dim < ndim(); dim++) {
        str += sprintf2( "%zu", shape_[dim]);
        if (dim < ndim() - 1) {
            str += sprintf2( ",");
        }
    }
    str += sprintf2(")\n");

    if (print_data) {

        str += sprintf2("\n");
            
        std::string order0str1 = "  " + data_format + "\n";
        std::string order1str1 = "  %5zu " + data_format + "\n";
        std::string order2str1 = " " + header_format;
        std::string order2str2 = " " + data_format;

        int order = ndim();
        size_t nelem = size();

        size_t page_size = 1L;
        size_t rows = 1;
        size_t cols = 1;
        if (order >= 1) {
            page_size *= shape_[order - 1];
            rows = shape_[order - 1];
        }
        if (order >= 2) {
            page_size *= shape_[order - 2];
            rows = shape_[order - 2];
            cols = shape_[order - 1];
        }

        str += sprintf2( "  Data:\n\n");

        if (nelem > 0){
            size_t pages = nelem / page_size;
            for (size_t page = 0L; page < pages; page++) {

                if (order > 2) {
                    str += sprintf2( "  Page (");
                    size_t num = page;
                    size_t den = pages;
                    size_t val;
                    for (int k = 0; k < order - 2; k++) {
                        den /= shape_[k];
                        val = num / den;
                        num -= val * den;
                        str += sprintf2("%zu,",val);
                    }
                    str += sprintf2( "*,*):\n\n");
                }

                const double* vp = data_.data() + page * page_size;
                if (order == 0) {
                    str += sprintf2( order0str1.c_str(), *(vp));
                } else if(order == 1) {
                    for (size_t i=0; i<page_size; ++i) {
                        str += sprintf2( order1str1.c_str(), i, *(vp + i));
                    }
                } else {
                    for (size_t j = 0; j < cols; j += maxcols) {
                        size_t ncols = (j + maxcols >= cols ? cols - j : maxcols);

                        // Column Header
                        str += sprintf2("  %5s", "");
                        for (size_t jj = j; jj < j+ncols; jj++) {
                            str += sprintf2(order2str1.c_str(), jj);
                        }
                        str += sprintf2("\n");

                        // Data
                        for (size_t i = 0; i < rows; i++) {
                            str += sprintf2("  %5zu", i);
                            for (size_t jj = j; jj < j+ncols; jj++) {
                                str += sprintf2(order2str2.c_str(), *(vp + i * cols + jj));
                            }
                            str += sprintf2("\n");
                        }

                        // Block separator
                        if (page < pages - 1 || j + maxcols < cols - 1) str += sprintf2("\n");
                    }
                }
            }
        }
    }
    return str;
}
void Tensor::print() const
{
    std::cout << string() << std::endl;
}
void Tensor::print(const std::string& name)
{
    std::string bak_name = name_;
    set_name(name);
    print();
    set_name(bak_name);
}

} // namespace lightspeed
