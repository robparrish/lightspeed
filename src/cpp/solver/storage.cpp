#include <lightspeed/solver.hpp>
#include <lightspeed/tensor.hpp>
#include "../util/string.hpp"
#include <stdexcept>
#include <sstream>
#include <cstdio>
#include <unistd.h>

namespace lightspeed {

// Default scratch path
std::string Storage::scratch_path__ = "./";

// Random unique file names
static size_t disk_next_id__ = 0L;
size_t disk_next_id() { return disk_next_id__++; }
std::string Storage::next_scratch_file()
{
    std::stringstream ss;
    ss << Storage::scratch_path();
    ss << "/";
    ss << "storage.";
    ss << getpid();
    ss << ".";
    ss << disk_next_id();
    ss << ".dat";
    return ss.str();
}

Storage::Storage(
    size_t size,
    bool is_disk):
    size_(size),
    is_disk_(is_disk)
{
    if (is_disk_) {
        filename_ = Storage::next_scratch_file();
        disk_data_ = fopen(filename_.c_str(),"wb+");
        if (disk_data_ == NULL) {
            throw std::runtime_error("Storage: cannot open disk file: " + filename_);
        }
        std::vector<double> temp(size_,0.0);
        size_t ret = fwrite(temp.data(),sizeof(double),size_,disk_data_);
        if (ret != size_) {
            throw std::runtime_error("Storage: cannot write to disk file: " + filename_);
        }
        fseek(disk_data_,0L,SEEK_SET);
    } else {
        core_data_.resize(size_,0.0);
    }
}
Storage::~Storage()
{
    if (is_disk_) {
        fclose(disk_data_);
        remove(filename_.c_str());
    }
}
std::string Storage::string() const 
{
    std::string s = "";
    s += "Storage:\n";
    s += sprintf2("  Size = %11zu\n", size_);
    s += sprintf2("  Disk = %11s\n", is_disk_ ? "Yes" : "No");
    if (is_disk_) {
        s += sprintf2("  File = %11s\n", filename_.c_str());
    }
    return s;
}

std::vector<double> Storage::disk_data()
{
    if (!is_disk_) throw std::runtime_error("Storage::disk_data: this storage is not disk.");
    std::vector<double> temp(size_);
    size_t ret = fread(temp.data(),sizeof(double),size_,disk_data_);
    if (ret != size_) {
        throw std::runtime_error("Storage: cannot read from disk file: " + filename_);
    }
    fseek(disk_data_,0L,SEEK_SET);
    return temp;
}

void Storage::to_disk_data(const std::vector<double>& data)
{
    if (!is_disk_) throw std::runtime_error("Storage::to_disk_data: this storage is not disk.");
    if (data.size() != size_) throw std::runtime_error("Storage::to_disk_data: data size is wrong");
    size_t ret = fwrite(data.data(),sizeof(double),size_,disk_data_);
    if (ret != size_) {
        throw std::runtime_error("Storage: cannot write to disk file: " + filename_);
    }
    fseek(disk_data_,0L,SEEK_SET);
}
std::vector<double>& Storage::core_data() 
{
    if (is_disk_) throw std::runtime_error("Storage::core_data: this storage is disk.");
    return core_data_;
}
std::shared_ptr<Storage> Storage::zeros_like(
    const std::shared_ptr<Storage>& x)
{
    return std::shared_ptr<Storage>(new Storage(x->size(),x->is_disk()));
}    

std::shared_ptr<Storage> Storage::copy(
    const std::shared_ptr<Storage>& other)
{
    std::shared_ptr<Storage> target = std::shared_ptr<Storage>(new Storage(other->size(),other->is_disk()));
  
    if (!other->is_disk()) {
        std::vector<double>& od = other->core_data();
        std::vector<double>& td = target->core_data();
        for (size_t ind = 0; ind < od.size(); ind++) {
            td[ind]=od[ind];
        }
    } else {
        std::vector<double> od = other->disk_data();
        std::vector<double> td = target->disk_data();
        for (size_t ind = 0; ind < od.size(); ind++) {
            td[ind]=od[ind];
        }
        target->to_disk_data(td);
    }


    return target;

}

std::shared_ptr<Storage> Storage::scale(
    const std::shared_ptr<Storage>& x,
    double a)
{
    if (!x->is_disk()) {
        std::vector<double>& xd = x->core_data();
        for (size_t ind = 0; ind < xd.size(); ind++) {
            xd[ind] *= a;
        }
    } else {
        std::vector<double> xd = x->disk_data();
        for (size_t ind = 0; ind < xd.size(); ind++) {
            xd[ind] *= a;
        }
        x->to_disk_data(xd);
    }
    return x;
}
double Storage::dot( 
    const std::shared_ptr<Storage>& x,
    const std::shared_ptr<Storage>& y)
{
    if (x->size() != y->size()) {
        throw std::runtime_error("Storage::dot: x and y are not the same size");
    }

    std::vector<double> xtemp;
    if (x->is_disk()) {
        xtemp = x->disk_data();
    }
    std::vector<double> ytemp;
    if (y->is_disk()) {
        ytemp = y->disk_data();
    }
    const std::vector<double>& xref = (x->is_disk() ? xtemp : x->core_data()); 
    const std::vector<double>& yref = (y->is_disk() ? ytemp : y->core_data()); 

    const double* xp = xref.data();
    const double* yp = yref.data();
    double val = 0.0;
    size_t size = x->size();
    for (size_t ind = 0; ind < size; ind++) {
        val += (*xp++) * (*yp++);
    } 

    return val;
}
std::shared_ptr<Storage> Storage::axpby(
    const std::shared_ptr<Storage>& x,
    const std::shared_ptr<Storage>& y,
    double a,
    double b)
{
    if (x->size() != y->size()) {
        throw std::runtime_error("Storage::dot: x and y are not the same size");
    }

    std::vector<double> xtemp;
    if (x->is_disk()) {
        xtemp = x->disk_data();
    }
    std::vector<double> ytemp;
    if (y->is_disk()) {
        ytemp = y->disk_data();
    }
    const std::vector<double>& xref = (x->is_disk() ? xtemp : x->core_data()); 
    std::vector<double>& yref = (y->is_disk() ? ytemp : y->core_data()); 
    
    const double* xp = xref.data();
    double* yp = yref.data();
    size_t size = x->size();
    for (size_t ind = 0; ind < size; ind++) {
        yp[ind] = a * xp[ind] + b * yp[ind];
    } 

    if (y->is_disk()) {
        y->to_disk_data(ytemp);
    }
    
    return y;
}

std::shared_ptr<Storage> Storage::from_tensor(
    const std::shared_ptr<Tensor>& T,
    bool use_disk)
{
    size_t size = T->size();
    std::shared_ptr<Storage> S(new Storage(size,use_disk));
    if (use_disk) {
        S->to_disk_data(T->data());
    } else {
        S->core_data() = T->data();
    }
    return S;
}
std::shared_ptr<Tensor> Storage::to_tensor(
    const std::shared_ptr<Storage>& S,
    const std::shared_ptr<Tensor>& T)
{
    if (T->size() != S->size()) {
        throw std::runtime_error("Storage::to_tensor: S->size() != T->size()");
    }

    if (S->is_disk()) {
        T->data() = S->disk_data();
    } else {
        T->data() = S->core_data();
    }
    
    return T;
}
std::shared_ptr<Storage> Storage::from_tensor_vec(
    const std::vector<std::shared_ptr<Tensor> >& T,
    bool use_disk)
{
    size_t size = 0;
    for (int Tind = 0; Tind < T.size(); Tind++) {
        size += T[Tind]->size();
    }

    std::shared_ptr<Storage> S(new Storage(size,use_disk));
    std::vector<double> Stemp;
    if (use_disk) {
        Stemp = std::vector<double>(size,0.0);
    }
    double* Sp = (use_disk ? Stemp.data() : S->core_data().data());
    
    size_t offset = 0;
    for (int Tind = 0; Tind < T.size(); Tind++) {
        double* S2p = Sp + offset;
        size_t Tsize = T[Tind]->size();
        const double* T2p = T[Tind]->data().data();
        for (size_t ind = 0; ind < Tsize; ind++) {
            (*S2p++) = (*T2p++);
        }
        offset += Tsize;
    }

    return S; 
}
std::vector<std::shared_ptr<Tensor> > Storage::to_tensor_vec(
    const std::shared_ptr<Storage>& S,
    const std::vector<std::shared_ptr<Tensor> >& T)
{
    size_t size = 0;
    for (int Tind = 0; Tind < T.size(); Tind++) {
        size += T[Tind]->size();
    }
    if (size != S->size()) {
        throw std::runtime_error("Storage::to_tensor_vec: S->size() != T->size()");
    }

    std::vector<double> Stemp;
    if (S->is_disk()) {
        Stemp = S->disk_data();
    }
    const double* Sp = (S->is_disk() ? Stemp.data() : S->core_data().data());

    size_t offset = 0;
    for (int Tind = 0; Tind < T.size(); Tind++) {
        const double* S2p = Sp + offset;
        size_t Tsize = T[Tind]->size();
        double* T2p = T[Tind]->data().data();
        for (size_t ind = 0; ind < Tsize; ind++) {
            (*T2p++) = (*S2p++);
        }
        offset += Tsize;
    }

    return T; 
}


} // namespace lightspeed 
