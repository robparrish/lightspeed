#ifndef LS_CONFIG_HPP
#define LS_CONFIG_HPP

#include <string>

namespace lightspeed {

/**
 * A simple static class that can be used at runtime to determine the optional
 * modules that LS was compiled with and the state of the repository (GIT SHA
 * number and clearn/dirty state) at compile time.
 **/
class Config {

public:

static std::string git_sha();
static bool git_dirty();

static bool has_cuda() {
#ifdef HAVE_CUDA
    return true;
#else
    return false;
#endif
}

static bool has_terachem() {
#ifdef HAVE_TERACHEM
    return true;
#else
    return false;
#endif
}

static bool has_libxc() {
#ifdef HAVE_LIBXC
    return true;
#else
    return false;
#endif
}

static bool has_openmp() {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

};

} // namespace lightspeed

#endif
