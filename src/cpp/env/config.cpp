#include <lightspeed/config.hpp>

#define _STRINGIZE(x) #x
#define STRINGIZE(x) _STRINGIZE(x)

// Flags to be passed in from compiler
#ifndef GIT_SHA
#define GIT_SHA "Unknown"
#endif
#ifndef GIT_DIRTY
#define GIT_DIRTY 0
#endif

namespace lightspeed {

std::string Config::git_sha()
{
    return std::string(STRINGIZE(GIT_SHA));
}
bool Config::git_dirty()
{
    return (bool) GIT_DIRTY;
}

} // namespace lightspeed 
