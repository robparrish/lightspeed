#include "string.hpp"
#include <cstdio>
#include <cstdarg>
#include <stdexcept>
#include <cstdlib>

namespace lightspeed {

std::string sprintf2(const char* fmt, ...)
{
    char* result = 0;
    va_list ap;
    va_start(ap, fmt);
    if(vasprintf(&result, fmt, ap) == -1)
        throw std::bad_alloc();
    va_end(ap);
    std::string str_result(result);
    free(result);
    return str_result;
}

} // namespace lightspeed
