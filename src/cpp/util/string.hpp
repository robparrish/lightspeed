#ifndef LS_STRING_HPP
#define LS_STRING_HPP

#include <string>

namespace lightspeed {

/**!
 * sprintf2 acts like printf, except that it returns a dynamically allocated
 * std::string (or throws a std::alloc_error if memory could not be obtained)
 *
 * This should be used as a replacement for printf or fprintf calls in 
 * print or print_header type methods - returning a std::string to the user
 * is much more portable.
 **/
std::string sprintf2(const char* fmt, ...);

} // namespace lightspeed

#endif
