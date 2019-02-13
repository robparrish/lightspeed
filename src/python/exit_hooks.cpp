#include "../cpp/intbox/tc/tcintbox/tcintbox.hpp"
#include "../cpp/casbox/gpu/tc_cibox.hpp"

namespace lightspeed {

/**
 * In certain cases, the unpredictable nature of destruction order of objects
 * and code in boost::python libraries can cause nasty segfaults as
 * lightspeed.so is unloaded. This particularly seems to be the case for
 * singleton pointers in classes that are obfuscated from python. These objects
 * should be manually deleted in this "exit_hooks" function, which will be
 * registered to the atexit module in __init__.py.
 **/
void exit_hooks()
{
    TCIntBox::clear_instance();
    TC_CIBox::clear_instance();
}

} // namespace lightspeed
