otool -L /Users/parrish/Code/lightspeed/build/python/lightspeed/lightspeed.so
install_name_tool -change libboost_python37.dylib /Users/parrish/Code/boost_1_69_0/stage/lib/libboost_python37.dylib /Users/parrish/Code/lightspeed/build/python/lightspeed/lightspeed.so
otool -L /Users/parrish/Code/lightspeed/build/python/lightspeed/lightspeed.so
