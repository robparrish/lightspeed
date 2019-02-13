# => Custom Makefile <= #

# The name of your plugin. 
NAME = pyplugin

# C++ source files for your plugin. By default we grab all *.cc files.
CXXSRC = $(notdir $(wildcard *.cpp))

# Header files for your plugin. By default we grab all *.h* files
DEPENDINCLUDE = $(notdir $(wildcard *.h*))

# Flags that were used to compile Lightspeed.
CXX = /opt/intel/composer_xe_2015.0.090/bin/intel64/icpc
CXXFLAGS = \
    -Xlinker \
    -export-dynamic \
    -fPIC \
    -std=c++11 \
    -O3 \
    -Wall \
    -fopenmp \
    -Wuninitialized \
    -Wno-unknown-pragmas \

INCLUDES = \
    -I/home/parrish/Code/lightspeed/build/include \
    -I/home/parrish/anaconda2/include/python2.7 \
    -I/home/parrish/Code/boost_1_59_0 \
    -I/usr/include \

LDFLAGS = \
    -headerpad_max_install_names

# Used to determine linking flags.
UNAME = $(shell uname)

# Add the flags needed for shared library creation
ifeq ($(UNAME), Linux)
    LDFLAGS += -shared
endif
ifeq ($(UNAME), Darwin)
    LDFLAGS += -shared -undefined dynamic_lookup
    CXXFLAGS += -fno-common
endif

TARGET = $(NAME).so

# Start the compilation rules
default:: $(TARGET)

# The object files
BINOBJ = $(CXXSRC:%.cpp=%.o)

LDLIBS = -L/home/parrish/Code/lightspeed/build/lib -llightspeed

%.o: %.cpp
	$(CXX) $(CXXDEFS) $(CXXFLAGS) $(INCLUDES) -c $<

$(TARGET): $(BINOBJ)
	$(CXX) $(LDFLAGS) -o $@ $^ $(CXXDEFS) $(LDLIBS)

# Erase all compiled intermediate files
clean:
	rm -f $(BINOBJ) $(TARGET) *.d *.pyc 

