# => Custom Makefile <= #

# The name of your plugin. 
NAME = pyplugin

# C++ source files for your plugin. By default we grab all *.cpp files.
CXXSRC = $(notdir $(wildcard *.cpp))
# CUDA source files for your plugin. By default we grab all *.cu files.
CUDASRC = $(notdir $(wildcard *.cu))
# Header files for your plugin. By default we grab all *.h* files
DEPENDINCLUDE = $(notdir $(wildcard *.h*))

# Flags that were used to compile Lightspeed.
CXX = /global/software/icc/2016.0.109/compilers_and_libraries_2016.0.109/linux/bin/intel64/icpc
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

NVCC = /global/software/CUDA/8.0.44/bin/nvcc

NVCCFLAGS = \

INCLUDES = \
    -I/global/user_software/lightspeed/1.01/build/include \
    -I/global/software/Anaconda2/4.3.0/include/python2.7 \
    -I/global/software/Boost/1.63.0-intel-2016.0.109/include \
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

LDLIBS = -L/global/user_software/lightspeed/1.01/build/lib -llightspeed

TARGET = $(NAME).so

# Start the compilation rules
default:: $(TARGET)

# The object files
BINOBJ = $(CXXSRC:%.cpp=%.o)
BINOBJ += $(CUDASRC:%.cu=%.o)

%.o: %.cpp
	$(CXX) $(CXXDEFS) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(CXXDEFS) $(CXXFLAGS) $(NVCCFLAGS) $(INCLUDES) -c $<

$(TARGET): $(BINOBJ)
	$(CXX) $(LDFLAGS) -o $@ $^ $(CXXDEFS) $(LDLIBS)

# Erase all compiled intermediate files
clean:
	rm -f $(BINOBJ) $(TARGET) *.d *.pyc 

