# Lightspeed Makefile
#
# We use a single global make session to build the Lightspeed C++ source.
#
# A common set of build parameters are defined in make.vars. These are LOCAL
# settings, and should not be checked into git. Instead a configure script is
# available to quickly generate default make.vars files for the various
# machines that we use regularly.

include make.vars

ifeq ($(VERBOSEMAKE), true) 
DEBPREF=
else
DEBPREF:=@
endif

LS := src

MODULES := $(LS) 
BINS    := $(patsubst %, %.bin, $(MODULES))
CLEANS  := $(patsubst %, %.clean, $(MODULES))

.PHONY : $(BINS) $(CLEANS)
default : src.bin

.PHONY : bins
bins : $(BINS)

.PHONY : clean
clean : $(CLEANS)

include $(patsubst %, %/module.mk, $(MODULES))

# Standard make compilation rules

$(OBJDIR)/%.o : %.cpp
	@printf "Compiling %-25s > %-25s\n" $< $@
	@mkdir -p $(dir $@)
	$(DEBPREF)$(CXX)     -c $(CPPFLAGS) $(CXXFLAGS) $< -o $@
	@$(CXX) -MM -c $(CPPFLAGS) $(CXXFLAGS) $< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
		sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)

