# Module: lightspeed

# C++/CUDA source files
LSSRC := $(LS)/cpp/env/config.cpp \
         $(LS)/cpp/env/gpu_context.cpp \
         $(LS)/cpp/env/resource_list.cpp \
         $(LS)/cpp/util/string.cpp \
         $(LS)/cpp/math/blas.cpp \
         $(LS)/cpp/math/lapack.cpp \
         $(LS)/cpp/tensor/tensor.cpp \
         $(LS)/cpp/tensor/einsum.cpp \
         $(LS)/cpp/tensor/linalg.cpp \
         $(LS)/cpp/solver/storage.cpp \
         $(LS)/cpp/solver/diis.cpp \
         $(LS)/cpp/solver/davidson.cpp \
         $(LS)/cpp/core/molecule.cpp \
         $(LS)/cpp/core/am.cpp \
         $(LS)/cpp/core/basis.cpp \
         $(LS)/cpp/core/ecp.cpp \
         $(LS)/cpp/core/pair_list.cpp \
         $(LS)/cpp/core/pair_list_util.cpp \
         $(LS)/cpp/core/pure_transform.cpp \
         $(LS)/cpp/core/hermite.cpp \
         $(LS)/cpp/core/hermite_util.cpp \
         $(LS)/cpp/core/kvec.cpp \
         $(LS)/cpp/core/molden.cpp \
         $(LS)/cpp/core/local.cpp \
         $(LS)/cpp/quad/gh.cpp \
         $(LS)/cpp/quad/gh_data.cpp \
         $(LS)/cpp/quad/rys.cpp \
         $(LS)/cpp/quad/rys_data.cpp \
         $(LS)/cpp/quad/peg_data.cpp \
         $(LS)/cpp/quad/boys.cpp \
         $(LS)/cpp/quad/boys_data.cpp \
         $(LS)/cpp/intbox/intbox.cpp \
         $(LS)/cpp/intbox/mo_eri.cpp \
         $(LS)/cpp/intbox/cpu/charge.cpp \
         $(LS)/cpp/intbox/cpu/overlap.cpp \
         $(LS)/cpp/intbox/cpu/overlap_grad.cpp \
         $(LS)/cpp/intbox/cpu/dipole.cpp \
         $(LS)/cpp/intbox/cpu/dipole_grad.cpp \
         $(LS)/cpp/intbox/cpu/quadrupole.cpp \
         $(LS)/cpp/intbox/cpu/nabla.cpp \
         $(LS)/cpp/intbox/cpu/angular_momentum.cpp \
         $(LS)/cpp/intbox/cpu/kinetic.cpp \
         $(LS)/cpp/intbox/cpu/kinetic_grad.cpp \
         $(LS)/cpp/intbox/cpu/potential.cpp \
         $(LS)/cpp/intbox/cpu/efield.cpp \
         $(LS)/cpp/intbox/cpu/esp.cpp \
         $(LS)/cpp/intbox/cpu/coulomb.cpp \
         $(LS)/cpp/intbox/cpu/coulomb_grad.cpp \
         $(LS)/cpp/intbox/cpu/exchange.cpp \
         $(LS)/cpp/intbox/cpu/exchange_grad.cpp \
         $(LS)/cpp/intbox/cpu/potential_grad.cpp \
         $(LS)/cpp/intbox/tc/coulomb.cpp \
         $(LS)/cpp/intbox/tc/exchange.cpp \
         $(LS)/cpp/intbox/tc/coulomb_grad.cpp \
         $(LS)/cpp/intbox/tc/exchange_grad.cpp \
         $(LS)/cpp/intbox/tc/ecp.cpp \
         $(LS)/cpp/intbox/tc/ecp_grad.cpp \
         $(LS)/cpp/intbox/tc/tcintbox/tcintbox.cpp \
         $(LS)/cpp/intbox/tc/tcintbox/tc2ls.cpp \
         $(LS)/cpp/blurbox/potential.cpp \
         $(LS)/cpp/blurbox/potential_grad.cpp \
         $(LS)/cpp/blurbox/esp.cpp \
         $(LS)/cpp/blurbox/esp_grad.cpp \
         $(LS)/cpp/blurbox/coulomb.cpp \
         $(LS)/cpp/blurbox/coulomb_grad.cpp \
         $(LS)/cpp/blurbox/lda.cpp \
         $(LS)/cpp/blurbox/lda2.cpp \
         $(LS)/cpp/blurbox/point.cpp \
         $(LS)/cpp/casbox/casbox.cpp \
         $(LS)/cpp/casbox/explicit_casbox.cpp \
         $(LS)/cpp/casbox/casbox_util.cpp \
         $(LS)/cpp/casbox/sblock.cpp \
         $(LS)/cpp/casbox/csf_basis.cpp \
         $(LS)/cpp/casbox/gpu/opdm.cpp \
         $(LS)/cpp/casbox/gpu/sigma.cpp \
         $(LS)/cpp/casbox/gpu/tc_cibox.cpp \
         $(LS)/cpp/casbox/gpu/tpdm.cpp \
         $(LS)/cpp/gridbox/hashed_grid.cpp \
         $(LS)/cpp/gridbox/lda.cpp \
         $(LS)/cpp/gridbox/gga.cpp \
         $(LS)/cpp/gridbox/meta.cpp \
         $(LS)/cpp/gridbox/orbitals.cpp \
         $(LS)/cpp/cubic/cubic.cpp \
         $(LS)/cpp/becke/lebedev.cpp \
         $(LS)/cpp/becke/radial.cpp \
         $(LS)/cpp/becke/atom.cpp \
         $(LS)/cpp/becke/becke.cpp \
         $(LS)/cpp/dftbox/functional.cpp \
         $(LS)/cpp/dftbox/functional_build.cpp \
         $(LS)/cpp/dftbox/libxc_functional_impl.cpp \
         $(LS)/cpp/dftbox/dftbox.cpp \
         $(LS)/cpp/sad/sad.cpp \
         $(LS)/cpp/sad/sad_util.cpp \
         $(LS)/cpp/lr/laplace.cpp \
         $(LS)/cpp/lr/laplace_data.cpp \
         $(LS)/cpp/lr/df.cpp \
         $(LS)/cpp/lr/thc.cpp \
	 $(LS)/cpp/lr/mints/potential4c.cpp \
         $(LS)/python/exit_hooks.cpp \
         $(LS)/python/export_collections.cpp \
         $(LS)/python/export_env.cpp \
         $(LS)/python/export_tensor.cpp \
         $(LS)/python/export_solver.cpp \
         $(LS)/python/export_core.cpp \
         $(LS)/python/export_intbox.cpp \
         $(LS)/python/export_blurbox.cpp \
         $(LS)/python/export_casbox.cpp \
         $(LS)/python/export_gridbox.cpp \
         $(LS)/python/export_cubic.cpp \
         $(LS)/python/export_becke.cpp \
         $(LS)/python/export_dftbox.cpp \
         $(LS)/python/export_sad.cpp \
         $(LS)/python/export_lr.cpp \
         $(LS)/python/python.cpp \

# External C++ header files
LSEXH := $(LS)/lightspeed/config.hpp \
         $(LS)/lightspeed/gpu_context.hpp \
         $(LS)/lightspeed/resource_list.hpp \
         $(LS)/lightspeed/math.hpp \
         $(LS)/lightspeed/tensor.hpp \
         $(LS)/lightspeed/solver.hpp \
         $(LS)/lightspeed/ewald.hpp \
         $(LS)/lightspeed/molecule.hpp \
         $(LS)/lightspeed/am.hpp \
         $(LS)/lightspeed/basis.hpp \
         $(LS)/lightspeed/ecp.hpp \
         $(LS)/lightspeed/pair_list.hpp \
         $(LS)/lightspeed/hermite.hpp \
         $(LS)/lightspeed/pure_transform.hpp \
         $(LS)/lightspeed/rys.hpp \
         $(LS)/lightspeed/boys.hpp \
         $(LS)/lightspeed/gh.hpp \
         $(LS)/lightspeed/molden.hpp \
         $(LS)/lightspeed/local.hpp \
         $(LS)/lightspeed/intbox.hpp \
         $(LS)/lightspeed/blurbox.hpp \
         $(LS)/lightspeed/casbox.hpp \
         $(LS)/lightspeed/gridbox.hpp \
         $(LS)/lightspeed/cubic.hpp \
         $(LS)/lightspeed/becke.hpp \
         $(LS)/lightspeed/dftbox.hpp \
         $(LS)/lightspeed/sad.hpp \
         $(LS)/lightspeed/lr.hpp \

# Python extensions and wrapper files
LSPY  := $(LS)/python/__init__.py \
         $(LS)/python/env.py \
         $(LS)/python/tensor.py \
         $(LS)/python/solver.py \
         $(LS)/python/core.py \
         $(LS)/python/intbox.py \
         $(LS)/python/casbox.py \
         $(LS)/python/gridbox.py \
         $(LS)/python/cubic.py \
         $(LS)/python/becke.py \
         $(LS)/python/dftbox.py \
         $(LS)/python/sad.py \
         $(LS)/python/lr.py \
         $(LS)/python/util.py \
         $(LS)/python/title.py \

LSOBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(filter %.cpp, $(LSSRC)))
LSDEP := $(LSOBJ:%.o=%.d)
-include $(LSDEP)

$(OBJDIR)/$(LS)/%.o : CPPFLAGS += -I$(LS) $(BOOSTINC) $(PYINC) $(CUDAINC) $(TCINC) $(XCINC) 
$(OBJDIR)/$(LS)/liblightspeed.so.1.0 : LDFLAGS :=  $(MATHLD) $(BOOSTLD) $(PYLD) $(CUDALD) $(TCLD) $(XCLD) 

# The string representation of the full GIT SHA for the LS repository
GIT_SHA :
	$(eval GIT_SHA = $(shell git rev-parse HEAD))
	@echo 'GIT_SHA: $(GIT_SHA)'

# An integer which is 0 if the respository is clean, >0 if it is dirty
GIT_DIRTY :
	$(eval GIT_DIRTY = $(shell git status --porcelain | wc -l))
	@echo 'GIT_DIRTY: $(GIT_DIRTY)'

$(OBJDIR)/$(LS)/cpp/env/config.o : GIT_SHA GIT_DIRTY
$(OBJDIR)/$(LS)/cpp/env/config.o : CPPFLAGS += -DGIT_SHA="$(GIT_SHA)"
$(OBJDIR)/$(LS)/cpp/env/config.o : CPPFLAGS += -DGIT_DIRTY=$(GIT_DIRTY)

$(OBJDIR)/$(LS)/liblightspeed.so.1.0 : $(LSOBJ)
	@echo Linking $@
	@mkdir -p $(dir $@)
	$(DEBPREF)$(CXX) $(SHAREDFLAG)liblightspeed.so.1 $(CXXFLAGS) -o $@ $(LSOBJ) $(LDFLAGS)

$(PREFIX)/lib/liblightspeed.so.1.0 : $(OBJDIR)/$(LS)/liblightspeed.so.1.0
	@printf "Grabbing %-25s > %-25s\n" $< $@
	@mkdir -p $(PREFIX)/lib
	@cp -f $(OBJDIR)/$(LS)/liblightspeed.so.1.0 $(PREFIX)/lib/
	@ln -sf liblightspeed.so.1.0 $(PREFIX)/lib/liblightspeed.so.1
	@ln -sf liblightspeed.so.1.0 $(PREFIX)/lib/liblightspeed.so

LSINC := $(patsubst $(LS)/lightspeed/%.hpp,$(PREFIX)/include/lightspeed/%.hpp, $(LSEXH))

$(PREFIX)/include/lightspeed/%.hpp : $(LS)/lightspeed/%.hpp
	@printf "Grabbing %-25s > %-25s\n" $< $@
	@mkdir -p $(PREFIX)/include/lightspeed
	@cp $^ $(PREFIX)/include/lightspeed/

$(PREFIX)/python/lightspeed/lightspeed.so : $(PREFIX)/lib/liblightspeed.so.1.0
	@printf "Grabbing %-25s > %-25s\n" $< $@
	@mkdir -p $(PREFIX)/python
	@mkdir -p $(PREFIX)/python/lightspeed
	@cp -f $(PREFIX)/lib/liblightspeed.so.1.0 $(PREFIX)/python/lightspeed/lightspeed.so

LSPY2 := $(patsubst $(LS)/python/%.py,$(PREFIX)/python/lightspeed/%.py, $(LSPY))

$(PREFIX)/python/lightspeed/%.py : $(LS)/python/%.py 
	@printf "Grabbing %-25s > %-25s\n" $< $@
	@mkdir -p $(PREFIX)/python
	@mkdir -p $(PREFIX)/python/lightspeed
	@cp $^ $(PREFIX)/python/lightspeed/

$(LS).bin : $(PREFIX)/lib/liblightspeed.so.1.0 $(LSINC) $(PREFIX)/python/lightspeed/lightspeed.so $(LSPY2)

$(LS).clean :
	@rm -rf $(OBJDIR)/$(LS) $(PREFIX)/doc/$(LS) $(PREFIX)/include/$(LS) $(PREFIX)/lib/liblightspeed.*

