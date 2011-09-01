# compiler configuration
# show list of all available configurations with make cfg_list
CFG ?= gcc

NVCC ?= /usr/local/cuda/bin/nvcc
PTXAS ?= /usr/local/cuda/bin/ptxas

# machine type: SEQ or PAR
PARTYPE ?= SEQ

# choose target: DEBUG|RELEASE
# some configurations have additional targets, see the respective configuration files for details
TARGET ?= DEBUG

# compile using depmode?
DEPS ?= 1

#build unit tests?
TESTS ?= 0

VTK ?= 0

CUDA_CONFIG ?= CUDA_DOUBLE_SORTED
NUM_WARPS ?= 4

NUM_LOCAL_STORAGE_WARPS ?= $(NUM_WARPS)

MAX_REGISTER_COUNT ?= 63 

CURDIR ?= $(realpath $(PWD) )
ifeq ($(findstring src,$(CURDIR)),)
  $(error ERROR: current directory $(CURDIR) seems not to be a source directory)
endif

#build name and include config file
MAKEFILEDIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
CFGFILE := $(realpath $(MAKEFILEDIR)cfg/$(CFG).in )

ifneq ($(CFGFILE),)
  $(info # Selected configuration: $(CFG) )
  $(info # Including file $(CFGFILE) )
  include $(CFGFILE)
else
  $(error ERROR: config file for $(CFG) not found - no include )
endif

SOURCES_COMMON = $(shell find ./ -name "*.cpp" | grep -v "parallel/" | grep -v "/tests/" | grep -v "/vtk/")
SOURCES_CUDA = $(shell find ./ -name "*.cu" | grep -v "parallel/" | grep -v "/tests/" | grep -v "/vtk/")
SOURCES_CUDA_MODULES = $(shell find ./ -name "*.cum" | grep -v "parallel/" | grep -v "/tests/" | grep -v "/vtk/")
SOURCES_SEQ = $(shell find parallel/ -name "*.cpp" | grep "Dummy" | grep -v "/tests/" | grep -v "/vtk/")
SOURCES_PAR = $(shell find parallel/ -name "*.cpp" | grep -v "Dummy" | grep -v "/tests/" | grep -v "/vtk/")
SOURCES = $(SOURCES_COMMON) $(SOURCES_$(PARTYPE)) 


OBJECTS = $(SOURCES:.cpp=.o) $(SOURCES_CUDA:.cu=.o)
# ptx files are stored in the output directory
PTXS = $(SOURCES_CUDA:.cu=.ptx)

INCLUDES = -I$(CURDIR) -I$(CURDIR)/External $(CPPUNIT_INCLUDES) -I../dependencies-external/nvidia

BINARYbasename = MarDyn.$(CUDA_CONFIG).$(NUM_WARPS).$(NUM_LOCAL_STORAGE_WARPS).$(MAX_REGISTER_COUNT)
ifneq ($(realpath .svn),)
  $(info Building from a svn version.)
  svnVERSION = $(shell svnversion | sed -e "s/:/-/")
  ifneq ($(svnVERSION),)
    BINARY = $(BINARYbasename)_$(svnVERSION).$(PARTYPE)_$(TARGET)
  else
    BINARY = $(BINARYbasename).$(PARTYPE)_$(TARGET)
  endif
else
  BINARY = $(BINARYbasename).$(PARTYPE)_$(TARGET)
endif

SHARED_OPTIONS=-DCONFIG_$(CUDA_CONFIG) -DMAX_NUM_WARPS=$(NUM_WARPS) -DNUM_LOCAL_STORAGE_WARPS=$(NUM_LOCAL_STORAGE_WARPS) -DMAX_REGISTER_COUNT=$(MAX_REGISTER_COUNT)
CXX = $(CXX_$(PARTYPE))
CXXFLAGS = $(CXXFLAGS_$(PARTYPE)_$(TARGET)) -DMPICH_IGNORE_CXX_SEEK $(SHARED_OPTIONS)
LINKFLAGS = $(LINKFLAGS_$(PARTYPE)_$(TARGET)) $(CPPUNIT_LINK_ARGS) -lrt -lcuda

ifeq ($(DEPS),1)
  DEPCOMP = depmode=$(DEPMODE) source=$< object=$@ $(MAKEFILEDIR)depcomp
else
  DEPCOMP =
endif

VTK_MAKEFILE := $(realpath $(MAKEFILEDIR)vtkwriter.mk ) 
include $(VTK_MAKEFILE)

$(BINARY): .deps $(OBJECTS) cppunit
	@echo "building $(BINARY)"
	$(CXX) $(LINKFLAGS) -o $@ $(OBJECTS)
	$(shell if [ ! -f $(BINARYbasename) -o -L $(BINARYbasename) ]; then ln -sf $(BINARY) $(BINARYbasename); fi)

%.o: %.cpp
	$(DEPCOMP) $(CXX) $(CXXFLAGS) -c $< -o $@

define CU_BUILDER
$(shell $(NVCC) --generate-dependencies $(2) --output-directory ./cuda | sed -e 's/.o/.ptx/g' > $(2:.cu=.deps))
endef

$(call CU_BUILDER,./cuda/kernel.ptx,./cuda/kernel.cu)
#include ./cuda/kernel.deps

cuda/kernel.ptx: $(SOURCES_CUDA_MODULES) cuda/config.h cuda/sharedDecls.h

%.ptx: %.cu 
	$(NVCC) -arch=sm_21 --ptx -Xptxas -v -Xcompiler $(SHARED_OPTIONS) $< --output-directory $(<D)
	$(PTXAS) -arch=sm_21 -maxrregcount $(MAX_REGISTER_COUNT) -v $@
%.o: %.ptx
	ld -r -b binary -o $@ $<

#$(foreach cufile,$(SOURCES_CUDA),$(eval $(call CU_BUILDER,$(cufile:.cu=.ptx),$(cufile))))
 
 
.PHONY: help all clean cleanall depend cfg_list test

help:
	@echo "configuration: $(CFG_INFO)"
	@echo
	@echo "options:"
	@echo "	CFG=...                 choose compiler (see make cfg_list)"
	@echo " PARTYPE=SEQ | PAR       sequential or parallel (MPI) build"
	@echo " TARGET=DEBUG | RELEASE  debugging / non optimized build"
	@echo " DEPS=0 | 1              compile using dependancy tracking"
	@echo " TESTS=0 | 1             compile with or without unit tests (and build cppunit if neccessary)"
	@echo " VTK=0 | 1	            compile with or without the vtk output plugin" 
	@echo
	@echo "targets:"
	@echo "make all         build $(BINARY)"
	@echo "make cfg_list    list available configurations"
	@echo "make clean       delete object files"
	@echo "make cleanall    delete object files and executable"
	@echo "make depend      generate makefile dependency list"

all: $(BINARY)

test: 
	./$(BINARY) -t

#list available configurations
cfg_list:
	@ls -1 ../makefile/cfg | sed -e 's/.in//'

clean:
	find -type f -name '*.o' -delete
	find -path '*/.deps/*' -type f -delete
	find -type d -name .deps -delete
	find -type f -name '*.ptx' -delete

clear: cleanall

cleanall: clean cppunit_clean
	rm -f $(BINARY)
	find -type f -name '*~' -delete
	if [ -L $(BINARYbasename) ]; then rm -f $(BINARYbasename); fi

# create .deps subdirectories containing dependency information
.deps:
ifeq ($(DEPS),1)
	$(shell find -type d ! -name .deps -execdir mkdir -p {}/.deps \;)
endif

# include potentially generated dependency information
-include $(shell find -path '*/.deps/*' -type f)
  
#include makefile for cppunit tests
CPPUNIT_MAKEFILE := $(realpath $(MAKEFILEDIR)cppunit.mk )
include $(CPPUNIT_MAKEFILE)

