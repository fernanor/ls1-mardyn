# std=c++11
CXXFLAGS += -std=c++11 -pedantic -pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-include-dirs -Wnoexcept -Woverloaded-virtual -Wredundant-decls -Wshadow -Wstrict-null-sentinel -Wstrict-overflow=5 -Wundef -Werror -Wno-unused # -Wsign-conversion -Wswitch-default -Wold-style-cast -Wsign-promo -Wmissing-declarations 

# Vectorization settings:
#########################################
ifeq ($(VECTORIZE_CODE),SSE)
CXXFLAGS_VECTORIZE = -msse3
endif
ifeq ($(VECTORIZE_CODE),AVX)
CXXFLAGS_VECTORIZE = -mavx
endif
ifeq ($(VECTORIZE_CODE),AVX2)
CXXFLAGS_VECTORIZE = -mavx2 -mfma
endif
ifeq ($(VECTORIZE_CODE),KNC_MASK)
$(error gcc does not support KNC)
endif
ifeq ($(VECTORIZE_CODE),KNC_G_S)
$(error gcc does not support KNC)
endif
ifeq ($(VECTORIZE_CODE),KNL_MASK)
CXXFLAGS_VECTORIZE = -march=knl
endif
ifeq ($(VECTORIZE_CODE),KNL_G_S)
CXXFLAGS_VECTORIZE = -march=knl -D__VCP_GATHER__
endif

# On AMD BUlldozer:
ifeq ($(VECTORIZE_CODE),SSEAMD)
CXXFLAGS_VECTORIZE = -msse3 -mfma4 -march=bdver1
endif
ifeq ($(VECTORIZE_CODE),AVXAMD)
CXXFLAGS_VECTORIZE = -mavx -mfma4 -march=bdver1
endif

# OpenMP settings:
#########################################
ifeq ($(OPENMP),1)
FLAGS_OPENMP += -fopenmp
else
  ifeq ($(OPENMP_SIMD),1)
  FLAGS_OPENMP += -fopenmp-simd
  endif
endif
