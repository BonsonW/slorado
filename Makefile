CC       = gcc
CXX		 = c++

LIBTORCH_DIR ?= thirdparty/torch/libtorch
CPPFLAGS += -I slow5lib/include/ \
			-I src/ \
			-I $(LIBTORCH_DIR)/include/torch/csrc/api/include \
			-I $(LIBTORCH_DIR)/include -I thirdparty/ \
			-I thirdparty/tomlc99/ \
			-I openfish/include
CFLAGS	+= 	-g -Wall -O2
CXXFLAGS   += -g -Wall -O2 -std=c++17
DEPFLAGS = -MMD -MP -MF $(@:.o=.d)
LIBS    +=  -Wl,-rpath,'$$ORIGIN/$(LIBTORCH_DIR)/lib' -Wl,-rpath,'$$ORIGIN/../lib' \
			-Wl,-rpath,$(LIBTORCH_DIR)/lib \
			-Wl,--as-needed,"$(LIBTORCH_DIR)/lib/libtorch_cpu.so"  \
			-Wl,--as-needed,"$(LIBTORCH_DIR)/lib/libtorch.so"  \
			-Wl,--as-needed $(LIBTORCH_DIR)/lib/libc10.so
LDFLAGS  += $(LIBS) -lz -lm -lpthread
BUILD_DIR = build

ifeq ($(f16c),1)
CXXFLAGS   += -mavx2 -mf16c
endif

ifeq ($(zstd),1)
LDFLAGS		+= -lzstd
endif

ifeq ($(zstd_local),)
else
LDFLAGS		+= zstd/lib/libzstd.a
endif

# https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html
ifeq ($(cxx11_abi),) #  cxx11_abi not defined
CXXFLAGS		+= -D_GLIBCXX_USE_CXX11_ABI=0
endif

# change the tool name to what you want
BINARY = slorado

OBJ = $(BUILD_DIR)/main.o \
      $(BUILD_DIR)/basecaller_main.o \
      $(BUILD_DIR)/slorado.o \
      $(BUILD_DIR)/thread.o \
	  $(BUILD_DIR)/misc.o \
	  $(BUILD_DIR)/error.o \
	  $(BUILD_DIR)/writer.o \
	  $(BUILD_DIR)/torchbox.o \
	  $(BUILD_DIR)/basecall.o \
	  $(BUILD_DIR)/calib.o \
	  $(BUILD_DIR)/quant.o \
	  $(BUILD_DIR)/sensitivity.o \
	  $(BUILD_DIR)/tensor_chunk_utils.o \
	  $(BUILD_DIR)/modbase.o \
	  $(BUILD_DIR)/CRFModel.o \
	  $(BUILD_DIR)/TxModel.o \
	  $(BUILD_DIR)/ModBaseModel.o \
	  $(BUILD_DIR)/model_config.o \
	  $(BUILD_DIR)/toml.o \
	  $(BUILD_DIR)/flute.o \

# add more objects here if needed

VERSION = `git describe --tags`

# make asan=1 enables address sanitiser
ifdef asan
	CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer
	CFLAGS += -fsanitize=address -fno-omit-frame-pointer
	LDFLAGS += -fsanitize=address -fno-omit-frame-pointer
endif

# make accel=1 enables the acceelerator (CUDA,OpenCL,FPGA etc if implemented)
ifdef cuda
    CPPFLAGS += -DUSE_GPU=1
    CPPFLAGS += -DHAVE_CUDA=1
	CUDA_ROOT ?= /usr/local/cuda
	CUDA_LIB ?= $(CUDA_ROOT)/lib64
	CUDA_INC ?= $(CUDA_ROOT)/include
	CPPFLAGS += -I $(CUDA_INC)
	# precompiled fused int8 kernels (cutedsl/CUTLASS) for the real quant inference path.
	# The shipped objects reference underscore-prefixed CUDA symbols; objcopy rewrites them
	# to the real ELF names (resolved from cudart_static + the driver stub libcuda).
	# (flute.o itself is always built — its CPU body is just a nullptr backend stub.)
	FLUTE_OBJ = $(BUILD_DIR)/gemm_i8_dual_silu_N2048_K512.o \
	            $(BUILD_DIR)/gemm_i8_rotary_N1536_K512_H8D64R64S1024.o
	LIBS += -Wl,--as-needed -lpthread -Wl,--no-as-needed,"$(LIBTORCH_DIR)/lib/libtorch_cuda.so" -Wl,--as-needed,"$(LIBTORCH_DIR)/lib/libc10_cuda.so"
	LDFLAGS += -L$(CUDA_LIB) -lcudart_static -L$(CUDA_LIB)/stubs -lcuda -lrt -ldl
else ifdef rocm
	CPPFLAGS += -DUSE_GPU=1 -DHAVE_ROCM=1 -D__HIP_PLATFORM_AMD__
	ROCM_ROOT ?= /opt/rocm
	ROCM_INC ?= $(ROCM_ROOT)/include
	ROCM_LIB ?= $(ROCM_ROOT)/lib
	CPPFLAGS += -I $(ROCM_INC)
	LIBS += -Wl,--as-needed -lpthread -Wl,--no-as-needed,"$(LIBTORCH_DIR)/lib/libtorch_hip.so" -Wl,--as-needed,"$(LIBTORCH_DIR)/lib/libc10_hip.so"
	LDFLAGS += -L$(ROCM_LIB) -lamdhip64 -lrt -ldl
endif

.PHONY: clean distclean test

#include ""
#include ""

# slorado
$(BINARY): $(OBJ) $(FLUTE_OBJ) slow5lib/lib/libslow5.a openfish/lib/libopenfish.a
	$(CXX) $(CFLAGS) $(OBJ) $(FLUTE_OBJ) slow5lib/lib/libslow5.a openfish/lib/libopenfish.a $(LDFLAGS) -o $@

$(BUILD_DIR)/main.o: src/main.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

$(BUILD_DIR)/basecaller_main.o: src/basecaller_main.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

$(BUILD_DIR)/slorado.o: src/slorado.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

$(BUILD_DIR)/thread.o: src/thread.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

$(BUILD_DIR)/misc.o: src/misc.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

$(BUILD_DIR)/error.o: src/error.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

$(BUILD_DIR)/writer.o: src/writer.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

$(BUILD_DIR)/torchbox.o: src/torchbox.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

$(BUILD_DIR)/basecall.o: src/basecall.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

$(BUILD_DIR)/calib.o: src/calib.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

$(BUILD_DIR)/quant.o: src/quant.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

$(BUILD_DIR)/sensitivity.o: src/sensitivity.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

# dorado
$(BUILD_DIR)/tensor_chunk_utils.o: thirdparty/dorado/tensor_chunk_utils.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

$(BUILD_DIR)/modbase.o: thirdparty/dorado/modbase.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

$(BUILD_DIR)/CRFModel.o: thirdparty/dorado/CRFModel.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

$(BUILD_DIR)/TxModel.o: thirdparty/dorado/TxModel.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

# flute — facade over the precompiled fused int8 kernels (cuda builds only)
$(BUILD_DIR)/flute.o: thirdparty/flute/flute.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

# Rewrite the underscore-prefixed CUDA symbol references in the precompiled kernel objects
# to their real ELF names so they resolve against cudart_static / libcuda.
$(BUILD_DIR)/%.o: thirdparty/flute/sm80/%.o
	objcopy \
	  --redefine-sym _cudaDeviceGetAttribute=cudaDeviceGetAttribute \
	  --redefine-sym _cudaFuncSetAttribute=cudaFuncSetAttribute \
	  --redefine-sym _cudaGetDevice=cudaGetDevice \
	  --redefine-sym _cudaKernelSetAttributeForDevice=cudaKernelSetAttributeForDevice \
	  --redefine-sym _cudaLaunchKernelEx=cudaLaunchKernelExC \
	  --redefine-sym _cudaLibraryGetKernel=cudaLibraryGetKernel \
	  --redefine-sym _cudaLibraryLoadData=cudaLibraryLoadData \
	  --redefine-sym _cuKernelGetAttribute=cuKernelGetAttribute \
	  $< $@

$(BUILD_DIR)/ModBaseModel.o: thirdparty/dorado/ModBaseModel.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

$(BUILD_DIR)/model_config.o: thirdparty/dorado/model_config.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

# toml
$(BUILD_DIR)/toml.o: thirdparty/tomlc99/toml.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

openfish/lib/libopenfish.a:
	$(MAKE) -C openfish cuda=$(cuda) rocm=$(rocm) ROCM_ROOT=$(ROCM_ROOT) ROCM_ARCH=$(ROCM_ARCH) CUDA_ROOT=$(CUDA_ROOT) CUDA_ARCH=$(CUDA_ARCH) lib/libopenfish.a

slow5lib/lib/libslow5.a:
	$(MAKE) -C slow5lib zstd=$(zstd) no_simd=$(no_simd) zstd_local=$(zstd_local) lib/libslow5.a

clean:
	rm -rf $(BINARY) $(BUILD_DIR)/*.o $(BUILD_DIR)/*.d
	make -C slow5lib clean
	make -C openfish clean

# Delete all gitignored files (but not directories)
distclean: clean
	git clean -f -X
	rm -rf $(BUILD_DIR)/* autom4te.cache

# make test with run a simple test
test: $(BINARY)
	./test/test.sh

# make mem with run a simple memory test using valgrind
mem: $(BINARY)
	./test/test.sh mem

-include $(OBJ:.o=.d)
