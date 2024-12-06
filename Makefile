CC       = cc
CXX		 = c++

LIBTORCH_DIR ?= thirdparty/torch/libtorch
CPPFLAGS += -I slow5lib/include/ \
			-I src/ \
			-I $(LIBTORCH_DIR)/include/torch/csrc/api/include \
			-I $(LIBTORCH_DIR)/include -I thirdparty/ \
			-I thirdparty/tomlc99/ \
			-I openfish/include
CFLAGS	+= 	-g -Wall -O2
CXXFLAGS   += -g -Wall -O2  -std=c++17
LIBS    +=  -Wl,-rpath,'$$ORIGIN/$(LIBTORCH_DIR)/lib' -Wl,-rpath,'$$ORIGIN/../lib' \
			-Wl,-rpath,$(LIBTORCH_DIR)/lib \
			-Wl,--as-needed,"$(LIBTORCH_DIR)/lib/libtorch_cpu.so"  \
			-Wl,--as-needed,"$(LIBTORCH_DIR)/lib/libtorch.so"  \
			-Wl,--as-needed $(LIBTORCH_DIR)/lib/libc10.so
LDFLAGS  += $(LIBS) -lz -lm -lpthread
BUILD_DIR = build

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
	  $(BUILD_DIR)/elephant.o \
	  $(BUILD_DIR)/basecall.o \
	  $(BUILD_DIR)/signal_prep_stitch_tensor_utils.o \
	  $(BUILD_DIR)/CRFModel.o \
	  $(BUILD_DIR)/TxModel.o \
	  $(BUILD_DIR)/model_config.o \
	  $(BUILD_DIR)/toml.o \

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
	LIBS += -Wl,--as-needed -lpthread -Wl,--no-as-needed,"$(LIBTORCH_DIR)/lib/libtorch_cuda.so" -Wl,--as-needed,"$(LIBTORCH_DIR)/lib/libc10_cuda.so"
	LDFLAGS += -L$(CUDA_LIB) -lcudart_static -lrt -ldl
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
$(BINARY): $(OBJ) slow5lib/lib/libslow5.a openfish/lib/libopenfish.a
	$(CXX) $(CFLAGS) $(OBJ) slow5lib/lib/libslow5.a openfish/lib/libopenfish.a $(LDFLAGS) -o $@

$(BUILD_DIR)/main.o: src/main.cpp src/error.h src/misc.h src/slorado.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

$(BUILD_DIR)/basecaller_main.o: src/basecaller_main.cpp src/error.h src/misc.h src/slorado.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

$(BUILD_DIR)/slorado.o: src/slorado.cpp src/misc.h src/error.h src/slorado.h src/basecall.h src/writer.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

$(BUILD_DIR)/thread.o: src/thread.cpp src/misc.h src/error.h src/slorado.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

$(BUILD_DIR)/misc.o: src/misc.cpp src/misc.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

$(BUILD_DIR)/error.o: src/error.cpp src/error.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

$(BUILD_DIR)/writer.o: src/writer.cpp src/error.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

$(BUILD_DIR)/elephant.o: src/elephant.cpp src/elephant.h src/slorado.h thirdparty/dorado/signal_prep_stitch_tensor_utils.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

$(BUILD_DIR)/basecall.o: src/basecall.cpp src/basecall.h src/misc.h src/error.h src/elephant.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

# dorado
$(BUILD_DIR)/signal_prep_stitch_tensor_utils.o: thirdparty/dorado/signal_prep_stitch_tensor_utils.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

$(BUILD_DIR)/CRFModel.o: thirdparty/dorado/CRFModel.cpp thirdparty/dorado/CRFModel.h src/error.h thirdparty/dorado/signal_prep_stitch_tensor_utils.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

$(BUILD_DIR)/TxModel.o: thirdparty/dorado/TxModel.cpp thirdparty/dorado/TxModel.h src/error.h thirdparty/dorado/signal_prep_stitch_tensor_utils.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

$(BUILD_DIR)/model_config.o: thirdparty/dorado/model_config.cpp thirdparty/dorado/model_config.h thirdparty/dorado/CRFModel.h src/error.h thirdparty/tomlc99
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

# toml
$(BUILD_DIR)/toml.o: thirdparty/tomlc99/toml.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -c -o $@

openfish/lib/libopenfish.a:
	$(MAKE) -C openfish cuda=$(cuda) rocm=$(rocm) ROCM_ROOT=$(ROCM_ROOT) ROCM_ARCH=$(ROCM_ARCH) CUDA_ROOT=$(CUDA_ROOT) CUDA_ARCH=$(CUDA_ARCH) lib/libopenfish.a

slow5lib/lib/libslow5.a:
	$(MAKE) -C slow5lib zstd=$(zstd) no_simd=$(no_simd) zstd_local=$(zstd_local) lib/libslow5.a

clean:
	rm -rf $(BINARY) $(BUILD_DIR)/*.o
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
	./test/mem.sh mem
