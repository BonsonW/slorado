CC       = gcc
CXX		 = c++

LIBTORCH_DIR ?= thirdparty/torch/libtorch
CPPFLAGS += -I slow5lib/include/ \
			-I src/ \
			-I $(LIBTORCH_DIR)/include/torch/csrc/api/include \
			-I $(LIBTORCH_DIR)/include -I thirdparty/ \
			-I thirdparty/tomlc99/ \
			-I openfish/include \
			-I fluke/include
CFLAGS	+= 	-g -Wall -O2
CXXFLAGS   += -g -Wall -O2 -std=c++17
DEPFLAGS = -MMD -MP -MF $(@:.o=.d)
ifdef metal
# macOS/Apple Silicon: libtorch ships dylibs; ld64 has no --as-needed and uses
# @loader_path (not $ORIGIN) for relative rpaths. The macOS arm64 package bundles
# the MPS backend inside libtorch_cpu.dylib, so no extra device library is linked.
LIBS    +=  -Wl,-rpath,@loader_path/$(LIBTORCH_DIR)/lib -Wl,-rpath,@loader_path/../lib \
			-Wl,-rpath,$(LIBTORCH_DIR)/lib \
			$(LIBTORCH_DIR)/lib/libtorch_cpu.dylib \
			$(LIBTORCH_DIR)/lib/libtorch.dylib \
			$(LIBTORCH_DIR)/lib/libc10.dylib
LDFLAGS  += $(LIBS) -lz -lm -lpthread
else
LIBS    +=  -Wl,-rpath,'$$ORIGIN/$(LIBTORCH_DIR)/lib' -Wl,-rpath,'$$ORIGIN/../lib' \
			-Wl,-rpath,$(LIBTORCH_DIR)/lib \
			-Wl,--as-needed,"$(LIBTORCH_DIR)/lib/libtorch_cpu.so"  \
			-Wl,--as-needed,"$(LIBTORCH_DIR)/lib/libtorch.so"  \
			-Wl,--as-needed $(LIBTORCH_DIR)/lib/libc10.so
LDFLAGS  += $(LIBS) -lz -lm -lpthread
endif
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
	  $(BUILD_DIR)/pipeline.o \
	  $(BUILD_DIR)/misc.o \
	  $(BUILD_DIR)/error.o \
	  $(BUILD_DIR)/writer.o \
	  $(BUILD_DIR)/torchbox.o \
	  $(BUILD_DIR)/basecall.o \
	  $(BUILD_DIR)/modcall.o \
	  $(BUILD_DIR)/calib.o \
	  $(BUILD_DIR)/quant.o \
	  $(BUILD_DIR)/sensitivity.o \
	  $(BUILD_DIR)/tensor_chunk_utils.o \
	  $(BUILD_DIR)/modbase.o \
	  $(BUILD_DIR)/lstm_model.o \
	  $(BUILD_DIR)/tx_model.o \
	  $(BUILD_DIR)/modbase_model.o \
	  $(BUILD_DIR)/model_config.o \
	  $(BUILD_DIR)/toml.o \
	  $(BUILD_DIR)/fluke_wrapper.o \

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
	# The fused int8 kernels + their arch dispatch now live in the fluke library (libfluke.a,
	# built below like libopenfish.a). fluke's own Makefile bundles the AOT kernel objects and
	# does the CUDA-symbol objcopy + CUDA-12 gating internally, so nothing extra to link here.
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
else ifdef metal
	# Apple Silicon GPU via libtorch's MPS backend. The MPS runtime lives inside
	# libtorch_cpu.dylib (linked in the base block above); here we only add the
	# Apple frameworks that openfish's Metal decode objects need.
	CPPFLAGS += -DUSE_GPU=1 -DHAVE_METAL=1
	# clang++ links libc++ implicitly; -lobjc is needed for openfish's Objective-C runtime symbols.
	# IOKit is used to query the GPU core count for the LSTM kernel dispatch.
	LDFLAGS += -framework Metal -framework Foundation -framework CoreFoundation -framework IOKit -lobjc
	# slorado's Objective-C++ Metal glue: scores->MTLBuffer for openfish decode, and the dorado
	# LSTM kernel driver (fast/hac v5 acceleration).
	OBJ += $(BUILD_DIR)/metal_utils.o $(BUILD_DIR)/lstm_model_metal.o
endif

.PHONY: clean distclean test

#include ""
#include ""

# slorado
$(BINARY): $(OBJ) slow5lib/lib/libslow5.a openfish/lib/libopenfish.a fluke/lib/libfluke.a
	$(CXX) $(CFLAGS) $(OBJ) slow5lib/lib/libslow5.a openfish/lib/libopenfish.a fluke/lib/libfluke.a $(LDFLAGS) -o $@

$(BUILD_DIR)/main.o: src/main.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

# Objective-C++ Metal glue (metal=1 builds only; added to OBJ in the metal branch above).
$(BUILD_DIR)/metal_utils.o: src/metal_utils.mm
	xcrun clang++ -x objective-c++ -fobjc-arc -std=c++17 $(CFLAGS) -I src/ $(DEPFLAGS) $< -c -o $@

# Embed the dorado LSTM Metal kernels as a C string (compiled at runtime via newLibraryWithSource:).
# Each line is wrapped as a string literal with a trailing \n; adjacent literals concatenate.
$(BUILD_DIR)/lstm_model_metal_src.h: thirdparty/dorado/lstm_model_metal.metal
	printf 'static const char KERNELS_METAL_SRC[] =\n' > $@
	sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/"/' -e 's/$$/\\n"/' $< >> $@
	printf ';\n' >> $@

$(BUILD_DIR)/lstm_model_metal.o: thirdparty/dorado/lstm_model_metal.mm $(BUILD_DIR)/lstm_model_metal_src.h
	xcrun clang++ -x objective-c++ -fobjc-arc -std=c++17 $(CFLAGS) -I src/ -I thirdparty/dorado/ -I $(BUILD_DIR) $(DEPFLAGS) $< -c -o $@

$(BUILD_DIR)/basecaller_main.o: src/basecaller_main.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

$(BUILD_DIR)/slorado.o: src/slorado.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

$(BUILD_DIR)/thread.o: src/thread.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

$(BUILD_DIR)/pipeline.o: src/pipeline.cpp
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

$(BUILD_DIR)/modcall.o: src/modcall.cpp
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

$(BUILD_DIR)/lstm_model.o: thirdparty/dorado/lstm_model.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

$(BUILD_DIR)/tx_model.o: thirdparty/dorado/tx_model.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

# fluke_wrapper — thin ATen facade over the fluke library's fused-int8 C ABI (libfluke.a).
$(BUILD_DIR)/fluke_wrapper.o: thirdparty/fluke_wrapper.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

$(BUILD_DIR)/modbase_model.o: thirdparty/dorado/modbase_model.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

$(BUILD_DIR)/model_config.o: thirdparty/dorado/model_config.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

# toml
$(BUILD_DIR)/toml.o: thirdparty/tomlc99/toml.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

openfish/lib/libopenfish.a:
	$(MAKE) -C openfish cuda=$(cuda) rocm=$(rocm) metal=$(metal) ROCM_ROOT="$(ROCM_ROOT)" ROCM_ARCH="$(ROCM_ARCH)" CUDA_ROOT="$(CUDA_ROOT)" CUDA_ARCH="$(CUDA_ARCH)" lib/libopenfish.a

fluke/lib/libfluke.a:
	$(MAKE) -C fluke cuda=$(cuda) rocm=$(rocm) ROCM_ROOT="$(ROCM_ROOT)" ROCM_ARCH="$(ROCM_ARCH)" CUDA_ROOT="$(CUDA_ROOT)" CUDA_ARCH="$(CUDA_ARCH)" lib/libfluke.a

slow5lib/lib/libslow5.a:
	$(MAKE) -C slow5lib zstd=$(zstd) no_simd=$(no_simd) zstd_local=$(zstd_local) lib/libslow5.a

clean:
	rm -rf $(BINARY) $(BUILD_DIR)/*.o $(BUILD_DIR)/*.d
	make -C slow5lib clean
	make -C openfish clean
	make -C fluke clean

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
