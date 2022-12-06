CC       = cc
CPPFLAGS += -I slow5lib/include/
CFLAGS   += -g -Wall -O2  -std=c99
LDFLAGS  += $(LIBS) -lz -lm -lpthread
BUILD_DIR = build

ifeq ($(zstd),1)
LDFLAGS		+= -lzstd
endif

# change the tool name to what you want
BINARY = xyztool

OBJ = $(BUILD_DIR)/main.o \
      $(BUILD_DIR)/xyztool.o \
      $(BUILD_DIR)/subtool1_main.o \
      $(BUILD_DIR)/thread.o \
	  $(BUILD_DIR)/misc.o \
	  $(BUILD_DIR)/error.o \
# add more objects here if needed

VERSION = `git describe --tags`

# make asan=1 enables address sanitiser
ifdef asan
	CFLAGS += -fsanitize=address -fno-omit-frame-pointer
	LDFLAGS += -fsanitize=address -fno-omit-frame-pointer
endif

# make accel=1 enables the acceelerator (CUDA,OpenCL,FPGA etc if implemented)
ifdef acc
    CPPFLAGS += -DHAVE_ACC=1
endif

.PHONY: clean distclean test

$(BINARY): $(OBJ) slow5lib/lib/libslow5.a
	$(CC) $(CFLAGS) $(OBJ) slow5lib/lib/libslow5.a $(LDFLAGS) -o $@

$(BUILD_DIR)/main.o: src/main.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -c -o $@

$(BUILD_DIR)/xyztool.o: src/xyztool.c src/misc.h src/error.h src/xyztool.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -c -o $@

$(BUILD_DIR)/subtool1_main.o: src/subtool1_main.c src/error.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -c -o $@

$(BUILD_DIR)/thread.o: src/thread.c src/xyztool.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -c -o $@

$(BUILD_DIR)/misc.o: src/misc.c src/misc.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -c -o $@

$(BUILD_DIR)/error.o: src/error.c src/error.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -c -o $@

# follow the main.o above and add more objects here if needed

slow5lib/lib/libslow5.a:
	$(MAKE) -C slow5lib zstd=$(zstd) no_simd=$(no_simd) zstd_local=$(zstd_local) lib/libslow5.a

clean:
	rm -rf $(BINARY) $(BUILD_DIR)/*.o
	make -C slow5lib clean

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