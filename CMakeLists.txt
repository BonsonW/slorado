cmake_minimum_required (VERSION 3.5)

project(slorado)
set(SLORADO_VERSION "0.0.1")
set(TORCH_VERSION 1.10.2)
#set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_EXTENSIONS OFF)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -rdynamic -Wall -std=c++14 ")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -rdynamic -Wall -std=gnu++11 ")
message(STATUS "Using CMAKE_CXX_STANDARD-${CMAKE_CXX_STANDARD}")
option(USE_GPU "Use GPU" ON) #OFF until koi linked
option(SLOW5_USE_ZSTD_CUSTOM "Use ZSTD custom" OFF) #OFF by default

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/torch)
include_directories(${CMAKE_SOURCE_DIR}/thirdparty/tomlc99/)
if(USE_GPU)
    message(STATUS " Using prebuilt Koi from ${CMAKE_SOURCE_DIR}/thirdparty/koi_lib")
    include_directories(${CMAKE_SOURCE_DIR}/thirdparty/koi_lib/include)
endif()

set(CUDNN_LIBRARY_PATH ${CMAKE_SOURCE_DIR}/thirdparty/fake_cudnn)
set(CUDNN_INCLUDE_PATH ${CMAKE_SOURCE_DIR}/thirdparty/fake_cudnn)
set(TORCH_LIB "${CMAKE_SOURCE_DIR}/thirdparty/torch/libtorch")
# #set(TORCH_LIB "${CMAKE_SOURCE_DIR}/3rdparty/torch-${TORCH_VERSION}-${CMAKE_SYSTEM_NAME}-cxx11_ABI/libtorch")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_LIB})
find_package(Torch REQUIRED)

file(GLOB TORCH_DLLS "${TORCH_LIB}/lib/*.so*")
install(FILES ${TORCH_DLLS} DESTINATION lib)

set(LIB_SOURCE_FILES
        ${CMAKE_SOURCE_DIR}/src/misc.cpp
        ${CMAKE_SOURCE_DIR}/src/globals.cpp
        ${CMAKE_SOURCE_DIR}/src/thread.cpp
        ${CMAKE_SOURCE_DIR}/src/error.cpp
        ${CMAKE_SOURCE_DIR}/src/slorado.cpp
        ${CMAKE_SOURCE_DIR}/src/basecaller_main.cpp
        ${CMAKE_SOURCE_DIR}/src/signal_prep.cpp
        ${CMAKE_SOURCE_DIR}/src/basecall.cpp
        ${CMAKE_SOURCE_DIR}/src/writer.cpp
        ${CMAKE_SOURCE_DIR}/src/Chunk.h
        ${CMAKE_SOURCE_DIR}/src/utils/tensor_utils.cpp
        ${CMAKE_SOURCE_DIR}/src/utils/module_utils.cpp
        ${CMAKE_SOURCE_DIR}/src/utils/stitch.cpp
        ${CMAKE_SOURCE_DIR}/src/utils/cuda_utils.cpp
        ${CMAKE_SOURCE_DIR}/src/nn/CRFModel.cpp
        ${CMAKE_SOURCE_DIR}/src/nn/ModelRunner.h
        ${CMAKE_SOURCE_DIR}/src/decode/beam_search.cpp
        ${CMAKE_SOURCE_DIR}/src/decode/CPUDecoder.cpp
        ${CMAKE_SOURCE_DIR}/src/decode/GPUDecoder.cpp
        ${CMAKE_SOURCE_DIR}/src/decode/Decoder.h
        ${CMAKE_SOURCE_DIR}/src/decode/fast_hash.cpp
        )


list(APPEND LIB_SOURCE_FILES ${CMAKE_SOURCE_DIR}/thirdparty/tomlc99/toml.c ${CMAKE_SOURCE_DIR}/thirdparty/tomlc99/toml.h)

if(USE_GPU)
 MESSAGE(STATUS "Adding GPU source files to the compiling queue")
 add_definitions(-DUSE_GPU)
 list(APPEND LIB_SOURCE_FILES
         ${CMAKE_SOURCE_DIR}/src/decode/GPUDecoder.cpp
         ${CMAKE_SOURCE_DIR}/src/decode/GPUDecoder.h)
endif()

if(SLOW5_USE_ZSTD_CUSTOM)
    add_compile_options("-DSLOW5_USE_ZSTD")
    message(STATUS "Using ZSTD custom")
    include_directories(${CMAKE_SOURCE_DIR}/3rdparty/zstd/lib)
endif()

include_directories(${CMAKE_SOURCE_DIR}/slow5lib/include)
include_directories(${CMAKE_SOURCE_DIR}/slow5lib/src)
add_subdirectory(${CMAKE_SOURCE_DIR}/slow5lib)

if(SLOW5_USE_ZSTD_CUSTOM)
    target_link_directories(slow5 PUBLIC ${CMAKE_SOURCE_DIR}/3rdparty/zstd/lib/)
endif()

add_executable(slorado ${CMAKE_SOURCE_DIR}/src/main.cpp ${LIB_SOURCE_FILES})

if(USE_GPU)
    message(STATUS " Using prebuilt Koi")
    target_include_directories(slorado PUBLIC thirdparty/koi_lib/include)
    target_link_directories(slorado PUBLIC thirdparty/koi_lib/lib)
endif()


message(STATUS "TORCH_LIBRARIES")
message("${TORCH_LIBRARIES}")
target_link_libraries(slorado ${TORCH_LIBRARIES})
if(USE_GPU)
    target_link_libraries(slorado koi)
endif()
target_link_libraries(slorado slow5 -lz -lzstd -lpthread -lm -lstdc++fs)
