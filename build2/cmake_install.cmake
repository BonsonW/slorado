# Install script for directory: /data/bonwon/slorado

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE FILE FILES
    "/data/bonwon/slorado/torch/libtorch/lib/libbackend_with_compiler.so"
    "/data/bonwon/slorado/torch/libtorch/lib/libc10.so"
    "/data/bonwon/slorado/torch/libtorch/lib/libc10_cuda.so"
    "/data/bonwon/slorado/torch/libtorch/lib/libc10d_cuda_test.so"
    "/data/bonwon/slorado/torch/libtorch/lib/libcaffe2_detectron_ops_gpu.so"
    "/data/bonwon/slorado/torch/libtorch/lib/libcaffe2_module_test_dynamic.so"
    "/data/bonwon/slorado/torch/libtorch/lib/libcaffe2_nvrtc.so"
    "/data/bonwon/slorado/torch/libtorch/lib/libcaffe2_observers.so"
    "/data/bonwon/slorado/torch/libtorch/lib/libcudart-80664282.so.10.2"
    "/data/bonwon/slorado/torch/libtorch/lib/libfbjni.so"
    "/data/bonwon/slorado/torch/libtorch/lib/libgomp-a34b3233.so.1"
    "/data/bonwon/slorado/torch/libtorch/lib/libjitbackend_test.so"
    "/data/bonwon/slorado/torch/libtorch/lib/libnnapi_backend.so"
    "/data/bonwon/slorado/torch/libtorch/lib/libnvToolsExt-3965bdd0.so.1"
    "/data/bonwon/slorado/torch/libtorch/lib/libnvrtc-08c4863f.so.10.2"
    "/data/bonwon/slorado/torch/libtorch/lib/libnvrtc-builtins.so"
    "/data/bonwon/slorado/torch/libtorch/lib/libpytorch_jni.so"
    "/data/bonwon/slorado/torch/libtorch/lib/libshm.so"
    "/data/bonwon/slorado/torch/libtorch/lib/libtorch.so"
    "/data/bonwon/slorado/torch/libtorch/lib/libtorch_cpu.so"
    "/data/bonwon/slorado/torch/libtorch/lib/libtorch_cuda.so"
    "/data/bonwon/slorado/torch/libtorch/lib/libtorch_global_deps.so"
    "/data/bonwon/slorado/torch/libtorch/lib/libtorch_python.so"
    "/data/bonwon/slorado/torch/libtorch/lib/libtorchbind_test.so"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/data/bonwon/slorado/build2/slow5lib/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/data/bonwon/slorado/build2/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
