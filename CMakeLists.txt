#******************************************************************************
#
# \file   CMakeLists.txt
# \author Daniel Strigl, Klaus Kofler
# \date   Jun 01 2009
#
# $Id: CMakeLists.txt 2411 2009-10-22 17:41:16Z dast $
#
#******************************************************************************

project(cnnplus)
cmake_minimum_required(VERSION 2.6.1)
cmake_policy(VERSION 2.6.1)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMakeModules;${PROJECT_SOURCE_DIR}/CMakeModules/cuda")

if(NOT EXECUTABLE_OUTPUT_PATH)
  set(EXECUTABLE_OUTPUT_PATH
    ${PROJECT_BINARY_DIR}/bin
    CACHE PATH
    "Single output directory for building all executables"
    )
endif()

if(NOT LIBRARY_OUTPUT_PATH)
  set(LIBRARY_OUTPUT_PATH
    ${PROJECT_BINARY_DIR}/bin
    CACHE PATH
    "Single output directory for building all libraries"
    )
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: None (CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used), Debug, Release, RelWithDebInfo and MinSizeRel"
    FORCE
    )
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(DEBUG_DEFAULT ON)
  set(RELEASE_DEFAULT OFF)
else()
  set(DEBUG_DEFAULT OFF)
  set(RELEASE_DEFAULT ON)
endif()

set(CNNPLUS_LIBRARY ${CMAKE_PROJECT_NAME})
set(CNNPLUS_LIBRARIES)

include(machine-settings)

#
# Matlab
#
if(MATLAB_FOUND)
  include_directories(${MATLAB_INC_PATH})
  link_directories(${MATLAB_LIB_PATH})
  list(APPEND CNNPLUS_LIBRARIES ${MATLAB_LIBS})
  add_definitions(-DCNNPLUS_MATLAB_FOUND)
endif()

#
# Intel Performance Libraries
#
if(INTEL_LIBS_FOUND)
  option(CNNPLUS_USE_INTEL_LIBS
    "Use Intel Performance Libraries"
    ON
    )
  if(CNNPLUS_USE_INTEL_LIBS)
    include_directories(${INTEL_MKL_INC_PATH} ${INTEL_IPP_INC_PATH})
    link_directories(${INTEL_LIB_PATH} ${INTEL_MKL_LIB_PATH} ${INTEL_IPP_LIB_PATH})
    list(APPEND CNNPLUS_LIBRARIES ${INTEL_LIBS} ${INTEL_MKL_LIBS} ${INTEL_IPP_LIBS})
    add_definitions(-DCNNPLUS_USE_INTEL_LIBS)
  endif()
else()
  message("Intel Performance Libraries not found, disabling it.")
endif()

#
# Unix libraries
#
if(UNIX)
  find_package(Threads REQUIRED)
  list(APPEND CNNPLUS_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
  find_package(X11 REQUIRED)
  list(APPEND CNNPLUS_LIBRARIES ${X11_LIBRARIES})
endif()

#
# CUDA
#
find_package(CUDA)
if(CUDA_FOUND)
  option(CNNPLUS_USE_CUDA
    "Use CUDA"
    ON
    )
  if(CNNPLUS_USE_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    list(APPEND CNNPLUS_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
    add_definitions(-DCNNPLUS_USE_CUDA)
  endif()
else()
  message("CUDA not found, disabling it.")
endif()

option(CNNPLUS_PRINT_PROGRESS
  "Print progress during test and training"
  OFF
  )
if(CNNPLUS_PRINT_PROGRESS)
  add_definitions(-DCNNPLUS_PRINT_PROGRESS)
endif()

if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

#
# cnnplus library
#
add_subdirectory(cnnplus)

set(CNNPLUS_LIBRARIES ${CNNPLUS_LIBRARY} ${CNNPLUS_LIBRARIES})
#message("CNNPLUS_LIBRARIES: " ${CNNPLUS_LIBRARIES})

#
# Benchmarks
#
if(NOT CMAKE_CROSSCOMPILING)
  option(CNNPLUS_ENABLE_BENCHMARKS
    "Enable and build the benchmarks"
    ON
    )
  if(CNNPLUS_ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
  endif()
endif(NOT CMAKE_CROSSCOMPILING)

#
# Examples
#
if(NOT CMAKE_CROSSCOMPILING)
  option(CNNPLUS_ENABLE_EXAMPLES
    "Enable and build the examples"
    ON
    )
  if(CNNPLUS_ENABLE_EXAMPLES)
    add_subdirectory(examples)
  endif()
endif(NOT CMAKE_CROSSCOMPILING)

#
# Tests
#
if(NOT CMAKE_CROSSCOMPILING)
  if(MSVC)
    if(MSVC90)
      set(CPPUNIT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/win32/include")
      set(CPPUNIT_LIBRARIES   "${PROJECT_SOURCE_DIR}/win32/libs/msvc90/cppunit.lib")
      set(CPPUNIT_FOUND       TRUE)
    elseif(MSVC80)
      set(CPPUNIT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/win32/include")
      set(CPPUNIT_LIBRARIES   "${PROJECT_SOURCE_DIR}/win32/libs/msvc80/cppunit.lib")
      set(CPPUNIT_FOUND       TRUE)
    else()
      set(CPPUNIT_FOUND       FALSE)
    endif()
  else()
    find_package(CppUnit)
  endif()
  if(CPPUNIT_FOUND)
    option(
      CNNPLUS_ENABLE_TESTS
      "Enable unit tests"
      ON
      )
  else()
    message("CppUnit not found, disabling it.")
  endif()
  if(CNNPLUS_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
  endif()
endif(NOT CMAKE_CROSSCOMPILING)
