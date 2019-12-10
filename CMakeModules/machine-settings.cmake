#******************************************************************************
#
# \file   CMakeLists.txt
# \author Daniel Strigl, Klaus Kofler
# \date   Jun 01 2009
#
# $Id: machine-settings.cmake 2467 2009-11-04 10:30:41Z dast $
#
#******************************************************************************

site_name(MACHINE_NAME)

#------------------------------------------------------------------------------
#- dast-nb-winxp --------------------------------------------------------------
#------------------------------------------------------------------------------
if(${MACHINE_NAME} STREQUAL "dast-nb-winxp")

set(MATLAB_INC_PATH    "C:\\Programme\\MATLAB\\R2007b\\extern\\include")
set(INTEL_MKL_INC_PATH "C:\\Programme\\Intel\\Compiler\\11.0\\072\\cpp\\mkl\\include")
set(INTEL_IPP_INC_PATH "C:\\Programme\\Intel\\Compiler\\11.0\\072\\cpp\\ipp\\ia32\\include")

set(MATLAB_LIB_PATH    "C:\\Programme\\MATLAB\\R2007b\\extern\\lib\\win32\\microsoft")
set(INTEL_LIB_PATH     "C:\\Programme\\Intel\\Compiler\\11.0\\072\\cpp\\lib\\ia32")
set(INTEL_MKL_LIB_PATH "C:\\Programme\\Intel\\Compiler\\11.0\\072\\cpp\\mkl\\ia32\\lib")
set(INTEL_IPP_LIB_PATH "C:\\Programme\\Intel\\Compiler\\11.0\\072\\cpp\\ipp\\ia32\\stublib")

set(MATLAB_LIBS        libmx libmex libmat)
set(INTEL_LIBS         libiomp5md)
set(INTEL_MKL_LIBS     mkl_c_dll)
set(INTEL_IPP_LIBS     ippcore ippm ipps ippsr ippvm ippi)

set(MATLAB_FOUND       TRUE)
set(INTEL_LIBS_FOUND   TRUE)

#------------------------------------------------------------------------------
#- ATRALA0310 -----------------------------------------------------------------
#------------------------------------------------------------------------------
elseif(${MACHINE_NAME} STREQUAL "ATRALA0310")

set(MATLAB_INC_PATH    "C:\\Programme\\MATLAB\\R2007b\\extern\\include")
set(INTEL_MKL_INC_PATH "C:\\Programme\\Intel\\Compiler\\11.0\\072\\cpp\\mkl\\include")
set(INTEL_IPP_INC_PATH "C:\\Programme\\Intel\\Compiler\\11.0\\072\\cpp\\ipp\\ia32\\include")

set(MATLAB_LIB_PATH    "C:\\Programme\\MATLAB\\R2007b\\extern\\lib\\win32\\microsoft")
set(INTEL_LIB_PATH     "C:\\Programme\\Intel\\Compiler\\11.0\\072\\cpp\\lib\\ia32")
set(INTEL_MKL_LIB_PATH "C:\\Programme\\Intel\\Compiler\\11.0\\072\\cpp\\mkl\\ia32\\lib")
set(INTEL_IPP_LIB_PATH "C:\\Programme\\Intel\\Compiler\\11.0\\072\\cpp\\ipp\\ia32\\stublib")

set(MATLAB_LIBS        libmx libmex libmat)
set(INTEL_LIBS         libiomp5md)
set(INTEL_MKL_LIBS     mkl_c_dll)
set(INTEL_IPP_LIBS     ippcore ippm ipps ippsr ippvm ippi)

set(MATLAB_FOUND       TRUE)
set(INTEL_LIBS_FOUND   TRUE)

#------------------------------------------------------------------------------
#- dast-nb-ubuntu -------------------------------------------------------------
#------------------------------------------------------------------------------
elseif(${MACHINE_NAME} STREQUAL "dast-nb-ubuntu")

#set(MATLAB_INC_PATH    "/home/dast/matu2k8b/extern/include")
set(INTEL_MKL_INC_PATH "/opt/intel/Compiler/11.1/056/mkl/include")
set(INTEL_IPP_INC_PATH "/opt/intel/Compiler/11.1/056/ipp/ia32/include")

#set(MATLAB_LIB_PATH    "/home/dast/matu2k8b/bin/glnx86")
set(INTEL_LIB_PATH     "/opt/intel/Compiler/11.1/056/lib/ia32")
set(INTEL_MKL_LIB_PATH "/opt/intel/Compiler/11.1/056/mkl/lib/32")
set(INTEL_IPP_LIB_PATH "/opt/intel/Compiler/11.1/056/ipp/ia32/sharedlib")

#set(MATLAB_LIBS        mx mex mat)
set(INTEL_LIBS         iomp5)
set(INTEL_MKL_LIBS     mkl_intel mkl_intel_thread mkl_core)
set(INTEL_IPP_LIBS     ippcore ippm ipps ippsr ippvm ippi)

set(MATLAB_FOUND       FALSE)
set(INTEL_LIBS_FOUND   TRUE)

#------------------------------------------------------------------------------
#- bob-machine ----------------------------------------------------------------
#------------------------------------------------------------------------------
elseif(${MACHINE_NAME} STREQUAL "bob-machine")

set(MATLAB_INC_PATH    "/home/klois/matlab/extern/include")
set(INTEL_MKL_INC_PATH "/opt/intel/Compiler/11.1/056/mkl/include")
set(INTEL_IPP_INC_PATH "/opt/intel/Compiler/11.1/056/ipp/ia32/include")

set(MATLAB_LIB_PATH    "/home/klois/matlab/bin/glnx86")
set(INTEL_LIB_PATH     "/opt/intel/Compiler/11.1/056/lib/ia32")
set(INTEL_MKL_LIB_PATH "/opt/intel/Compiler/11.1/056/mkl/lib/32")
set(INTEL_IPP_LIB_PATH "/opt/intel/Compiler/11.1/056/ipp/ia32/sharedlib")

set(MATLAB_LIBS        mx mex mat)
set(INTEL_LIBS         iomp5)
set(INTEL_MKL_LIBS     mkl_intel mkl_intel_thread mkl_core)
set(INTEL_IPP_LIBS     ippcore ippm ipps ippsr ippvm ippi)

set(MATLAB_FOUND       TRUE)
set(INTEL_LIBS_FOUND   TRUE)

#------------------------------------------------------------------------------
#- nexoc ----------------------------------------------------------------------
#------------------------------------------------------------------------------
elseif(${MACHINE_NAME} STREQUAL "nexoc")

set(MATLAB_INC_PATH    "C:\\uni\\progs\\matlab\\extern\\include")
set(INTEL_MKL_INC_PATH "C:\\Programme\\Intel\\Compiler\\11.0\\066\\cpp\\mkl\\include")
set(INTEL_IPP_INC_PATH "C:\\Programme\\Intel\\Compiler\\11.0\\066\\cpp\\ipp\\ia32\\include")

set(MATLAB_LIB_PATH    "C:\\uni\\progs\\matlab\\extern\\lib\\win32\\microsoft")
set(INTEL_LIB_PATH     "C:\\Programme\\Intel\\Compiler\\11.0\\066\\cpp\\lib\\ia32")
set(INTEL_MKL_LIB_PATH "C:\\Programme\\Intel\\Compiler\\11.0\\066\\cpp\\mkl\\ia32\\lib")
set(INTEL_IPP_LIB_PATH "C:\\Programme\\Intel\\Compiler\\11.0\\066\\cpp\\ipp\\ia32\\stublib")

set(MATLAB_LIBS        libmx libmex libmat)
set(INTEL_LIBS         libiomp5md)
set(INTEL_MKL_LIBS     mkl_c_dll)
set(INTEL_IPP_LIBS     ippcore ippm ipps ippsr ippvm ippi)

set(MATLAB_FOUND       TRUE)
set(INTEL_LIBS_FOUND   TRUE)

#------------------------------------------------------------------------------
#- schareck.dps.uibk.ac.at ----------------------------------------------------
#------------------------------------------------------------------------------
elseif(${MACHINE_NAME} STREQUAL "schareck.dps.uibk.ac.at")

set(INTEL_MKL_INC_PATH "/opt/intel/Compiler/11.0/074/mkl/include")
set(INTEL_IPP_INC_PATH "/opt/intel/Compiler/11.0/074/ipp/ia32/include")

set(INTEL_LIB_PATH     "/opt/intel/Compiler/11.0/074/lib/ia32")
set(INTEL_MKL_LIB_PATH "/opt/intel/Compiler/11.0/074/mkl/lib/32")
set(INTEL_IPP_LIB_PATH "/opt/intel/Compiler/11.0/074/ipp/ia32/sharedlib")

set(INTEL_LIBS         iomp5)
set(INTEL_MKL_LIBS     mkl_ia32)
set(INTEL_IPP_LIBS     ippcore ippm ipps ippsr ippvm ippi)

set(MATLAB_FOUND       FALSE)
set(INTEL_LIBS_FOUND   TRUE)

#------------------------------------------------------------------------------
#- Unknown system -------------------------------------------------------------
#------------------------------------------------------------------------------
else()
set(MATLAB_FOUND       FALSE)
set(INTEL_LIBS_FOUND   FALSE)
endif()
