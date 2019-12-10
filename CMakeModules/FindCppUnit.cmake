#******************************************************************************
#
# \file   CMakeLists.txt
# \author Daniel Strigl, Klaus Kofler
# \date   Jun 01 2009
#
# $Id: FindCppUnit.cmake 1396 2009-06-04 06:33:11Z dast $
#
# \brief  cmake-macro to find cppunit.
#
#******************************************************************************

# Find the native cppunit includes and library.
# This module defines:
#   CPPUNIT_INCLUDE_DIR, where to find cppunit/Test.h, etc.
#   CPPUNIT_LIBRARIES, the libraries needed to use cppunit.
#   CPPUNIT_FOUND, if false, do not try to use cppunit.
# Also defined, but not for general use are:
#   CPPUNIT_LIBRARY, where to find the cppunit library.

find_path(CPPUNIT_INCLUDE_DIR cppunit/Test.h)

set(CPPUNIT_NAMES ${CPPUNIT_NAMES} cppunit)
find_library(CPPUNIT_LIBRARY NAMES ${CPPUNIT_NAMES})

# handle the QUIETLY and REQUIRED arguments and set CPPUNIT_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CPPUNIT DEFAULT_MSG
  CPPUNIT_LIBRARY
  CPPUNIT_INCLUDE_DIR
  )

if(CPPUNIT_FOUND)
  set(CPPUNIT_LIBRARIES ${CPPUNIT_LIBRARY})
endif()

mark_as_advanced(CPPUNIT_LIBRARY CPPUNIT_INCLUDE_DIR)
