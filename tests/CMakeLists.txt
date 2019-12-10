#******************************************************************************
#
# \file   CMakeLists.txt
# \author Daniel Strigl, Klaus Kofler
# \date   Jun 01 2009
#
# $Id: CMakeLists.txt 1723 2009-07-12 19:17:58Z dast $
#
#******************************************************************************

include_directories(${CPPUNIT_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/cnnplus/include)

set(TESTS_TO_RUN
  test_matvecli
  test_mlpnet
  test_simardnet
  test_jahrernet
  test_lenet5
)
if(CNNPLUS_USE_CUDA)
  list(APPEND TESTS_TO_RUN test_cumvli)
endif()

create_test_sourcelist(DUMMY_NAME
  test_driver.cc
  ${TESTS_TO_RUN}
)

add_executable(test_driver test_driver.cc ${TESTS_TO_RUN})
if(MSVC)
  target_link_libraries(test_driver ${CNNPLUS_LIBRARIES})
  # http://www.vtk.org/Bug/view.php?id=6973
  foreach(_lib ${CPPUNIT_LIBRARIES})
    # do a simple string replacement to obtain the debugged libname
    string(REPLACE ".lib" "d.lib" _libd ${_lib})
    target_link_libraries(test_driver debug ${_libd})
    target_link_libraries(test_driver optimized ${_lib})
  endforeach()
else()
  target_link_libraries(test_driver ${CNNPLUS_LIBRARIES} ${CPPUNIT_LIBRARIES})
endif()

foreach(test ${TESTS_TO_RUN})
  get_filename_component(TNAME ${test} NAME_WE)
  add_test(${test} ${EXECUTABLE_OUTPUT_PATH}/test_driver ${TNAME})
endforeach()
