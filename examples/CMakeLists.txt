#******************************************************************************
#
# \file   CMakeLists.txt
# \author Daniel Strigl, Klaus Kofler
# \date   Jun 01 2009
#
# $Id: CMakeLists.txt 3145 2010-01-05 13:22:37Z dast $
#
#******************************************************************************

include_directories(${PROJECT_SOURCE_DIR}/cnnplus/include)
include_directories(${PROJECT_SOURCE_DIR}/cnnplus/cimg)

#
# show_mnist
#
add_executable(show_mnist show_mnist.cc)
target_link_libraries(show_mnist ${CNNPLUS_LIBRARIES})

#
# show_cbclfaces
#
add_executable(show_cbclfaces show_cbclfaces.cc)
target_link_libraries(show_cbclfaces ${CNNPLUS_LIBRARIES})

#
# show_matsrc
#
if(MATLAB_FOUND)
  add_executable(show_matsrc show_matsrc.cc)
  target_link_libraries(show_matsrc ${CNNPLUS_LIBRARIES})
endif()

#
# train_lenet5
#
add_executable(train_lenet5 train_lenet5.cc)
target_link_libraries(train_lenet5 ${CNNPLUS_LIBRARIES})

#
# train_lenet5_bench
#
add_executable(train_lenet5_bench train_lenet5_bench.cc)
target_link_libraries(train_lenet5_bench ${CNNPLUS_LIBRARIES})

#
# train_mlpnet
#
add_executable(train_mlpnet train_mlpnet.cc)
target_link_libraries(train_mlpnet ${CNNPLUS_LIBRARIES})

#
# train_simardnet
#
add_executable(train_simardnet train_simardnet.cc)
target_link_libraries(train_simardnet ${CNNPLUS_LIBRARIES})

#
# train_jahrernet
#
add_executable(train_jahrernet train_jahrernet.cc)
target_link_libraries(train_jahrernet ${CNNPLUS_LIBRARIES})

#
# train_lolnet
#
add_executable(train_lolnet train_lolnet.cc)
target_link_libraries(train_lolnet ${CNNPLUS_LIBRARIES})

#
# train_garcianet
#
add_executable(train_garcianet train_garcianet.cc)
target_link_libraries(train_garcianet ${CNNPLUS_LIBRARIES})

#
# train_phungnet
#
if(MATLAB_FOUND)
  add_executable(train_phungnet train_phungnet.cc)
  target_link_libraries(train_phungnet ${CNNPLUS_LIBRARIES})
endif()

#
# train_facefindernet
#
if(MATLAB_FOUND)
  add_executable(train_facefindernet train_facefindernet.cc)
  target_link_libraries(train_facefindernet ${CNNPLUS_LIBRARIES})
endif()

#
# train_facefindernet2
#
if(MATLAB_FOUND)
  add_executable(train_facefindernet2 train_facefindernet2.cc)
  target_link_libraries(train_facefindernet2 ${CNNPLUS_LIBRARIES})
endif()

#
# test_precision
#
if(MATLAB_FOUND)
  add_executable(test_precision test_precision.cc)
  target_link_libraries(test_precision ${CNNPLUS_LIBRARIES})
endif()

#
# facefinder
#
if(MATLAB_FOUND AND INTEL_LIBS_FOUND)
  find_package(OpenCV)
  if (OPENCV_FOUND)
    include_directories(${OPENCV_INCLUDE_DIR})
    add_executable(facefinder facefinder.cc)
    target_link_libraries(facefinder ${CNNPLUS_LIBRARIES} ${INTEL_LIBS} ${INTEL_IPP_LIBS} ${OPENCV_LIBRARIES})
  endif()
endif()
