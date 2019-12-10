#******************************************************************************
#
# \file   CMakeLists.txt
# \author Daniel Strigl, Klaus Kofler
# \date   Jun 01 2009
#
# $Id: CMakeLists.txt 2870 2009-12-07 20:12:53Z dast $
#
#******************************************************************************

include_directories(${PROJECT_SOURCE_DIR}/cnnplus/include)

#
# bench_jahrernet
#
add_executable(bench_jahrernet bench_jahrernet.cc)
target_link_libraries(bench_jahrernet ${CNNPLUS_LIBRARIES})

#
# bench_lenet5
#
add_executable(bench_lenet5 bench_lenet5.cc)
target_link_libraries(bench_lenet5 ${CNNPLUS_LIBRARIES})

#
# bench_simardnet
#
add_executable(bench_simardnet bench_simardnet.cc)
target_link_libraries(bench_simardnet ${CNNPLUS_LIBRARIES})

#
# bench_mlpnet
#
add_executable(bench_mlpnet bench_mlpnet.cc)
target_link_libraries(bench_mlpnet ${CNNPLUS_LIBRARIES})

#
# bench_garcianet
#
add_executable(bench_garcianet bench_garcianet.cc)
target_link_libraries(bench_garcianet ${CNNPLUS_LIBRARIES})
