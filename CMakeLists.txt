cmake_minimum_required(VERSION 2.8)
project(ProjectName)
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_calib3d)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
#FIND_PACKAGE(Qt4 REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
#find_package(TBB)

#if(NOT TBB_FOUND)
#MESSAGE(STATUS "TBB not found!");
#else(NOT TBB_FOUND)
#include_directories(${TBB_INCLUDE_DIRS})
#link_directories(${TBB_LIBRARY_DIRS})
#endif(NOT TBB_FOUND)


set(CMAKE_C_FLAGS
-frounding-math 
#-Wall
)

execute_process(COMMAND cat /proc/cpuinfo OUTPUT_VARIABLE CPU_INFO)
string(REGEX MATCHALL "avx2" AVX_STRING ${CPU_INFO})
list(LENGTH AVX_STRING AVX_STRING_LENGTH)
# message(STATUS "AVX_STRING: " ${AVX_STRING})
message(STATUS "Number of avx2 occurrences in /proc/cpuinfo: " ${AVX_STRING_LENGTH})


if(${AVX_STRING_LENGTH} GREATER 0)
message(STATUS "Compiling with AVX2 support")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -mavx2 -funroll-loops -DOWNAVX")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -mavx2 -funroll-loops -DOWNAVX")
else()
message(STATUS "Compiling without AVX2 support")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -msse4.2 -funroll-loops")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -msse4.2 -funroll-loops")
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPRINT_DEBUG")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPRINT_DEBUG")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)



include_directories(${PROJECT_SOURCE_DIR}/src)
message(STATUS "PROJECT_SOURCE_DIR is " ${PROJECT_SOURCE_DIR})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories("/usr/include/eigen3")
add_subdirectory(src)

