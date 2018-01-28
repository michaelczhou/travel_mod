PROJECT(PrjTest)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FIND_PACKAGE( OpenCV 2.4.13 REQUIRED HINTS /home/zc/tools/opencv-2.4.13)
FIND_PACKAGE( CUDA )

INCLUDE_DIRECTORIES(src)
INCLUDE_DIRECTORIES(./src/gslic)
INCLUDE_DIRECTORIES(./src/cvblob)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(/home/zc/local/include/eigen3)#/usr/include/eigen3)

FILE(GLOB srcfile  src/*.c*)
#FILE(GLOB srcfile  src/*.c)
FILE(GLOB headfile src/*.h)
FILE(GLOB gslicfile src/gslic/*.*)
FILE(GLOB cvblobfile src/cvblob/*.*)

LIST(REMOVE_ITEM  srcfile
${CMAKE_CURRENT_SOURCE_DIR}/src/main_travelablev3.cpp
)

#if(NOT CUDA_FOUND)
#	message( STATUS "The PROJECT will run without GPU")
#    ADD_EXECUTABLE(train ${srcfile} ${headfile} ${cvblobfile})
#elseif( CUDA_FOUND )
#    CUDA_ADD_EXECUTABLE(train ${srcfile} ${headfile} ${gslicfile} ${cvblobfile})
#    #MESSAGE(STATUS 'OpenCV_LIBS: ${CUDA_LIBS}')
#endif(NOT CUDA_FOUND)

#TARGET_LINK_LIBRARIES(train ${OpenCV_LIBS})


ADD_EXECUTABLE(call1 ${srcfile} ${headfile} ${cvblobfile})
TARGET_LINK_LIBRARIES(call1 ${OpenCV_LIBS})

# Set optimized building:
if( CMAKE_BUILD_TYPE MATCHES "Debug" )
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fpic -Wextra -pedantic -std=c++11 -pthread -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3 -std=c++11 -pthread -g")
else( CMAKE_BUILD_TYPE MATCHES "Release" )
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fpic -Wextra -pedantic -std=c++11 -pthread -O3 -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3 -std=c++11 -pthread -O3 -march=native")
endif( CMAKE_BUILD_TYPE MATCHES "Debug" )
