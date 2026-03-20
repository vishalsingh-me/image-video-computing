
get_filename_component( hough_SRC_ROOT ${CMAKE_CURRENT_LIST_DIR} ABSOLUTE )


set( hough_HEADERS
    ${INC_ROOT}/hough/morphology.h
    ${INC_ROOT}/hough/hough.h
)

set( hough_SRCS
    ${hough_SRC_ROOT}/morphology.cc
    ${hough_SRC_ROOT}/hough.cc
)

add_library( ${PROJECT_NAME}-hough SHARED ${hough_HEADERS} ${hough_SRCS} )
target_link_libraries( ${PROJECT_NAME}-hough ${HoughLibs_LIBRARIES} Eigen3::Eigen )

# tests
add_subdirectory( ${hough_SRC_ROOT}/test )

