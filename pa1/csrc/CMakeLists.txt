
get_filename_component( canny_SRC_ROOT ${CMAKE_CURRENT_LIST_DIR} ABSOLUTE )


set( canny_HEADERS
    ${INC_ROOT}/canny/filtering.h
    ${INC_ROOT}/canny/edge.h
)

set( canny_SRCS
    ${canny_SRC_ROOT}/filtering.cc
    ${canny_SRC_ROOT}/edge.cc
)

add_library( ${PROJECT_NAME}-canny SHARED ${canny_HEADERS} ${canny_SRCS} )
target_link_libraries( ${PROJECT_NAME}-canny ${CannyLibs_LIBRARIES} Eigen3::Eigen )

# tests
add_subdirectory( ${canny_SRC_ROOT}/test )

