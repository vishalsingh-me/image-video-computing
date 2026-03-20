get_filename_component( CMAKE_ROOT ${CMAKE_CURRENT_LIST_DIR} ABSOLUTE )
get_filename_component( PKG_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../ ABSOLUTE )

find_path( 585_INCLUDE_DIR
    NAMES 585/common/types.h
          585/io/io.h
          585/filtering/filtering.h
          585/imgproc/imgproc.h
          585/hough/hough.h
          585/grad/grad.h
    PATHS ${PKG_ROOT}/include
    NO_DEFAULT_PATH
)

find_library( 585_COMMON_SHARED_LIB 585-ivc-common
    PATHS ${PKG_ROOT}/lib
)
find_library( 585_IO_SHARED_LIB 585-ivc-io
    PATHS ${PKG_ROOT}/lib
)
find_library( 585_FILTERING_SHARED_LIB 585-ivc-filtering
    PATHS ${PKG_ROOT}/lib
)
find_library( 585_IMGPROC_SHARED_LIB 585-ivc-imgproc
    PATHS ${PKG_ROOT}/lib
)
find_library( 585_HOUGH_SHARED_LIB 585-ivc-hough
    PATHS ${PKG_ROOT}/lib
)
find_library( 585_GRAD_SHARED_LIB 585-ivc-grad
    PATHS ${PKG_ROOT}/lib
)
find_library( 585_OPENCV_SHARED_LIB 585-ivc-opencv
    PATHS ${PKG_ROOT}/lib
)

include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( HoughLibs
    REQUIRED_VARS 585_INCLUDE_DIR 585_COMMON_SHARED_LIB 585_IO_SHARED_LIB
                  585_FILTERING_SHARED_LIB 585_IMGPROC_SHARED_LIB 585_HOUGH_SHARED_LIB 585_GRAD_SHARED_LIB )

if( HoughLibs_FOUND )
    set( HoughLibs_INCLUDE_DIR ${585_INCLUDE_DIR} )
    set( HoughLibs_LIBRARIES ${585_COMMON_SHARED_LIB} ${585_IO_SHARED_LIB}
                             ${585_FILTERING_SHARED_LIB} ${585_IMGPROC_SHARED_LIB} ${585_HOUGH_SHARED_LIB}
                             ${585_GRAD_SHARED_LIB} )


    find_package( OpenCV )
    if( OpenCV_FOUND AND 585_OPENCV_SHARED_LIB )
        # append it to HoughLibs_LIBRARIES
        list( APPEND HoughLibs_LIBRARIES ${585_OPENCV_SHARED_LIB} )
    endif()
endif()

