
enable_testing()

get_filename_component( test_SRC_ROOT ${CMAKE_CURRENT_LIST_DIR} ABSOLUTE )


add_executable( ${PROJECT_NAME}-filter_test ${test_SRC_ROOT}/filter_test.cc )
target_link_libraries( ${PROJECT_NAME}-filter_test
                       ${PROJECT_NAME}-canny                    # your library
                       ${CannyLibs_LIBRARIES}                   # my libraries
                       Eigen3::Eigen GTest::GTest GTest::Main ) # third party dependencies
gtest_discover_tests( ${PROJECT_NAME}-filter_test )


find_package( OpenCV )
if( ${OpenCV_FOUND} )
    include_directories( ${OpenCV_INCLUDE_DIRS} )


    add_executable( ${PROJECT_NAME}-read_render_save ${test_SRC_ROOT}/read_render_save.cc )

    target_link_libraries( ${PROJECT_NAME}-read_render_save
                           ${PROJECT_NAME}-canny                                    # your library
                           ${CannyLibs_LIBRARIES}                                   # my libraries
                           Eigen3::Eigen ${OpenCV_LIBS} GTest::GTest GTest::Main )  # third party dependencies

endif()

