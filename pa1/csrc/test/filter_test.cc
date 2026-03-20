// SYSTEM INCLUDES
#include <iostream>
#include <gtest/gtest.h>
#include <string>

#include <585/common/types.h>
#include <585/io/io.h>

// UNCOMMENT IF YOU WANT TO DO SOME OPENCV THINGS!
//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>


// C++ PROJECT INCLUDES
#include "canny/filtering.h"


TEST(test_pixel_difference_filter, test_horizontal_creation)
{
    ivc::student::Filter2d x_filter = ivc::student::make_pixel_difference_filter(ivc::student::dir_t::HORIZONTAL);

    EXPECT_EQ(1, x_filter.filter.rows());
    EXPECT_EQ(2, x_filter.filter.cols());

    EXPECT_EQ(-1, x_filter.filter(0));
    EXPECT_EQ(+1, x_filter.filter(1));
}


TEST(test_pixel_difference_filter, test_vertical_creation)
{
    ivc::student::Filter2d x_filter = ivc::student::make_pixel_difference_filter(ivc::student::dir_t::VERTICAL);

    EXPECT_EQ(2, x_filter.filter.rows());
    EXPECT_EQ(1, x_filter.filter.cols());

    EXPECT_EQ(-1, x_filter.filter(0));
    EXPECT_EQ(+1, x_filter.filter(1));
}

