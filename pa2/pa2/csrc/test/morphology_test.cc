// SYSTEM INCLUDES
#include <iostream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include <585/common/types.h>
#include <585/imgproc/imgproc.h>

// UNCOMMENT IF YOU WANT TO DO SOME OPENCV THINGS!
//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>


// C++ PROJECT INCLUDES
#include "hough/morphology.h"


TEST(test_morphology, test_strel_square_fixed_shapes)
{
    // test 3x3
    {
        ivc::se_polarity_t f = ivc::se_polarity_t::FOREGROUND;

        ivc::StructureElement expected(3, 3);
        expected << f, f, f,
                    f, f, f,
                    f, f, f;

        ivc::StructureElement se = ivc::student::strel_square(3);

        EXPECT_EQ(expected.rows(), se.rows());
        EXPECT_EQ(expected.cols(), se.cols());

        // std::cout << "square(" << dim << "):" << std::endl << ivc::to_byte(se) << std::endl;
        for(size_t width_idx = 0; width_idx < ivc::get_width(se); ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < ivc::get_height(se); ++height_idx)
            {
                EXPECT_EQ(ivc::get_pixel(expected, width_idx, height_idx),
                          ivc::get_pixel(se, width_idx, height_idx));
            }
        }
    }
}


TEST(test_morphology, test_strel_cross_fixed_shapes)
{
    // test 3x3 with thickness=1
    {
        ivc::se_polarity_t f = ivc::se_polarity_t::FOREGROUND;
        ivc::se_polarity_t b = ivc::se_polarity_t::BACKGROUND;

        ivc::StructureElement expected(3, 3);
        expected << b, f, b,
                    f, f, f,
                    b, f, b;

        ivc::StructureElement se = ivc::student::strel_cross(3);

        EXPECT_EQ(expected.rows(), se.rows());
        EXPECT_EQ(expected.cols(), se.cols());

        // std::cout << "square(" << dim << "):" << std::endl << ivc::to_byte(se) << std::endl;
        for(size_t width_idx = 0; width_idx < ivc::get_width(se); ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < ivc::get_height(se); ++height_idx)
            {
                EXPECT_EQ(ivc::get_pixel(expected, width_idx, height_idx),
                          ivc::get_pixel(se, width_idx, height_idx));
            }
        }
    }

    // test 5x5 with thickness=1
    {
        ivc::se_polarity_t f = ivc::se_polarity_t::FOREGROUND;
        ivc::se_polarity_t b = ivc::se_polarity_t::BACKGROUND;

        ivc::StructureElement expected(5, 5);
        expected << b, b, f, b, b,
                    b, b, f, b, b,
                    f, f, f, f, f,
                    b, b, f, b, b,
                    b, b, f, b, b;

        ivc::StructureElement se = ivc::student::strel_cross(5);

        EXPECT_EQ(expected.rows(), se.rows());
        EXPECT_EQ(expected.cols(), se.cols());

        // std::cout << "square(" << dim << "):" << std::endl << ivc::to_byte(se) << std::endl;
        for(size_t width_idx = 0; width_idx < ivc::get_width(se); ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < ivc::get_height(se); ++height_idx)
            {
                EXPECT_EQ(ivc::get_pixel(expected, width_idx, height_idx),
                          ivc::get_pixel(se, width_idx, height_idx));
            }
        }
    }

    // test 5x5 with thickness=2
    {
        ivc::se_polarity_t f = ivc::se_polarity_t::FOREGROUND;
        ivc::se_polarity_t b = ivc::se_polarity_t::BACKGROUND;

        ivc::StructureElement expected(5, 5);
        expected << b, f, f, f, b,
                    f, f, f, f, f,
                    f, f, f, f, f,
                    f, f, f, f, f,
                    b, f, f, f, b;

        ivc::StructureElement se = ivc::student::strel_cross(5, 2);

        EXPECT_EQ(expected.rows(), se.rows());
        EXPECT_EQ(expected.cols(), se.cols());

        // std::cout << "square(" << dim << "):" << std::endl << ivc::to_byte(se) << std::endl;
        for(size_t width_idx = 0; width_idx < ivc::get_width(se); ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < ivc::get_height(se); ++height_idx)
            {
                EXPECT_EQ(ivc::get_pixel(expected, width_idx, height_idx),
                          ivc::get_pixel(se, width_idx, height_idx));
            }
        }
    }
}


TEST(test_morphology, test_strel_diamond_fixed_shapes)
{
    // test 3x3
    {
        ivc::se_polarity_t f = ivc::se_polarity_t::FOREGROUND;
        ivc::se_polarity_t b = ivc::se_polarity_t::BACKGROUND;

        ivc::StructureElement expected(3, 3);
        expected << b, f, b,
                    f, f, f,
                    b, f, b;

        ivc::StructureElement se = ivc::student::strel_diamond(3);

        EXPECT_EQ(expected.rows(), se.rows());
        EXPECT_EQ(expected.cols(), se.cols());

        // std::cout << "square(" << dim << "):" << std::endl << ivc::to_byte(se) << std::endl;
        for(size_t width_idx = 0; width_idx < ivc::get_width(se); ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < ivc::get_height(se); ++height_idx)
            {
                EXPECT_EQ(ivc::get_pixel(expected, width_idx, height_idx),
                          ivc::get_pixel(se, width_idx, height_idx));
            }
        }
    }

    // test 5x5
    {
        ivc::se_polarity_t f = ivc::se_polarity_t::FOREGROUND;
        ivc::se_polarity_t b = ivc::se_polarity_t::BACKGROUND;

        ivc::StructureElement expected(5, 5);
        expected << b, b, f, b, b,
                    b, f, f, f, b,
                    f, f, f, f, f,
                    b, f, f, f, b,
                    b, b, f, b, b;

        ivc::StructureElement se = ivc::student::strel_diamond(5);

        EXPECT_EQ(expected.rows(), se.rows());
        EXPECT_EQ(expected.cols(), se.cols());

        // std::cout << "square(" << dim << "):" << std::endl << ivc::to_byte(se) << std::endl;
        for(size_t width_idx = 0; width_idx < ivc::get_width(se); ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < ivc::get_height(se); ++height_idx)
            {
                EXPECT_EQ(ivc::get_pixel(expected, width_idx, height_idx),
                          ivc::get_pixel(se, width_idx, height_idx));
            }
        }
    }

    // test 7x7
    {
        ivc::se_polarity_t f = ivc::se_polarity_t::FOREGROUND;
        ivc::se_polarity_t b = ivc::se_polarity_t::BACKGROUND;

        ivc::StructureElement expected(7, 7);
        expected << b, b, b, f, b, b, b,
                    b, b, f, f, f, b, b,
                    b, f, f, f, f, f, b,
                    f, f, f, f, f, f, f,
                    b, f, f, f, f, f, b,
                    b, b, f, f, f, b, b,
                    b, b, b, f, b, b, b;

        ivc::StructureElement se = ivc::student::strel_diamond(7);

        EXPECT_EQ(expected.rows(), se.rows());
        EXPECT_EQ(expected.cols(), se.cols());

        // std::cout << "square(" << dim << "):" << std::endl << ivc::to_byte(se) << std::endl;
        for(size_t width_idx = 0; width_idx < ivc::get_width(se); ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < ivc::get_height(se); ++height_idx)
            {
                EXPECT_EQ(ivc::get_pixel(expected, width_idx, height_idx),
                          ivc::get_pixel(se, width_idx, height_idx));
            }
        }
    }
}


TEST(test_morphology, test_strel_disk_fixed_shapes)
{
    // test 3x3
    {
        ivc::se_polarity_t f = ivc::se_polarity_t::FOREGROUND;

        ivc::StructureElement expected(3, 3);
        expected << f, f, f,
                    f, f, f,
                    f, f, f;

        ivc::StructureElement se = ivc::student::strel_disk(3);

        EXPECT_EQ(expected.rows(), se.rows());
        EXPECT_EQ(expected.cols(), se.cols());

        // std::cout << "square(" << dim << "):" << std::endl << ivc::to_byte(se) << std::endl;
        for(size_t width_idx = 0; width_idx < ivc::get_width(se); ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < ivc::get_height(se); ++height_idx)
            {
                EXPECT_EQ(ivc::get_pixel(expected, width_idx, height_idx),
                          ivc::get_pixel(se, width_idx, height_idx));
            }
        }
    }

    // test 5x5
    {
        ivc::se_polarity_t f = ivc::se_polarity_t::FOREGROUND;
        ivc::se_polarity_t b = ivc::se_polarity_t::BACKGROUND;

        ivc::StructureElement expected(5, 5);
        expected << b, f, f, f, b,
                    f, f, f, f, f,
                    f, f, f, f, f,
                    f, f, f, f, f,
                    b, f, f, f, b;

        ivc::StructureElement se = ivc::student::strel_disk(5);

        EXPECT_EQ(expected.rows(), se.rows());
        EXPECT_EQ(expected.cols(), se.cols());

        // std::cout << "square(" << dim << "):" << std::endl << ivc::to_byte(se) << std::endl;
        for(size_t width_idx = 0; width_idx < ivc::get_width(se); ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < ivc::get_height(se); ++height_idx)
            {
                EXPECT_EQ(ivc::get_pixel(expected, width_idx, height_idx),
                          ivc::get_pixel(se, width_idx, height_idx));
            }
        }
    }

    // test 7x7
    {
        ivc::se_polarity_t f = ivc::se_polarity_t::FOREGROUND;
        ivc::se_polarity_t b = ivc::se_polarity_t::BACKGROUND;

        ivc::StructureElement expected(7, 7);
        expected << b, b, f, f, f, b, b,
                    b, f, f, f, f, f, b,
                    f, f, f, f, f, f, f,
                    f, f, f, f, f, f, f,
                    f, f, f, f, f, f, f,
                    b, f, f, f, f, f, b,
                    b, b, f, f, f, b, b;

        ivc::StructureElement se = ivc::student::strel_disk(7);

        EXPECT_EQ(expected.rows(), se.rows());
        EXPECT_EQ(expected.cols(), se.cols());

        // std::cout << "square(" << dim << "):" << std::endl << ivc::to_byte(se) << std::endl;
        for(size_t width_idx = 0; width_idx < ivc::get_width(se); ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < ivc::get_height(se); ++height_idx)
            {
                EXPECT_EQ(ivc::get_pixel(expected, width_idx, height_idx),
                          ivc::get_pixel(se, width_idx, height_idx));
            }
        }
    }
}


TEST(test_morphology, test_strel_octagon_fixed_shapes)
{
    // test dim=3
    {
        ivc::se_polarity_t f = ivc::se_polarity_t::FOREGROUND;
        ivc::se_polarity_t b = ivc::se_polarity_t::BACKGROUND;

        ivc::StructureElement expected(7, 7);
        expected << b, b, f, f, f, b, b,
                    b, f, f, f, f, f, b,
                    f, f, f, f, f, f, f,
                    f, f, f, f, f, f, f,
                    f, f, f, f, f, f, f,
                    b, f, f, f, f, f, b,
                    b, b, f, f, f, b, b;

        ivc::StructureElement se = ivc::student::strel_octagon(3);

        EXPECT_EQ(expected.rows(), se.rows());
        EXPECT_EQ(expected.cols(), se.cols());

        // std::cout << "square(" << dim << "):" << std::endl << ivc::to_byte(se) << std::endl;
        for(size_t width_idx = 0; width_idx < ivc::get_width(se); ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < ivc::get_height(se); ++height_idx)
            {
                EXPECT_EQ(ivc::get_pixel(expected, width_idx, height_idx),
                          ivc::get_pixel(se, width_idx, height_idx));
            }
        }
    }

    // test dim=5
    {
        ivc::se_polarity_t f = ivc::se_polarity_t::FOREGROUND;
        ivc::se_polarity_t b = ivc::se_polarity_t::BACKGROUND;

        ivc::StructureElement expected(13, 13);
        expected << b, b, b, b, f, f, f, f, f, b, b, b, b,
                    b, b, b, f, f, f, f, f, f, f, b, b, b,
                    b, b, f, f, f, f, f, f, f, f, f, b, b,
                    b, f, f, f, f, f, f, f, f, f, f, f, b,
                    f, f, f, f, f, f, f, f, f, f, f, f, f,
                    f, f, f, f, f, f, f, f, f, f, f, f, f,
                    f, f, f, f, f, f, f, f, f, f, f, f, f,
                    f, f, f, f, f, f, f, f, f, f, f, f, f,
                    f, f, f, f, f, f, f, f, f, f, f, f, f,
                    b, f, f, f, f, f, f, f, f, f, f, f, b,
                    b, b, f, f, f, f, f, f, f, f, f, b, b,
                    b, b, b, f, f, f, f, f, f, f, b, b, b,
                    b, b, b, b, f, f, f, f, f, b, b, b, b;

        ivc::StructureElement se = ivc::student::strel_octagon(5);

        EXPECT_EQ(expected.rows(), se.rows());
        EXPECT_EQ(expected.cols(), se.cols());

        // std::cout << "square(" << dim << "):" << std::endl << ivc::to_byte(se) << std::endl;
        for(size_t width_idx = 0; width_idx < ivc::get_width(se); ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < ivc::get_height(se); ++height_idx)
            {
                EXPECT_EQ(ivc::get_pixel(expected, width_idx, height_idx),
                          ivc::get_pixel(se, width_idx, height_idx));
            }
        }
    }
}

