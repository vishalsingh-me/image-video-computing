// SYSTEM INCLUDES
#include <iostream>
#include <gtest/gtest.h>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <585/common/types.h>
#include <585/imgproc/imgproc.h>

// UNCOMMENT IF YOU WANT TO DO SOME OPENCV THINGS!
//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>


// C++ PROJECT INCLUDES
#include "vbow/vbow.h"


namespace
{
    ivc::GrayscaleByteImg make_img(const size_t width,
                                   const size_t height,
                                   const std::vector<uint8_t>& values)
    {
        ivc::GrayscaleByteImg img(height, width);
        img.setZero();

        size_t idx = 0;
        for(size_t height_idx = 0; height_idx < height; ++height_idx)
        {
            for(size_t width_idx = 0; width_idx < width; ++width_idx)
            {
                ivc::set_pixel(img, width_idx, height_idx, values[idx++]);
            }
        }

        return img;
    }

    std::vector<uint8_t> flatten(const ivc::GrayscaleByteImg& img)
    {
        std::vector<uint8_t> values;
        for(size_t height_idx = 0; height_idx < ivc::get_height(img); ++height_idx)
        {
            for(size_t width_idx = 0; width_idx < ivc::get_width(img); ++width_idx)
            {
                values.push_back(ivc::get_pixel(img,
                                                static_cast<int>(width_idx),
                                                static_cast<int>(height_idx)));
            }
        }

        return values;
    }
}


TEST(test_vbow, test_vocab)
{
    ivc::student::Vocab vocab;
    const ivc::GrayscaleByteImg a = make_img(1, 1, {2});
    const ivc::GrayscaleByteImg b = make_img(1, 1, {5});

    vocab.add(a);
    vocab.add(a);
    vocab.add(b);

    EXPECT_EQ(2u, vocab.size());
    EXPECT_TRUE(vocab.contains(a));
    EXPECT_TRUE(vocab.contains(b));
    EXPECT_EQ(0u, vocab.get_idx(a));
    EXPECT_EQ(1u, vocab.get_idx(b));
}


TEST(test_vbow, test_binary_vbow)
{
    const ivc::GrayscaleByteImg img = make_img(2, 2,
                                               {1, 2,
                                                1, 3});
    const ivc::ByteDataset X = {img};
    ivc::student::BinaryBagOfWords bow(X, 1);

    const ivc::FloatDataset D = bow.transform(X);
    ASSERT_EQ(1, D.rows());
    ASSERT_EQ(3, D.cols());
    EXPECT_FLOAT_EQ(1.0f, D(0, 0));
    EXPECT_FLOAT_EQ(1.0f, D(0, 1));
    EXPECT_FLOAT_EQ(1.0f, D(0, 2));
}


TEST(test_vbow, test_counting_vbow)
{
    const ivc::GrayscaleByteImg img = make_img(2, 2,
                                               {1, 2,
                                                1, 3});
    const ivc::ByteDataset X = {img};
    ivc::student::CountingBagOfWords bow(X, 1);

    const ivc::FloatDataset D = bow.transform(X);
    ASSERT_EQ(1, D.rows());
    ASSERT_EQ(3, D.cols());
    EXPECT_FLOAT_EQ(2.0f, D(0, 0));
    EXPECT_FLOAT_EQ(1.0f, D(0, 1));
    EXPECT_FLOAT_EQ(1.0f, D(0, 2));
}


TEST(test_vbow, test_tile_dataset)
{
    const ivc::GrayscaleByteImg img = make_img(4, 4,
                                               { 1,  2,  3,  4,
                                                 5,  6,  7,  8,
                                                 9, 10, 11, 12,
                                                13, 14, 15, 16});
    const ivc::ByteDataset tiles = ivc::student::tile_dataset(ivc::ByteDataset{img}, 1);
    ASSERT_EQ(4u, tiles.size());

    auto it = tiles.begin();
    EXPECT_EQ(std::vector<uint8_t>({1, 2, 5, 6}), flatten(*it++));
    EXPECT_EQ(std::vector<uint8_t>({3, 4, 7, 8}), flatten(*it++));
    EXPECT_EQ(std::vector<uint8_t>({9, 10, 13, 14}), flatten(*it++));
    EXPECT_EQ(std::vector<uint8_t>({11, 12, 15, 16}), flatten(*it++));
}


TEST(test_vbow, test_balance_oversample)
{
    const ivc::ByteDataset X = {
        make_img(1, 1, {1}),
        make_img(1, 1, {2}),
        make_img(1, 1, {3})
    };
    ivc::ProbVector y(3);
    y << 0.0f, 0.0f, 1.0f;

    const auto [X_balanced, y_balanced] = ivc::student::balance(X, y, ivc::student::OVERSAMPLE);
    ASSERT_EQ(4u, X_balanced.size());
    ASSERT_EQ(4, y_balanced.rows());

    std::map<int, int> counts;
    for(Eigen::Index row_idx = 0; row_idx < y_balanced.rows(); ++row_idx)
    {
        counts[static_cast<int>(y_balanced(row_idx))] += 1;
    }

    EXPECT_EQ(2, counts[0]);
    EXPECT_EQ(2, counts[1]);
}
