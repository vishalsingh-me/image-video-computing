#pragma once
#ifndef _585_TYPES_H_
#define _585_TYPES_H_

// SYSTEM INCLUDES
#include <Eigen/Dense>
#include <string>
#include <tuple>


// C++ PROJECT INCLUDES


namespace ivc
{

    using ColorByteImg = Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic>;
    using GrayscaleByteImg = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>;

    using ColorFloatImg = Eigen::Matrix<unsigned __int128, Eigen::Dynamic, Eigen::Dynamic>;
    using GrayscaleFloatImg = Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic>;

    using BinaryImg = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>;

    using Vec3b = Eigen::Matrix<uint8_t, 3, 1>;
    using Vec4b = Eigen::Matrix<uint8_t, 4, 1>;

    using Vec3f = Eigen::Matrix<float_t, 3, 1>;
    using Vec4f = Eigen::Matrix<float_t, 4, 1>;

    typedef enum channel_t
    {
        ONE = 0,
        TWO = 1,
        THREE = 2,
    } channel_t;

    typedef enum pad_t
    {
        ZEROS,
        WRAP,
        COPY_EDGE,
        REFLECT_EDGE,
    } pad_t;

    typedef enum symm_pad_dir_t
    {
        SYMMETRIC_ALL,
        SYMMETRIC_HORIZONTAL,
        SYMMETRIC_VERTICAL,
    } symm_pad_dir_t;

    size_t get_width(const ColorByteImg& img);
    size_t get_width(const ColorFloatImg& img);
    size_t get_width(const GrayscaleByteImg& img);
    size_t get_width(const GrayscaleFloatImg& img);
    size_t get_width(const BinaryImg& img);

    size_t get_height(const ColorByteImg& img);
    size_t get_height(const ColorFloatImg& img);
    size_t get_height(const GrayscaleByteImg& img);
    size_t get_height(const GrayscaleFloatImg& img);
    size_t get_height(const BinaryImg& img);

    std::tuple<size_t, size_t> get_size(const ColorByteImg& img);
    std::tuple<size_t, size_t> get_size(const ColorFloatImg& img);
    std::tuple<size_t, size_t> get_size(const GrayscaleByteImg& img);
    std::tuple<size_t, size_t> get_size(const GrayscaleFloatImg& img);
    std::tuple<size_t, size_t> get_size(const BinaryImg& img);

    ivc::Vec3b get_pixel(const ColorByteImg& img,
                         const int width_idx,
                         const int height_idx);
    uint8_t get_pixel(const GrayscaleByteImg& img,
                      const int width_idx,
                      const int height_idx);

    ivc::Vec3f get_pixel(const ColorFloatImg& img,
                         const int width_idx,
                         const int height_idx);
    float_t get_pixel(const GrayscaleFloatImg& img,
                      const int width_idx,
                      const int height_idx);
    bool get_pixel(const BinaryImg& img,
                   const int width_idx,
                   const int height_idx);


    void set_pixel(ColorByteImg& img,
                   const int width_idx,
                   const int height_idx,
                   const ivc::Vec3b& pixel);
    void set_pixel(GrayscaleByteImg& img,
                   const int width_idx,
                   const int height_idx,
                   const uint8_t pixel);

    void set_pixel(ColorFloatImg& img,
                   const int width_idx,
                   const int height_idx,
                   const ivc::Vec3f& pixel);
    void set_pixel(GrayscaleFloatImg& img,
                   const int width_idx,
                   const int height_idx,
                   const float_t pixel);
    void set_pixel(BinaryImg& img,
                   const int width_idx,
                   const int height_idx,
                   const bool pixel);

    // ivc::Vec3b scale_pixel(const ivc::Vec3b& a,
    //                        const float_t scale);

    ivc::ColorFloatImg      byte_to_float(const ColorByteImg& img);
    ivc::GrayscaleFloatImg  byte_to_float(const GrayscaleByteImg& img);

    ivc::ColorByteImg       float_to_byte(const ColorFloatImg& img);
    ivc::GrayscaleByteImg   float_to_byte(const GrayscaleFloatImg& img);

    ivc::ColorFloatImg      rgb_to_hsv(const ColorFloatImg& rgb_img);
    ivc::ColorFloatImg      hsv_to_rgb(const ColorFloatImg& hsv_img);

    ivc::GrayscaleByteImg   rgb_to_grayscale(const ColorByteImg& rgb_img);
    ivc::GrayscaleFloatImg  rgb_to_grayscale(const ColorFloatImg& rgb_img);
    ivc::ColorFloatImg      grayscale_to_rgb(const GrayscaleFloatImg& grayscale_img);

    ivc::GrayscaleFloatImg  grayscale_up(const ivc::GrayscaleFloatImg& img);


    ivc::ColorByteImg       symmetric_pad(const ColorByteImg& img,
                                          const size_t pad_amount,
                                          const pad_t pad_type,
                                          const symm_pad_dir_t pad_dir);
    ivc::GrayscaleByteImg   symmetric_pad(const GrayscaleByteImg& img,
                                          const size_t pad_amount,
                                          const pad_t pad_type,
                                          const symm_pad_dir_t pad_dir);
    ivc::BinaryImg          symmetric_pad(const BinaryImg& img,
                                          const size_t pad_amount,
                                          const pad_t pad_type,
                                          const symm_pad_dir_t pad_dir);


}


#endif // end of _585_TYPES_H_

