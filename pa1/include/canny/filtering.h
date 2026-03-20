#pragma once
#ifndef _CANNY_FILTERING_H_
#define _CANNY_FILTERING_H_

// SYSTEM INCLUDES
#include <585/common/types.h>


// C++ PROJECT INCLUDES


namespace ivc
{
namespace student
{

    // the eigen type to describe a filter
    using Filter2dType = Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic>;

    // the filter type
    typedef struct Filter2d
    {
        const Filter2dType filter;
        const size_t center_width_idx;      // the location of the "center" of the filter
        const size_t center_height_idx;     // the location of the "center" of the filter
    } Filter2d;

    // when creating derivative filters we need to know whether its d/dx or d/dy
    typedef enum dir_t
    {
        HORIZONTAL,     // d/dx
        VERTICAL        // d/dy
    } dir_t;

    // some functions for creating derivative filters
    ivc::student::Filter2d make_pixel_difference_filter(const ivc::student::dir_t filter_dir); // I did this one for you
    ivc::student::Filter2d make_central_difference_filter_3x3(const ivc::student::dir_t filter_dir);
    ivc::student::Filter2d make_prewitt_filter_3x3(const ivc::student::dir_t filter_dir);
    ivc::student::Filter2d make_sobel_filter_3x3();
    ivc::student::Filter2d make_sobel_filter_3x3(const ivc::student::dir_t filter_dir);

    // some functions for making smoothing filters
    ivc::student::Filter2d make_blur_box_filter(const size_t dim);
    ivc::student::Filter2d make_gaussian_filter(const size_t dim,
                                                const float_t stddev);

    // sharpening filter
    ivc::student::Filter2d make_sharpening_filter(const size_t dim,
                                                  const size_t sharpen_coeff);

    // convolution for grayscale & color imgs
    ivc::GrayscaleFloatImg conv(const ivc::GrayscaleByteImg& img,
                                const ivc::student::Filter2d& filter);
    ivc::ColorFloatImg     conv(const ivc::ColorByteImg& img,
                                const ivc::student::Filter2d& filter);


    // some functions for converting floating-poing images back to byte images (done for you)
    ivc::ColorByteImg      convert_scale_abs(const ivc::ColorFloatImg& img,
                                             const float_t alpha,
                                             const float_t beta);
    ivc::ColorByteImg      convert_scale_abs(const ivc::ColorFloatImg& img);


    // I did these for you...coverts float images back to rendering range.
    ivc::GrayscaleByteImg  convert_scale_abs(const ivc::GrayscaleFloatImg& img,
                                             const float_t alpha,
                                             const float_t beta);
    ivc::GrayscaleByteImg  convert_scale_abs(const ivc::GrayscaleFloatImg& img);

} // end of namespace student
} // end of namespace ivc


#endif // end of _CANNY_FILTERING_H_
