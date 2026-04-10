#pragma once
#ifndef _585_GRAD_H_
#define _585_GRAD_H_

// SYSTEM INCLUDES
#include <585/common/types.h>
#include <585/filtering/filtering.h>


// C++ PROJECT INCLUDES


namespace ivc
{
    // these functions require you to pad the input image beforehand
    ivc::GrayscaleFloatImg get_gradient_magnitudes(const ivc::GrayscaleByteImg& img,
                                                   const ivc::Filter2d& filter_x,
                                                   const ivc::Filter2d& filter_y);
    ivc::ColorFloatImg get_gradient_magnitudes(const ivc::ColorByteImg& img,
                                               const ivc::Filter2d& filter_x,
                                               const ivc::Filter2d& filter_y);

    // these functions require you to pad the input image beforehand
    ivc::GrayscaleFloatImg get_gradient_angles(const ivc::GrayscaleByteImg& img,
                                               const ivc::Filter2d& filter_x,
                                               const ivc::Filter2d& filter_y);
    ivc::ColorFloatImg get_gradient_angles(const ivc::ColorByteImg& img,
                                           const ivc::Filter2d& filter_x,
                                           const ivc::Filter2d& filter_y);

    // these functions do *not* require you to pad the input image beforehand
    // they will pad on your behalf
    ivc::GrayscaleFloatImg get_sobel_3x3_gradient_magnitudes(const ivc::GrayscaleByteImg& img);
    ivc::ColorFloatImg get_sobel_3x3_gradient_magnitudes(const ivc::ColorByteImg& img);

    ivc::GrayscaleFloatImg get_sobel_3x3_gradient_angles(const ivc::GrayscaleByteImg& img);
    ivc::ColorFloatImg get_sobel_3x3_gradient_angles(const ivc::ColorByteImg& img);
} // end of namespace ivc


#endif // end of _585_GRAD_H_

