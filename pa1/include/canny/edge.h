#pragma once
#ifndef _CANNY_EDGE_H_
#define _CANNY_EDGE_H_

// SYSTEM INCLUDES
#include <585/common/types.h>


// C++ PROJECT INCLUDES
#include "canny/filtering.h"


namespace ivc
{
namespace student
{

    // ----------------------------- REQUIRED BY ALL STUDENTS -------------------------------------
    ivc::GrayscaleFloatImg get_gradient_magnitudes(const ivc::GrayscaleByteImg& img,
                                                   const ivc::student::Filter2d& filter_x,
                                                   const ivc::student::Filter2d& filter_y);

    ivc::GrayscaleFloatImg get_gradient_angles(const ivc::GrayscaleByteImg& img,
                                               const ivc::student::Filter2d& filter_x,
                                               const ivc::student::Filter2d& filter_y);

    ivc::GrayscaleFloatImg non_maximum_suppression(const ivc::GrayscaleFloatImg& grad_magnitudes,
                                                   const ivc::GrayscaleFloatImg& grad_angles);

    ivc::GrayscaleFloatImg hysteresis_threshold(const ivc::GrayscaleFloatImg& thin_edge_img,
                                                const float min_val,
                                                const float max_val);

    ivc::GrayscaleFloatImg canny_edge(const ivc::GrayscaleByteImg& img,
                                      const float min_val,
                                      const float max_val);


    // ---------------------- REQUIRED BY GRAD / BONUS FOR UNDERGRAD ------------------------------
    ivc::ColorFloatImg get_gradient_magnitudes(const ivc::ColorByteImg& img,
                                               const ivc::student::Filter2d& filter_x,
                                               const ivc::student::Filter2d& filter_y);

    ivc::ColorFloatImg get_gradient_angles(const ivc::ColorByteImg& img,
                                           const ivc::student::Filter2d& filter_x,
                                           const ivc::student::Filter2d& filter_y);

    ivc::ColorFloatImg non_maximum_suppression(const ivc::ColorFloatImg& grad_magnitudes,
                                               const ivc::ColorFloatImg& grad_angles);

    ivc::ColorFloatImg hysteresis_threshold(const ivc::ColorFloatImg& thin_edge_img,
                                            const float min_val,
                                            const float max_val);

    ivc::ColorFloatImg canny_edge(const ivc::ColorByteImg& img,
                                  const float min_val,
                                  const float max_val);


    // ---------------------- BONUS FOR GRAD / NO CREDIT FOR UNDERGRAD -----------------------------
    ivc::GrayscaleFloatImg canny_edge_autotune(const ivc::GrayscaleByteImg& img);
    ivc::ColorFloatImg     canny_edge_autotune(const ivc::ColorByteImg& img);

    

} // end of namespace student
} // end of namespace ivc


#endif // end of _CANNY_EDGE_H_

