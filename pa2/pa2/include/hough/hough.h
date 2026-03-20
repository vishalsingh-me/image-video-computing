#pragma once
#ifndef _HOUGH_HOUGH_H_
#define _HOUGH_HOUGH_H_

// SYSTEM INCLUDES
#include <set>                      // std::set
#include <tuple>                    // std::tuple
#include <585/common/types.h>
#include <585/imgproc/imgproc.h>
#include <585/hough/hough.h>


// C++ PROJECT INCLUDES
#include "hough/hough.h"


namespace ivc
{
namespace student
{

    // ----------------------------- REQUIRED BY ALL STUDENTS -------------------------------------

    // implements the normal hough transform where the shape is parameterized
    void                    parameterized_hough(const ivc::BinaryImg& img,
                                                const PixelParamSuggestionFunc& func,
                                                ivc::CooAccumulator& accumulator);

    // hough transform for lines (should use <code>parameterized_hough</code> as a subroutine)
    ivc::ParamCombinations  hough_line(const ivc::BinaryImg& img,
                                       const ivc::Vec3f& r_values,              // (min, max, delta)
                                       const ivc::Vec3f& theta_values,          // (min, max, delta)
                                       const uint64_t threshold);


    // hough transform for circles/ellipses (should use <code>parameterized_hough</code> as a subroutine)
    ivc::ParamCombinations  hough_circle(const ivc::BinaryImg& img,
                                         const ivc::Vec3f& x_center_values,     // (min, max, delta)
                                         const ivc::Vec3f& y_center_values,     // (min, max, delta)
                                         const ivc::Vec3f& r_values,            // (min, max, delta)
                                         const uint64_t threshold);
    ivc::ParamCombinations  hough_ellipse(const ivc::BinaryImg& img,
                                          const ivc::Vec3f& x_center_values,    // (min, max, delta)
                                          const ivc::Vec3f& y_center_values,    // (min, max, delta)
                                          const ivc::Vec3f& x_stretch_values,   // (min, max, delta)
                                          const ivc::Vec3f& y_stretch_values,   // (min, max, delta)
                                          const uint64_t threshold);

    // ---------------------- REQUIRED BY GRAD / BONUS FOR UNDERGRAD ------------------------------
    // R table for generalized hough transform
    // assume the shape is convex and that it is centered in the image
    // (so we can use the center of the image as the center of the object)
    class RTable
    {
    public:
        RTable(ivc::BinaryImg& obj_img);     // the image containing the object

        // for a specific gradient angle phi return the potential (r, theta) values
        std::set<std::tuple<float_t, float_t> > get_r_theta_values(const float_t phi);

        // TODO: whatever public fields/methods you want

    private:
        const ivc::BinaryImg                _obj_img;
        const std::tuple<size_t, size_t>    _obj_img_center_idxs;   // the "center" of the object

        // TODO: whatever private fields/methods you want
    };

    // generalized hough transform
    ivc::ParamCombinations generalized_hough_fixed_obj(const ivc::BinaryImg& img,
                                                       const ivc::BinaryImg& object_img,    // make an RTable from this
                                                       const ivc::Vec3f& x_center_values,   // (min, max, delta)
                                                       const ivc::Vec3f& y_center_values,   // (min, max, delta)
                                                       ivc::CooAccumulator& accumulator,
                                                       const uint64_t threshold);


    // ---------------------- BONUS FOR GRAD / NO CREDIT FOR UNDERGRAD -----------------------------
    // generalized hough transform with scale/rotation
    ivc::ParamCombinations generalized_hough(const ivc::BinaryImg& img,
                                             const ivc::BinaryImg& object_img,              // make an RTable from this
                                             const ivc::Vec3f& x_center_values,             // (min, max, delta)
                                             const ivc::Vec3f& y_center_values,             // (min, max, delta)
                                             const ivc::Vec3f& scale_values,                // (min, max, delta)
                                             const ivc::Vec3f& theta_values,                // (min, max, delta)
                                             ivc::CooAccumulator& accumulator,
                                             const uint64_t threshold);
    

} // end of namespace student
} // end of namespace ivc


#endif // end of _HOUGH_HOUGH_H_

