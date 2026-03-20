// SYSTEM INCLUDES
#include <585/common/types.h>
#include <585/imgproc/imgproc.h>
#include <585/hough/hough.h>


// C++ PROJECT INCLUDES
#include "hough/hough.h"


namespace ivc
{
namespace student
{

    void parameterized_hough(const ivc::BinaryImg& img,
                             const PixelParamSuggestionFunc& func,
                             ivc::CooAccumulator& accumulator)
    {

    }

    ivc::ParamCombinations  hough_line(const ivc::BinaryImg& img,
                                       const ivc::Vec3f& r_values,              // (min, max, delta)
                                       const ivc::Vec3f& theta_values,          // (min, max, delta)
                                       const uint64_t threshold)
    {
        ivc::ParamCombinations all_combs(1, 1);
        return all_combs;
    }

    ivc::ParamCombinations  hough_circle(const ivc::BinaryImg& img,
                                         const ivc::Vec3f& x_center_values,     // (min, max, delta)
                                         const ivc::Vec3f& y_center_values,     // (min, max, delta)
                                         const ivc::Vec3f& r_values,            // (min, max, delta)
                                         const uint64_t threshold)
    {
        ivc::ParamCombinations all_combs(1, 1);
        return all_combs;
    }

    ivc::ParamCombinations  hough_ellipse(const ivc::BinaryImg& img,
                                          const ivc::Vec3f& x_center_values,    // (min, max, delta)
                                          const ivc::Vec3f& y_center_values,    // (min, max, delta)
                                          const ivc::Vec3f& x_stretch_values,   // (min, max, delta)
                                          const ivc::Vec3f& y_stretch_values,   // (min, max, delta)
                                          const uint64_t threshold)
    {
        ivc::ParamCombinations all_combs(1, 1);
        return all_combs;
    }

    RTable::RTable(ivc::BinaryImg& obj_img) // the image containing the object
        : _obj_img(obj_img),
          _obj_img_center_idxs(std::make_tuple(ivc::get_height(obj_img)/2, ivc::get_width(obj_img)/2))
    {

    }

    std::set<std::tuple<float_t, float_t> > RTable::get_r_theta_values(const float_t phi)
    {
        std::set<std::tuple<float_t, float_t> > r_theta_values;
        return r_theta_values;
    }

    ivc::ParamCombinations generalized_hough_fixed_obj(const ivc::BinaryImg& img,
                                                       const ivc::BinaryImg& object_img,    // make an RTable from this
                                                       const ivc::Vec3f& x_center_values,   // (min, max, delta)
                                                       const ivc::Vec3f& y_center_values,   // (min, max, delta)
                                                       ivc::CooAccumulator& accumulator,
                                                       const uint64_t threshold)
    {
        ivc::ParamCombinations all_combs(1, 1);
        return all_combs;
    }

    ivc::ParamCombinations generalized_hough(const ivc::BinaryImg& img,
                                             const ivc::BinaryImg& object_img,              // make an RTable from this
                                             const ivc::Vec3f& x_center_values,             // (min, max, delta)
                                             const ivc::Vec3f& y_center_values,             // (min, max, delta)
                                             const ivc::Vec3f& scale_values,                // (min, max, delta)
                                             const ivc::Vec3f& theta_values,                // (min, max, delta)
                                             ivc::CooAccumulator& accumulator,
                                             const uint64_t threshold)
    {
        ivc::ParamCombinations all_combs(1, 1);
        return all_combs;
    }

} // end of namespace student
} // end of namespace ivc

