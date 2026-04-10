#pragma once
#ifndef _585_HOUGH_H_
#define _585_HOUGH_H_

// SYSTEM INCLUDES
#include <Eigen/Dense>              // Eigen::Matrix
#include <functional>               // std::function
#include <map>                      // std::map
#include <set>                      // std::set
#include <tuple>                    // std::tuple


// C++ PROJECT INCLUDES
#include "585/common/types.h"
#include "585/imgproc/imgproc.h"


namespace ivc
{

    // a point for a specific parameter combination. This point is what you will use to lookup
    // the value for different parameter combinations
    using ParamCombination = Eigen::Matrix<float_t, Eigen::Dynamic, 1>;

    // if you ever want to pre-calculate the discretized values for a parameter you can do so with this datatype
    // (see CooAccumulator)
    using ParamRange = Eigen::Matrix<float_t, Eigen::Dynamic, 1>;

    // if you want to store multiple param combinations you can do so with this type.
    // one param combination per column (e.g. the first dimension of this should match the first dimension of
    // type ParamCombination)
    using ParamCombinations = Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic>;


    // want to use ivc::ParamCombinations as a key in a dictionary (e.g. std::map) but need to know
    // how to compare them against each other. Treat params as if they are in lexigraphical ordering:
    // thanks https://stackoverflow.com/questions/27048146/using-stdmap-with-eigen-3
    // don't want to add to the std namespace though
    class ParamComb_comp_t
    {
    public:
        // returns true iff a < b and false otherwise
        bool operator()(const ivc::ParamCombination& a,
                        const ivc::ParamCombination& b) const;
    };


    /**
     * A class that implements the n-d Accumulator matrix. Rather than store this as a dense n-d matrix we will
     * instead store this in sparse "coordinate" (COO) format. The idea of COO format is only to store the coordinate
     * and value of nonzero entries. This should hopefully save a lot of time and memory IFF the accumulator is
     * actually sparse. If the image you have produces really dense votes then COO format is more expensive memory-wise
     * than a dense n-d matrix.
     */
    class CooAccumulator
    {
    public:
        CooAccumulator(const ivc::ParamCombination& min_vals,
                       const ivc::ParamCombination& max_vals,
                       const ivc::ParamCombination& discretization_amounts);

        uint64_t                get_count(const ivc::ParamCombination& param_combination);
        void                    set_count(const ivc::ParamCombination& param_combination,
                                          uint64_t new_count);
        ivc::ParamCombinations  threshold(const uint64_t threshold);

        const std::vector<ivc::ParamRange>                                  _param_all_values;

    private: // don't mess with these directly
        const ivc::ParamCombination                                         _param_max_values;
        const ivc::ParamCombination                                         _param_min_values;
        const ivc::ParamCombination                                         _param_discretization_amounts;

        // counts are uint64_t meaning we can use this for images with <= 2^64 pixels
        // (or even larger images if we can guarantee that every pixel votes for the same param combination)
        std::map<ivc::ParamCombination, uint64_t, ivc::ParamComb_comp_t>    _coo_counts;
    };

    // if you have the coordinates of a foreground pixel (e.g. <code>true</code>) you need to use it
    // to suggest plausible parameter combinations for that pixel
    using PixelParamSuggestionFunc = std::function<ivc::ParamCombinations(const ivc::CooAccumulator& accumulator,
                                                                          const ivc::BinaryImg& img,
                                                                          const size_t width_idx,
                                                                          const size_t height_idx)>;

    // now you need to do so but you have the image gradient present so you can lookup (r, theta) values
    // indexed by the gradient angle (phi) in the generalized hough transform
    using GeneralPixelParamsSuggestionFunc = std::function<ivc::ParamCombinations(const ivc::CooAccumulator& accumulator,
                                                                                  const ivc::BinaryImg& img,
                                                                                  const ivc::GrayscaleFloatImg& img_grad,
                                                                                  const size_t width_idx,
                                                                                  const size_t height_idx)>;


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


    // R table for generalized hough transform
    // assume the shape is convex and that it is centered in the image
    // (so we can use the center of the image as the center of the object)
    class RTable
    {
    public:
        RTable(ivc::BinaryImg& obj_img);    // the image containing the object

        // for a specific gradient angle phi return the potential (r, theta) values
        std::set<std::tuple<float_t, float_t> > get_r_theta_values(const float_t phi);

        // TODO: whatever public fields/methods you want
        const ivc::GrayscaleFloatImg        _grad_angles_degrees;

    private:
        const ivc::BinaryImg                _obj_img;
        const std::tuple<size_t, size_t>    _obj_img_center_idxs;   // the "center" of the object (h, w)

        // TODO: whatever private fields/methods you want
        std::map<float_t, std::set<std::tuple<float_t, float_t> > > _phi_to_r_theta_degrees;
    };

    // generalized hough transform
    ivc::ParamCombinations generalized_hough_fixed_obj(const ivc::BinaryImg& img,
                                                       const ivc::BinaryImg& object_img,    // make an RTable from this
                                                       const ivc::Vec3f& x_center_values,   // (min, max, delta)
                                                       const ivc::Vec3f& y_center_values,   // (min, max, delta)
                                                       const uint64_t threshold);

    ivc::ParamCombinations generalized_hough(const ivc::BinaryImg& img,
                                             const ivc::BinaryImg& object_img,              // make an RTable from this
                                             const ivc::Vec3f& x_center_values,             // (min, max, delta)
                                             const ivc::Vec3f& y_center_values,             // (min, max, delta)
                                             const ivc::Vec3f& scale_values,                // (min, max, delta)
                                             const ivc::Vec3f& theta_values,                // (min, max, delta)
                                             const uint64_t threshold);
}


#endif // end of _585_HOUGH_H_

