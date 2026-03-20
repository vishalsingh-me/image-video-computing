#pragma once
#ifndef _585_HOUGH_H_
#define _585_HOUGH_H_

// SYSTEM INCLUDES
#include <Eigen/Dense>              // Eigen::Matrix
#include <functional>               // std::function
#include <map>                      // std::map
#include <tuple>                    // std::tuple
#include <585/common/types.h>


// C++ PROJECT INCLUDES
#include "585/common/types.h"


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
}


#endif // end of _585_HOUGH_H_

