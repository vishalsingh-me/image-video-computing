#pragma once
#ifndef _585_IMGPROC_H_
#define _585_IMGPROC_H_

// SYSTEM INCLUDES
#include <Eigen/Dense>
#include <tuple>
#include <585/common/types.h>


// C++ PROJECT INCLUDES
#include "585/common/types.h"


namespace ivc
{

    // structuring element polarity type
    // can't print this out directly inside Eigen types
    typedef enum se_polarity_t
    {
        BACKGROUND = 0,
        FOREGROUND = 1,
        BLANK = 2
    } se_polarity_t;

    // the eigen type to describe structuring element
    using StructureElement = Eigen::Matrix<se_polarity_t, Eigen::Dynamic, Eigen::Dynamic>;

    // dimensionality getters
    size_t get_width(const ivc::StructureElement& se);
    size_t get_height(const ivc::StructureElement& se);
    std::tuple<size_t, size_t> get_size(const ivc::StructureElement& se);

    // pixel getters/setters
    se_polarity_t get_pixel(const ivc::StructureElement& se,
                            const int width_idx,
                            const int height_idx);
    void set_pixel(ivc::StructureElement& se,
                   const int width_idx,
                   const int height_idx,
                   const se_polarity_t pixel);

    // eigen types containing se_polarity_t values cannot be printed (and I didn't overload an operator)
    // to let them be printed anyways. Instead you can use these to convert structuring elements to
    // byte/float form and then print them out. Remember that BLANK will be printed as 0
    ivc::GrayscaleByteImg  to_byte(ivc::StructureElement& se);
    ivc::GrayscaleFloatImg to_float(ivc::StructureElement& se);

}


#endif // end of _585_IMGPROC_H_

