#pragma once
#ifndef _585_OPENCV_H_
#define _585_OPENCV_H_

// SYSTEM INCLUDES
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>


// C++ PROJECT INCLUDES
#include "585/common/types.h"


namespace ivc
{

    cv::Mat to_opencv(const ivc::ColorByteImg& rgb_img);
    cv::Mat to_opencv(const ivc::GrayscaleByteImg& grayscale_img);

}


#endif // end of _585_OPENCV_H_

