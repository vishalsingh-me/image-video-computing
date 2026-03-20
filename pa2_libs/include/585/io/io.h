#pragma once
#ifndef _585_IO_H_
#define _585_IO_H_

// SYSTEM INCLUDES
// #include <Eigen/Dense>
#include <string>


// C++ PROJECT INCLUDES
#include "585/common/types.h"


namespace ivc
{

    ivc::ColorByteImg imread_rgba(const std::string& filepath);
    ivc::ColorByteImg imread_rgb(const std::string& filepath);
    ivc::GrayscaleByteImg imread_grayscale(const std::string& filepath);

    const int imwrite_rgba(const std::string& filepath,
                           const ivc::ColorByteImg& img);
    const int imwrite_rgb(const std::string& filepath,
                          const ivc::ColorByteImg& img);

    const int imwrite_grayscale(const std::string& filepath,
                                const ivc::GrayscaleByteImg& img);

}


#endif // end of _585_IO_H_

