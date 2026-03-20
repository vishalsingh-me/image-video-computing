// SYSTEM INCLUDES
#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <585/common/types.h>
#include <585/io/io.h>
#include <585/opencv/opencv.h>


// C++ PROJECT INCLUDES


int main(int argc,
         char** argv)
{
    std::string file_path = __FILE__;

    // Find the last path separator ('\\' for Windows, '/' for Linux/macOS)
    size_t last_separator = file_path.find_last_of('/');

    // get dir
    std::string cd = file_path.substr(0, last_separator);

    std::string img_path = cd + "/imgs/breakfast2.png";
    std::cout << "looking for file at path: " << img_path << std::endl;

    ivc::ColorByteImg ivc_img = ivc::imread_rgb(img_path);

    cv::Mat cv_img = ivc::to_opencv(ivc_img);
    cv::imshow("Window", cv_img);
    cv::waitKey(0);

    ivc::imwrite_rgb(cd + "/breakfast2_out.png", ivc_img);

    return 0;
}

