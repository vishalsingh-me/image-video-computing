// SYSTEM INCLUDES
#include <585/common/types.h>


// C++ PROJECT INCLUDES
#include "canny/filtering.h"



namespace ivc
{
namespace student
{

    ivc::student::Filter2d make_pixel_difference_filter(const ivc::student::dir_t filter_dir)
    {
        // did this one for you
        int num_rows = -1;
        int num_cols = -1;
        if(filter_dir == ivc::student::dir_t::HORIZONTAL)
        {
            num_rows = 1;
            num_cols = 2;
        } else
        {
            num_rows = 2;
            num_cols = 1;
        }

        ivc::student::Filter2dType filter(num_rows, num_cols);
        filter(0) = -1;
        filter(1) = +1;
        return ivc::student::Filter2d{filter, 0, 0};
    }

    ivc::student::Filter2d make_central_difference_filter_3x3(const ivc::student::dir_t filter_dir)
    {
        ivc::student::Filter2dType filter = ivc::student::Filter2dType::Zero(3, 3);
        if(filter_dir == ivc::student::dir_t::HORIZONTAL)
        {
            filter(1, 0) = -1.0f;
            filter(1, 2) = 1.0f;
        } else
        {
            filter(0, 1) = -1.0f;
            filter(2, 1) = 1.0f;
        }
        return ivc::student::Filter2d{filter, 1, 1};
    }

    ivc::student::Filter2d make_prewitt_filter_3x3(const ivc::student::dir_t filter_dir)
    {
        ivc::student::Filter2dType filter(3, 3);
        if(filter_dir == ivc::student::dir_t::HORIZONTAL)
        {
            filter << -1, 0, 1,
                       -1, 0, 1,
                       -1, 0, 1;
        } else
        {
            filter <<  1,  1,  1,
                        0,  0,  0,
                       -1, -1, -1;
        }
        return ivc::student::Filter2d{filter, 1, 1};
    }

    ivc::student::Filter2d make_sobel_filter_3x3(const ivc::student::dir_t filter_dir)
    {
        ivc::student::Filter2dType filter(3, 3);
        if(filter_dir == ivc::student::dir_t::HORIZONTAL)
        {
            filter << -1, 0, 1,
                      -2, 0, 2,
                      -1, 0, 1;
        } else
        {
            filter <<  1,  2,  1,
                        0,  0,  0,
                       -1, -2, -1;
        }
        return ivc::student::Filter2d{filter, 1, 1};
    }

    ivc::student::Filter2d make_sobel_filter_3x3()
    {
        return ivc::student::make_sobel_filter_3x3(ivc::student::dir_t::HORIZONTAL);
    }


    ivc::student::Filter2d make_blur_box_filter(const size_t dim)
    {
        if(dim == 0 || dim % 2 == 0)
        {
            throw std::runtime_error("Blur box filter dimension must be positive and odd");
        }

        const float_t val = 1.0f / static_cast<float_t>(dim * dim);
        ivc::student::Filter2dType filter = ivc::student::Filter2dType::Constant(dim, dim, val);
        return ivc::student::Filter2d{filter, dim/2, dim/2};
    }

    ivc::student::Filter2d make_gaussian_filter(const size_t dim,
                                       const float_t stddev)
    {
        if(dim == 0 || dim % 2 == 0)
        {
            throw std::runtime_error("Gaussian filter dimension must be positive and odd");
        }
        if(stddev <= 0)
        {
            throw std::runtime_error("Gaussian filter stddev must be positive");
        }

        const size_t center = dim / 2;
        const float_t two_sigma_sq = static_cast<float_t>(2.0) * stddev * stddev;

        Eigen::Matrix<float_t, Eigen::Dynamic, 1> g(dim);
        for(size_t i = 0; i < dim; ++i)
        {
            const float_t x = static_cast<float_t>(static_cast<long>(i) - static_cast<long>(center));
            g(static_cast<Eigen::Index>(i)) = std::exp(-(x * x) / two_sigma_sq);
        }

        const float_t g_sum = g.sum();
        if(g_sum != 0)
        {
            g /= g_sum;
        }

        ivc::student::Filter2dType filter = g * g.transpose();
        const float_t f_sum = filter.sum();
        if(f_sum != 0)
        {
            filter /= f_sum;
        }
        return ivc::student::Filter2d{filter, dim/2, dim/2};
    }

    ivc::student::Filter2d make_sharpening_filter(const size_t dim,
                                         const size_t sharpen_coeff)
    {
        if(dim == 0 || dim % 2 == 0)
        {
            throw std::runtime_error("Sharpening filter dimension must be positive and odd");
        }

        // Sharpen by subtracting a blurred version and adding a scaled identity:
        //   output = sharpen_coeff * I - blur
        // This matches the reference tests: off-center weights are -1/(dim*dim)
        // and center weight is sharpen_coeff - 1/(dim*dim).
        const float_t a = static_cast<float_t>(sharpen_coeff);
        const auto blur = ivc::student::make_blur_box_filter(dim).filter; // normalized to sum=1

        ivc::student::Filter2dType filter = (-1.0f) * blur;
        filter(dim/2, dim/2) += a;
        return ivc::student::Filter2d{filter, dim/2, dim/2};
    }

    ivc::ColorFloatImg conv(const ivc::ColorByteImg& img,
                            const ivc::student::Filter2d& filter)
    {
        const size_t img_w = ivc::get_width(img);
        const size_t img_h = ivc::get_height(img);
        const size_t filt_w = static_cast<size_t>(filter.filter.cols());
        const size_t filt_h = static_cast<size_t>(filter.filter.rows());

        if(filt_w > img_w || filt_h > img_h)
        {
            throw std::runtime_error("Filter larger than image in conv (color)");
        }

        const size_t out_w = img_w - filt_w + 1;
        const size_t out_h = img_h - filt_h + 1;

        ivc::ColorFloatImg out(out_h, out_w);
        for(size_t y = 0; y < out_h; ++y)
        {
            for(size_t x = 0; x < out_w; ++x)
            {
                ivc::Vec3f acc = ivc::Vec3f::Zero();
                for(size_t fy = 0; fy < filt_h; ++fy)
                {
                    for(size_t fx = 0; fx < filt_w; ++fx)
                    {
                        const float_t w = filter.filter(static_cast<Eigen::Index>(fy),
                                                        static_cast<Eigen::Index>(fx));
                        const ivc::Vec3b pix = ivc::get_pixel(img, x + fx, y + fy);
                        acc(0) += w * static_cast<float_t>(pix(0));
                        acc(1) += w * static_cast<float_t>(pix(1));
                        acc(2) += w * static_cast<float_t>(pix(2));
                    }
                }
                ivc::set_pixel(out, x, y, acc);
            }
        }
        return out;
    }

    ivc::GrayscaleFloatImg conv(const ivc::GrayscaleByteImg& img,
                                const ivc::student::Filter2d& filter)
    {
        const size_t img_w = ivc::get_width(img);
        const size_t img_h = ivc::get_height(img);
        const size_t filt_w = static_cast<size_t>(filter.filter.cols());
        const size_t filt_h = static_cast<size_t>(filter.filter.rows());

        if(filt_w > img_w || filt_h > img_h)
        {
            throw std::runtime_error("Filter larger than image in conv (grayscale)");
        }

        const size_t out_w = img_w - filt_w + 1;
        const size_t out_h = img_h - filt_h + 1;

        ivc::GrayscaleFloatImg out(out_h, out_w);
        for(size_t y = 0; y < out_h; ++y)
        {
            for(size_t x = 0; x < out_w; ++x)
            {
                float_t acc = 0.0f;
                for(size_t fy = 0; fy < filt_h; ++fy)
                {
                    for(size_t fx = 0; fx < filt_w; ++fx)
                    {
                        const float_t w = filter.filter(static_cast<Eigen::Index>(fy),
                                                        static_cast<Eigen::Index>(fx));
                        const float_t pix = static_cast<float_t>(ivc::get_pixel(img, x + fx, y + fy));
                        acc += w * pix;
                    }
                }
                ivc::set_pixel(out, x, y, acc);
            }
        }
        return out;
    }

    uint8_t clamp_8bit(const float_t val)
    {
        return static_cast<uint8_t>(std::max(std::min(val, 255.0f), 0.0f));
    }

    ivc::ColorByteImg convert_scale_abs(const ivc::ColorFloatImg& img,
                                        const float_t alpha,
                                        const float_t beta)
    {
        const size_t width = ivc::get_width(img);
        const size_t height = ivc::get_height(img);

        ivc::ColorByteImg out_img(img.rows(), img.cols());
        for(size_t width_idx = 0; width_idx < width; ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < height; ++height_idx)
            {
                ivc::Vec3f img_pixel = ivc::get_pixel(img, width_idx, height_idx);

                ivc::Vec3b out_pixel(0);
                out_pixel(0) = clamp_8bit(std::abs(alpha * img_pixel(0) + beta));
                out_pixel(1) = clamp_8bit(std::abs(alpha * img_pixel(1) + beta));
                out_pixel(2) = clamp_8bit(std::abs(alpha * img_pixel(2) + beta));
                ivc::set_pixel(out_img, width_idx, height_idx, out_pixel);
            }
        }

        return out_img;
    }

    ivc::ColorByteImg convert_scale_abs(const ivc::ColorFloatImg& img)
    {
        return ivc::student::convert_scale_abs(img, 1.0, 0.0);
    }

    ivc::GrayscaleByteImg convert_scale_abs(const ivc::GrayscaleFloatImg& img,
                                            const float_t alpha,
                                            const float_t beta)
    {
        const size_t width = ivc::get_width(img);
        const size_t height = ivc::get_height(img);

        ivc::GrayscaleByteImg out_img(img.rows(), img.cols());
        for(size_t width_idx = 0; width_idx < width; ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < height; ++height_idx)
            {
                float_t img_pixel = ivc::get_pixel(img, width_idx, height_idx);

                ivc::set_pixel(out_img, width_idx, height_idx, clamp_8bit(std::abs(alpha * img_pixel + beta)));
            }
        }

        return out_img;
    }

    ivc::GrayscaleByteImg convert_scale_abs(const ivc::GrayscaleFloatImg& img)
    {
        return ivc::student::convert_scale_abs(img, 1.0, 0.0);
    }

} // end of namespace student
} // end of namespace ivc
