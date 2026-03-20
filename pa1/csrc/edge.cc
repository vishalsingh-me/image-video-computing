// SYSTEM INCLUDES
#include <585/common/types.h>


// C++ PROJECT INCLUDES
#include "canny/filtering.h"
#include "canny/edge.h"


namespace ivc
{
namespace student
{

    ivc::GrayscaleFloatImg get_gradient_magnitudes(const ivc::GrayscaleByteImg& img,
                                                   const ivc::student::Filter2d& filter_x,
                                                   const ivc::student::Filter2d& filter_y)
    {
        ivc::GrayscaleFloatImg grad_x = ivc::student::conv(img, filter_x);
        ivc::GrayscaleFloatImg grad_y = ivc::student::conv(img, filter_y);

        const size_t width = ivc::get_width(grad_x);
        const size_t height = ivc::get_height(grad_x);

        ivc::GrayscaleFloatImg magnitudes(height, width);
        for(size_t x = 0; x < width; ++x)
        {
            for(size_t y = 0; y < height; ++y)
            {
                const float_t gx = ivc::get_pixel(grad_x, x, y);
                const float_t gy = ivc::get_pixel(grad_y, x, y);
                ivc::set_pixel(magnitudes, x, y, std::sqrt(gx * gx + gy * gy));
            }
        }

        return magnitudes;
    }

    ivc::GrayscaleFloatImg get_gradient_angles(const ivc::GrayscaleByteImg& img,
                                               const ivc::student::Filter2d& filter_x,
                                               const ivc::student::Filter2d& filter_y)
    {
        ivc::GrayscaleFloatImg grad_x = ivc::student::conv(img, filter_x);
        ivc::GrayscaleFloatImg grad_y = ivc::student::conv(img, filter_y);

        const size_t width = ivc::get_width(grad_x);
        const size_t height = ivc::get_height(grad_x);

        ivc::GrayscaleFloatImg angles(height, width);
        const float_t rad2deg = static_cast<float_t>(180.0 / std::acos(-1.0));
        for(size_t x = 0; x < width; ++x)
        {
            for(size_t y = 0; y < height; ++y)
            {
                const float_t gx = ivc::get_pixel(grad_x, x, y);
                const float_t gy = ivc::get_pixel(grad_y, x, y);
                float_t ang = std::atan2(gy, gx) * rad2deg;
                if(ang < 0) ang += 360.0f;
                ivc::set_pixel(angles, x, y, ang);
            }
        }

        return angles;
    }

    ivc::GrayscaleFloatImg non_maximum_suppression(const ivc::GrayscaleFloatImg& grad_magnitudes,
                                                   const ivc::GrayscaleFloatImg& grad_angles)
    {
        const size_t width = ivc::get_width(grad_magnitudes);
        const size_t height = ivc::get_height(grad_magnitudes);

        ivc::GrayscaleFloatImg thin_edge_img(height, width);

        for(size_t x = 0; x < width; ++x)
        {
            for(size_t y = 0; y < height; ++y)
            {
                const float_t mag = ivc::get_pixel(grad_magnitudes, x, y);
                float_t angle = ivc::get_pixel(grad_angles, x, y);

                // normalize angle to [0,180)
                while(angle < 0) angle += 180.0f;
                while(angle >= 180.0f) angle -= 180.0f;

                int dx1 = 0, dy1 = 0, dx2 = 0, dy2 = 0;
                if(angle < 22.5f || angle >= 157.5f)
                {
                    dx1 = -1; dy1 = 0; dx2 = 1; dy2 = 0;    // 0 deg
                } else if(angle < 67.5f)
                {
                    dx1 = -1; dy1 = 1; dx2 = 1; dy2 = -1;  // 45 deg
                } else if(angle < 112.5f)
                {
                    dx1 = 0; dy1 = -1; dx2 = 0; dy2 = 1;   // 90 deg
                } else
                {
                    dx1 = -1; dy1 = -1; dx2 = 1; dy2 = 1;  // 135 deg
                }

                float_t n1 = 0.0f, n2 = 0.0f;
                long nx1 = static_cast<long>(x) + dx1;
                long ny1 = static_cast<long>(y) + dy1;
                long nx2 = static_cast<long>(x) + dx2;
                long ny2 = static_cast<long>(y) + dy2;
                if(nx1 >= 0 && ny1 >= 0 && nx1 < static_cast<long>(width) && ny1 < static_cast<long>(height))
                    n1 = ivc::get_pixel(grad_magnitudes, static_cast<size_t>(nx1), static_cast<size_t>(ny1));
                if(nx2 >= 0 && ny2 >= 0 && nx2 < static_cast<long>(width) && ny2 < static_cast<long>(height))
                    n2 = ivc::get_pixel(grad_magnitudes, static_cast<size_t>(nx2), static_cast<size_t>(ny2));

                if(mag >= n1 && mag >= n2)
                {
                    ivc::set_pixel(thin_edge_img, x, y, mag);
                } else
                {
                    ivc::set_pixel(thin_edge_img, x, y, 0.0f);
                }
            }
        }

        return thin_edge_img;
    }

    ivc::GrayscaleFloatImg hysteresis_threshold(const ivc::GrayscaleFloatImg& thin_edge_img,
                                                const float min_val,
                                                const float max_val)
    {
        const size_t width = ivc::get_width(thin_edge_img);
        const size_t height = ivc::get_height(thin_edge_img);

        ivc::GrayscaleFloatImg edges(height, width);
        edges.setZero();
        std::vector<bool> visited(width * height, false);
        std::vector<std::pair<size_t, size_t>> stack;

        auto idx = [width](size_t x, size_t y){ return y * width + x; };

        for(size_t x = 0; x < width; ++x)
        {
            for(size_t y = 0; y < height; ++y)
            {
                const float_t v = ivc::get_pixel(thin_edge_img, x, y);
                if(v > max_val)
                {
                    ivc::set_pixel(edges, x, y, v);
                    visited[idx(x, y)] = true;
                    stack.push_back({x, y});
                }
            }
        }

        // Use 8-neighborhood connectivity so weak pixels diagonally connected
        // to a strong edge are also preserved.
        const int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
        const int dy[8] = {-1,-1,-1,  0, 0,  1, 1, 1};

        while(!stack.empty())
        {
            auto [cx, cy] = stack.back();
            stack.pop_back();
            for(int k = 0; k < 8; ++k)
            {
                long nx = static_cast<long>(cx) + dx[k];
                long ny = static_cast<long>(cy) + dy[k];
                if(nx < 0 || ny < 0 || nx >= static_cast<long>(width) || ny >= static_cast<long>(height))
                    continue;
                size_t sx = static_cast<size_t>(nx);
                size_t sy = static_cast<size_t>(ny);
                if(visited[idx(sx, sy)])
                    continue;
                const float_t v = ivc::get_pixel(thin_edge_img, sx, sy);
                if(v >= min_val)
                {
                    visited[idx(sx, sy)] = true;
                    ivc::set_pixel(edges, sx, sy, v);
                    stack.push_back({sx, sy});
                }
            }
        }

        return edges;
    }

    ivc::GrayscaleFloatImg canny_edge(const ivc::GrayscaleByteImg& img,
                                      const float min_val,
                                      const float max_val)
    {
        const auto gaussian = ivc::student::make_gaussian_filter(5, 1.0f);
        ivc::GrayscaleFloatImg smoothed_f = ivc::student::conv(img, gaussian);
        ivc::GrayscaleByteImg smoothed = ivc::student::convert_scale_abs(smoothed_f, 1.0f, 0.0f);

        const auto sobel_x = ivc::student::make_sobel_filter_3x3();
        ivc::student::Filter2d sobel_y{sobel_x.filter.transpose(),
                                       sobel_x.center_height_idx,
                                       sobel_x.center_width_idx};

        ivc::GrayscaleFloatImg grad_x = ivc::student::conv(smoothed, sobel_x);
        ivc::GrayscaleFloatImg grad_y = ivc::student::conv(smoothed, sobel_y);

        const size_t width = ivc::get_width(grad_x);
        const size_t height = ivc::get_height(grad_x);

        ivc::GrayscaleFloatImg magnitudes(height, width);
        ivc::GrayscaleFloatImg angles(height, width);
        const float_t rad2deg = static_cast<float_t>(180.0 / std::acos(-1.0));
        for(size_t x = 0; x < width; ++x)
        {
            for(size_t y = 0; y < height; ++y)
            {
                const float_t gx = ivc::get_pixel(grad_x, x, y);
                const float_t gy = ivc::get_pixel(grad_y, x, y);
                ivc::set_pixel(magnitudes, x, y, std::sqrt(gx * gx + gy * gy));
                float_t ang = std::atan2(gy, gx) * rad2deg;
                if(ang < 0) ang += 360.0f;
                ivc::set_pixel(angles, x, y, ang);
            }
        }

        ivc::GrayscaleFloatImg thin = ivc::student::non_maximum_suppression(magnitudes, angles);
        return ivc::student::hysteresis_threshold(thin, min_val, max_val);
    }

    ivc::ColorFloatImg get_gradient_magnitudes(const ivc::ColorByteImg& img,
                                               const ivc::student::Filter2d& filter_x,
                                               const ivc::student::Filter2d& filter_y)
    {
        // Reuse the well-tested grayscale path per channel to avoid any
        // inadvertent channel/stride issues in color convolutions.
        const size_t width = ivc::get_width(img);
        const size_t height = ivc::get_height(img);
        const size_t filt_w = static_cast<size_t>(filter_x.filter.cols());
        const size_t filt_h = static_cast<size_t>(filter_x.filter.rows());
        ivc::ColorFloatImg magnitudes(height - filt_h + 1,
                                      width  - filt_w + 1);
        magnitudes.setZero();

        for(int c = 0; c < 3; ++c)
        {
            // Extract channel c into a grayscale image
            ivc::GrayscaleByteImg channel(height, width);
            for(size_t x = 0; x < width; ++x)
            {
                for(size_t y = 0; y < height; ++y)
                {
                    ivc::Vec3b pix = ivc::get_pixel(img, x, y);
                    ivc::set_pixel(channel, x, y, pix(c));
                }
            }

            ivc::GrayscaleFloatImg mag_c = ivc::student::get_gradient_magnitudes(channel, filter_x, filter_y);

            const size_t out_w = ivc::get_width(mag_c);
            const size_t out_h = ivc::get_height(mag_c);
            for(size_t x = 0; x < out_w; ++x)
            {
                for(size_t y = 0; y < out_h; ++y)
                {
                    ivc::Vec3f cur = ivc::get_pixel(magnitudes, x, y);
                    cur(c) = ivc::get_pixel(mag_c, x, y);
                    ivc::set_pixel(magnitudes, x, y, cur);
                }
            }
        }

        return magnitudes;
    }

    ivc::ColorFloatImg get_gradient_angles(const ivc::ColorByteImg& img,
                                           const ivc::student::Filter2d& filter_x,
                                           const ivc::student::Filter2d& filter_y)
    {
        const size_t width = ivc::get_width(img);
        const size_t height = ivc::get_height(img);
        const size_t filt_w = static_cast<size_t>(filter_x.filter.cols());
        const size_t filt_h = static_cast<size_t>(filter_x.filter.rows());
        ivc::ColorFloatImg angles(height - filt_h + 1,
                                  width  - filt_w + 1);
        angles.setZero();

        for(int c = 0; c < 3; ++c)
        {
            ivc::GrayscaleByteImg channel(height, width);
            for(size_t x = 0; x < width; ++x)
            {
                for(size_t y = 0; y < height; ++y)
                {
                    ivc::Vec3b pix = ivc::get_pixel(img, x, y);
                    ivc::set_pixel(channel, x, y, pix(c));
                }
            }

            ivc::GrayscaleFloatImg ang_c = ivc::student::get_gradient_angles(channel, filter_x, filter_y);

            const size_t out_w = ivc::get_width(ang_c);
            const size_t out_h = ivc::get_height(ang_c);
            for(size_t x = 0; x < out_w; ++x)
            {
                for(size_t y = 0; y < out_h; ++y)
                {
                    ivc::Vec3f cur = ivc::get_pixel(angles, x, y);
                    cur(c) = ivc::get_pixel(ang_c, x, y);
                    ivc::set_pixel(angles, x, y, cur);
                }
            }
        }

        return angles;
    }

    ivc::ColorFloatImg non_maximum_suppression(const ivc::ColorFloatImg& grad_magnitudes,
                                               const ivc::ColorFloatImg& grad_angles)
    {
        const size_t width = ivc::get_width(grad_magnitudes);
        const size_t height = ivc::get_height(grad_magnitudes);

        ivc::ColorFloatImg thin_edge_img(height, width);

        for(size_t x = 0; x < width; ++x)
        {
            for(size_t y = 0; y < height; ++y)
            {
                ivc::Vec3f mag = ivc::get_pixel(grad_magnitudes, x, y);
                ivc::Vec3f ang = ivc::get_pixel(grad_angles, x, y);
                ivc::Vec3f out_pix = ivc::Vec3f::Zero();

                for(int c = 0; c < 3; ++c)
                {
                    float_t angle = ang(c);
                    while(angle < 0) angle += 180.0f;
                    while(angle >= 180.0f) angle -= 180.0f;

                    int dx1 = 0, dy1 = 0, dx2 = 0, dy2 = 0;
                    if(angle < 22.5f || angle >= 157.5f)
                    {
                        dx1 = -1; dy1 = 0; dx2 = 1; dy2 = 0;
                    } else if(angle < 67.5f)
                    {
                        dx1 = -1; dy1 = 1; dx2 = 1; dy2 = -1;
                    } else if(angle < 112.5f)
                    {
                        dx1 = 0; dy1 = -1; dx2 = 0; dy2 = 1;
                    } else
                    {
                        dx1 = -1; dy1 = -1; dx2 = 1; dy2 = 1;
                    }

                    float_t n1 = 0.0f, n2 = 0.0f;
                    long nx1 = static_cast<long>(x) + dx1;
                    long ny1 = static_cast<long>(y) + dy1;
                    long nx2 = static_cast<long>(x) + dx2;
                    long ny2 = static_cast<long>(y) + dy2;
                    if(nx1 >= 0 && ny1 >= 0 && nx1 < static_cast<long>(width) && ny1 < static_cast<long>(height))
                        n1 = ivc::get_pixel(grad_magnitudes, static_cast<size_t>(nx1), static_cast<size_t>(ny1))(c);
                    if(nx2 >= 0 && ny2 >= 0 && nx2 < static_cast<long>(width) && ny2 < static_cast<long>(height))
                        n2 = ivc::get_pixel(grad_magnitudes, static_cast<size_t>(nx2), static_cast<size_t>(ny2))(c);

                    if(mag(c) >= n1 && mag(c) >= n2)
                        out_pix(c) = mag(c);
                    else
                        out_pix(c) = 0.0f;
                }

                ivc::set_pixel(thin_edge_img, x, y, out_pix);
            }
        }

        return thin_edge_img;
    }

    ivc::ColorFloatImg hysteresis_threshold(const ivc::ColorFloatImg& thin_edge_img,
                                            const float min_val,
                                            const float max_val)
    {
        const size_t width = ivc::get_width(thin_edge_img);
        const size_t height = ivc::get_height(thin_edge_img);

        ivc::ColorFloatImg edges(height, width);
        edges.setZero();
        const int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
        const int dy[8] = {-1,-1,-1,  0, 0,  1, 1, 1};

        for(int channel = 0; channel < 3; ++channel)
        {
            std::vector<bool> visited(width * height, false);
            std::vector<std::pair<size_t, size_t>> stack;
            auto idx = [width](size_t x, size_t y){ return y * width + x; };

            for(size_t x = 0; x < width; ++x)
            {
                for(size_t y = 0; y < height; ++y)
                {
                    const float_t v = ivc::get_pixel(thin_edge_img, x, y)(channel);
                    if(v > max_val)
                    {
                        ivc::Vec3f pix = ivc::get_pixel(edges, x, y);
                        pix(channel) = ivc::get_pixel(thin_edge_img, x, y)(channel);
                        ivc::set_pixel(edges, x, y, pix);
                        visited[idx(x, y)] = true;
                        stack.push_back({x, y});
                    }
                }
            }

            while(!stack.empty())
            {
                auto [cx, cy] = stack.back();
                stack.pop_back();
                for(int k = 0; k < 8; ++k)
                {
                    long nx = static_cast<long>(cx) + dx[k];
                    long ny = static_cast<long>(cy) + dy[k];
                    if(nx < 0 || ny < 0 || nx >= static_cast<long>(width) || ny >= static_cast<long>(height))
                        continue;
                    size_t sx = static_cast<size_t>(nx);
                    size_t sy = static_cast<size_t>(ny);
                    if(visited[idx(sx, sy)])
                        continue;
                    const float_t v = ivc::get_pixel(thin_edge_img, sx, sy)(channel);
                    if(v >= min_val)
                    {
                        visited[idx(sx, sy)] = true;
                        ivc::Vec3f pix = ivc::get_pixel(edges, sx, sy);
                        pix(channel) = v;
                        ivc::set_pixel(edges, sx, sy, pix);
                        stack.push_back({sx, sy});
                    }
                }
            }
        }

        return edges;
    }

    ivc::ColorFloatImg canny_edge(const ivc::ColorByteImg& img,
                                  const float min_val,
                                  const float max_val)
    {
        const auto gaussian = ivc::student::make_gaussian_filter(5, 1.0f);
        ivc::ColorFloatImg smoothed_f = ivc::student::conv(img, gaussian);
        ivc::ColorByteImg smoothed = ivc::student::convert_scale_abs(smoothed_f, 1.0f, 0.0f);

        const auto sobel_x = ivc::student::make_sobel_filter_3x3();
        ivc::student::Filter2d sobel_y{sobel_x.filter.transpose(),
                                       sobel_x.center_height_idx,
                                       sobel_x.center_width_idx};

        ivc::ColorFloatImg grad_x = ivc::student::conv(smoothed, sobel_x);
        ivc::ColorFloatImg grad_y = ivc::student::conv(smoothed, sobel_y);

        const size_t width = ivc::get_width(grad_x);
        const size_t height = ivc::get_height(grad_x);

        ivc::ColorFloatImg magnitudes(height, width);
        ivc::ColorFloatImg angles(height, width);
        const float_t rad2deg = static_cast<float_t>(180.0 / std::acos(-1.0));
        for(size_t x = 0; x < width; ++x)
        {
            for(size_t y = 0; y < height; ++y)
            {
                ivc::Vec3f gx = ivc::get_pixel(grad_x, x, y);
                ivc::Vec3f gy = ivc::get_pixel(grad_y, x, y);
                ivc::Vec3f m = ivc::Vec3f::Zero();
                ivc::Vec3f a = ivc::Vec3f::Zero();
                for(int c = 0; c < 3; ++c)
                {
                    m(c) = std::sqrt(gx(c) * gx(c) + gy(c) * gy(c));
                    float_t ang = std::atan2(gy(c), gx(c)) * rad2deg;
                    if(ang < 0) ang += 360.0f;
                    a(c) = ang;
                }
                ivc::set_pixel(magnitudes, x, y, m);
                ivc::set_pixel(angles, x, y, a);
            }
        }

        ivc::ColorFloatImg thin = ivc::student::non_maximum_suppression(magnitudes, angles);
        return ivc::student::hysteresis_threshold(thin, min_val, max_val);
    }

    ivc::GrayscaleFloatImg canny_edge_autotune(const ivc::GrayscaleByteImg& img)
    {
        const auto gaussian = ivc::student::make_gaussian_filter(5, 1.0f);
        ivc::GrayscaleFloatImg smoothed_f = ivc::student::conv(img, gaussian);
        ivc::GrayscaleByteImg smoothed = ivc::student::convert_scale_abs(smoothed_f, 1.0f, 0.0f);

        const auto sobel_x = ivc::student::make_sobel_filter_3x3();
        ivc::student::Filter2d sobel_y{sobel_x.filter.transpose(),
                                       sobel_x.center_height_idx,
                                       sobel_x.center_width_idx};

        ivc::GrayscaleFloatImg grad_x = ivc::student::conv(smoothed, sobel_x);
        ivc::GrayscaleFloatImg grad_y = ivc::student::conv(smoothed, sobel_y);

        const size_t width = ivc::get_width(grad_x);
        const size_t height = ivc::get_height(grad_x);

        ivc::GrayscaleFloatImg magnitudes(height, width);
        ivc::GrayscaleFloatImg angles(height, width);
        std::vector<float_t> mags;
        mags.reserve(width * height);
        const float_t rad2deg = static_cast<float_t>(180.0 / std::acos(-1.0));

        for(size_t x = 0; x < width; ++x)
        {
            for(size_t y = 0; y < height; ++y)
            {
                const float_t gx = ivc::get_pixel(grad_x, x, y);
                const float_t gy = ivc::get_pixel(grad_y, x, y);
                const float_t m = std::sqrt(gx * gx + gy * gy);
                mags.push_back(m);
                ivc::set_pixel(magnitudes, x, y, m);
                ivc::set_pixel(angles, x, y, std::atan2(gy, gx) * rad2deg);
            }
        }

        if(mags.empty())
        {
            ivc::GrayscaleFloatImg empty(height, width);
            empty.setZero();
            return empty;
        }

        std::sort(mags.begin(), mags.end());
        const float percentile = 0.9f;
        const size_t idx = static_cast<size_t>(percentile * static_cast<float>(mags.size() - 1));
        const float_t high = mags[idx];
        const float_t low = 0.4f * high;

        ivc::GrayscaleFloatImg thin = ivc::student::non_maximum_suppression(magnitudes, angles);
        return ivc::student::hysteresis_threshold(thin, low, high);
    }

    ivc::ColorFloatImg     canny_edge_autotune(const ivc::ColorByteImg& img)
    {
        const auto gaussian = ivc::student::make_gaussian_filter(5, 1.0f);
        ivc::ColorFloatImg smoothed_f = ivc::student::conv(img, gaussian);
        ivc::ColorByteImg smoothed = ivc::student::convert_scale_abs(smoothed_f, 1.0f, 0.0f);

        const auto sobel_x = ivc::student::make_sobel_filter_3x3();
        ivc::student::Filter2d sobel_y{sobel_x.filter.transpose(),
                                       sobel_x.center_height_idx,
                                       sobel_x.center_width_idx};

        ivc::ColorFloatImg grad_x = ivc::student::conv(smoothed, sobel_x);
        ivc::ColorFloatImg grad_y = ivc::student::conv(smoothed, sobel_y);

        const size_t width = ivc::get_width(grad_x);
        const size_t height = ivc::get_height(grad_x);

        ivc::ColorFloatImg magnitudes(height, width);
        ivc::ColorFloatImg angles(height, width);
        std::vector<float_t> mags;
        mags.reserve(width * height);
        const float_t rad2deg = static_cast<float_t>(180.0 / std::acos(-1.0));

        for(size_t x = 0; x < width; ++x)
        {
            for(size_t y = 0; y < height; ++y)
            {
                ivc::Vec3f gx = ivc::get_pixel(grad_x, x, y);
                ivc::Vec3f gy = ivc::get_pixel(grad_y, x, y);
                ivc::Vec3f m = ivc::Vec3f::Zero();
                ivc::Vec3f a = ivc::Vec3f::Zero();
                for(int c = 0; c < 3; ++c)
                {
                    m(c) = std::sqrt(gx(c) * gx(c) + gy(c) * gy(c));
                    a(c) = std::atan2(gy(c), gx(c)) * rad2deg;
                }
                mags.push_back(std::max({m(0), m(1), m(2)}));
                ivc::set_pixel(magnitudes, x, y, m);
                ivc::set_pixel(angles, x, y, a);
            }
        }

        if(mags.empty())
        {
            ivc::ColorFloatImg empty(height, width);
            empty.setZero();
            return empty;
        }

        std::sort(mags.begin(), mags.end());
        const float percentile = 0.9f;
        const size_t idx = static_cast<size_t>(percentile * static_cast<float>(mags.size() - 1));
        const float_t high = mags[idx];
        const float_t low = 0.4f * high;

        ivc::ColorFloatImg thin = ivc::student::non_maximum_suppression(magnitudes, angles);
        return ivc::student::hysteresis_threshold(thin, low, high);
    }

} // end of namespace student
} // end of namespace ivc
