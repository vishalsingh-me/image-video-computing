// SYSTEM INCLUDES
#include <cmath>
#include <new>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

#include <585/common/types.h>
#include <585/grad/grad.h>
#include <585/hough/hough.h>
#include <585/imgproc/imgproc.h>


// C++ PROJECT INCLUDES
#include "hough/hough.h"


namespace
{
    constexpr float_t kPi = 3.14159265358979323846f;
    constexpr float_t kTwoPi = 2.0f * kPi;
    constexpr float_t kDegreesPerCircle = 360.0f;
    constexpr float_t kSnapEps = 1e-4f;

    bool get_binary_pixel(const ivc::BinaryImg& img,
                          const size_t width_idx,
                          const size_t height_idx)
    {
        return ivc::get_pixel(static_cast<const ivc::BinaryImg&>(img),
                              static_cast<int>(width_idx),
                              static_cast<int>(height_idx));
    }

    float_t get_gray_float_pixel(const ivc::GrayscaleFloatImg& img,
                                 const size_t width_idx,
                                 const size_t height_idx)
    {
        return ivc::get_pixel(static_cast<const ivc::GrayscaleFloatImg&>(img),
                              static_cast<int>(width_idx),
                              static_cast<int>(height_idx));
    }

    ivc::ParamCombination make_param_combination(const std::initializer_list<float_t>& values)
    {
        ivc::ParamCombination params(values.size());
        size_t idx = 0;
        for(const float_t value : values)
        {
            params(idx++) = value;
        }

        return params;
    }

    ivc::ParamCombinations set_to_param_combinations(
        const std::set<ivc::ParamCombination, ivc::ParamComb_comp_t>& values,
        const size_t num_rows)
    {
        ivc::ParamCombinations out(num_rows, values.size());
        size_t col_idx = 0;
        for(const ivc::ParamCombination& value : values)
        {
            out.col(col_idx++) = value;
        }

        return out;
    }

    void add_vote(ivc::CooAccumulator& accumulator,
                  const ivc::ParamCombination& param_combination)
    {
        accumulator.set_count(param_combination,
                              accumulator.get_count(param_combination) + 1);
    }

    float_t degrees_to_radians(const float_t degrees)
    {
        return degrees * kPi / 180.0f;
    }

    float_t radians_to_degrees(const float_t radians)
    {
        return radians * 180.0f / kPi;
    }

    float_t normalize_angle_degrees(float_t angle)
    {
        while(angle < 0.0f)
        {
            angle += kDegreesPerCircle;
        }

        while(angle >= kDegreesPerCircle)
        {
            angle -= kDegreesPerCircle;
        }

        return angle;
    }

    float_t normalize_angle_radians(float_t angle)
    {
        while(angle < 0.0f)
        {
            angle += kTwoPi;
        }

        while(angle >= kTwoPi)
        {
            angle -= kTwoPi;
        }

        return angle;
    }

    float_t normalize_phi_degrees(const float_t phi)
    {
        // Internal gradient images use radians, but direct tests may query by degrees.
        if(std::abs(phi) > (kTwoPi + 1.0f))
        {
            return normalize_angle_degrees(phi);
        }

        return normalize_angle_degrees(radians_to_degrees(phi));
    }

    std::optional<float_t> snap_to_range(const ivc::ParamRange& range,
                                         const float_t value)
    {
        if(range.rows() == 0)
        {
            return std::nullopt;
        }

        if(range.rows() == 1)
        {
            if(std::abs(value - range(0)) <= kSnapEps)
            {
                return range(0);
            }

            return std::nullopt;
        }

        const float_t step = range(1) - range(0);
        const float_t continuous_idx = (value - range(0)) / step;
        const long snapped_idx = std::lround(continuous_idx);
        if(snapped_idx < 0 || snapped_idx >= range.rows())
        {
            return std::nullopt;
        }

        const float_t snapped_value = range(snapped_idx);
        if(std::abs(value - snapped_value) > kSnapEps)
        {
            return std::nullopt;
        }

        return snapped_value;
    }

    ivc::GrayscaleByteImg binary_to_grayscale_byte(const ivc::BinaryImg& img)
    {
        ivc::GrayscaleByteImg out(ivc::get_height(img), ivc::get_width(img));
        out.setZero();

        for(size_t width_idx = 0; width_idx < ivc::get_width(img); ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < ivc::get_height(img); ++height_idx)
            {
                ivc::set_pixel(out,
                               width_idx,
                               height_idx,
                               get_binary_pixel(img, width_idx, height_idx) ? static_cast<uint8_t>(255)
                                                                             : static_cast<uint8_t>(0));
            }
        }

        return out;
    }

    void rebuild_accumulator(ivc::CooAccumulator& accumulator,
                             const ivc::ParamCombination& min_vals,
                             const ivc::ParamCombination& max_vals,
                             const ivc::ParamCombination& deltas)
    {
        accumulator.~CooAccumulator();
        new (&accumulator) ivc::CooAccumulator(min_vals, max_vals, deltas);
    }
}


namespace ivc
{
namespace student
{

    void parameterized_hough(const ivc::BinaryImg& img,
                             const PixelParamSuggestionFunc& func,
                             ivc::CooAccumulator& accumulator)
    {
        for(size_t width_idx = 0; width_idx < ivc::get_width(img); ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < ivc::get_height(img); ++height_idx)
            {
                if(!get_binary_pixel(img, width_idx, height_idx))
                {
                    continue;
                }

                const ivc::ParamCombinations suggested_params = func(accumulator,
                                                                     img,
                                                                     width_idx,
                                                                     height_idx);

                for(int col_idx = 0; col_idx < suggested_params.cols(); ++col_idx)
                {
                    add_vote(accumulator, suggested_params.col(col_idx));
                }
            }
        }
    }

    ivc::ParamCombinations  hough_line(const ivc::BinaryImg& img,
                                       const ivc::Vec3f& r_values,              // (min, max, delta)
                                       const ivc::Vec3f& theta_values,          // (min, max, delta)
                                       const uint64_t threshold)
    {
        const ivc::ParamCombination min_vals = make_param_combination({theta_values(0), r_values(0)});
        const ivc::ParamCombination max_vals = make_param_combination({theta_values(1), r_values(1)});
        const ivc::ParamCombination deltas = make_param_combination({theta_values(2), r_values(2)});
        ivc::CooAccumulator accumulator(min_vals, max_vals, deltas);

        parameterized_hough(img,
                            [](const ivc::CooAccumulator& acc,
                               const ivc::BinaryImg&,
                               const size_t width_idx,
                               const size_t height_idx)
                            {
                                std::set<ivc::ParamCombination, ivc::ParamComb_comp_t> candidates;
                                const ivc::ParamRange& theta_range = acc._param_all_values[0];
                                const ivc::ParamRange& r_range = acc._param_all_values[1];

                                for(int theta_idx = 0; theta_idx < theta_range.rows(); ++theta_idx)
                                {
                                    const float_t theta_degrees = theta_range(theta_idx);
                                    const float_t theta_radians = degrees_to_radians(theta_degrees);
                                    const float_t r = (static_cast<float_t>(width_idx) * std::cos(theta_radians)) +
                                                      (static_cast<float_t>(height_idx) * std::sin(theta_radians));
                                    const std::optional<float_t> snapped_r = snap_to_range(r_range, r);
                                    if(!snapped_r.has_value())
                                    {
                                        continue;
                                    }

                                    candidates.insert(make_param_combination({theta_degrees, snapped_r.value()}));
                                }

                                return set_to_param_combinations(candidates, 2);
                            },
                            accumulator);

        return accumulator.threshold(threshold);
    }

    ivc::ParamCombinations  hough_circle(const ivc::BinaryImg& img,
                                         const ivc::Vec3f& x_center_values,     // (min, max, delta)
                                         const ivc::Vec3f& y_center_values,     // (min, max, delta)
                                         const ivc::Vec3f& r_values,            // (min, max, delta)
                                         const uint64_t threshold)
    {
        const ivc::ParamCombination min_vals = make_param_combination(
            {x_center_values(0), y_center_values(0), r_values(0)});
        const ivc::ParamCombination max_vals = make_param_combination(
            {x_center_values(1), y_center_values(1), r_values(1)});
        const ivc::ParamCombination deltas = make_param_combination(
            {x_center_values(2), y_center_values(2), r_values(2)});
        ivc::CooAccumulator accumulator(min_vals, max_vals, deltas);

        parameterized_hough(img,
                            [](const ivc::CooAccumulator& acc,
                               const ivc::BinaryImg&,
                               const size_t width_idx,
                               const size_t height_idx)
                            {
                                std::set<ivc::ParamCombination, ivc::ParamComb_comp_t> candidates;
                                const ivc::ParamRange& x_center_range = acc._param_all_values[0];
                                const ivc::ParamRange& y_center_range = acc._param_all_values[1];
                                const ivc::ParamRange& r_range = acc._param_all_values[2];

                                for(int x_idx = 0; x_idx < x_center_range.rows(); ++x_idx)
                                {
                                    const float_t x_center = x_center_range(x_idx);
                                    for(int y_idx = 0; y_idx < y_center_range.rows(); ++y_idx)
                                    {
                                        const float_t y_center = y_center_range(y_idx);
                                        const float_t radius = std::hypot(static_cast<float_t>(width_idx) - x_center,
                                                                          static_cast<float_t>(height_idx) - y_center);
                                        const std::optional<float_t> snapped_r = snap_to_range(r_range, radius);
                                        if(!snapped_r.has_value())
                                        {
                                            continue;
                                        }

                                        candidates.insert(make_param_combination(
                                            {x_center, y_center, snapped_r.value()}));
                                    }
                                }

                                return set_to_param_combinations(candidates, 3);
                            },
                            accumulator);

        return accumulator.threshold(threshold);
    }

    ivc::ParamCombinations  hough_ellipse(const ivc::BinaryImg& img,
                                          const ivc::Vec3f& x_center_values,    // (min, max, delta)
                                          const ivc::Vec3f& y_center_values,    // (min, max, delta)
                                          const ivc::Vec3f& x_stretch_values,   // (min, max, delta)
                                          const ivc::Vec3f& y_stretch_values,   // (min, max, delta)
                                          const uint64_t threshold)
    {
        const ivc::ParamCombination min_vals = make_param_combination(
            {x_center_values(0), y_center_values(0), x_stretch_values(0), y_stretch_values(0)});
        const ivc::ParamCombination max_vals = make_param_combination(
            {x_center_values(1), y_center_values(1), x_stretch_values(1), y_stretch_values(1)});
        const ivc::ParamCombination deltas = make_param_combination(
            {x_center_values(2), y_center_values(2), x_stretch_values(2), y_stretch_values(2)});
        ivc::CooAccumulator accumulator(min_vals, max_vals, deltas);

        parameterized_hough(img,
                            [](const ivc::CooAccumulator& acc,
                               const ivc::BinaryImg&,
                               const size_t width_idx,
                               const size_t height_idx)
                            {
                                std::set<ivc::ParamCombination, ivc::ParamComb_comp_t> candidates;
                                const ivc::ParamRange& x_center_range = acc._param_all_values[0];
                                const ivc::ParamRange& y_center_range = acc._param_all_values[1];
                                const ivc::ParamRange& x_stretch_range = acc._param_all_values[2];
                                const ivc::ParamRange& y_stretch_range = acc._param_all_values[3];

                                for(int x_idx = 0; x_idx < x_center_range.rows(); ++x_idx)
                                {
                                    const float_t x_center = x_center_range(x_idx);
                                    const float_t dx = static_cast<float_t>(width_idx) - x_center;
                                    const float_t dx_sq = dx * dx;

                                    for(int y_idx = 0; y_idx < y_center_range.rows(); ++y_idx)
                                    {
                                        const float_t y_center = y_center_range(y_idx);
                                        const float_t dy = static_cast<float_t>(height_idx) - y_center;
                                        const float_t dy_sq = dy * dy;

                                        if(std::abs(dx) <= kSnapEps && std::abs(dy) <= kSnapEps)
                                        {
                                            continue;
                                        }

                                        if(std::abs(dy) <= kSnapEps)
                                        {
                                            const std::optional<float_t> snapped_x_stretch =
                                                snap_to_range(x_stretch_range, std::abs(dx));
                                            if(!snapped_x_stretch.has_value())
                                            {
                                                continue;
                                            }

                                            for(int y_stretch_idx = 0; y_stretch_idx < y_stretch_range.rows(); ++y_stretch_idx)
                                            {
                                                candidates.insert(make_param_combination(
                                                    {x_center,
                                                     y_center,
                                                     snapped_x_stretch.value(),
                                                     y_stretch_range(y_stretch_idx)}));
                                            }

                                            continue;
                                        }

                                        for(int x_stretch_idx = 0; x_stretch_idx < x_stretch_range.rows(); ++x_stretch_idx)
                                        {
                                            const float_t x_stretch = x_stretch_range(x_stretch_idx);
                                            const float_t x_stretch_sq = x_stretch * x_stretch;
                                            const float_t denom = 1.0f - (dx_sq / x_stretch_sq);

                                            if(denom <= kSnapEps)
                                            {
                                                continue;
                                            }

                                            const float_t y_stretch = std::sqrt(dy_sq / denom);
                                            const std::optional<float_t> snapped_y_stretch =
                                                snap_to_range(y_stretch_range, y_stretch);
                                            if(!snapped_y_stretch.has_value())
                                            {
                                                continue;
                                            }

                                            candidates.insert(make_param_combination(
                                                {x_center, y_center, x_stretch, snapped_y_stretch.value()}));
                                        }
                                    }
                                }

                                return set_to_param_combinations(candidates, 4);
                            },
                            accumulator);

        return accumulator.threshold(threshold);
    }

    RTable::RTable(ivc::BinaryImg& obj_img) // the image containing the object
        : _obj_img(obj_img),
          _obj_img_center_idxs(std::make_tuple(ivc::get_height(obj_img)/2, ivc::get_width(obj_img)/2))
    {
        const ivc::GrayscaleByteImg grayscale_obj = binary_to_grayscale_byte(_obj_img);
        const ivc::GrayscaleFloatImg grad_angles = ivc::get_sobel_3x3_gradient_angles(grayscale_obj);
        const ivc::GrayscaleFloatImg grad_magnitudes = ivc::get_sobel_3x3_gradient_magnitudes(grayscale_obj);
        const float_t center_y = static_cast<float_t>(std::get<0>(_obj_img_center_idxs));
        const float_t center_x = static_cast<float_t>(std::get<1>(_obj_img_center_idxs));

        for(size_t width_idx = 0; width_idx < ivc::get_width(_obj_img); ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < ivc::get_height(_obj_img); ++height_idx)
            {
                if(!get_binary_pixel(_obj_img, width_idx, height_idx))
                {
                    continue;
                }

                if(std::abs(get_gray_float_pixel(grad_magnitudes, width_idx, height_idx)) <= kSnapEps)
                {
                    continue;
                }

                const float_t phi = get_gray_float_pixel(grad_angles, width_idx, height_idx);
                const float_t dx = center_x - static_cast<float_t>(width_idx);
                const float_t dy = center_y - static_cast<float_t>(height_idx);
                const float_t r = std::hypot(dx, dy);
                const float_t theta = normalize_angle_degrees(std::atan2(dy, dx) * 180.0f / kPi);

                _phi_to_r_theta_values[quantize_phi(phi)].insert(std::make_tuple(r, theta));
            }
        }
    }

    std::set<std::tuple<float_t, float_t> > RTable::get_r_theta_values(const float_t phi) const
    {
        const int quantized_phi = quantize_phi(phi);
        const auto r_theta_itr = _phi_to_r_theta_values.find(quantized_phi);
        if(r_theta_itr == _phi_to_r_theta_values.end())
        {
            return {};
        }

        return r_theta_itr->second;
    }

    int RTable::quantize_phi(const float_t phi)
    {
        const float_t normalized_phi = normalize_phi_degrees(phi);
        const int quantized_phi = static_cast<int>(std::lround(normalized_phi));
        return quantized_phi >= static_cast<int>(kDegreesPerCircle)
                   ? quantized_phi - static_cast<int>(kDegreesPerCircle)
                   : quantized_phi;
    }

    ivc::ParamCombinations generalized_hough_fixed_obj(const ivc::BinaryImg& img,
                                                       const ivc::BinaryImg& object_img,    // make an RTable from this
                                                       const ivc::Vec3f& x_center_values,   // (min, max, delta)
                                                       const ivc::Vec3f& y_center_values,   // (min, max, delta)
                                                       ivc::CooAccumulator& accumulator,
                                                       const uint64_t threshold)
    {
        const ivc::ParamCombination min_vals = make_param_combination(
            {x_center_values(0), y_center_values(0)});
        const ivc::ParamCombination max_vals = make_param_combination(
            {x_center_values(1), y_center_values(1)});
        const ivc::ParamCombination deltas = make_param_combination(
            {x_center_values(2), y_center_values(2)});
        rebuild_accumulator(accumulator, min_vals, max_vals, deltas);

        ivc::BinaryImg object_copy = object_img;
        const RTable r_table(object_copy);
        const ivc::GrayscaleByteImg grayscale_img = binary_to_grayscale_byte(img);
        const ivc::GrayscaleFloatImg grad_angles = ivc::get_sobel_3x3_gradient_angles(grayscale_img);
        const ivc::GrayscaleFloatImg grad_magnitudes = ivc::get_sobel_3x3_gradient_magnitudes(grayscale_img);

        for(size_t width_idx = 0; width_idx < ivc::get_width(img); ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < ivc::get_height(img); ++height_idx)
            {
                if(!get_binary_pixel(img, width_idx, height_idx))
                {
                    continue;
                }

                if(std::abs(get_gray_float_pixel(grad_magnitudes, width_idx, height_idx)) <= kSnapEps)
                {
                    continue;
                }

                const float_t phi = get_gray_float_pixel(grad_angles, width_idx, height_idx);
                const std::set<std::tuple<float_t, float_t> > r_theta_values =
                    r_table.get_r_theta_values(phi);

                for(const std::tuple<float_t, float_t>& r_theta : r_theta_values)
                {
                    const float_t r = std::get<0>(r_theta);
                    const float_t theta_degrees = std::get<1>(r_theta);
                    const float_t theta_radians = degrees_to_radians(theta_degrees);
                    const float_t x_center = static_cast<float_t>(width_idx) + (r * std::cos(theta_radians));
                    const float_t y_center = static_cast<float_t>(height_idx) + (r * std::sin(theta_radians));

                    const std::optional<float_t> snapped_x_center =
                        snap_to_range(accumulator._param_all_values[0], x_center);
                    const std::optional<float_t> snapped_y_center =
                        snap_to_range(accumulator._param_all_values[1], y_center);
                    if(!snapped_x_center.has_value() || !snapped_y_center.has_value())
                    {
                        continue;
                    }

                    add_vote(accumulator,
                             make_param_combination({snapped_x_center.value(),
                                                     snapped_y_center.value()}));
                }
            }
        }

        return accumulator.threshold(threshold);
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
        const ivc::ParamCombination min_vals = make_param_combination(
            {x_center_values(0), y_center_values(0), scale_values(0), theta_values(0)});
        const ivc::ParamCombination max_vals = make_param_combination(
            {x_center_values(1), y_center_values(1), scale_values(1), theta_values(1)});
        const ivc::ParamCombination deltas = make_param_combination(
            {x_center_values(2), y_center_values(2), scale_values(2), theta_values(2)});
        rebuild_accumulator(accumulator, min_vals, max_vals, deltas);

        ivc::BinaryImg object_copy = object_img;
        const RTable r_table(object_copy);
        const ivc::GrayscaleByteImg grayscale_img = binary_to_grayscale_byte(img);
        const ivc::GrayscaleFloatImg grad_angles = ivc::get_sobel_3x3_gradient_angles(grayscale_img);
        const ivc::GrayscaleFloatImg grad_magnitudes = ivc::get_sobel_3x3_gradient_magnitudes(grayscale_img);

        for(size_t width_idx = 0; width_idx < ivc::get_width(img); ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < ivc::get_height(img); ++height_idx)
            {
                if(!get_binary_pixel(img, width_idx, height_idx))
                {
                    continue;
                }

                if(std::abs(get_gray_float_pixel(grad_magnitudes, width_idx, height_idx)) <= kSnapEps)
                {
                    continue;
                }

                const float_t target_phi = get_gray_float_pixel(grad_angles, width_idx, height_idx);

                for(int scale_idx = 0; scale_idx < accumulator._param_all_values[2].rows(); ++scale_idx)
                {
                    const float_t scale = accumulator._param_all_values[2](scale_idx);
                    for(int rotation_idx = 0; rotation_idx < accumulator._param_all_values[3].rows(); ++rotation_idx)
                    {
                        const float_t rotation = accumulator._param_all_values[3](rotation_idx);
                        const std::set<std::tuple<float_t, float_t> > r_theta_values =
                            r_table.get_r_theta_values(target_phi - degrees_to_radians(rotation));

                        for(const std::tuple<float_t, float_t>& r_theta : r_theta_values)
                        {
                            const float_t r = std::get<0>(r_theta);
                            const float_t theta_degrees = std::get<1>(r_theta) + rotation;
                            const float_t theta_radians = degrees_to_radians(theta_degrees);
                            const float_t x_center = static_cast<float_t>(width_idx) +
                                                     (scale * r * std::cos(theta_radians));
                            const float_t y_center = static_cast<float_t>(height_idx) +
                                                     (scale * r * std::sin(theta_radians));

                            const std::optional<float_t> snapped_x_center =
                                snap_to_range(accumulator._param_all_values[0], x_center);
                            const std::optional<float_t> snapped_y_center =
                                snap_to_range(accumulator._param_all_values[1], y_center);
                            if(!snapped_x_center.has_value() || !snapped_y_center.has_value())
                            {
                                continue;
                            }

                            add_vote(accumulator,
                                     make_param_combination({snapped_x_center.value(),
                                                             snapped_y_center.value(),
                                                             scale,
                                                             rotation}));
                        }
                    }
                }
            }
        }

        return accumulator.threshold(threshold);
    }

} // end of namespace student
} // end of namespace ivc
