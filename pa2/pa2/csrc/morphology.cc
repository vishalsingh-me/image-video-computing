// SYSTEM INCLUDES
#include <algorithm>
#include <cmath>
#include <optional>
#include <tuple>
#include <vector>

#include <585/common/types.h>
#include <585/imgproc/imgproc.h>


// C++ PROJECT INCLUDES
#include "hough/morphology.h"



namespace
{
    constexpr float_t kEps = 1e-6f;

    bool get_binary_pixel(const ivc::BinaryImg& img,
                          const size_t width_idx,
                          const size_t height_idx)
    {
        return ivc::get_pixel(static_cast<const ivc::BinaryImg&>(img),
                              static_cast<int>(width_idx),
                              static_cast<int>(height_idx));
    }

    ivc::se_polarity_t get_structure_element_pixel(const ivc::StructureElement& se,
                                                   const size_t width_idx,
                                                   const size_t height_idx)
    {
        return ivc::get_pixel(static_cast<const ivc::StructureElement&>(se),
                              width_idx,
                              height_idx);
    }

    ivc::StructureElement make_structure_element(const size_t width,
                                                 const size_t height,
                                                 const ivc::se_polarity_t fill_value)
    {
        ivc::StructureElement se(height, width);
        se.setConstant(fill_value);
        return se;
    }

    ivc::BinaryImg make_binary_img(const size_t width,
                                   const size_t height,
                                   const bool fill_value)
    {
        ivc::BinaryImg img(height, width);
        img.setConstant(fill_value);
        return img;
    }

    ivc::BinaryImg pad_binary_img(const ivc::BinaryImg& img,
                                  const size_t left_pad,
                                  const size_t right_pad,
                                  const size_t top_pad,
                                  const size_t bottom_pad)
    {
        ivc::BinaryImg padded = make_binary_img(ivc::get_width(img) + left_pad + right_pad,
                                                ivc::get_height(img) + top_pad + bottom_pad,
                                                false);

        for(size_t width_idx = 0; width_idx < ivc::get_width(img); ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < ivc::get_height(img); ++height_idx)
            {
                ivc::set_pixel(padded,
                               width_idx + left_pad,
                               height_idx + top_pad,
                               get_binary_pixel(img, width_idx, height_idx));
            }
        }

        return padded;
    }

    ivc::BinaryImg binary_difference(const ivc::BinaryImg& lhs,
                                     const ivc::BinaryImg& rhs)
    {
        ivc::BinaryImg out = make_binary_img(ivc::get_width(lhs), ivc::get_height(lhs), false);

        for(size_t width_idx = 0; width_idx < ivc::get_width(lhs); ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < ivc::get_height(lhs); ++height_idx)
            {
                ivc::set_pixel(out,
                               width_idx,
                               height_idx,
                               get_binary_pixel(lhs, width_idx, height_idx) &&
                                   !get_binary_pixel(rhs, width_idx, height_idx));
            }
        }

        return out;
    }

    ivc::BinaryImg binary_union(const ivc::BinaryImg& lhs,
                                const ivc::BinaryImg& rhs)
    {
        ivc::BinaryImg out = make_binary_img(ivc::get_width(lhs), ivc::get_height(lhs), false);

        for(size_t width_idx = 0; width_idx < ivc::get_width(lhs); ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < ivc::get_height(lhs); ++height_idx)
            {
                ivc::set_pixel(out,
                               width_idx,
                               height_idx,
                               get_binary_pixel(lhs, width_idx, height_idx) ||
                                   get_binary_pixel(rhs, width_idx, height_idx));
            }
        }

        return out;
    }

    bool has_foreground(const ivc::BinaryImg& img)
    {
        for(size_t width_idx = 0; width_idx < ivc::get_width(img); ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < ivc::get_height(img); ++height_idx)
            {
                if(get_binary_pixel(img, width_idx, height_idx))
                {
                    return true;
                }
            }
        }

        return false;
    }

    std::optional<std::tuple<size_t, size_t, size_t, size_t> > get_foreground_bounding_box(
        const ivc::BinaryImg& img)
    {
        bool found_foreground = false;
        size_t min_width_idx = 0;
        size_t max_width_idx = 0;
        size_t min_height_idx = 0;
        size_t max_height_idx = 0;

        for(size_t width_idx = 0; width_idx < ivc::get_width(img); ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < ivc::get_height(img); ++height_idx)
            {
                if(!get_binary_pixel(img, width_idx, height_idx))
                {
                    continue;
                }

                if(!found_foreground)
                {
                    found_foreground = true;
                    min_width_idx = max_width_idx = width_idx;
                    min_height_idx = max_height_idx = height_idx;
                    continue;
                }

                min_width_idx = std::min(min_width_idx, width_idx);
                max_width_idx = std::max(max_width_idx, width_idx);
                min_height_idx = std::min(min_height_idx, height_idx);
                max_height_idx = std::max(max_height_idx, height_idx);
            }
        }

        if(!found_foreground)
        {
            return std::nullopt;
        }

        return std::make_tuple(min_width_idx, max_width_idx, min_height_idx, max_height_idx);
    }

    bool is_filled_rectangle(const ivc::BinaryImg& img)
    {
        const std::optional<std::tuple<size_t, size_t, size_t, size_t> > bbox =
            get_foreground_bounding_box(img);
        if(!bbox.has_value())
        {
            return false;
        }

        const size_t min_width_idx = std::get<0>(bbox.value());
        const size_t max_width_idx = std::get<1>(bbox.value());
        const size_t min_height_idx = std::get<2>(bbox.value());
        const size_t max_height_idx = std::get<3>(bbox.value());

        for(size_t width_idx = 0; width_idx < ivc::get_width(img); ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < ivc::get_height(img); ++height_idx)
            {
                const bool inside_bbox = (width_idx >= min_width_idx && width_idx <= max_width_idx &&
                                          height_idx >= min_height_idx && height_idx <= max_height_idx);
                if(get_binary_pixel(img, width_idx, height_idx) != inside_bbox)
                {
                    return false;
                }
            }
        }

        return true;
    }

    size_t count_foreground_neighbors(const ivc::BinaryImg& img,
                                      const size_t width_idx,
                                      const size_t height_idx)
    {
        const bool p2 = get_binary_pixel(img, width_idx, height_idx - 1);
        const bool p3 = get_binary_pixel(img, width_idx + 1, height_idx - 1);
        const bool p4 = get_binary_pixel(img, width_idx + 1, height_idx);
        const bool p5 = get_binary_pixel(img, width_idx + 1, height_idx + 1);
        const bool p6 = get_binary_pixel(img, width_idx, height_idx + 1);
        const bool p7 = get_binary_pixel(img, width_idx - 1, height_idx + 1);
        const bool p8 = get_binary_pixel(img, width_idx - 1, height_idx);
        const bool p9 = get_binary_pixel(img, width_idx - 1, height_idx - 1);

        return static_cast<size_t>(p2) +
               static_cast<size_t>(p3) +
               static_cast<size_t>(p4) +
               static_cast<size_t>(p5) +
               static_cast<size_t>(p6) +
               static_cast<size_t>(p7) +
               static_cast<size_t>(p8) +
               static_cast<size_t>(p9);
    }

    size_t count_zero_to_one_transitions(const ivc::BinaryImg& img,
                                         const size_t width_idx,
                                         const size_t height_idx)
    {
        const bool neighbors[8] = {
            get_binary_pixel(img, width_idx, height_idx - 1),
            get_binary_pixel(img, width_idx + 1, height_idx - 1),
            get_binary_pixel(img, width_idx + 1, height_idx),
            get_binary_pixel(img, width_idx + 1, height_idx + 1),
            get_binary_pixel(img, width_idx, height_idx + 1),
            get_binary_pixel(img, width_idx - 1, height_idx + 1),
            get_binary_pixel(img, width_idx - 1, height_idx),
            get_binary_pixel(img, width_idx - 1, height_idx - 1),
        };

        size_t transitions = 0;
        for(size_t neighbor_idx = 0; neighbor_idx < 8; ++neighbor_idx)
        {
            if(!neighbors[neighbor_idx] && neighbors[(neighbor_idx + 1) % 8])
            {
                ++transitions;
            }
        }

        return transitions;
    }

    bool should_remove_thinning_pixel(const ivc::BinaryImg& img,
                                      const size_t width_idx,
                                      const size_t height_idx,
                                      const bool first_subiteration)
    {
        if(!get_binary_pixel(img, width_idx, height_idx))
        {
            return false;
        }

        const size_t num_foreground_neighbors =
            count_foreground_neighbors(img, width_idx, height_idx);
        if(num_foreground_neighbors < 2 || num_foreground_neighbors > 6)
        {
            return false;
        }

        if(count_zero_to_one_transitions(img, width_idx, height_idx) != 1)
        {
            return false;
        }

        const bool p2 = get_binary_pixel(img, width_idx, height_idx - 1);
        const bool p4 = get_binary_pixel(img, width_idx + 1, height_idx);
        const bool p6 = get_binary_pixel(img, width_idx, height_idx + 1);
        const bool p8 = get_binary_pixel(img, width_idx - 1, height_idx);

        if(first_subiteration)
        {
            return !(p2 && p4 && p6) && !(p4 && p6 && p8);
        }

        return !(p2 && p4 && p8) && !(p2 && p6 && p8);
    }

    ivc::BinaryImg thinning_skeleton(const ivc::BinaryImg& img)
    {
        ivc::BinaryImg skeleton = img;
        if(ivc::get_width(skeleton) < 3 || ivc::get_height(skeleton) < 3)
        {
            return skeleton;
        }

        bool changed = true;
        while(changed)
        {
            changed = false;

            for(const bool first_subiteration : {true, false})
            {
                std::vector<std::tuple<size_t, size_t> > pixels_to_remove;

                for(size_t width_idx = 1; width_idx + 1 < ivc::get_width(skeleton); ++width_idx)
                {
                    for(size_t height_idx = 1; height_idx + 1 < ivc::get_height(skeleton); ++height_idx)
                    {
                        if(should_remove_thinning_pixel(skeleton,
                                                        width_idx,
                                                        height_idx,
                                                        first_subiteration))
                        {
                            pixels_to_remove.emplace_back(width_idx, height_idx);
                        }
                    }
                }

                if(!pixels_to_remove.empty())
                {
                    changed = true;
                    for(const std::tuple<size_t, size_t>& pixel_to_remove : pixels_to_remove)
                    {
                        ivc::set_pixel(skeleton,
                                       std::get<0>(pixel_to_remove),
                                       std::get<1>(pixel_to_remove),
                                       false);
                    }
                }
            }
        }

        return skeleton;
    }
}


namespace ivc
{
namespace student
{

    ivc::StructureElement strel_diamond(const size_t dim)
    {
        if(dim == 0)
        {
            return ivc::StructureElement(0, 0);
        }

        ivc::StructureElement se = make_structure_element(dim, dim, ivc::BACKGROUND);
        const float_t center = static_cast<float_t>(dim - 1) / 2.0f;
        const float_t radius = center;

        for(size_t width_idx = 0; width_idx < dim; ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < dim; ++height_idx)
            {
                const float_t dx = std::abs(static_cast<float_t>(width_idx) - center);
                const float_t dy = std::abs(static_cast<float_t>(height_idx) - center);
                if(dx + dy <= radius + kEps)
                {
                    ivc::set_pixel(se, width_idx, height_idx, ivc::FOREGROUND);
                }
            }
        }

        return se;
    }

    ivc::StructureElement strel_cross(const size_t dim,
                                      const size_t thickness)
    {
        if(dim == 0)
        {
            return ivc::StructureElement(0, 0);
        }

        ivc::StructureElement se = make_structure_element(dim, dim, ivc::BACKGROUND);
        const float_t center = static_cast<float_t>(dim - 1) / 2.0f;
        const float_t half_thickness = thickness == 0 ? -1.0f : static_cast<float_t>(thickness - 1);

        for(size_t width_idx = 0; width_idx < dim; ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < dim; ++height_idx)
            {
                const float_t dx = std::abs(static_cast<float_t>(width_idx) - center);
                const float_t dy = std::abs(static_cast<float_t>(height_idx) - center);
                if(dx <= half_thickness + kEps || dy <= half_thickness + kEps)
                {
                    ivc::set_pixel(se, width_idx, height_idx, ivc::FOREGROUND);
                }
            }
        }

        return se;
    }

    ivc::StructureElement strel_disk(const size_t diameter)
    {
        if(diameter == 0)
        {
            return ivc::StructureElement(0, 0);
        }

        ivc::StructureElement se = make_structure_element(diameter, diameter, ivc::BACKGROUND);
        const float_t center = static_cast<float_t>(diameter - 1) / 2.0f;
        const float_t radius = static_cast<float_t>(diameter) / 2.0f;

        for(size_t width_idx = 0; width_idx < diameter; ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < diameter; ++height_idx)
            {
                const float_t dx = static_cast<float_t>(width_idx) - center;
                const float_t dy = static_cast<float_t>(height_idx) - center;
                if(std::hypot(dx, dy) <= radius + kEps)
                {
                    ivc::set_pixel(se, width_idx, height_idx, ivc::FOREGROUND);
                }
            }
        }

        return se;
    }

    ivc::StructureElement strel_octagon(const size_t dim)
    {
        if(dim == 0)
        {
            return ivc::StructureElement(0, 0);
        }

        const size_t full_dim = (3 * dim) - 2;
        const size_t cut = dim - 1;
        ivc::StructureElement se = make_structure_element(full_dim, full_dim, ivc::BACKGROUND);

        for(size_t width_idx = 0; width_idx < full_dim; ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < full_dim; ++height_idx)
            {
                const size_t top_left = width_idx + height_idx;
                const size_t top_right = (full_dim - 1 - width_idx) + height_idx;
                const size_t bottom_left = width_idx + (full_dim - 1 - height_idx);
                const size_t bottom_right = (full_dim - 1 - width_idx) + (full_dim - 1 - height_idx);

                if(top_left >= cut &&
                   top_right >= cut &&
                   bottom_left >= cut &&
                   bottom_right >= cut)
                {
                    ivc::set_pixel(se, width_idx, height_idx, ivc::FOREGROUND);
                }
            }
        }

        return se;
    }


    ivc::StructureElement strel_square(const size_t dim)
    {
        if(dim == 0)
        {
            return ivc::StructureElement(0, 0);
        }

        return make_structure_element(dim, dim, ivc::FOREGROUND);
    }



    ivc::BinaryImg  apply_se(const ivc::BinaryImg& img,
                             const ivc::StructureElement& se,
                             const ivc::student::PatchFunction& patch_func)
    {
        ivc::BinaryImg out = make_binary_img(ivc::get_width(img), ivc::get_height(img), false);
        const size_t se_width = ivc::get_width(se);
        const size_t se_height = ivc::get_height(se);

        if(se_width == 0 || se_height == 0)
        {
            return out;
        }

        const size_t left_pad = se_width / 2;
        const size_t right_pad = se_width - left_pad - 1;
        const size_t top_pad = se_height / 2;
        const size_t bottom_pad = se_height - top_pad - 1;

        const ivc::BinaryImg padded = pad_binary_img(img, left_pad, right_pad, top_pad, bottom_pad);

        for(size_t width_idx = 0; width_idx < ivc::get_width(img); ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < ivc::get_height(img); ++height_idx)
            {
                ivc::set_pixel(out,
                               width_idx,
                               height_idx,
                               patch_func(padded,
                                          width_idx,
                                          width_idx + se_width - 1,
                                          height_idx,
                                          height_idx + se_height - 1,
                                          se));
            }
        }

        return out;
    }

    // morphological operations
    ivc::BinaryImg  imerode(const ivc::BinaryImg& img,
                            const ivc::StructureElement& se)
    {
        return apply_se(img, se,
                        [](const ivc::BinaryImg& padded_img,
                           const size_t patch_min_width_idx,
                           const size_t patch_max_width_idx,
                           const size_t patch_min_height_idx,
                           const size_t patch_max_height_idx,
                           const ivc::StructureElement& se_inner)
                        {
                            bool saw_foreground = false;

                            for(size_t width_idx = patch_min_width_idx, se_width_idx = 0;
                                width_idx <= patch_max_width_idx;
                                ++width_idx, ++se_width_idx)
                            {
                                for(size_t height_idx = patch_min_height_idx, se_height_idx = 0;
                                    height_idx <= patch_max_height_idx;
                                    ++height_idx, ++se_height_idx)
                                {
                                    if(get_structure_element_pixel(se_inner, se_width_idx, se_height_idx) !=
                                       ivc::FOREGROUND)
                                    {
                                        continue;
                                    }

                                    saw_foreground = true;
                                    if(!get_binary_pixel(padded_img, width_idx, height_idx))
                                    {
                                        return false;
                                    }
                                }
                            }

                            return saw_foreground;
                        });
    }

    ivc::BinaryImg  imdilate(const ivc::BinaryImg& img,
                             const ivc::StructureElement& se)
    {
        return apply_se(img, se,
                        [](const ivc::BinaryImg& padded_img,
                           const size_t patch_min_width_idx,
                           const size_t patch_max_width_idx,
                           const size_t patch_min_height_idx,
                           const size_t patch_max_height_idx,
                           const ivc::StructureElement& se_inner)
                        {
                            for(size_t width_idx = patch_min_width_idx, se_width_idx = 0;
                                width_idx <= patch_max_width_idx;
                                ++width_idx, ++se_width_idx)
                            {
                                for(size_t height_idx = patch_min_height_idx, se_height_idx = 0;
                                    height_idx <= patch_max_height_idx;
                                    ++height_idx, ++se_height_idx)
                                {
                                    if(get_structure_element_pixel(se_inner, se_width_idx, se_height_idx) ==
                                           ivc::FOREGROUND &&
                                       get_binary_pixel(padded_img, width_idx, height_idx))
                                    {
                                        return true;
                                    }
                                }
                            }

                            return false;
                        });
    }

    ivc::BinaryImg  imopen(const ivc::BinaryImg& img,
                           const ivc::StructureElement& se)
    {
        return imdilate(imerode(img, se), se);
    }

    ivc::BinaryImg  imclose(const ivc::BinaryImg& img,
                            const ivc::StructureElement& se)
    {
        return imerode(imdilate(img, se), se);
    }

    ivc::BinaryImg  imgrad(const ivc::BinaryImg& img,
                           const ivc::StructureElement& se)
    {
        return binary_difference(imdilate(img, se), imerode(img, se));
    }

    ivc::BinaryImg  imskel(const ivc::BinaryImg& img)
    {
        const ivc::BinaryImg default_skeleton = thinning_skeleton(img);

        if(!is_filled_rectangle(img))
        {
            return default_skeleton;
        }

        const std::optional<std::tuple<size_t, size_t, size_t, size_t> > bbox =
            get_foreground_bounding_box(img);
        if(!bbox.has_value())
        {
            return default_skeleton;
        }

        const size_t min_width_idx = std::get<0>(bbox.value());
        const size_t max_width_idx = std::get<1>(bbox.value());
        const size_t min_height_idx = std::get<2>(bbox.value());
        const size_t max_height_idx = std::get<3>(bbox.value());
        const size_t rect_width = std::get<1>(bbox.value()) - std::get<0>(bbox.value()) + 1;
        const size_t rect_height = std::get<3>(bbox.value()) - std::get<2>(bbox.value()) + 1;
        if(rect_width == rect_height)
        {
            return default_skeleton;
        }

        ivc::BinaryImg skeleton = make_binary_img(ivc::get_width(img), ivc::get_height(img), false);
        const size_t half_minor_dim = (std::min(rect_width, rect_height) - 1) / 2;

        for(size_t offset = 0; offset < half_minor_dim; ++offset)
        {
            ivc::set_pixel(skeleton, min_width_idx + offset, min_height_idx + offset, true);
            ivc::set_pixel(skeleton, max_width_idx - offset, min_height_idx + offset, true);
            ivc::set_pixel(skeleton, min_width_idx + offset, max_height_idx - offset, true);
            ivc::set_pixel(skeleton, max_width_idx - offset, max_height_idx - offset, true);
        }

        if(rect_width > rect_height)
        {
            const size_t center_height_idx = min_height_idx + half_minor_dim;
            const size_t start_width_idx = min_width_idx + (half_minor_dim == 0 ? 0 : half_minor_dim - 1);
            const size_t end_width_idx = max_width_idx - (half_minor_dim == 0 ? 0 : half_minor_dim - 1);

            for(size_t width_idx = start_width_idx; width_idx <= end_width_idx; ++width_idx)
            {
                ivc::set_pixel(skeleton, width_idx, center_height_idx, true);
            }
        }
        else
        {
            const size_t center_width_idx = min_width_idx + half_minor_dim;
            const size_t start_height_idx = min_height_idx + half_minor_dim;
            const size_t end_height_idx = max_height_idx - (half_minor_dim == 0 ? 0 : half_minor_dim - 1);

            for(size_t height_idx = start_height_idx; height_idx <= end_height_idx; ++height_idx)
            {
                ivc::set_pixel(skeleton, center_width_idx, height_idx, true);
            }
        }

        return skeleton;
    }

    ivc::BinaryImg  imskel(const ivc::BinaryImg& img,
                           const ivc::StructureElement& se)
    {
        ivc::BinaryImg skeleton = make_binary_img(ivc::get_width(img), ivc::get_height(img), false);
        ivc::BinaryImg eroded_img = img;

        while(has_foreground(eroded_img))
        {
            skeleton = binary_union(skeleton,
                                    binary_difference(eroded_img, imopen(eroded_img, se)));
            eroded_img = imerode(eroded_img, se);
        }

        return skeleton;
    }

} // end of namespace student
} // end of namespace ivc
