#pragma once
#ifndef _HOUGH_SOLUTION_MORPHOLOGY_H_
#define _HOUGH_SOLUTION_MORPHOLOGY_H_

// SYSTEM INCLUDES
#include <functional>           // std::function
#include <585/common/types.h>
#include <585/imgproc/imgproc.h>


// C++ PROJECT INCLUDES


namespace ivc
{

    // some functions for creating structure elements
    ivc::StructureElement strel_diamond(const size_t dim);          // the length of the diamond (filled in)
    ivc::StructureElement strel_cross(const size_t dim,             // the length of the cross (filled in)
                                      const size_t thickness=1);
    ivc::StructureElement strel_disk(const size_t diameter);        // the diameter of the disk (filled in)
    ivc::StructureElement strel_octagon(const size_t dim);          // the odd-dim of a side of the octagon (filled in)
    ivc::StructureElement strel_square(const size_t dim);           // completely full of FOREGROUND pixels


    // the backbone of morphological operations (technically you could make this work for convolutions too)
    using PatchFunction = std::function<bool(const ivc::BinaryImg& img,
                                             const size_t patch_min_width_idx,
                                             const size_t patch_max_width_idx,
                                             const size_t patch_min_height_idx,
                                             const size_t patch_max_height_idx,
                                             const ivc::StructureElement& se)>;

    // generic function that implements a convolution-esque style of rubbing a filter across an image
    // the <code>PatchFunction</code> is responsible for calculating the output pixel from the patch + filter
    ivc::BinaryImg  apply_se(const ivc::BinaryImg& img,
                             const ivc::StructureElement& se,
                             const ivc::PatchFunction& patch_func);

    // morphological operations (should use <code>apply_se</code> as a subroutine)
    ivc::BinaryImg  imerode(const ivc::BinaryImg& img,
                            const ivc::StructureElement& se);
    ivc::BinaryImg  imdilate(const ivc::BinaryImg& img,
                             const ivc::StructureElement& se);
    ivc::BinaryImg  imopen(const ivc::BinaryImg& img,
                           const ivc::StructureElement& se);
    ivc::BinaryImg  imclose(const ivc::BinaryImg& img,
                            const ivc::StructureElement& se);
    ivc::BinaryImg  imgrad(const ivc::BinaryImg& img,
                           const ivc::StructureElement& se);
    ivc::BinaryImg  imskel(const ivc::BinaryImg& img);

} // end of namespace ivc


#endif // end of _HOUGH_SOLUTION_MORPHOLOGY_H_

