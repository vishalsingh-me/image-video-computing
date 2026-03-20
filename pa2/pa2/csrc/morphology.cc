// SYSTEM INCLUDES
#include <iostream>
#include <vector>

#include <cstdlib>
#include <585/common/types.h>
#include <585/imgproc/imgproc.h>


// C++ PROJECT INCLUDES
#include "hough/morphology.h"



namespace ivc
{
namespace student
{

    ivc::StructureElement strel_diamond(const size_t dim)
    {
        ivc::StructureElement se(1, 1);
        return se;
    }

    ivc::StructureElement strel_cross(const size_t dim,
                                      const size_t thickness)
    {
        ivc::StructureElement se(1, 1);
        return se;
    }

    ivc::StructureElement strel_disk(const size_t diameter)
    {
        ivc::StructureElement se(1, 1);
        return se;
    }

    ivc::StructureElement strel_octagon(const size_t dim)
    {
        ivc::StructureElement se(1, 1);
        return se;
    }


    ivc::StructureElement strel_square(const size_t dim)
    {
        ivc::StructureElement se(1, 1);
        return se;
    }



    ivc::BinaryImg  apply_se(const ivc::BinaryImg& img,
                             const ivc::StructureElement& se,
                             const ivc::student::PatchFunction& patch_func)
    {
        ivc::BinaryImg out(1,1);
        return out;
    }

    // morphological operations
    ivc::BinaryImg  imerode(const ivc::BinaryImg& img,
                            const ivc::StructureElement& se)
    {
        ivc::BinaryImg out(1,1);
        return out;
    }

    ivc::BinaryImg  imdilate(const ivc::BinaryImg& img,
                             const ivc::StructureElement& se)
    {
        ivc::BinaryImg out(1,1);
        return out;
    }

    ivc::BinaryImg  imopen(const ivc::BinaryImg& img,
                           const ivc::StructureElement& se)
    {
        ivc::BinaryImg out(1,1);
        return out;
    }

    ivc::BinaryImg  imclose(const ivc::BinaryImg& img,
                            const ivc::StructureElement& se)
    {
        ivc::BinaryImg out(1,1);
        return out;
    }

    ivc::BinaryImg  imgrad(const ivc::BinaryImg& img,
                           const ivc::StructureElement& se)
    {
        ivc::BinaryImg out(1,1);
        return out;
    }

    ivc::BinaryImg  imskel(const ivc::BinaryImg& img,
                           const ivc::StructureElement& se)
    {
        ivc::BinaryImg out(1,1);
        return out;
    }

} // end of namespace student
} // end of namespace ivc

