// SYSTEM INCLUDES
#include <585/common/types.h>
#include <585/vbow/vbow.h>


// C++ PROJECT INCLUDES
#include "vbow/vbow.h"


namespace ivc
{
namespace student
{

    Vocab::Vocab()
        : _patch_to_idx(), _idx_to_patch()
    {}

    void Vocab::add(const ivc::GrayscaleByteImg& e)
    {

    }

    const size_t Vocab::size() const
    {
        return this->_patch_to_idx.size();
    }

    std::list<ivc::GrayscaleByteImg> Vocab::ordered_elements() const
    {
        std::list<ivc::GrayscaleByteImg> elements;
        return elements;
    }

    BagOfWords::BagOfWords(const ivc::ByteDataset& X,
                           const size_t patch_size)
        : _vocab(), _patch_size(patch_size)
    {}

    BinaryBagOfWords::BinaryBagOfWords(const ivc::ByteDataset& X,
                                       const size_t patch_size)
        : BagOfWords(X, patch_size)
    {}

    ivc::FloatDataset BinaryBagOfWords::transform(const ivc::ByteDataset& X) const
    {
        ivc::FloatDataset D(X.size(), this->_vocab.size());
        return D;
    }

    CountingBagOfWords::CountingBagOfWords(const ivc::ByteDataset& X,
                                           const size_t patch_size)
        : BagOfWords(X, patch_size)
    {}

    ivc::FloatDataset CountingBagOfWords::transform(const ivc::ByteDataset& X) const
    {
        ivc::FloatDataset D(X.size(), this->_vocab.size());
        return D;
    }


    OVR::OVR()
    {}

    ivc::ProbVector OVR::predict(const ivc::FloatDataset& X) const
    {
        const size_t num_rows = X.rows();
        ivc::ProbVector y_pred = ivc::ProbVector::Random(num_rows);
        return y_pred;
    }

    float_t OVR::cost(const ivc::FloatDataset& X,
                      const ivc::ProbVector& y_gt) const
    {
        return 0.0f;
    }

    void OVR::train(const ivc::FloatDataset& X,
                    const ivc::ProbVector& y_gt,
                    const float_t lr,
                    const size_t max_epochs)
    {

    }

    ivc::ByteDataset tile_dataset(const ivc::ByteDataset& X,
                                  const size_t level)
    {
        return X;
    }

    HistogramOfGradients::HistogramOfGradients()
    {}

    ivc::FloatDataset HistogramOfGradients::transform(const ivc::ByteDataset& X) const
    {
        ivc::FloatDataset D(X.size(), 1);
        return D;
    }

    std::tuple<ivc::ByteDataset,
               ivc::ProbVector> balance(const ivc::ByteDataset& X,
                                        const ivc::ProbVector& y_gt,
                                        const ivc::student::balance_type_t balance_type)
    {
        return std::make_tuple(X, y_gt);
    }

} // end of namespace student
} // end of namespace ivc

