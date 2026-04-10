#pragma once
#ifndef _585_IMCLS_VBOG_H_
#define _585_IMCLS_VBOG_H_

// SYSTEM INCLUDES
#include <Eigen/Dense>
#include <list>           // std::list is a linked list
#include <tuple>


// C++ PROJECT INCLUDES
#include <585/common/types.h>


namespace ivc
{

    class GrayscaleByteImg_comp_t
    {
    public:
        // returns true iff a < b and false otherwise
        bool operator()(const ivc::GrayscaleByteImg& a,
                        const ivc::GrayscaleByteImg& b) const;
    };

    // dataset datatypes.
    using FeatureVector = Eigen::Matrix<float_t, 1, Eigen::Dynamic>;
    using ByteDataset = std::list<ivc::GrayscaleByteImg>;   // images stacked together
    using FloatDataset = Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic>;    // one row per sample, one col per feature
    using ProbVector = Eigen::VectorXf;

    // logistic regression model
    class LogReg
    {
    public:
        LogReg(const size_t num_features);

        ivc::ProbVector predict(const ivc::FloatDataset& X) const;
        float_t cost(const ivc::FloatDataset& X,
                     const ivc::ProbVector& y_gt) const;

        void train(const ivc::FloatDataset& X,
                   const ivc::ProbVector& y_gt,
                   const float_t lr,
                   const size_t max_epochs);

    protected:
        Eigen::VectorXf     _w;
        float_t             _b;
    };

}


#endif // end of _585_IMCLS_VBOG_H_

