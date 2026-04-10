// SYSTEM INCLUDES
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

#include <585/common/types.h>
#include <585/grad/grad.h>
#include <585/vbow/vbow.h>


// C++ PROJECT INCLUDES
#include "vbow/vbow.h"


namespace
{
    uint8_t get_byte_pixel(const ivc::GrayscaleByteImg& img,
                           const size_t width_idx,
                           const size_t height_idx)
    {
        return ivc::get_pixel(img,
                              static_cast<int>(width_idx),
                              static_cast<int>(height_idx));
    }

    float_t get_float_pixel(const ivc::GrayscaleFloatImg& img,
                            const size_t width_idx,
                            const size_t height_idx)
    {
        return ivc::get_pixel(img,
                              static_cast<int>(width_idx),
                              static_cast<int>(height_idx));
    }

    ivc::GrayscaleByteImg extract_patch(const ivc::GrayscaleByteImg& img,
                                        const size_t start_width_idx,
                                        const size_t start_height_idx,
                                        const size_t patch_size)
    {
        ivc::GrayscaleByteImg patch(patch_size, patch_size);
        patch.setZero();

        for(size_t width_idx = 0; width_idx < patch_size; ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < patch_size; ++height_idx)
            {
                ivc::set_pixel(patch,
                               width_idx,
                               height_idx,
                               get_byte_pixel(img,
                                              start_width_idx + width_idx,
                                              start_height_idx + height_idx));
            }
        }

        return patch;
    }

    float_t normalize_angle(float_t angle)
    {
        while(angle < 0.0f)
        {
            angle += 360.0f;
        }

        while(angle >= 360.0f)
        {
            angle -= 360.0f;
        }

        return angle;
    }

    size_t get_angle_bin(const float_t angle)
    {
        return static_cast<size_t>(normalize_angle(angle + 22.5f) / 45.0f) % 8;
    }

    void append_tiles_recursive(const ivc::GrayscaleByteImg& img,
                                const size_t level,
                                ivc::ByteDataset& out)
    {
        if(level == 0)
        {
            out.push_back(img);
            return;
        }

        const size_t width = ivc::get_width(img);
        const size_t height = ivc::get_height(img);
        if(width < 2 || height < 2)
        {
            out.push_back(img);
            return;
        }

        const size_t mid_width = width / 2;
        const size_t mid_height = height / 2;
        const size_t widths[2] = {mid_width, width - mid_width};
        const size_t heights[2] = {mid_height, height - mid_height};
        const size_t x_starts[2] = {0, mid_width};
        const size_t y_starts[2] = {0, mid_height};

        for(size_t y_block = 0; y_block < 2; ++y_block)
        {
            for(size_t x_block = 0; x_block < 2; ++x_block)
            {
                if(widths[x_block] == 0 || heights[y_block] == 0)
                {
                    continue;
                }

                ivc::GrayscaleByteImg tile(heights[y_block], widths[x_block]);
                tile.setZero();
                for(size_t width_idx = 0; width_idx < widths[x_block]; ++width_idx)
                {
                    for(size_t height_idx = 0; height_idx < heights[y_block]; ++height_idx)
                    {
                        ivc::set_pixel(tile,
                                       width_idx,
                                       height_idx,
                                       get_byte_pixel(img,
                                                      x_starts[x_block] + width_idx,
                                                      y_starts[y_block] + height_idx));
                    }
                }

                append_tiles_recursive(tile, level - 1, out);
            }
        }
    }

    double squared_distance(const ivc::GrayscaleByteImg& a,
                            const ivc::GrayscaleByteImg& b)
    {
        double distance = 0.0;

        for(size_t width_idx = 0; width_idx < ivc::get_width(a); ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < ivc::get_height(a); ++height_idx)
            {
                const double diff = static_cast<double>(get_byte_pixel(a, width_idx, height_idx)) -
                                    static_cast<double>(get_byte_pixel(b, width_idx, height_idx));
                distance += diff * diff;
            }
        }

        return distance;
    }

    ivc::GrayscaleByteImg smote_mix(const ivc::GrayscaleByteImg& a,
                                    const ivc::GrayscaleByteImg& b,
                                    std::mt19937& rng)
    {
        std::uniform_real_distribution<float_t> alpha_dist(0.0f, 1.0f);
        const float_t alpha = alpha_dist(rng);

        ivc::GrayscaleByteImg mixed(a.rows(), a.cols());
        mixed.setZero();

        for(size_t width_idx = 0; width_idx < ivc::get_width(a); ++width_idx)
        {
            for(size_t height_idx = 0; height_idx < ivc::get_height(a); ++height_idx)
            {
                const float_t value = alpha * static_cast<float_t>(get_byte_pixel(a, width_idx, height_idx)) +
                                      (1.0f - alpha) * static_cast<float_t>(get_byte_pixel(b, width_idx, height_idx));
                ivc::set_pixel(mixed,
                               width_idx,
                               height_idx,
                               static_cast<uint8_t>(std::clamp(std::lround(value), 0l, 255l)));
            }
        }

        return mixed;
    }
}


namespace ivc
{
namespace student
{

    Vocab::Vocab()
        : _patch_to_idx(), _idx_to_patch()
    {}

    void Vocab::add(const ivc::GrayscaleByteImg& e)
    {
        if(this->_patch_to_idx.find(e) != this->_patch_to_idx.end())
        {
            return;
        }

        const uint64_t idx = static_cast<uint64_t>(this->_patch_to_idx.size());
        this->_patch_to_idx.emplace(e, idx);
        this->_idx_to_patch.emplace(idx, e);
    }

    bool Vocab::contains(const ivc::GrayscaleByteImg& e) const
    {
        return this->_patch_to_idx.find(e) != this->_patch_to_idx.end();
    }

    uint64_t Vocab::get_idx(const ivc::GrayscaleByteImg& e) const
    {
        const auto it = this->_patch_to_idx.find(e);
        if(it == this->_patch_to_idx.end())
        {
            throw std::runtime_error("Patch not found in vocabulary");
        }

        return it->second;
    }

    const size_t Vocab::size() const
    {
        return this->_patch_to_idx.size();
    }

    std::list<ivc::GrayscaleByteImg> Vocab::ordered_elements() const
    {
        std::list<ivc::GrayscaleByteImg> elements;
        for(const auto& [idx, patch] : this->_idx_to_patch)
        {
            elements.push_back(patch);
        }

        return elements;
    }

    BagOfWords::BagOfWords(const ivc::ByteDataset& X,
                           const size_t patch_size)
        : _vocab(), _patch_size(patch_size)
    {
        if(patch_size == 0 || patch_size % 2 == 0)
        {
            throw std::runtime_error("Patch size must be positive and odd");
        }

        for(const ivc::GrayscaleByteImg& img : X)
        {
            const ivc::GrayscaleByteImg padded =
                ivc::symmetric_pad(img, patch_size / 2, ivc::ZEROS, ivc::SYMMETRIC_ALL);

            for(size_t width_idx = 0; width_idx < ivc::get_width(img); ++width_idx)
            {
                for(size_t height_idx = 0; height_idx < ivc::get_height(img); ++height_idx)
                {
                    this->_vocab.add(extract_patch(padded, width_idx, height_idx, this->_patch_size));
                }
            }
        }
    }

    BinaryBagOfWords::BinaryBagOfWords(const ivc::ByteDataset& X,
                                       const size_t patch_size)
        : BagOfWords(X, patch_size)
    {}

    ivc::FloatDataset BinaryBagOfWords::transform(const ivc::ByteDataset& X) const
    {
        ivc::FloatDataset D(X.size(), this->_vocab.size());
        D.setZero();

        size_t row_idx = 0;
        for(const ivc::GrayscaleByteImg& img : X)
        {
            const ivc::GrayscaleByteImg padded =
                ivc::symmetric_pad(img, this->_patch_size / 2, ivc::ZEROS, ivc::SYMMETRIC_ALL);

            for(size_t width_idx = 0; width_idx < ivc::get_width(img); ++width_idx)
            {
                for(size_t height_idx = 0; height_idx < ivc::get_height(img); ++height_idx)
                {
                    const ivc::GrayscaleByteImg patch =
                        extract_patch(padded, width_idx, height_idx, this->_patch_size);
                    if(this->_vocab.contains(patch))
                    {
                        D(static_cast<Eigen::Index>(row_idx),
                          static_cast<Eigen::Index>(this->_vocab.get_idx(patch))) = 1.0f;
                    }
                }
            }

            ++row_idx;
        }

        return D;
    }

    CountingBagOfWords::CountingBagOfWords(const ivc::ByteDataset& X,
                                           const size_t patch_size)
        : BagOfWords(X, patch_size)
    {}

    ivc::FloatDataset CountingBagOfWords::transform(const ivc::ByteDataset& X) const
    {
        ivc::FloatDataset D(X.size(), this->_vocab.size());
        D.setZero();

        size_t row_idx = 0;
        for(const ivc::GrayscaleByteImg& img : X)
        {
            const ivc::GrayscaleByteImg padded =
                ivc::symmetric_pad(img, this->_patch_size / 2, ivc::ZEROS, ivc::SYMMETRIC_ALL);

            for(size_t width_idx = 0; width_idx < ivc::get_width(img); ++width_idx)
            {
                for(size_t height_idx = 0; height_idx < ivc::get_height(img); ++height_idx)
                {
                    const ivc::GrayscaleByteImg patch =
                        extract_patch(padded, width_idx, height_idx, this->_patch_size);
                    if(this->_vocab.contains(patch))
                    {
                        D(static_cast<Eigen::Index>(row_idx),
                          static_cast<Eigen::Index>(this->_vocab.get_idx(patch))) += 1.0f;
                    }
                }
            }

            ++row_idx;
        }

        return D;
    }


    OVR::OVR()
        : _labels(), _models()
    {}

    ivc::ProbVector OVR::predict(const ivc::FloatDataset& X) const
    {
        if(this->_models.empty())
        {
            throw std::runtime_error("OVR model has not been trained");
        }

        ivc::ProbVector y_pred(X.rows());
        ivc::ProbVector best_scores = ivc::ProbVector::Constant(X.rows(),
                                                                -std::numeric_limits<float_t>::infinity());

        for(size_t model_idx = 0; model_idx < this->_models.size(); ++model_idx)
        {
            const ivc::ProbVector scores = this->_models[model_idx].predict(X);
            for(Eigen::Index row_idx = 0; row_idx < X.rows(); ++row_idx)
            {
                if(scores(row_idx) > best_scores(row_idx))
                {
                    best_scores(row_idx) = scores(row_idx);
                    y_pred(row_idx) = this->_labels[model_idx];
                }
            }
        }

        return y_pred;
    }

    float_t OVR::cost(const ivc::FloatDataset& X,
                      const ivc::ProbVector& y_gt) const
    {
        if(this->_models.empty())
        {
            throw std::runtime_error("OVR model has not been trained");
        }

        if(y_gt.rows() == 0)
        {
            return 0.0f;
        }

        float_t total_cost = 0.0f;
        for(size_t model_idx = 0; model_idx < this->_models.size(); ++model_idx)
        {
            ivc::ProbVector y_binary(y_gt.rows());
            for(Eigen::Index row_idx = 0; row_idx < y_gt.rows(); ++row_idx)
            {
                y_binary(row_idx) = (std::abs(y_gt(row_idx) - this->_labels[model_idx]) <= 1e-6f) ? 1.0f : 0.0f;
            }

            total_cost += this->_models[model_idx].cost(X, y_binary);
        }

        return total_cost / static_cast<float_t>(this->_models.size());
    }

    void OVR::train(const ivc::FloatDataset& X,
                    const ivc::ProbVector& y_gt,
                    const float_t lr,
                    const size_t max_epochs)
    {
        if(X.rows() != y_gt.rows())
        {
            throw std::runtime_error("Dataset and label sizes do not match");
        }

        std::map<float_t, bool> labels;
        for(Eigen::Index row_idx = 0; row_idx < y_gt.rows(); ++row_idx)
        {
            labels[y_gt(row_idx)] = true;
        }

        std::vector<float_t> current_labels;
        current_labels.reserve(labels.size());
        for(const auto& [label, _] : labels)
        {
            current_labels.push_back(label);
        }

        if(this->_models.empty() || this->_labels != current_labels)
        {
            this->_labels = current_labels;
            this->_models.clear();

            for(size_t model_idx = 0; model_idx < this->_labels.size(); ++model_idx)
            {
                this->_models.emplace_back(X.cols());
            }
        }

        for(size_t model_idx = 0; model_idx < this->_labels.size(); ++model_idx)
        {
            ivc::ProbVector y_binary(y_gt.rows());
            for(Eigen::Index row_idx = 0; row_idx < y_gt.rows(); ++row_idx)
            {
                y_binary(row_idx) = (std::abs(y_gt(row_idx) - this->_labels[model_idx]) <= 1e-6f) ? 1.0f : 0.0f;
            }

            this->_models[model_idx].train(X, y_binary, lr, max_epochs);
        }
    }

    ivc::ByteDataset tile_dataset(const ivc::ByteDataset& X,
                                  const size_t level)
    {
        ivc::ByteDataset out;
        for(const ivc::GrayscaleByteImg& img : X)
        {
            append_tiles_recursive(img, level, out);
        }

        return out;
    }

    HistogramOfGradients::HistogramOfGradients()
    {}

    ivc::FloatDataset HistogramOfGradients::transform(const ivc::ByteDataset& X) const
    {
        ivc::FloatDataset D(X.size(), 8);
        D.setZero();

        size_t row_idx = 0;
        for(const ivc::GrayscaleByteImg& img : X)
        {
            const ivc::GrayscaleFloatImg angles = ivc::get_sobel_3x3_gradient_angles(img);

            for(size_t width_idx = 0; width_idx < ivc::get_width(img); ++width_idx)
            {
                for(size_t height_idx = 0; height_idx < ivc::get_height(img); ++height_idx)
                {
                    D(static_cast<Eigen::Index>(row_idx),
                      static_cast<Eigen::Index>(get_angle_bin(get_float_pixel(angles,
                                                                               width_idx,
                                                                               height_idx)))) += 1.0f;
                }
            }

            ++row_idx;
        }

        return D;
    }

    std::tuple<ivc::ByteDataset,
               ivc::ProbVector> balance(const ivc::ByteDataset& X,
                                        const ivc::ProbVector& y_gt,
                                        const ivc::student::balance_type_t balance_type)
    {
        if(X.size() != static_cast<size_t>(y_gt.rows()))
        {
            throw std::runtime_error("Dataset and label sizes do not match");
        }

        if(X.empty())
        {
            return std::make_tuple(X, y_gt);
        }

        std::vector<ivc::GrayscaleByteImg> images(X.begin(), X.end());
        std::vector<float_t> labels(y_gt.rows());
        for(Eigen::Index row_idx = 0; row_idx < y_gt.rows(); ++row_idx)
        {
            labels[static_cast<size_t>(row_idx)] = y_gt(row_idx);
        }

        std::map<float_t, std::vector<size_t> > class_indices;
        for(size_t idx = 0; idx < labels.size(); ++idx)
        {
            class_indices[labels[idx]].push_back(idx);
        }

        size_t max_count = 0;
        size_t min_count = std::numeric_limits<size_t>::max();
        for(const auto& [label, indices] : class_indices)
        {
            max_count = std::max(max_count, indices.size());
            min_count = std::min(min_count, indices.size());
        }

        std::mt19937 rng(585);
        std::vector<ivc::GrayscaleByteImg> balanced_images;
        std::vector<float_t> balanced_labels;

        if(balance_type == ivc::student::OVERSAMPLE)
        {
            balanced_images = images;
            balanced_labels = labels;

            for(const auto& [label, indices] : class_indices)
            {
                size_t count = indices.size();
                size_t pick = 0;
                while(count < max_count)
                {
                    balanced_images.push_back(images[indices[pick % indices.size()]]);
                    balanced_labels.push_back(label);
                    ++count;
                    ++pick;
                }
            }
        }
        else if(balance_type == ivc::student::UNDERSAMPLE)
        {
            for(const auto& [label, indices] : class_indices)
            {
                std::vector<size_t> chosen = indices;
                std::shuffle(chosen.begin(), chosen.end(), rng);
                chosen.resize(min_count);

                for(const size_t idx : chosen)
                {
                    balanced_images.push_back(images[idx]);
                    balanced_labels.push_back(label);
                }
            }
        }
        else
        {
            balanced_images = images;
            balanced_labels = labels;

            for(const auto& [label, indices] : class_indices)
            {
                size_t count = indices.size();
                while(count < max_count)
                {
                    std::uniform_int_distribution<size_t> sample_dist(0, indices.size() - 1);
                    const size_t base_idx = indices[sample_dist(rng)];

                    if(indices.size() == 1)
                    {
                        balanced_images.push_back(images[base_idx]);
                        balanced_labels.push_back(label);
                        ++count;
                        continue;
                    }

                    std::vector<std::pair<double, size_t> > neighbors;
                    for(const size_t idx : indices)
                    {
                        if(idx == base_idx)
                        {
                            continue;
                        }

                        neighbors.push_back(std::make_pair(squared_distance(images[base_idx], images[idx]), idx));
                    }

                    std::sort(neighbors.begin(),
                              neighbors.end(),
                              [](const std::pair<double, size_t>& a,
                                 const std::pair<double, size_t>& b)
                              {
                                  return a.first < b.first;
                              });

                    const size_t num_neighbors = std::min(static_cast<size_t>(5), neighbors.size());
                    std::uniform_int_distribution<size_t> neighbor_dist(0, num_neighbors - 1);
                    balanced_images.push_back(smote_mix(images[base_idx],
                                                        images[neighbors[neighbor_dist(rng)].second],
                                                        rng));
                    balanced_labels.push_back(label);
                    ++count;
                }
            }
        }

        ivc::ByteDataset X_balanced;
        for(const ivc::GrayscaleByteImg& img : balanced_images)
        {
            X_balanced.push_back(img);
        }

        ivc::ProbVector y_balanced(balanced_labels.size());
        for(size_t idx = 0; idx < balanced_labels.size(); ++idx)
        {
            y_balanced(static_cast<Eigen::Index>(idx)) = balanced_labels[idx];
        }

        return std::make_tuple(X_balanced, y_balanced);
    }

} // end of namespace student
} // end of namespace ivc
