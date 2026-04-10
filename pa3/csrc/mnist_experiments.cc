// SYSTEM INCLUDES
#include <algorithm>
#include <cstdlib>
#include <dirent.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <585/io/io.h>


// C++ PROJECT INCLUDES
#include "vbow/vbow.h"


namespace
{
    struct Config
    {
        std::string mnist_root;
        std::string descriptor = "binary";
        std::string balance = "none";
        size_t patch_size = 1;
        size_t tile_level = 0;
        float_t lr = 0.1f;
        size_t epochs = 100;
    };

    struct Split
    {
        ivc::ByteDataset X;
        ivc::ProbVector y;
    };

    void usage()
    {
        std::cout << "Usage: ./hough-vbow-mnist --mnist-root <dir> [options]\n"
                  << "  --descriptor <binary|counting|hog>\n"
                  << "  --patch-size <odd int>\n"
                  << "  --tile-level <0|1|2>\n"
                  << "  --lr <float>\n"
                  << "  --epochs <int>\n"
                  << "  --balance <none|oversample|undersample|smote>\n";
    }

    std::vector<std::string> list_pngs(const std::string& dirpath)
    {
        DIR* dir = opendir(dirpath.c_str());
        if(dir == nullptr)
        {
            throw std::runtime_error("Could not open " + dirpath);
        }

        std::vector<std::string> files;
        for(dirent* entry = readdir(dir); entry != nullptr; entry = readdir(dir))
        {
            const std::string name(entry->d_name);
            if(name == "." || name == "..")
            {
                continue;
            }

            if(name.size() >= 4 && name.substr(name.size() - 4) == ".png")
            {
                files.push_back(dirpath + "/" + name);
            }
        }

        closedir(dir);
        std::sort(files.begin(), files.end());
        return files;
    }

    Split load_split(const std::string& root,
                     const std::string& split_name)
    {
        ivc::ByteDataset X;
        std::vector<float_t> labels;

        for(int digit = 0; digit <= 9; ++digit)
        {
            const std::string class_dir = root + "/" + split_name + "/" + std::to_string(digit);
            for(const std::string& path : list_pngs(class_dir))
            {
                X.push_back(ivc::imread_grayscale(path));
                labels.push_back(static_cast<float_t>(digit));
            }
        }

        ivc::ProbVector y(labels.size());
        for(size_t idx = 0; idx < labels.size(); ++idx)
        {
            y(static_cast<Eigen::Index>(idx)) = labels[idx];
        }

        return Split{X, y};
    }

    size_t num_tiles(const size_t level)
    {
        size_t total = 1;
        for(size_t idx = 0; idx < level; ++idx)
        {
            total *= 4;
        }

        return total;
    }

    ivc::FloatDataset combine_tiles(const ivc::FloatDataset& tiled,
                                    const size_t tiles_per_image)
    {
        if(tiles_per_image == 1)
        {
            return tiled;
        }

        const Eigen::Index rows = tiled.rows() / static_cast<Eigen::Index>(tiles_per_image);
        const Eigen::Index cols = tiled.cols() * static_cast<Eigen::Index>(tiles_per_image);
        ivc::FloatDataset out(rows, cols);
        out.setZero();

        for(Eigen::Index row_idx = 0; row_idx < rows; ++row_idx)
        {
            for(size_t tile_idx = 0; tile_idx < tiles_per_image; ++tile_idx)
            {
                out.block(row_idx,
                          static_cast<Eigen::Index>(tile_idx) * tiled.cols(),
                          1,
                          tiled.cols()) =
                    tiled.row(row_idx * static_cast<Eigen::Index>(tiles_per_image) +
                              static_cast<Eigen::Index>(tile_idx));
            }
        }

        return out;
    }

    void make_features(const ivc::ByteDataset& train_X,
                       const ivc::ByteDataset& test_X,
                       const Config& config,
                       ivc::FloatDataset& X_train,
                       ivc::FloatDataset& X_test)
    {
        if(config.descriptor == "binary")
        {
            ivc::student::BinaryBagOfWords bow(train_X, config.patch_size);
            X_train = bow.transform(train_X);
            X_test = bow.transform(test_X);
            return;
        }

        if(config.descriptor == "counting")
        {
            ivc::student::CountingBagOfWords bow(train_X, config.patch_size);
            X_train = bow.transform(train_X);
            X_test = bow.transform(test_X);
            return;
        }

        if(config.descriptor == "hog")
        {
            const size_t tiles = num_tiles(config.tile_level);
            ivc::student::HistogramOfGradients hog;
            X_train = combine_tiles(hog.transform(ivc::student::tile_dataset(train_X, config.tile_level)),
                                    tiles);
            X_test = combine_tiles(hog.transform(ivc::student::tile_dataset(test_X, config.tile_level)),
                                   tiles);
            return;
        }

        throw std::runtime_error("Unknown descriptor");
    }

    Config parse_args(const int argc,
                      char** argv)
    {
        Config config;

        for(int idx = 1; idx < argc; ++idx)
        {
            const std::string arg(argv[idx]);
            if(arg == "--help")
            {
                usage();
                std::exit(0);
            }

            if(idx + 1 >= argc)
            {
                throw std::runtime_error("Missing value for " + arg);
            }

            const std::string value(argv[++idx]);
            if(arg == "--mnist-root")
            {
                config.mnist_root = value;
            }
            else if(arg == "--descriptor")
            {
                config.descriptor = value;
            }
            else if(arg == "--patch-size")
            {
                config.patch_size = static_cast<size_t>(std::stoul(value));
            }
            else if(arg == "--tile-level")
            {
                config.tile_level = static_cast<size_t>(std::stoul(value));
            }
            else if(arg == "--lr")
            {
                config.lr = std::stof(value);
            }
            else if(arg == "--epochs")
            {
                config.epochs = static_cast<size_t>(std::stoul(value));
            }
            else if(arg == "--balance")
            {
                config.balance = value;
            }
            else
            {
                throw std::runtime_error("Unknown option " + arg);
            }
        }

        if(config.mnist_root.empty())
        {
            throw std::runtime_error("--mnist-root is required");
        }

        if(config.descriptor != "binary" &&
           config.descriptor != "counting" &&
           config.descriptor != "hog")
        {
            throw std::runtime_error("Descriptor must be binary, counting, or hog");
        }

        if(config.balance != "none" &&
           config.balance != "oversample" &&
           config.balance != "undersample" &&
           config.balance != "smote")
        {
            throw std::runtime_error("Balance must be none, oversample, undersample, or smote");
        }

        return config;
    }

    ivc::student::balance_type_t get_balance_type(const std::string& name)
    {
        if(name == "oversample")
        {
            return ivc::student::OVERSAMPLE;
        }

        if(name == "undersample")
        {
            return ivc::student::UNDERSAMPLE;
        }

        if(name != "smote")
        {
            throw std::runtime_error("Unknown balance mode");
        }

        return ivc::student::SMOTE;
    }
}


int main(const int argc,
         char** argv)
{
    try
    {
        const Config config = parse_args(argc, argv);
        Split train = load_split(config.mnist_root, "training");
        Split test = load_split(config.mnist_root, "testing");

        if(config.balance != "none")
        {
            std::tie(train.X, train.y) =
                ivc::student::balance(train.X, train.y, get_balance_type(config.balance));
        }

        ivc::FloatDataset X_train;
        ivc::FloatDataset X_test;
        make_features(train.X, test.X, config, X_train, X_test);

        ivc::student::OVR model;
        model.train(X_train, train.y, config.lr, config.epochs);

        const float_t train_error = model.cost(X_train, train.y);
        const float_t test_error = model.cost(X_test, test.y);

        std::cout << "descriptor=" << config.descriptor << "\n";
        std::cout << "patch_size=" << config.patch_size << "\n";
        std::cout << "tile_level=" << config.tile_level << "\n";
        std::cout << "balance=" << config.balance << "\n";
        std::cout << "learning_rate=" << config.lr << "\n";
        std::cout << "epochs=" << config.epochs << "\n";
        std::cout << "train_accuracy=" << (1.0f - train_error) << "\n";
        std::cout << "test_accuracy=" << (1.0f - test_error) << "\n";
        std::cout << "feature_dim=" << X_train.cols() << "\n";
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << "\n";
        return 1;
    }

    return 0;
}
