/**************************************************************************//**
 *
 * \file   test_precision.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Nov 02 2009
 *
 * $Id: test_precision.cc 2708 2009-11-26 14:59:37Z dast $
 *
 *****************************************************************************/

#include "lenet5.hh"
#ifdef CNNPLUS_USE_CUDA
#include "culenet5.hh"
#endif // CNNPLUS_USE_CUDA
#include "mnistsource.hh"
#include "subsetdatasrc.hh"
#include "batchtrainer.hh"
#include "timer.hh"
#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
using namespace cnnplus;

template<typename T>
class MyAdapter : public BatchTrainer<T>::Adapter
{
public:
    MyAdapter(NeuralNet<T> const & net, std::string const & filename) : BatchTrainer<T>::Adapter(1000), net_(net), file_(filename.c_str())
    {}
    virtual bool update(BatchTrainer<T> & trainer, DataSource<T> & dsTrain, size_t epoch, int &, T & eta)
    {
        if (epoch == 0)
        {
            // Get current date and time
            char strDateTime[256];
            time_t const timer = time(NULL);
            strftime(strDateTime, countof(strDateTime), "Started on %a %d %b %Y at %H:%M:%S.", localtime(&timer));

            // Create header
            std::stringstream ss;
            ss << net_.toString() << std::endl;
            ss << "[" << net_.numTrainableParam() << " trainable parameters, " << net_.numConnections() << " connections]" << std::endl;
            ss << trainer.toString() << std::endl;
            ss << dsTrain.toString() << std::endl;
            ss << std::endl;
            ss << strDateTime << std::endl;
            ss << std::endl;
            ss << "------+--------------+-----------------------------+----------------------" << std::endl;
            ss << "epoch | eta          | train-set                   |    elapsed-time [sec]" << std::endl;
            ss << "------+--------------+--------------+--------------+----------------------" << std::endl;
            ss << "      |              | error        | err-rate [%] |                      " << std::endl;
            ss << "------+--------------+--------------+--------------+----------------------" << std::endl;

            // Write header to file
            file_ << ss.str();
            file_.flush();

            // Write header to console
            std::cout << ss.str();
            std::cout.flush();

            // Start timer
            timer_.tic();
        }

        // Run the tests
        double errRateTrn = 0;
        T const errTrn = trainer.test(dsTrain, errRateTrn);

        // Measure elapsed time
        double const elapsed = timer_.toc();

        // Get current time
        static char strTime[256];
        time_t const timer = time(NULL);
        strftime(strTime, countof(strTime), "%H:%M:%S", localtime(&timer));

        // Create result string
        static char strLine[256];
        sprintf(strLine, "%5d | %12.10f | %12.10f | %12.4f | %10.2f [%s]", epoch, eta, errTrn, errRateTrn, elapsed, strTime);

        // Write result to file
        file_ << strLine << std::endl;
        file_.flush();

        // Write result to console
        std::cout << strLine << std::endl;
        std::cout.flush();

#if 1   // Update learning-rate
        if (epoch > 0 && epoch % 100 == 0)
            eta *= T(0.8);
#endif
        return (epoch < this->maxEpochs_);
    }
private:
    NeuralNet<T> const & net_;
    std::ofstream file_;
    Timer timer_;
};

int main(int argc, char *argv[])
{
    if (argc < 3) {
        printf("Usage: %s <train-img-file> <train-lbl-file>\n", argv[0]);
        return -1;
    }
    else {
        LeNet5<float> net;
        try {
            net.load("lenet5.mat");
        }
        catch (...) {
            net.forget();
            net.save("lenet5.mat");
        }
    }

    float const eta = 1e-5f;

    // ******************** CPU SP ********************
    try {
        // Initialize the network
        LeNet5<float> net;
        net.load("lenet5.mat");

        // Load the MNIST datasets
        MnistMemSource<float> mnistTrain(Size(32, 32));
        mnistTrain.load(argv[1], argv[2]);

        // Use only 1k of the training-set
        SubsetDataSrc<float> mnistTrn(mnistTrain, 0, 1000);

        // Start training
        MyAdapter<float> adapter(net, "lenet5-cpu-sp.txt");
        BatchTrainer<float> trainer(net, eta, adapter);
        trainer.train(mnistTrn);
    }
    catch (std::exception const & ex) {
        std::cerr << ex.what() << std::endl;
        return -1;
    }

    // ******************** CPU DP ********************
    try {
        // Initialize the network
        LeNet5<double> net;
        net.load("lenet5.mat");

        // Load the MNIST datasets
        MnistMemSource<double> mnistTrain(Size(32, 32));
        mnistTrain.load(argv[1], argv[2]);

        // Use only 1k of the training-set
        SubsetDataSrc<double> mnistTrn(mnistTrain, 0, 1000);

        // Start training
        MyAdapter<double> adapter(net, "lenet5-cpu-dp.txt");
        BatchTrainer<double> trainer(net, eta, adapter);
        trainer.train(mnistTrn);
    }
    catch (std::exception const & ex) {
        std::cerr << ex.what() << std::endl;
        return -1;
    }

#ifdef CNNPLUS_USE_CUDA
    // ******************** GPU SP ********************
    try {
        // Initialize the network
        CuLeNet5<float> net;
        net.load("lenet5.mat");

        // Load the MNIST datasets
        MnistMemSource<float> mnistTrain(Size(32, 32));
        mnistTrain.load(argv[1], argv[2]);

        // Use only 1k of the training-set
        SubsetDataSrc<float> mnistTrn(mnistTrain, 0, 1000);

        // Start training
        MyAdapter<float> adapter(net, "lenet5-gpu-sp.txt");
        BatchTrainer<float> trainer(net, eta, adapter);
        trainer.train(mnistTrn);
    }
    catch (std::exception const & ex) {
        std::cerr << ex.what() << std::endl;
        return -1;
    }
#endif // CNNPLUS_USE_CUDA

    return 0;
}
