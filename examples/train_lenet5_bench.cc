/**************************************************************************//**
 *
 * \file   train_lenet5_bench.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Dec 02 2009
 *
 * $Id: train_lenet5_bench.cc 2868 2009-12-07 20:06:25Z dast $
 *
 *****************************************************************************/

#include "lenet5.hh"
#include "mnistsource.hh"
#include "normdatasrc.hh"
#include "onlinetrainer.hh"
#include "timer.hh"
#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
using namespace cnnplus;

template<typename T, class ErrFnc>
class MyAdapter : public OnlineTrainer<T, ErrFnc>::Adapter
{
public:
    MyAdapter(NeuralNet<T> const & net, DataSource<T> & dsTst, std::string const & filename)
        : OnlineTrainer<T, ErrFnc>::Adapter(25), net_(net), dsTst_(dsTst), timer_(), filename_(filename), file_(filename_.c_str())
    {}
    virtual bool update(OnlineTrainer<T, ErrFnc> & trainer, DataSource<T> & dsTrain, size_t epoch, T & eta)
    {
        if (epoch == 0)
        {
            // Get current date and time
            char strDateTime[256];
            time_t const timer = time(NULL);
            strftime(strDateTime, countof(strDateTime), "Started on %a %d %b %Y at %H:%M:%S.", localtime(&timer));

            // Create header
            std::stringstream ss;
            ss << "#!/usr/bin/python" << std::endl;
            ss << std::endl;
            ss << "import matplotlib.pyplot as plt" << std::endl;
            ss << "import numpy as np" << std::endl;
            ss << std::endl;
            ss << "# " << net_.toString() << std::endl;
            ss << "# [" << net_.numTrainableParam() << " trainable parameters, " << net_.numConnections() << " connections]" << std::endl;
            ss << "# " << trainer.toString() << std::endl;
            ss << "# trn-set: " << dsTrain.toString() << std::endl;
            ss << "# tst-set: " << dsTst_.toString() << std::endl;
            ss << "#" << std::endl;
            ss << "# " << strDateTime << std::endl;
            ss << std::endl;
            ss << "train = np.array([" << std::endl;
            ss << "# ------+--------------+---------------------------+---------------------------+-------------------" << std::endl;
            ss << "# epoch | eta          | train-set                 | test-set                  | elapsed-time [sec]" << std::endl;
            ss << "# ------+--------------+------------+--------------+------------+--------------+-------------------" << std::endl;
            ss << "#       |              | error      | err-rate [%] | error      | err-rate [%] |                   " << std::endl;
            ss << "# ------+--------------+------------+--------------+------------+--------------+-------------------" << std::endl;

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
        double errRateTrn = 0, errRateTst = 0;
        T const errTrn = trainer.test(dsTrain, errRateTrn);
        T const errTst = trainer.test(dsTst_,  errRateTst);

        // Measure elapsed time
        double const elapsed = timer_.toc();

        // Get current time
        static char strTime[256];
        time_t const timer = time(NULL);
        strftime(strTime, countof(strTime), "%H:%M:%S", localtime(&timer));

        // Create result string
        static char strLine[256];
        sprintf(strLine, "[ %5d , %12.10f , %10.8f , %12.4f , %10.8f , %12.4f , %18.2f %s",
            epoch, eta, errTrn, errRateTrn, errTst, errRateTst, elapsed,
            epoch < this->maxEpochs_ ? "]," : "]])");

        // Write result to file
        file_ << strLine << std::endl;
        file_.flush();

        // Write result to console
        std::cout << strLine << std::endl;
        std::cout.flush();

        if (epoch == this->maxEpochs_)
        {
            std::stringstream ss;
            ss << std::endl;
            ss << "fig = plt.figure()" << std::endl;
            ss << std::endl;
            ss << "ax = fig.add_subplot(211)" << std::endl;
            ss << "ax.plot(train[:,0], train[:,2])" << std::endl;
            ss << "ax.plot(train[:,0], train[:,4])" << std::endl;
            ss << "ax.grid(True)" << std::endl;
            ss << "plt.xlabel('Number of training epochs')" << std::endl;
            ss << "plt.ylabel('MSE')" << std::endl;
            ss << "plt.legend(('training-set', 'test-set'), loc='upper right')" << std::endl;
            ss << std::endl;
            ss << "ax = fig.add_subplot(212)" << std::endl;
            ss << "ax.plot(train[:,0], train[:,3])" << std::endl;
            ss << "ax.plot(train[:,0], train[:,5])" << std::endl;
            ss << "ax.grid(True)" << std::endl;
            ss << "plt.xlabel('Number of training epochs')" << std::endl;
            ss << "plt.ylabel('Classification error-rate (%)')" << std::endl;
            ss << "plt.legend(('training-set', 'test-set'), loc='upper right')" << std::endl;
            ss << std::endl;
            ss << "plt.show()" << std::endl;

            file_ << ss.str(); file_.flush();
            std::cout << ss.str(); std::cout.flush();
        }

        return (epoch < this->maxEpochs_);
    }
private:
    NeuralNet<T> const & net_;
    DataSource<T> & dsTst_;
    Timer timer_;
    std::string const filename_;
    std::ofstream file_;
};

int main(int argc, char *argv[])
{
    if (argc < 5) {
        printf("Usage: %s <train-img-file> <train-lbl-file> <test-img-file> <test-lbl-file> [log-file]\n", argv[0]);
        return -1;
    }

    typedef float DataType;
    typedef MeanSquaredError<DataType> ErrFnc;

    try {
        // Load the MNIST datasets
        MnistMemSource<DataType> mnistTrn(Size(32, 32));
        MnistMemSource<DataType> mnistTst(Size(32, 32));
        mnistTrn.load(argv[1], argv[2]);
        mnistTst.load(argv[3], argv[4]);

        // Initialize the network
        LeNet5<DataType> net;
        net.forget();
        //net.save("lenet5.mat"); return 0;
        //net.load("lenet5.mat");

#if 0   // Normalize the datasets
        NormDataSrc<DataType> nrmMnistTrn(mnistTrn);
        NormDataSrc<DataType> nrmMnistTst(mnistTst, nrmMnistTrn);
#endif
        // Start training
        MyAdapter<DataType, ErrFnc> adapter(net, mnistTst, argc >= 6 ? argv[5] : "train.py");
        OnlineTrainer<DataType, ErrFnc> trainer(net, 0.001f, adapter);
        trainer.train(mnistTrn);
    }
    catch (std::exception const & ex) {
        std::cerr << ex.what() << std::endl;
        return -1;
    }

    return 0;
}
