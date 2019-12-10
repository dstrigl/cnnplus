/**************************************************************************//**
 *
 * \file   train_phungnet.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jul 15 2009
 *
 * $Id: train_phungnet.cc 2868 2009-12-07 20:06:25Z dast $
 *
 *****************************************************************************/

#include "phungnet.hh"
#include "matdatasrc.hh"
#include "subsetdatasrc.hh"
#include "rproptrainer.hh"
#include "timer.hh"
#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <fstream>
using namespace cnnplus;

template<typename T, class ErrFnc>
class MyAdapter : public RpropTrainer<T, ErrFnc>::Adapter
{
public:
    MyAdapter(NeuralNet<T> const & net, DataSource<T> & dsVal, DataSource<T> & dsTst, std::string const & filename)
        : RpropTrainer<T, ErrFnc>::Adapter(1000), net_(net), dsVal_(dsVal), dsTst_(dsTst), timer_(), filename_(filename), file_(filename_.c_str())
    {}
    virtual bool update(RpropTrainer<T, ErrFnc> & trainer, DataSource<T> & dsTrain, size_t epoch)
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
            ss << "# val-set: " << dsVal_.toString() << std::endl;
            ss << "# tst-set: " << dsTst_.toString() << std::endl;
            ss << "#" << std::endl;
            ss << "# " << strDateTime << std::endl;
            ss << std::endl;
            ss << "train = np.array([" << std::endl;
            ss << "# ------+---------------------------+---------------------------+---------------------------+-------------------" << std::endl;
            ss << "# epoch | train-set                 | validation-set            | test-set                  | elapsed-time [sec]" << std::endl;
            ss << "# ------+------------+--------------+------------+--------------+------------+--------------+-------------------" << std::endl;
            ss << "#       | error      | err-rate [%] | error      | err-rate [%] | error      | err-rate [%] |                   " << std::endl;
            ss << "# ------+------------+--------------+------------+--------------+------------+--------------+-------------------" << std::endl;

            // Write header to file
            file_ << ss.str();
            file_.flush();

            // Write header to console
            std::cout << ss.str();
            std::cout.flush();

            // Start timer
            timer_.tic();
        }

#if 0   // Save neural-net data
        static char filename[256];
        sprintf(filename, "epoch%03d.mat", epoch);
        net_.save(filename);
#endif
        // Run the tests
        double errRateTrn = 0, errRateVal = 0, errRateTst = 0;
        T const errTrn = trainer.test(dsTrain, errRateTrn);
        T const errVal = trainer.test(dsVal_,  errRateVal);
        T const errTst = trainer.test(dsTst_,  errRateTst);

        // Measure elapsed time
        double const elapsed = timer_.toc();

        // Get current time
        static char strTime[256];
        time_t const timer = time(NULL);
        strftime(strTime, countof(strTime), "%H:%M:%S", localtime(&timer));

        // Create result string
        static char strLine[256];
        sprintf(strLine, "[ %5d , %10.8f , %12.4f , %10.8f , %12.4f , %10.8f , %12.4f , %18.2f %s",
            epoch, errTrn, errRateTrn, errVal, errRateVal, errTst, errRateTst, elapsed,
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
            ss << "ax.plot(train[:,0], train[:,1])" << std::endl;
            ss << "ax.plot(train[:,0], train[:,3])" << std::endl;
            ss << "ax.plot(train[:,0], train[:,5])" << std::endl;
            ss << "ax.grid(True)" << std::endl;
            ss << "plt.xlabel('Number of training epochs')" << std::endl;
            ss << "plt.ylabel('MSE')" << std::endl;
            ss << "plt.legend(('training-set', 'validation-set', 'test-set'), loc='upper right')" << std::endl;
            ss << std::endl;
            ss << "ax = fig.add_subplot(212)" << std::endl;
            ss << "ax.plot(train[:,0], train[:,2])" << std::endl;
            ss << "ax.plot(train[:,0], train[:,4])" << std::endl;
            ss << "ax.plot(train[:,0], train[:,6])" << std::endl;
            ss << "ax.grid(True)" << std::endl;
            ss << "plt.xlabel('Number of training epochs')" << std::endl;
            ss << "plt.ylabel('Classification error-rate (%)')" << std::endl;
            ss << "plt.legend(('training-set', 'validation-set', 'test-set'), loc='upper right')" << std::endl;
            ss << std::endl;
            ss << "plt.show()" << std::endl;

            file_ << ss.str(); file_.flush();
            std::cout << ss.str(); std::cout.flush();
        }

        return (epoch < this->maxEpochs_);
    }
private:
    NeuralNet<T> const & net_;
    DataSource<T> & dsVal_;
    DataSource<T> & dsTst_;
    Timer timer_;
    std::string const filename_;
    std::ofstream file_;
};

int main(int argc, char *argv[])
{
    if (argc < 3) {
        printf("Usage: %s <train-mat-file> <test-mat-file> [log-file]\n", argv[0]);
        return -1;
    }

    typedef float DataType;
    typedef MeanSquaredError<DataType> ErrFnc;

    try {
        // Load the face datasets
        MatDataSrc<DataType> facesTrain(-1, 1);
        MatDataSrc<DataType> facesTst(-1, 1);
        facesTrain.load(argv[1], "x", "d");
        facesTst.load(argv[2], "x_test", "d_test");

        // Use 200 samples (10%) of the training-set for validation
        SubsetDataSrc<DataType> facesTrn(facesTrain,    0, 1800);
        SubsetDataSrc<DataType> facesVal(facesTrain, 1800,  200);

        // Initialize the network
        PhungNet<DataType> net;
        net.forget();

        // Start training
        MyAdapter<DataType, ErrFnc> adapter(net, facesVal, facesTst, argc >= 4 ? argv[3] : "train.py");
        RpropTrainer<DataType, ErrFnc> trainer(net, 1.01f, 0.99f, 0.01f, 0.0f, 10.0f, adapter);
        trainer.train(facesTrn);
    }
    catch (std::exception const & ex) {
        std::cerr << ex.what() << std::endl;
        return -1;
    }

    return 0;
}
