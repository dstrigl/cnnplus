/**************************************************************************//**
 *
 * \file   train_facefindernet.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jul 17 2009
 *
 * $Id: train_facefindernet.cc 1940 2009-08-04 19:44:22Z dast $
 *
 *****************************************************************************/

#include "facefindernet.hh"
#include "matdatasrc.hh"
#include "onlinetrainer.hh"
#include "timer.hh"
#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <fstream>
using namespace cnnplus;

template<typename T, class ErrFnc>
class MyAdapter : public OnlineTrainer<T, ErrFnc>::Adapter
{
public:
    MyAdapter(NeuralNet<T> const & net, DataSource<T> & dsTst)
        : OnlineTrainer<T, ErrFnc>::Adapter(1000), net_(net), dsTst_(dsTst), timer_(), filename_("train.txt"), file_(filename_.c_str())
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
            ss << net_.toString() << std::endl;
            ss << "[" << net_.numTrainableParam() << " trainable parameters, " << net_.numConnections() << " connections]" << std::endl;
            ss << trainer.toString() << std::endl;
            ss << "trn-set: " << dsTrain.toString() << std::endl;
            ss << "tst-set: " << dsTst_.toString() << std::endl;
            ss << std::endl;
            ss << strDateTime << std::endl;
            ss << std::endl;
            ss << "------+--------------+---------------------------+---------------------------+----------------------" << std::endl;
            ss << "epoch | eta          | train-set                 | test-set                  |    elapsed-time [sec]" << std::endl;
            ss << "------+--------------+------------+--------------+------------+--------------+----------------------" << std::endl;
            ss << "      |              | error      | err-rate [%] | error      | err-rate [%] |                      " << std::endl;
            ss << "------+--------------+------------+--------------+------------+--------------+----------------------" << std::endl;

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
        sprintf(strLine, "%5d | %12.10f | %10.8f | %12.4f | %10.8f | %12.4f | %10.2f [%s]",
            epoch, eta, errTrn, errRateTrn, errTst, errRateTst, elapsed, strTime);

        // Write result to file
        file_ << strLine << std::endl;
        file_.flush();

        // Write result to console
        std::cout << strLine << std::endl;
        std::cout.flush();

#if 1   // Update learning-rate
        if (epoch > 0 && epoch % 20 == 0)
            eta *= T(0.9);
#endif

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
    if (argc < 3) {
        printf("Usage: %s <train-mat-file> <test-mat-file> [data-range-min] [data-range-max]\n", argv[0]);
        return -1;
    }

    typedef float DataType;
    typedef MeanSquaredError<DataType> ErrFnc;

    try {
        // Load the face datasets
        MatDataSrc<DataType> facesTrn(DataType(argc >= 4 ? atof(argv[3]) : -1), DataType(argc >= 5 ? atof(argv[4]) : 1));
        MatDataSrc<DataType> facesTst(DataType(argc >= 4 ? atof(argv[3]) : -1), DataType(argc >= 5 ? atof(argv[4]) : 1));
        facesTrn.load(argv[1], "x",      "d"     );
        facesTst.load(argv[2], "x_test", "d_test");

        // Initialize the network
        FaceFinderNet<DataType> net;
        net.forget();

        // Start training
        MyAdapter<DataType, ErrFnc> adapter(net, facesTst);
        OnlineTrainer<DataType, ErrFnc> trainer(net, 0.003f, adapter);
        trainer.train(facesTrn);
    }
    catch (std::exception const & ex) {
        std::cerr << ex.what() << std::endl;
        return -1;
    }

    return 0;
}
