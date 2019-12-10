/**************************************************************************//**
 *
 * \file   bench_lenet5.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jun 30 2009
 *
 * $Id: bench_lenet5.cc 2461 2009-10-30 13:14:15Z dast $
 *
 *****************************************************************************/

#include <iostream>
#include <iomanip>
#include "lenet5.hh"
#ifdef CNNPLUS_USE_CUDA
#include "culenet5.hh"
#endif // CNNPLUS_USE_CUDA
#include "propsbench.hh"
using namespace cnnplus;

template<typename T>
void run(Size const & sizeImgIn = Size(32, 32), size_t const numRuns = 1000)
{
    LeNet5<T> net(sizeImgIn);

    double t1 = 0;
    for (size_t i = 0; i < 3; ++i)
        t1 += PropsBench(numRuns).run(net, true);
    t1 /= 3;

    //std::cout << net.toString() << ": " << t1 << " sec" << std::endl;
    //std::cout << "CPU " << sizeImgIn.toString() << ": " << t1 << " sec" << std::endl;
    std::cout << std::setw(10) << std::setprecision(5) << t1;

#ifdef CNNPLUS_USE_CUDA
    CuLeNet5<T> cuNet(sizeImgIn);

    double t2 = 0;
    for (size_t i = 0; i < 3; ++i)
        t2 += PropsBench(numRuns).run(cuNet, true);
    t2 /= 3;

    //std::cout << cuNet.toString() << ": " << t2 << " sec" << std::endl;
    //std::cout << "GPU " << sizeImgIn.toString() << ": " << t2 << " sec" << std::endl;
    //std::cout << "CPU/GPU: " << (t1 / t2) << std::endl << std::endl;
    std::cout << "\t" << std::setw(10) << std::setprecision(5) << t2;
    std::cout << "\t" << std::setw(10) << std::setprecision(5) << (t1/t2);
#endif // CNNPLUS_USE_CUDA

    std::cout << std::endl;
}

int main(int argc, char *argv[])
{
    typedef float DataType;
#if 1
    for (size_t i = 0; i < 10; ++i)
        run<DataType>(Size(32 + i * 8, 32 + i * 8));
#else
    run<DataType>(Size(32, 32));
    run<DataType>(Size(44, 44));
    run<DataType>(Size(56, 56));
    run<DataType>(Size(64, 64));
#endif
    return 0;
}
