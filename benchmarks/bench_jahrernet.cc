/**************************************************************************//**
 *
 * \file   bench_jahrernet.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jun 30 2009
 *
 * $Id: bench_jahrernet.cc 1722 2009-07-12 19:08:32Z dast $
 *
 *****************************************************************************/

#include <iostream>
#include "jahrernet.hh"
#ifdef CNNPLUS_USE_CUDA
#include "cujahrernet.hh"
#endif // CNNPLUS_USE_CUDA
#include "propsbench.hh"
using namespace cnnplus;

int main(int argc, char *argv[])
{
    JahrerNet<float> net;
    double const t1 = PropsBench().run(net);
    std::cout << net.toString() << ": " << t1 << " sec" << std::endl;
#ifdef CNNPLUS_USE_CUDA
    CuJahrerNet<float> cuNet;
    double const t2 = PropsBench().run(cuNet);
    std::cout << cuNet.toString() << ": " << t2 << " sec" << std::endl;
    std::cout << "CPU/GPU: " << (t1 / t2) << std::endl << std::endl;
#endif // CNNPLUS_USE_CUDA

    return 0;
}
