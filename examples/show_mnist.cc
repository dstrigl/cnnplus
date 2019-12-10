/**************************************************************************//**
 *
 * \file   show_mnist.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jun 01 2009
 *
 * $Id: show_mnist.cc 1411 2009-06-04 12:47:36Z dast $
 *
 *****************************************************************************/

#include "mnistsource.hh"
#include "datasrcview.hh"
#include <iostream>
using namespace cnnplus;

int main(int argc, char *argv[])
{
    if (argc < 3) {
        printf("Usage: %s <mnist-img-file> <mnist-lbl-file> [delay]\n", argv[0]);
        return -1;
    }

    try {
        MnistMemSource<double> mnist;
        mnist.load(argv[1], argv[2]);
        DataSrcView view(10, argc >= 4 ? atoi(argv[3]) : 0);
        view.run(mnist);
    }
    catch (std::exception const & ex) {
        std::cerr << ex.what() << std::endl;
        return -1;
    }

    return 0;
}
