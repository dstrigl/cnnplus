/**************************************************************************//**
 *
 * \file   show_matsrc.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jul 11 2009
 *
 * $Id: show_matsrc.cc 1940 2009-08-04 19:44:22Z dast $
 *
 *****************************************************************************/

#include "matdatasrc.hh"
#include "datasrcview.hh"
#include <iostream>
using namespace cnnplus;

int main(int argc, char *argv[])
{
    if (argc < 4) {
        printf(
            "Usage: %s <mat-file> <pat-arr-name> <lab-arr-name> [data-range-min] [data-range-max] [delay]\n",
            argv[0]);
        return -1;
    }

    try {
        MatDataSrc<double> matsrc(
            argc >= 5 ? atof(argv[4]) : -1, argc >= 6 ? atof(argv[5]) : 1);
        matsrc.load(argv[1], argv[2], argv[3]);
        matsrc.shuffle();
        DataSrcView view(10, argc >= 7 ? atoi(argv[6]) : 0);
        view.run(matsrc);
    }
    catch (std::exception const & ex) {
        std::cerr << ex.what() << std::endl;
        return -1;
    }

    return 0;
}
