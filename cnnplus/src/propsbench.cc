/**************************************************************************//**
 *
 * \file   propsbench.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jan 04 2009
 *
 * $Id: propsbench.cc 2103 2009-08-19 17:49:54Z dast $
 *
 * \brief  Implementation of cnnplus::PropsBench.
 *
 *****************************************************************************/

#include "propsbench.hh"
#include "neuralnet.hh"
#include "matvecli.hh"
#include "timer.hh"

CNNPLUS_NS_BEGIN

template<typename T> double
PropsBench::run(NeuralNet<T> & net, bool const withUpdate)
{
    size_t const sizeIn  = net.sizeIn();
    size_t const sizeOut = net.sizeOut();

    // Allocate memory
    T * in  = matvecli::allocv<T>(sizeIn);
    T * out = matvecli::allocv<T>(sizeOut);

    // Initialize input with random values
    matvecli::randv<T>(in, sizeIn);

    // Initialize weights with random values
    net.forget();

    // Measure time for a number of fwd- and back-props
    Timer tm;
    if (withUpdate) {
        for (size_t i = 0; i < numRuns_; ++i) {
            net.fprop(in, out);
            net.bprop(in, out);
            net.update(T(0.001));
        }
    }
    else {
        for (size_t i = 0; i < numRuns_; ++i) {
            net.fprop(in, out);
            net.bprop(in, out);
        }
    }
    tm.toc();

    // Deallocate memory
    matvecli::free<T>(in);
    matvecli::free<T>(out);

    return tm.elapsed();
}

/*! \addtogroup eti_grp Explicit Template Instantiation
 @{
 */
template double PropsBench::run<float>(NeuralNet<float> & net, bool withUpdate);
template double PropsBench::run<double>(NeuralNet<double> & net, bool withUpdate);
/*! @} */

CNNPLUS_NS_END
