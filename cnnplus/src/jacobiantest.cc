/**************************************************************************//**
 *
 * \file   jacobiantest.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jan 02 2009
 *
 * $Id: jacobiantest.cc 1555 2009-06-18 09:29:17Z dast $
 *
 * \brief  Implementation of cnnplus::JacobianTest.
 *
 *****************************************************************************/

#include "jacobiantest.hh"
#include "neuralnet.hh"
#include "error.hh"
#include "matvecli.hh"

CNNPLUS_NS_BEGIN

template<>
JacobianTest<float>::JacobianTest()
    : accThresh_(1.0e-005f), epsilon_(1.0e-001f), err_(0)
{}

template<>
JacobianTest<double>::JacobianTest()
    : accThresh_(1.0e-011), epsilon_(1.0e-004), err_(0)
{}

template<typename T>
JacobianTest<T>::JacobianTest(T const accThresh, T const epsilon)
    : accThresh_(accThresh), epsilon_(epsilon), err_(0)
{
    if (accThresh < 0)
        throw ParameterError("accThresh", "must be greater or equal to zero.");
    else if (epsilon < 0)
        throw ParameterError("epsilon", "must be greater or equal to zero.");
}

template<typename T> bool
JacobianTest<T>::run(NeuralNet<T> & net)
{
    size_t const sizeIn  = net.sizeIn();
    size_t const sizeOut = net.sizeOut();

    //
    // Allocate memory
    //
    T * j_fprop = matvecli::allocv<T>(sizeOut * sizeIn);
    T * j_bprop = matvecli::allocv<T>(sizeOut * sizeIn);

    T * in  = matvecli::allocv<T>(sizeIn);
    T * out = matvecli::allocv<T>(sizeOut);

    //
    // Initialize input with random values
    //
    matvecli::randv<T>(in, sizeIn);

    //
    // Initialize weights with random values
    //
    net.forget();

    //
    // Compute Jacobian Matrix via perturbation
    //
    for (size_t i = 0; i < sizeIn; ++i)
    {
        T const tmp = in[i];

        in[i] = tmp + epsilon_;
        net.fprop(in, out);

        for (size_t j = 0; j < sizeOut; ++j)
            j_fprop[j * sizeIn + i] = out[j];

        in[i] = tmp - epsilon_;
        net.fprop(in, out);

        for (size_t j = 0; j < sizeOut; ++j) {
            j_fprop[j * sizeIn + i] -= out[j];
            j_fprop[j * sizeIn + i] /= 2 * epsilon_;
        }

        in[i] = tmp;
    }

    //
    // Compute Jacobian Matrix via back-propagation
    //
    net.fprop(in, out);

    for (size_t j = 0; j < sizeOut; ++j)
    {
        matvecli::zerov<T>(out, sizeOut);
        out[j] = 1;

        net.bprop(j_bprop + j * sizeIn, out);
    }

    matvecli::free<T>(in);
    matvecli::free<T>(out);

    //
    // Compute error
    //
    matvecli::subv<T>(j_bprop, j_fprop, sizeOut * sizeIn);
    err_= matvecli::absmaxv<T>(j_bprop, sizeOut * sizeIn);

    matvecli::free<T>(j_fprop);
    matvecli::free<T>(j_bprop);

    return (err_ <= accThresh_);
}

/*! \addtogroup eti_grp Explicit Template Instantiation
 @{
 */
template class JacobianTest<float>;
template class JacobianTest<double>;
/*! @} */

CNNPLUS_NS_END
