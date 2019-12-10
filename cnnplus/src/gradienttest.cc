/**************************************************************************//**
 *
 * \file   gradienttest.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jan 25 2009
 *
 * $Id: gradienttest.cc 1555 2009-06-18 09:29:17Z dast $
 *
 * \brief  Implementation of cnnplus::GradientTest.
 *
 *****************************************************************************/

#include "gradienttest.hh"
#include "neuralnet.hh"
#include "error.hh"
#include "matvecli.hh"
#include "mathli.hh"

CNNPLUS_NS_BEGIN

template<>
GradientTest< float, MeanSquaredError<float> >::GradientTest()
    : accThresh_(1.0e-003f), epsilon_(1.0e-002f), err_(0)
{}

template<>
GradientTest< float, CrossEntropy<float> >::GradientTest()
    : accThresh_(1.0e-003f), epsilon_(1.0e-002f), err_(0)
{}

template<>
GradientTest< double, MeanSquaredError<double> >::GradientTest()
    : accThresh_(1.0e-008), epsilon_(1.0e-005), err_(0)
{}

template<>
GradientTest< double, CrossEntropy<double> >::GradientTest()
    : accThresh_(1.0e-008), epsilon_(1.0e-005), err_(0)
{}

template<typename T, class ErrFnc>
GradientTest<T, ErrFnc>::GradientTest(T const accThresh, T const epsilon)
    : accThresh_(accThresh), epsilon_(epsilon), err_(0)
{
    if (accThresh < 0)
        throw ParameterError("accThresh", "must be greater or equal to zero.");
    else if (epsilon < 0)
        throw ParameterError("epsilon", "must be greater or equal to zero.");
}

template<> void
GradientTest< float, MeanSquaredError<float> >::initDesired(float * des, size_t const size)
{
    matvecli::randv<float>(des, size, -1, 1);
}

template<> void
GradientTest< double, MeanSquaredError<double> >::initDesired(double * des, size_t const size)
{
    matvecli::randv<double>(des, size, -1, 1);
}

template<> void
GradientTest< float, CrossEntropy<float> >::initDesired(float * des, size_t const size)
{
    matvecli::randv<float>(des, size, 0, 1);
    float const sum = matvecli::sumv<float>(des, size);
    matvecli::divvc<float>(des, sum, size);
}

template<> void
GradientTest< double, CrossEntropy<double> >::initDesired(double * des, size_t const size)
{
    matvecli::randv<double>(des, size, 0, 1);
    double const sum = matvecli::sumv<double>(des, size);
    matvecli::divvc<double>(des, sum, size);
}

template<typename T, class ErrFnc> bool
GradientTest<T, ErrFnc>::run(NeuralNet<T> & net, bool const accumGradients)
{
    size_t const sizeIn  = net.sizeIn();
    size_t const sizeOut = net.sizeOut();

    //
    // Allocate memory
    //
    T * in  = matvecli::allocv<T>(sizeIn);
    T * out = matvecli::allocv<T>(sizeOut);
    T * des = matvecli::allocv<T>(sizeOut);

    //
    // Initialize input and desired output with random values
    //
    matvecli::randv<T>(in, sizeIn);
    initDesired(des, sizeOut);

    ErrFnc errFnc(sizeOut);

    //
    // Compute gradients via back-propagation
    //
    net.reset();
    net.forget();
    net.fprop(in, out);
    errFnc.fprop(out, des);
    errFnc.bprop(out);
    net.bprop(NULL, out, accumGradients);

    //
    // Compute gradients via perturbation
    //
    err_ = 0;

    typename NeuralNet<T>::TrainableParam param;
    net.trainableParam(param);

    for (int i = static_cast<int>(param.size()) - 1; i >= 0; --i)
    {
        //
        // Compute gradients of weights: O(W^2)
        //
        size_t const rows = param[i].weights.rows;
        size_t const cols = param[i].weights.cols;

        T       * const weights  = param[i].weights.val;
        T const * const dWeights = param[i].weights.dVal;
        T const * const mask     = param[i].weights.mask;

        size_t const strideWeights  = param[i].weights.strideVal;
        size_t const strideDWeights = param[i].weights.strideDVal;
        size_t const strideMask     = param[i].weights.strideMask;

        for (size_t r = 0; r < rows; ++r)
        {
            for (size_t c = 0; c < cols; ++c)
            {
                if (mask && matvecli::getelm<T>(mask, strideMask, r, c) == 0)
                    continue;

                T const tmp = matvecli::getelm<T>(weights, strideWeights, r, c);

                // E(w_rc + e)
                matvecli::setelm<T>(weights, strideWeights, r, c, tmp + epsilon_);
                net.fprop(in, out);
                T const eplus = errFnc.fprop(out, des);

                // E(w_rc - e)
                matvecli::setelm<T>(weights, strideWeights, r, c, tmp - epsilon_);
                net.fprop(in, out);
                T const eminus = errFnc.fprop(out, des);

                // dE/dw_rc = [E(w_rc + e) - E(w_rc - e)] / (2*e)
                T const dw = (eplus - eminus) / (2 * epsilon_);
                T const t = mathli::abs(
                    dw - matvecli::getelm<T>(dWeights, strideDWeights, r, c));

                if (t > err_) err_ = t;

                matvecli::setelm<T>(weights, strideWeights, r, c, tmp);
            }
        }

        //
        // Compute gradients of biases: O(B^2)
        //
        size_t    const len     = param[i].biases.len;
        T       * const biases  = param[i].biases.val;
        T const * const dBiases = param[i].biases.dVal;

        for (size_t j = 0; j < len; ++j)
        {
            T const tmp = matvecli::getelv<T>(biases, j);

            // E(b_j + e)
            matvecli::setelv<T>(biases, j, tmp + epsilon_);
            net.fprop(in, out);
            T const eplus = errFnc.fprop(out, des);

            // E(b_j - e)
            matvecli::setelv<T>(biases, j, tmp - epsilon_);
            net.fprop(in, out);
            T const eminus = errFnc.fprop(out, des);

            // dE/db_j = [E(b_j + e) - E(b_j - e)] / (2*e)
            T const db = (eplus - eminus) / (2 * epsilon_);
            T const t = mathli::abs(db - matvecli::getelv<T>(dBiases, j));

            if (t > err_) err_ = t;

            matvecli::setelv<T>(biases, j, tmp);
        }
    }

    //
    // Cleanup
    //
    matvecli::free<T>(in);
    matvecli::free<T>(out);
    matvecli::free<T>(des);

    return (err_ <= accThresh_);
}

/*! \addtogroup eti_grp Explicit Template Instantiation
 @{
 */
template class GradientTest< float,  MeanSquaredError<float>  >;
template class GradientTest< float,  CrossEntropy<float>      >;
template class GradientTest< double, MeanSquaredError<double> >;
template class GradientTest< double, CrossEntropy<double>     >;
/*! @} */

CNNPLUS_NS_END
