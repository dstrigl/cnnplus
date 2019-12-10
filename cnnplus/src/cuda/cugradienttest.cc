/**************************************************************************//**
 *
 * \file   cugradienttest.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jun 05 2009
 *
 * $Id: cugradienttest.cc 1726 2009-07-13 07:37:14Z dast $
 *
 * \brief  Implementation of cnnplus::CuGradientTest.
 *
 *****************************************************************************/

#include "cugradienttest.hh"
#include "cuneuralnet.hh"
#include "cumvli.hh"
#include "matvecli.hh"
#include "mathli.hh"

CNNPLUS_NS_BEGIN

template<typename T, class ErrFnc> bool
CuGradientTest<T, ErrFnc>::run(NeuralNet<T> & net, bool const accumGradients)
{
    if (!dynamic_cast<CuNeuralNet<T>*>(&net))
        throw ParameterError("net", "no CUDA enabled neural-net.");

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
    this->err_ = 0;

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
                if (mask && cumvli::getelm<T>(mask, strideMask, r, c) == 0)
                    continue;

                T const tmp = cumvli::getelm<T>(weights, strideWeights, r, c);

                // E(w_rc + e)
                cumvli::setelm<T>(weights, strideWeights, r, c, tmp + this->epsilon_);
                net.fprop(in, out);
                T const eplus = errFnc.fprop(out, des);

                // E(w_rc - e)
                cumvli::setelm<T>(weights, strideWeights, r, c, tmp - this->epsilon_);
                net.fprop(in, out);
                T const eminus = errFnc.fprop(out, des);

                // dE/dw_rc = [E(w_rc + e) - E(w_rc - e)] / (2*e)
                T const dw = (eplus - eminus) / (2 * this->epsilon_);
                T const t = mathli::abs(
                    dw - cumvli::getelm<T>(dWeights, strideDWeights, r, c));

                if (t > this->err_) this->err_ = t;

                cumvli::setelm<T>(weights, strideWeights, r, c, tmp);
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
            T const tmp = cumvli::getelv<T>(biases, j);

            // E(b_j + e)
            cumvli::setelv<T>(biases, j, tmp + this->epsilon_);
            net.fprop(in, out);
            T const eplus = errFnc.fprop(out, des);

            // E(b_j - e)
            cumvli::setelv<T>(biases, j, tmp - this->epsilon_);
            net.fprop(in, out);
            T const eminus = errFnc.fprop(out, des);

            // dE/db_j = [E(b_j + e) - E(b_j - e)] / (2*e)
            T const db = (eplus - eminus) / (2 * this->epsilon_);
            T const t = mathli::abs(db - cumvli::getelv<T>(dBiases, j));

            if (t > this->err_) this->err_ = t;

            cumvli::setelv<T>(biases, j, tmp);
        }
    }

    //
    // Cleanup
    //
    matvecli::free<T>(in);
    matvecli::free<T>(out);
    matvecli::free<T>(des);

    return (this->err_ <= this->accThresh_);
}

/*! \addtogroup eti_grp Explicit Template Instantiation
 @{
 */
template class CuGradientTest< float,  MeanSquaredError<float>  >;
template class CuGradientTest< float,  CrossEntropy<float>      >;
template class CuGradientTest< double, MeanSquaredError<double> >;
template class CuGradientTest< double, CrossEntropy<double>     >;
/*! @} */

CNNPLUS_NS_END
