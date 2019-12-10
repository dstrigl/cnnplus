/**************************************************************************//**
 *
 * \file   fulllayer.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Dec 19 2008
 *
 * $Id: fulllayer.cc 1905 2009-07-30 20:21:22Z dast $
 *
 * \brief  Implementation of cnnplus::FullLayer.
 *
 *****************************************************************************/

#include "fulllayer.hh"
#include "error.hh"
#include "matvecli.hh"
#include "mathli.hh"
#include <sstream>

CNNPLUS_NS_BEGIN

template<typename T, class SquFnc>
FullLayer<T, SquFnc>::FullLayer(Size const & sizeIn, Size const & sizeOut)
    : sizeIn_(sizeIn), sizeOut_(sizeOut), squasher_(sizeOut_)
{
    // Allocate memory
    in_ = matvecli::allocv<T>(sizeIn_.area());
    weights_ = matvecli::allocm<T>(sizeOut_.area(), sizeIn_.area(), strideWeights_);
    dWeights_ = matvecli::allocm<T>(sizeOut_.area(), sizeIn_.area(), strideDWeights_);
    biases_ = matvecli::allocv<T>(sizeOut_.area());
    dBiases_ = matvecli::allocv<T>(sizeOut_.area());
    sum_ = matvecli::allocv<T>(sizeOut_.area());
    delta_ = matvecli::allocv<T>(sizeOut_.area());
    tmp_ = matvecli::allocv<T>(sizeIn_.area());

    reset(); // Reset gradients to zero
}

template<typename T, class SquFnc>
FullLayer<T, SquFnc>::~FullLayer()
{
    // Deallocate memory
    matvecli::free<T>(in_);
    matvecli::free<T>(weights_);
    matvecli::free<T>(dWeights_);
    matvecli::free<T>(biases_);
    matvecli::free<T>(dBiases_);
    matvecli::free<T>(sum_);
    matvecli::free<T>(delta_);
    matvecli::free<T>(tmp_);
}

template<typename T, class SquFnc> void
FullLayer<T, SquFnc>::forget(T const sigma, bool scale)
{
    // Initialize weights with random values
    matvecli::randm<T>(weights_, strideWeights_,
                       sizeOut_.area(), sizeIn_.area(),
                       sigma);
    if (scale) {
        matvecli::divmc<T>(weights_, strideWeights_,
                           static_cast<T>(sizeIn_.area()),
                           sizeOut_.area(), sizeIn_.area());
    }

    // Initialize biases with random values
    matvecli::randv<T>(biases_, sizeOut_.area(), sigma);
}

//! Initializes weights
/*! Initializes weights with random values as recommended by Yann LeCun in
    "Efficient BackProp" in Neural Networks: Tricks of the Trade, 1998,
    1.4.6 Initializing the Weights.
    \see http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
 */
template<typename T, class SquFnc> void
FullLayer<T, SquFnc>::forget()
{
    // [...] weights should be randomly drawn from a distribution (e.g. uniform)
    // with mean zero and standard deviation 'sigma = m^(-1/2)', where 'm' is the
    // fan-in (the number of connections feeding into the node).
    //T const sigma = mathli::pow(static_cast<T>(sizeIn_.area()), T(-.5));
    T const sigma = 1 / mathli::sqrt(static_cast<T>(sizeIn_.area()));

    // Initialize weights with random values
    matvecli::randm<T>(weights_, strideWeights_,
                       sizeOut_.area(), sizeIn_.area(),
                       sigma);

    // Set biases to zero (as in lush 1.2.1, \gblearn2\gb-modules-nn.lsh)
    matvecli::zerov<T>(biases_, sizeOut_.area());
}

template<typename T, class SquFnc> void
FullLayer<T, SquFnc>::reset()
{
    matvecli::zerom<T>(dWeights_, strideDWeights_, sizeOut_.area(), sizeIn_.area());
    matvecli::zerov<T>(dBiases_, sizeOut_.area());
}

template<typename T, class SquFnc> void
FullLayer<T, SquFnc>::update(T const eta)
{
    // Compute: weights_ += eta * dWeights_
    for (size_t r = 0; r < sizeOut_.area(); ++r)
    {
        matvecli::axpy<T>(dWeights_ + r * strideDWeights_,
                          sizeIn_.area(),
                          weights_  + r * strideWeights_,
                          eta);
    }

    // Compute: biases_ += eta * dBiases_
    matvecli::axpy<T>(dBiases_, sizeOut_.area(), biases_, eta);
}

template<typename T, class SquFnc> void
FullLayer<T, SquFnc>::fprop(T const * in, size_t const strideIn,
                            T * out, size_t const strideOut)
{
    CNNPLUS_ASSERT(in && strideIn >= sizeIn_.width());
    CNNPLUS_ASSERT(!out || (out && strideOut >= sizeOut_.width()));

    // Store matrix 'in' in vector 'in_'
    matvecli::copymv<T>(in, strideIn, in_, sizeIn_.height(), sizeIn_.width());
#if 0
    // Copy vector 'biases_' to 'sum_'
    matvecli::copyvv<T>(biases_, sum_, sizeOut_.area());

    // Compute: sum_ = weights_ * in_ + sum_
    matvecli::gemv<T,'n'>(weights_, strideWeights_,
                          sizeOut_.area(), sizeIn_.area(),
                          in_, sizeIn_.area(), sum_);
#else
    // Compute: sum_ = weights_ * in_ + biases_
    matvecli::gemv<T>(weights_, strideWeights_,
                      sizeOut_.area(), sizeIn_.area(),
                      in_, sizeIn_.area(),
                      biases_, sizeOut_.area(),
                      sum_);
#endif
    if (out) {
        // Compute: out = f(sum_)
        squasher_.fprop(sum_, sizeOut_.width(), out, strideOut);
    }
}

template<typename T, class SquFnc> void
FullLayer<T, SquFnc>::bprop(T * in, size_t const strideIn,
                            T const * out, size_t const strideOut,
                            bool accumGradients)
{
    CNNPLUS_ASSERT(!in || (in && strideIn >= sizeIn_.width()));
    CNNPLUS_ASSERT(out && strideOut >= sizeOut_.width());

    // Compute: delta_ = f'(sum_) .* out
    matvecli::copyvv<T>(sum_, delta_, sizeOut_.area());
    squasher_.bprop(delta_, sizeOut_.width(), out, strideOut);

    if (accumGradients) {
        // Compute: dWeights_ += delta_ * in_'
        matvecli::ger<T>(delta_, sizeOut_.area(),
                         in_, sizeIn_.area(),
                         dWeights_, strideDWeights_);

        // Compute: dBiases_ += delta_
        matvecli::addv<T>(dBiases_, delta_, sizeOut_.area());
    }
    else {
        // Compute: dWeights_ = delta_ * in_'
        matvecli::mulmm<T,'n','n'>(delta_, 1, sizeOut_.area(), 1,
                                   in_, sizeIn_.area(), 1, sizeIn_.area(),
                                   dWeights_, strideDWeights_);
        //matvecli::gemm<T,'n','n'>(delta_, 1, sizeOut_.area(), 1,
        //                          in_, sizeIn_.area(), 1, sizeIn_.area(),
        //                          dWeights_, strideDWeights_, 1, 0);

        // Compute: dBiases_ = delta_
        matvecli::copyvv<T>(delta_, dBiases_, sizeOut_.area());
    }

    if (in) {
        // Compute: tmp_ = weights_' * delta_
        //matvecli::mulmv<T,'t'>(weights_, strideWeights_,
        //                       sizeOut_.area(), sizeIn_.area(),
        //                       delta_, sizeOut_.area(), tmp_);
        matvecli::gemv<T,'t'>(weights_, strideWeights_,
                              sizeOut_.area(), sizeIn_.area(),
                              delta_, sizeOut_.area(), tmp_, 1, 0);

        // Copy vector 'tmp_' to matrix 'in'
        matvecli::copyvm<T>(tmp_, in, strideIn, sizeIn_.height(), sizeIn_.width());
    }
}

#ifdef CNNPLUS_MATLAB_FOUND
template<typename T, class SquFnc> void
FullLayer<T, SquFnc>::load(mxArray const * arr)
{
    if (!arr || !mxIsStruct(arr) || !this->checkType(arr, "f"))
        throw MatlabError("Failed to read fully-connected layer.");

    // Read Matlab array with weight values
    mxArray const * arrW = mxGetField(arr, 0, "weights");
    {
        mwSize const dims[] = { sizeOut_.area(), sizeIn_.area() };
        if (!this->checkArr(arrW, countof(dims), dims))
            throw MatlabError("Failed to read 'weights'.");
    }

    // Read Matlab array with bias values
    mxArray const * arrB = mxGetField(arr, 0, "biases");
    {
        mwSize const dims[] = { sizeOut_.area(), 1 };
        if (!this->checkArr(arrB, countof(dims), dims))
            throw MatlabError("Failed to read 'biases'.");
    }

    double const * pArrW = static_cast<double const *>(mxGetData(arrW));
    double const * pArrB = static_cast<double const *>(mxGetData(arrB));

    // Read weight values from Matlab array
    for (size_t r = 0; r < sizeOut_.area(); ++r) {
        for (size_t c = 0; c < sizeIn_.area(); ++c) {
            weights_[r * strideWeights_ + c] =
                static_cast<T>(pArrW[r + c * sizeOut_.area()]);
        }
    }

    // Read bias values from Matlab array
    for (size_t i = 0; i < sizeOut_.area(); ++i)
        biases_[i] = static_cast<T>(pArrB[i]);
}

template<typename T, class SquFnc> mxArray *
FullLayer<T, SquFnc>::save() const
{
    char const * fieldnames[] = { "type", "weights", "biases" };
    mxArray * arr = mxCreateStructMatrix(1, 1, countof(fieldnames), fieldnames);
    if (!arr) throw MatlabError("Failed to create array.");

    mxArray * arrW = NULL, * arrB = NULL;

    try {
        // Create Matlab arrays
        arrW = mxCreateDoubleMatrix(sizeOut_.area(), sizeIn_.area(), mxREAL);
        if (!arrW) throw MatlabError("Failed to create array.");
        arrB = mxCreateDoubleMatrix(sizeOut_.area(), 1, mxREAL);
        if (!arrB) throw MatlabError("Failed to create array.");
    }
    catch (...) {
        if (arrW) mxDestroyArray(arrW);
        if (arrB) mxDestroyArray(arrB);
        throw;
    }

    double * pArrW = static_cast<double *>(mxGetData(arrW));
    double * pArrB = static_cast<double *>(mxGetData(arrB));

    // Copy weight values to Matlab array
    for (size_t r = 0; r < sizeOut_.area(); ++r) {
        for (size_t c = 0; c < sizeIn_.area(); ++c) {
            pArrW[r + c * sizeOut_.area()] = weights_[r * strideWeights_ + c];
        }
    }

    // Copy bias values to Matlab array
    for (size_t i = 0; i < sizeOut_.area(); ++i)
        pArrB[i] = biases_[i];

    // Write Matlab arrays to Matlab structure
    mxSetField(arr, 0, "type",    mxCreateString("f"));
    mxSetField(arr, 0, "weights", arrW);
    mxSetField(arr, 0, "biases",  arrB);

    return arr;
}

template<typename T, class SquFnc> mxArray *
FullLayer<T, SquFnc>::writeOut(T * const out, size_t const strideOut) const
{
    CNNPLUS_ASSERT(out && strideOut >= sizeOut_.width());

    // Create Matlab array
    mxArray * arrOut = mxCreateDoubleMatrix(sizeOut_.area(), 1, mxREAL);
    if (!arrOut) throw MatlabError("Failed to create array.");

    double * pArrOut = static_cast<double *>(mxGetData(arrOut));

    // Copy output data to Matlab array
    for (size_t r = 0; r < sizeOut_.height(); ++r) {
        for (size_t c = 0; c < sizeOut_.width(); ++c) {
            pArrOut[r * sizeOut_.width() + c] = out[r * strideOut + c];
        }
    }

    return arrOut;
}
#endif // CNNPLUS_MATLAB_FOUND

template<typename T, class SquFnc> void
FullLayer<T, SquFnc>::trainableParam(typename Layer<T>::TrainableParam & param)
{
    // Weights
    param.weights.val        = weights_;
    param.weights.dVal       = dWeights_;
    param.weights.mask       = NULL;
    param.weights.strideVal  = strideWeights_;
    param.weights.strideDVal = strideDWeights_;
    param.weights.strideMask = 0;
    param.weights.rows       = sizeOut_.area();
    param.weights.cols       = sizeIn_.area();

    // Biases
    param.biases.val         = biases_;
    param.biases.dVal        = dBiases_;
    param.biases.len         = sizeOut_.area();
}

template<typename T, class SquFnc> std::string
FullLayer<T, SquFnc>::toString() const
{
    std::stringstream ss;
    ss << "FullLayer"
       << "<" << sizeIn_.area()
       << "," << sizeOut_.area()
       << ";" << squasher_.toString()
       << ">";
    return ss.str();
}

template<typename T, class SquFnc> size_t
FullLayer<T, SquFnc>::numTrainableParam() const
{
    return (sizeIn_.area() * sizeOut_.area() + sizeOut_.area());
}

template<typename T, class SquFnc> size_t
FullLayer<T, SquFnc>::numConnections() const
{
    return (sizeIn_.area() * sizeOut_.area() + sizeOut_.area());
}

/*! \addtogroup eti_grp Explicit Template Instantiation
 @{
 */
template class FullLayer< float,  Tanh<float>        >;
template class FullLayer< float,  StdSigmoid<float>  >;
template class FullLayer< float,  LogSigmoid<float>  >;
template class FullLayer< float,  Identity<float>    >;

template class FullLayer< double, Tanh<double>       >;
template class FullLayer< double, StdSigmoid<double> >;
template class FullLayer< double, LogSigmoid<double> >;
template class FullLayer< double, Identity<double>   >;
/*! @} */

CNNPLUS_NS_END
