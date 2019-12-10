/**************************************************************************//**
 *
 * \file   sublayer.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Feb 23 2009
 *
 * $Id: sublayer.cc 2161 2009-09-09 11:49:16Z dast $
 *
 * \brief  Implementation of cnnplus::SubLayer.
 *
 *****************************************************************************/

#include "sublayer.hh"
#include "error.hh"
#include "matvecli.hh"
#include "mathli.hh"
#include <sstream>

CNNPLUS_NS_BEGIN

//! Computes the size of the output feature maps
/*! \param sizeMapsIn size of input feature maps
    \param sizeSample sample size
*/
inline Size
outputMapsSize(Size const & sizeMapsIn, Size const & sizeSample)
{
    // Check parameters
    if (!(sizeSample.height() > 0 && sizeSample.height() <= sizeMapsIn.height()     ) ||
        !(sizeSample.width()  > 0 && sizeSample.width()  <= sizeMapsIn.width()      ) ||
        !(sizeMapsIn.height() > 0 && sizeMapsIn.height() %  sizeSample.height() == 0) ||
        !(sizeMapsIn.width()  > 0 && sizeMapsIn.width()  %  sizeSample.width()  == 0)) {
        throw ParameterError("Inconsistent input size and sample size.");
    }

    return Size(sizeMapsIn.height() / sizeSample.height(),
                sizeMapsIn.width()  / sizeSample.width());
}

template<typename T, class SquFnc>
SubLayer<T, SquFnc>::SubLayer(Size const & sizeMapsIn,
                              size_t const numMaps,
                              Size const & sizeSample)
    : sizeMapsIn_(sizeMapsIn), numMaps_(numMaps), sizeSample_(sizeSample),
    sizeMapsOut_(outputMapsSize(sizeMapsIn_, sizeSample_)),
    squasher_(Size(numMaps_, sizeMapsOut_.area()))
{
    // Allocate memory
    inSat_ = matvecli::allocv<T>((sizeMapsIn_ + Size(1, 1)).area());
    matvecli::zerov<T>(inSat_, (sizeMapsIn_ + Size(1, 1)).area());
    inSub_ = matvecli::allocm<T>(numMaps, sizeMapsOut_.area(), strideInSub_);
    tmp_ = matvecli::allocv<T>(sizeMapsOut_.area());
    weights_ = matvecli::allocv<T>(numMaps_);
    dWeights_ = matvecli::allocv<T>(numMaps_);
    biases_ = matvecli::allocv<T>(numMaps_);
    dBiases_ = matvecli::allocv<T>(numMaps_);
    sum_ = matvecli::allocm<T>(numMaps_, sizeMapsOut_.area(), strideSum_);
    delta_ = matvecli::allocm<T>(numMaps_, sizeMapsOut_.area(), strideDelta_);

    reset(); // Reset gradients to zero
}

template<typename T, class SquFnc>
SubLayer<T, SquFnc>::~SubLayer()
{
    // Deallocate memory
    matvecli::free<T>(inSat_);
    matvecli::free<T>(inSub_);
    matvecli::free<T>(tmp_);
    matvecli::free<T>(weights_);
    matvecli::free<T>(dWeights_);
    matvecli::free<T>(biases_);
    matvecli::free<T>(dBiases_);
    matvecli::free<T>(sum_);
    matvecli::free<T>(delta_);
}

template<typename T, class SquFnc> void
SubLayer<T, SquFnc>::forget(T const sigma, bool scale)
{
    // Initialize weights and biases with random values
    matvecli::randv<T>(weights_, numMaps_, sigma);
    if (scale) {
        matvecli::divvc<T>(weights_,
                           static_cast<T>(sizeSample_.area()),
                           numMaps_);
    }
    matvecli::randv<T>(biases_,  numMaps_, sigma);
}

//! Initializes weights
template<typename T, class SquFnc> void
SubLayer<T, SquFnc>::forget()
{
    // Set weights to 'coeff = m^(-1/2)', where 'm' is the fan-in
    // (the number of connections feeding into the node)
    //T const coeff = mathli::pow(static_cast<T>(sizeSample_.area()), T(-.5));
    T const coeff = 1 / mathli::sqrt(static_cast<T>(sizeSample_.area()));
    matvecli::setv<T>(weights_, numMaps_, coeff);

    // Set biases to zero (as in lush 1.2.1, \gblearn2\gb-modules-nn.lsh)
    matvecli::zerov<T>(biases_, numMaps_);
}

template<typename T, class SquFnc> void
SubLayer<T, SquFnc>::reset()
{
    matvecli::zerov<T>(dWeights_, numMaps_);
    matvecli::zerov<T>(dBiases_,  numMaps_);
}

template<typename T, class SquFnc> void
SubLayer<T, SquFnc>::update(T const eta)
{
    // Compute: weights_ += eta * dWeights_
    matvecli::axpy<T>(dWeights_, numMaps_, weights_, eta);

    // Compute: biases_ += eta * dBiases_
    matvecli::axpy<T>(dBiases_, numMaps_, biases_, eta);
}

template<typename T, class SquFnc> void
SubLayer<T, SquFnc>::fprop(T const * in, size_t const strideIn,
                           T * out, size_t const strideOut)
{
    CNNPLUS_ASSERT(in && strideIn >= sizeMapsIn_.area());
    CNNPLUS_ASSERT(!out || (out && strideOut >= sizeMapsOut_.area()));

    // TODO doc
    if (sizeSample_ == Size(2, 2)) {
        for (size_t i = 0; i < numMaps_; ++i)
        {
            T const * map = in + i * strideIn;

            for (size_t y = 0, r = 0; y < sizeMapsOut_.height(); ++y, r += 2)
            {
                for (size_t x = 0, c = 0; x < sizeMapsOut_.width(); ++x, c += 2)
                {
                    inSub_[i * strideInSub_ + y * sizeMapsOut_.width() + x]
                        = map[(r + 0) * sizeMapsIn_.width() + (c + 0)]
                        + map[(r + 0) * sizeMapsIn_.width() + (c + 1)]
                        + map[(r + 1) * sizeMapsIn_.width() + (c + 0)]
                        + map[(r + 1) * sizeMapsIn_.width() + (c + 1)];
                }
            }

            matvecli::setv<T>(sum_ + i * strideSum_, sizeMapsOut_.area(), biases_[i]);
            matvecli::axpy<T>(inSub_ + i * strideInSub_, sizeMapsOut_.area(),
                              sum_ + i * strideSum_, weights_[i]);
        }
    }
    else {
        for (size_t i = 0; i < numMaps_; ++i)
        {
            // Compute summed-area table (integral image) of 'in'
            matvecli::sat<T>(in + i * strideIn, sizeMapsIn_.width(),
                             inSat_ + sizeMapsIn_.width() + 2, sizeMapsIn_.width() + 1,
                             sizeMapsIn_.height(), sizeMapsIn_.width());

            size_t const h = sizeSample_.height(), w = sizeSample_.width();
            size_t const stride = sizeMapsIn_.width() + 1;

            for (size_t y = 0, r = 0; y < sizeMapsOut_.height(); ++y, r += h)
            {
                for (size_t x = 0, c = 0; x < sizeMapsOut_.width(); ++x, c += w)
                {
                    inSub_[i * strideInSub_ + y * sizeMapsOut_.width() + x]
                        = inSat_[(r + h) * stride + (c + w)]
                        + inSat_[(r + 0) * stride + (c + 0)]
                        - inSat_[(r + 0) * stride + (c + w)]
                        - inSat_[(r + h) * stride + (c + 0)];
                }
            }

            matvecli::setv<T>(sum_ + i * strideSum_, sizeMapsOut_.area(), biases_[i]);
            matvecli::axpy<T>(inSub_ + i * strideInSub_, sizeMapsOut_.area(),
                              sum_ + i * strideSum_, weights_[i]);
        }
    }
    if (out) {
        // Compute: out = f(sum_)
        squasher_.fprop(sum_, strideSum_, out, strideOut);
    }
}

template<typename T, class SquFnc> void
SubLayer<T, SquFnc>::bprop(T * in, size_t const strideIn,
                           T const * out, size_t const strideOut,
                           bool accumGradients)
{
    CNNPLUS_ASSERT(!in || (in && strideIn >= sizeMapsIn_.area()));
    CNNPLUS_ASSERT(out && strideOut >= sizeMapsOut_.area());

    // Compute: delta_ = f'(sum_) .* out
    matvecli::copymm<T>(sum_, strideSum_, delta_, strideDelta_,
                        numMaps_, sizeMapsOut_.area());
    squasher_.bprop(delta_, strideDelta_, out, strideOut);

    if (in) {
        // TODO doc, optimize
        for (size_t i = 0; i < numMaps_; ++i)
        {
            matvecli::mulvc<T>(delta_ + i * strideDelta_, weights_[i],
                               tmp_, sizeMapsOut_.area());

            for (size_t r = 0, j = 0; r < sizeMapsIn_.height(); ++r)
            {
                size_t const y = r / sizeSample_.height();

                for (size_t c = 0; c < sizeMapsIn_.width(); ++c, ++j)
                {
                    size_t const x = c / sizeSample_.width();

                    in[i * strideIn + j] = tmp_[y * sizeMapsOut_.width() + x];
                }
            }
        }
    }

    if (accumGradients) {
        // Compute: dBiases_ += sums of row vectors in delta_
        matvecli::sumrowacc<T>(delta_, strideDelta_, dBiases_,
                               numMaps_, sizeMapsOut_.area());

        // Compute: delta_ = delta_ .* inSub_
        matvecli::pmulmm<T>(delta_, strideDelta_, inSub_, strideInSub_,
                            numMaps_, sizeMapsOut_.area());

        // Compute: dWeights_ += sums of row vectors in delta_
        matvecli::sumrowacc<T>(delta_, strideDelta_, dWeights_,
                               numMaps_, sizeMapsOut_.area());
    }
    else {
        // Compute: dBiases_ = sums of row vectors in delta_
        matvecli::sumrow<T>(delta_, strideDelta_, dBiases_,
                            numMaps_, sizeMapsOut_.area());

        // Compute: delta_ = delta_ .* inSub_
        matvecli::pmulmm<T>(delta_, strideDelta_, inSub_, strideInSub_,
                            numMaps_, sizeMapsOut_.area());

        // Compute: dWeights_ = sums of row vectors in delta_
        matvecli::sumrow<T>(delta_, strideDelta_, dWeights_,
                            numMaps_, sizeMapsOut_.area());
    }
}

#ifdef CNNPLUS_MATLAB_FOUND
template<typename T, class SquFnc> void
SubLayer<T, SquFnc>::load(mxArray const * arr)
{
    if (!arr || !mxIsStruct(arr) || !this->checkType(arr, "s"))
        throw MatlabError("Failed to read subsampling layer.");

    // Read Matlab array with weight values
    mxArray const * arrW = mxGetField(arr, 0, "weights");
    {
        mwSize const dims[] = { numMaps_, 1 };
        if (!this->checkArr(arrW, countof(dims), dims))
            throw MatlabError("Failed to read 'weights'.");
    }

    // Read Matlab array with bias values
    mxArray const * arrB = mxGetField(arr, 0, "biases");
    {
        mwSize const dims[] = { numMaps_, 1 };
        if (!this->checkArr(arrB, countof(dims), dims))
            throw MatlabError("Failed to read 'biases'.");
    }

    double const * pArrW = static_cast<double const *>(mxGetData(arrW));
    double const * pArrB = static_cast<double const *>(mxGetData(arrB));

    // Read weight and bias values from Matlab array
    for (size_t i = 0; i < numMaps_; ++i) {
        weights_[i] = static_cast<T>(pArrW[i]);
        biases_[i]  = static_cast<T>(pArrB[i]);
    }
}

template<typename T, class SquFnc> mxArray *
SubLayer<T, SquFnc>::save() const
{
    char const * fieldnames[] = { "type", "weights", "biases" };
    mxArray * arr = mxCreateStructMatrix(1, 1, countof(fieldnames), fieldnames);
    if (!arr) throw MatlabError("Failed to create array.");

    mxArray * arrW = NULL, * arrB = NULL;

    try {
        // Create Matlab arrays
        arrW = mxCreateDoubleMatrix(numMaps_, 1, mxREAL);
        if (!arrW) throw MatlabError("Failed to create array.");
        arrB = mxCreateDoubleMatrix(numMaps_, 1, mxREAL);
        if (!arrB) throw MatlabError("Failed to create array.");
    }
    catch (...) {
        if (arrW) mxDestroyArray(arrW);
        if (arrB) mxDestroyArray(arrB);
        throw;
    }

    double * pArrW = static_cast<double *>(mxGetData(arrW));
    double * pArrB = static_cast<double *>(mxGetData(arrB));

    // Copy weight and bias values to Matlab array
    for (size_t i = 0; i < numMaps_; ++i) {
        pArrW[i] = weights_[i];
        pArrB[i] = biases_[i];
    }

    // Write Matlab arrays to Matlab structure
    mxSetField(arr, 0, "type",    mxCreateString("s"));
    mxSetField(arr, 0, "weights", arrW);
    mxSetField(arr, 0, "biases",  arrB);

    return arr;
}

template<typename T, class SquFnc> mxArray *
SubLayer<T, SquFnc>::writeOut(T * const out, size_t const strideOut) const
{
    CNNPLUS_ASSERT(out && strideOut >= sizeMapsOut_.area());

    // Create Matlab array
    mwSize const dims[] = { sizeMapsOut_.height(), sizeMapsOut_.width(), numMaps_ };
    mxArray * arrOut = mxCreateNumericArray(countof(dims), dims, mxDOUBLE_CLASS, mxREAL);
    if (!arrOut) throw MatlabError("Failed to create array.");

    double * pArrOut = static_cast<double *>(mxGetData(arrOut));

    // Copy data of output feature maps to Matlab array
    for (size_t j = 0; j < numMaps_; ++j)
    {
        for (size_t r = 0; r < sizeMapsOut_.height(); ++r)
        {
            for (size_t c = 0; c < sizeMapsOut_.width(); ++c)
            {
                pArrOut[j * sizeMapsOut_.area() + r + c * sizeMapsOut_.height()]
                    = out[j * strideOut + r * sizeMapsOut_.width() + c];
            }
        }
    }

    return arrOut;
}
#endif // CNNPLUS_MATLAB_FOUND

template<typename T, class SquFnc> void
SubLayer<T, SquFnc>::trainableParam(typename Layer<T>::TrainableParam & param)
{
    // Weights
    param.weights.val        = weights_;
    param.weights.dVal       = dWeights_;
    param.weights.mask       = NULL;
    param.weights.strideVal  = numMaps_;
    param.weights.strideDVal = numMaps_;
    param.weights.strideMask = 0;
    param.weights.rows       = 1;
    param.weights.cols       = numMaps_;

    // Biases
    param.biases.val         = biases_;
    param.biases.dVal        = dBiases_;
    param.biases.len         = numMaps_;
}

template<typename T, class SquFnc> std::string
SubLayer<T, SquFnc>::toString() const
{
    std::stringstream ss;
    ss << "SubLayer"
       << "<" << numMaps_
       << "x" << sizeMapsIn_.toString()
       << "," << numMaps_
       << "x" << sizeMapsOut_.toString()
       << "," << sizeSample_.toString()
       << ";" << squasher_.toString()
       << ">";
    return ss.str();
}

template<typename T, class SquFnc> size_t
SubLayer<T, SquFnc>::numTrainableParam() const
{
    return (numMaps_ * 2);
}

template<typename T, class SquFnc> size_t
SubLayer<T, SquFnc>::numConnections() const
{
    return (numMaps_ * sizeMapsIn_.area() + numMaps_ * sizeMapsOut_.area());
}

/*! \addtogroup eti_grp Explicit Template Instantiation
 @{
 */
template class SubLayer< float,  Tanh<float>        >;
template class SubLayer< float,  StdSigmoid<float>  >;
template class SubLayer< float,  LogSigmoid<float>  >;
template class SubLayer< float,  Identity<float>    >;

template class SubLayer< double, Tanh<double>       >;
template class SubLayer< double, StdSigmoid<double> >;
template class SubLayer< double, LogSigmoid<double> >;
template class SubLayer< double, Identity<double>   >;
/*! @} */

CNNPLUS_NS_END
