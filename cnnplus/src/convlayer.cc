/**************************************************************************//**
 *
 * \file   convlayer.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Dec 09 2008
 *
 * $Id: convlayer.cc 1905 2009-07-30 20:21:22Z dast $
 *
 * \brief  Implementation of cnnplus::ConvLayer.
 *
 *****************************************************************************/

#include "convlayer.hh"
#include "error.hh"
#include "matvecli.hh"
#include "mathli.hh"
#include <sstream>

CNNPLUS_NS_BEGIN

//! Computes the size of the output feature maps
/*! \param sizeMapsIn size of input feature maps
    \param sizeKernel kernel size
    \param stepV vertical step size of the kernel
    \param stepH horizontal step size of the kernel
*/
inline Size
outputMapsSize(Size const & sizeMapsIn, Size const & sizeKernel,
               size_t const stepV, size_t const stepH)
{
    // Check parameters
    if (!(sizeMapsIn.area() > 0 && sizeKernel.area() > 0 && stepV > 0 && stepH > 0  ) ||
        !(stepV <= sizeKernel.height() && sizeKernel.height() <= sizeMapsIn.height()) ||
        !(stepH <= sizeKernel.width()  && sizeKernel.width()  <= sizeMapsIn.width() ) ||
        !((sizeMapsIn.height() - (sizeKernel.height() - stepV)) % stepV == 0        ) ||
        !((sizeMapsIn.width()  - (sizeKernel.width()  - stepH)) % stepH == 0       )) {
        throw ParameterError(
            "Inconsistent input size, kernel size and step size.");
    }

    return Size((sizeMapsIn.height() - (sizeKernel.height() - stepV)) / stepV,
                (sizeMapsIn.width()  - (sizeKernel.width()  - stepH)) / stepH);
}

template<typename T, class SquFnc>
ConvLayer<T, SquFnc>::ConvLayer(Size const & sizeMapsIn, size_t const numMapsIn,
                                size_t const numMapsOut, Size const & sizeKernel,
                                size_t const stepV, size_t const stepH,
                                double const connProp)
    : sizeMapsIn_(sizeMapsIn), numMapsIn_(numMapsIn), numMapsOut_(numMapsOut),
    sizeKernel_(sizeKernel), stepV_(stepV), stepH_(stepH),
    sizeMapsOut_(outputMapsSize(sizeMapsIn_, sizeKernel_, stepV_, stepH_)),
    conTbl_(numMapsOut_, numMapsIn_, connProp), squasher_(Size(numMapsOut_, sizeMapsOut_.area()))
{
    // Allocate memory
    inUnfolded_ = matvecli::allocm<T>(numMapsIn_ * sizeKernel_.area(), sizeMapsOut_.area(), strideInUnfolded_);
    weights_ = matvecli::allocm<T>(numMapsOut_, numMapsIn_ * sizeKernel_.area(), strideWeights_);
    dWeights_ = matvecli::allocm<T>(numMapsOut_, numMapsIn_ * sizeKernel_.area(), strideDWeights_);
    weightsMask_ = matvecli::allocm<T>(numMapsOut_, numMapsIn_ * sizeKernel_.area(), strideWeightsMask_);
    biases_ = matvecli::allocv<T>(numMapsOut_);
    dBiases_ = matvecli::allocv<T>(numMapsOut_);
    sum_ = matvecli::allocm<T>(numMapsOut_, sizeMapsOut_.area(), strideSum_);
    delta_ = matvecli::allocm<T>(numMapsOut_, sizeMapsOut_.area(), strideDelta_);
    tmp_ = matvecli::allocm<T>(numMapsIn_ * sizeKernel_.area(), sizeMapsOut_.area(), strideTmp_);

    setConTbl(conTbl_); // Initialize weights mask for connection table
    reset();            // Reset gradients to zero
}

template<typename T, class SquFnc>
ConvLayer<T, SquFnc>::~ConvLayer()
{
    // Deallocate memory
    matvecli::free<T>(inUnfolded_);
    matvecli::free<T>(weights_);
    matvecli::free<T>(dWeights_);
    matvecli::free<T>(weightsMask_);
    matvecli::free<T>(biases_);
    matvecli::free<T>(dBiases_);
    matvecli::free<T>(sum_);
    matvecli::free<T>(delta_);
    matvecli::free<T>(tmp_);
}

template<typename T, class SquFnc> void
ConvLayer<T, SquFnc>::forget(T const sigma, bool scale)
{
    // Initialize weights with random values
    for (size_t r = 0; r < numMapsOut_; ++r)
    {
        matvecli::randv<T>(weights_ + r * strideWeights_,
                           numMapsIn_ * sizeKernel_.area(),
                           sigma);
        if (scale) {
            size_t const m = conTbl_.numInConn(r) * sizeKernel_.area();
            CNNPLUS_ASSERT(m > 0);
            matvecli::divvc<T>(weights_ + r * strideWeights_,
                               static_cast<T>(m),
                               numMapsIn_ * sizeKernel_.area());
        }
    }
    matvecli::pmulmm<T>(weights_, strideWeights_, weightsMask_, strideWeightsMask_,
                        numMapsOut_, numMapsIn_ * sizeKernel_.area());

    // Initialize biases with random values
    matvecli::randv<T>(biases_, numMapsOut_, sigma);
}

//! Initializes weights
/*! Initializes weights with random values as recommended by Yann LeCun in
    "Efficient BackProp" in Neural Networks: Tricks of the Trade, 1998,
    1.4.6 Initializing the Weights.
    \see http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
 */
template<typename T, class SquFnc> void
ConvLayer<T, SquFnc>::forget()
{
    // [...] weights should be randomly drawn from a distribution (e.g. uniform)
    // with mean zero and standard deviation 'sigma = m^(-1/2)', where 'm' is the
    // fan-in (the number of connections feeding into the node).
    for (size_t r = 0; r < numMapsOut_; ++r)
    {
        size_t const m = conTbl_.numInConn(r) * sizeKernel_.area();
        //T const sigma = mathli::pow(static_cast<T>(m), T(-.5));
        T const sigma = 1 / mathli::sqrt(static_cast<T>(m));

        matvecli::randv<T>(weights_ + r * strideWeights_,
                           numMapsIn_ * sizeKernel_.area(),
                           sigma);
    }
    matvecli::pmulmm<T>(weights_, strideWeights_, weightsMask_, strideWeightsMask_,
                        numMapsOut_, numMapsIn_ * sizeKernel_.area());

    // Set biases to zero (as in lush 1.2.1, \gblearn2\gb-modules-nn.lsh)
    matvecli::zerov<T>(biases_, numMapsOut_);
}

template<typename T, class SquFnc> void
ConvLayer<T, SquFnc>::reset()
{
    matvecli::zerom<T>(dWeights_, strideDWeights_, numMapsOut_, numMapsIn_ * sizeKernel_.area());
    matvecli::zerov<T>(dBiases_, numMapsOut_);
}

template<typename T, class SquFnc> void
ConvLayer<T, SquFnc>::update(T const eta)
{
    // Compute: weights_ += eta * dWeights_ .* weightsMask_
    for (size_t r = 0; r < numMapsOut_; ++r)
    {
        matvecli::axpy<T>(dWeights_ + r * strideDWeights_,
                          numMapsIn_ * sizeKernel_.area(),
                          weights_  + r * strideWeights_,
                          eta);
        matvecli::mulv<T>(weights_ + r * strideWeights_,
                          weightsMask_ + r * strideWeightsMask_,
                          numMapsIn_ * sizeKernel_.area());
    }

    // Compute: biases_ += eta * dBiases_
    matvecli::axpy<T>(dBiases_, numMapsOut_, biases_, eta);
}

template<typename T, class SquFnc> void
ConvLayer<T, SquFnc>::fprop(T const * in, size_t const strideIn,
                            T * out, size_t const strideOut)
{
    CNNPLUS_ASSERT(in && strideIn >= sizeMapsIn_.area());
    CNNPLUS_ASSERT(!out || (out && strideOut >= sizeMapsOut_.area()));

    // Create a unfolded 'convolution matrix' for 'in'
    unfold(in, strideIn, inUnfolded_, strideInUnfolded_);

#ifndef NDEBUG
    for (size_t r = 0; r < numMapsOut_; ++r) {
        for (size_t i = 0; i < numMapsIn_; ++i) {
            if (!conTbl_.at(r, i)) {
                T const * ptr = weights_ + r * strideWeights_ + i * sizeKernel_.area();
                CNNPLUS_ASSERT(matvecli::absmaxv<T>(ptr, sizeKernel_.area()) == 0);
            }
        }
    }
#endif

    // Copy vector 'biases_' in each column of matrix 'sum_'
    matvecli::setcol<T>(sum_, strideSum_, numMapsOut_, sizeMapsOut_.area(), biases_);

    // Compute: sum_ = weights_ * inUnfolded_ + sum_
    matvecli::gemm<T,'n','n'>(weights_, strideWeights_,
                              numMapsOut_, numMapsIn_ * sizeKernel_.area(),
                              inUnfolded_, strideInUnfolded_,
                              numMapsIn_ * sizeKernel_.area(), sizeMapsOut_.area(),
                              sum_, strideSum_);
    if (out) {
        // Compute: out = f(sum_)
        squasher_.fprop(sum_, strideSum_, out, strideOut);
    }
}

template<typename T, class SquFnc> void
ConvLayer<T, SquFnc>::bprop(T * in, size_t const strideIn,
                            T const * out, size_t const strideOut,
                            bool accumGradients)
{
    CNNPLUS_ASSERT(!in || (in && strideIn >= sizeMapsIn_.area()));
    CNNPLUS_ASSERT(out && strideOut >= sizeMapsOut_.area());

    // Compute: delta_ = f'(sum_) .* out
    matvecli::copymm<T>(sum_, strideSum_, delta_, strideDelta_,
                        numMapsOut_, sizeMapsOut_.area());
    squasher_.bprop(delta_, strideDelta_, out, strideOut);

    if (accumGradients) {
        // Compute: dWeights_ += delta_ * inUnfolded_'
        matvecli::gemm<T,'n','t'>(delta_, strideDelta_,
                                  numMapsOut_, sizeMapsOut_.area(),
                                  inUnfolded_, strideInUnfolded_,
                                  numMapsIn_ * sizeKernel_.area(), sizeMapsOut_.area(),
                                  dWeights_, strideDWeights_, 1, 1);

        // Compute: dBiases_ += sums of row vectors in delta_
        matvecli::sumrowacc<T>(delta_, strideDelta_, dBiases_,
                               numMapsOut_, sizeMapsOut_.area());
    }
    else {
        // Compute: dWeights_ = delta_ * inUnfolded_'
        //matvecli::mulmm<T,'n','t'>(delta_, strideDelta_,
        //                           numMapsOut_, sizeMapsOut_.area(),
        //                           inUnfolded_, strideInUnfolded_,
        //                           numMapsIn_ * sizeKernel_.area(), sizeMapsOut_.area(),
        //                           dWeights_, strideDWeights_);
        matvecli::gemm<T,'n','t'>(delta_, strideDelta_,
                                  numMapsOut_, sizeMapsOut_.area(),
                                  inUnfolded_, strideInUnfolded_,
                                  numMapsIn_ * sizeKernel_.area(), sizeMapsOut_.area(),
                                  dWeights_, strideDWeights_, 1, 0);

        // Compute: dBiases_ = sums of row vectors in delta_
        matvecli::sumrow<T>(delta_, strideDelta_, dBiases_,
                            numMapsOut_, sizeMapsOut_.area());
    }

#ifndef NDEBUG
    for (size_t r = 0; r < numMapsOut_; ++r) {
        for (size_t i = 0; i < numMapsIn_; ++i) {
            if (!conTbl_.at(r, i)) {
                T const * ptr = weights_ + r * strideWeights_ + i * sizeKernel_.area();
                CNNPLUS_ASSERT(matvecli::absmaxv<T>(ptr, sizeKernel_.area()) == 0);
            }
        }
    }
#endif

    if (in) {
        // Compute: tmp_ = weights_' * delta_
        //matvecli::mulmm<T,'t','n'>(weights_, strideWeights_,
        //                           numMapsOut_, numMapsIn_ * sizeKernel_.area(),
        //                           delta_, strideDelta_,
        //                           numMapsOut_, sizeMapsOut_.area(),
        //                           tmp_, strideTmp_);
        matvecli::gemm<T,'t','n'>(weights_, strideWeights_,
                                  numMapsOut_, numMapsIn_ * sizeKernel_.area(),
                                  delta_, strideDelta_,
                                  numMapsOut_, sizeMapsOut_.area(),
                                  tmp_, strideTmp_, 1, 0);

        // Fold the convolution matrix 'tmp' back to matrix 'in'
        foldback(in, strideIn, tmp_, strideTmp_);
    }
}

#ifdef CNNPLUS_MATLAB_FOUND
template<typename T, class SquFnc> void
ConvLayer<T, SquFnc>::load(mxArray const * arr)
{
    if (!arr || !mxIsStruct(arr) || !this->checkType(arr, "c"))
        throw MatlabError("Failed to read convolutional layer.");

    conTbl_.load(arr);
    setConTbl(conTbl_); // Initialize weights mask for connection table

    // Read Matlab array with weight values
    mxArray const * arrW = mxGetField(arr, 0, "weights");
    {
        mwSize const dims[] = { sizeKernel_.height(), sizeKernel_.width(), numMapsIn_, numMapsOut_ };
        if (!this->checkArr(arrW, countof(dims), dims))
            throw MatlabError("Failed to read 'weights'.");
    }

    // Read Matlab array with bias values
    mxArray const * arrB = mxGetField(arr, 0, "biases");
    {
        mwSize const dims[] = { numMapsOut_, 1 };
        if (!this->checkArr(arrB, countof(dims), dims))
            throw MatlabError("Failed to read 'biases'.");
    }

    double const * pArrW = static_cast<double const *>(mxGetData(arrW));
    double const * pArrB = static_cast<double const *>(mxGetData(arrB));

    // Read weight values from Matlab array
    for (size_t j = 0; j < numMapsOut_; ++j)
    {
        for (size_t i = 0; i < numMapsIn_; ++i)
        {
            for (size_t r = 0; r < sizeKernel_.height(); ++r)
            {
                for (size_t c = 0; c < sizeKernel_.width(); ++c)
                {
                    weights_[j * strideWeights_ + i * sizeKernel_.area() + r * sizeKernel_.width() + c] =
                        static_cast<T>(pArrW[(j * numMapsIn_ + i) * sizeKernel_.area() + r + c * sizeKernel_.height()]);
                }
            }
        }
    }

    // Read bias values from Matlab array
    for (size_t i = 0; i < numMapsOut_; ++i)
        biases_[i] = static_cast<T>(pArrB[i]);
}

template<typename T, class SquFnc> mxArray *
ConvLayer<T, SquFnc>::save() const
{
    char const * fieldnames[] = { "type", "contbl", "weights", "biases" };
    mxArray * arr = mxCreateStructMatrix(1, 1, countof(fieldnames), fieldnames);
    if (!arr) throw MatlabError("Failed to create array.");

    mxArray * arrW = NULL, * arrB = NULL;

    try {
        // Create Matlab arrays
        mwSize const dims[] = { sizeKernel_.height(), sizeKernel_.width(), numMapsIn_, numMapsOut_ };
        arrW = mxCreateNumericArray(countof(dims), dims, mxDOUBLE_CLASS, mxREAL);
        if (!arrW) throw MatlabError("Failed to create array.");
        arrB = mxCreateDoubleMatrix(numMapsOut_, 1, mxREAL);
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
    for (size_t j = 0; j < numMapsOut_; ++j)
    {
        for (size_t i = 0; i < numMapsIn_; ++i)
        {
            for (size_t r = 0; r < sizeKernel_.height(); ++r)
            {
                for (size_t c = 0; c < sizeKernel_.width(); ++c)
                {
                    pArrW[(j * numMapsIn_ + i) * sizeKernel_.area() + r + c * sizeKernel_.height()] =
                        weights_[j * strideWeights_ + i * sizeKernel_.area() + r * sizeKernel_.width() + c];
                }
            }
        }
    }

    // Copy bias values to Matlab array
    for (size_t i = 0; i < numMapsOut_; ++i)
        pArrB[i] = biases_[i];

    // Write Matlab arrays to Matlab structure
    mxSetField(arr, 0, "type",    mxCreateString("c"));
    mxSetField(arr, 0, "weights", arrW);
    mxSetField(arr, 0, "biases",  arrB);
    conTbl_.save(arr);

    return arr;
}

template<typename T, class SquFnc> mxArray *
ConvLayer<T, SquFnc>::writeOut(T * const out, size_t const strideOut) const
{
    CNNPLUS_ASSERT(out && strideOut >= sizeMapsOut_.area());

    // Create Matlab array
    mwSize const dims[] = { sizeMapsOut_.height(), sizeMapsOut_.width(), numMapsOut_ };
    mxArray * arrOut = mxCreateNumericArray(countof(dims), dims, mxDOUBLE_CLASS, mxREAL);
    if (!arrOut) throw MatlabError("Failed to create array.");

    double * pArrOut = static_cast<double *>(mxGetData(arrOut));

    // Copy data of output feature maps to Matlab array
    for (size_t j = 0; j < numMapsOut_; ++j)
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

/*! Creates a <em>convolution matrix</em> to transform the traditional
    convolution into a matrix multiplication.

\see Chellapilla K., Puri S., Simard P., "High Performance Convolutional
     Neural Networks for Document Processing", 10th International Workshop
     on Frontiers in Handwriting Recognition (IWFHR'2006) will be held in
     La Baule, France on October 23-26, 2006.
 */
template<typename T, class SquFnc> void
ConvLayer<T, SquFnc>::unfold(T const * in, size_t const strideIn,
                             T * out, size_t const strideOut)
{
    CNNPLUS_ASSERT(in  && strideIn  >= sizeMapsIn_.area());
    CNNPLUS_ASSERT(out && strideOut >= sizeMapsOut_.area());

    for (size_t r = 0; r < sizeMapsOut_.height(); ++r)
    {
        for (size_t c = 0; c < sizeMapsOut_.width(); ++c)
        {
            for (size_t j = 0; j < numMapsIn_; ++j)
            {
                for (size_t y = 0; y < sizeKernel_.height(); ++y)
                {
                    for (size_t x = 0; x < sizeKernel_.width(); ++x)
                    {
                        out[(j * sizeKernel_.area() + y * sizeKernel_.width() + x) * strideOut + r * sizeMapsOut_.width() + c]
                            = in[j * strideIn + (r * stepV_ + y) * sizeMapsIn_.width() + c * stepH_ + x];
                    }
                }
            }
        }
    }
}

/*! \todo doc
 */
template<typename T, class SquFnc> void
ConvLayer<T, SquFnc>::foldback(T * in, size_t const strideIn,
                               T const * out, size_t const strideOut)
{
    CNNPLUS_ASSERT(in  && strideIn  >= sizeMapsIn_.area());
    CNNPLUS_ASSERT(out && strideOut >= sizeMapsOut_.area());

    matvecli::zerom<T>(in, strideIn, numMapsIn_, sizeMapsIn_.area());

    for (size_t r = 0; r < sizeMapsOut_.height(); ++r)
    {
        for (size_t c = 0; c < sizeMapsOut_.width(); ++c)
        {
            for (size_t j = 0; j < numMapsIn_; ++j)
            {
                for (size_t y = 0; y < sizeKernel_.height(); ++y)
                {
                    for (size_t x = 0; x < sizeKernel_.width(); ++x)
                    {
                        in[j * strideIn + (r * stepV_ + y) * sizeMapsIn_.width() + c * stepH_ + x] +=
                            out[(j * sizeKernel_.area() + y * sizeKernel_.width() + x) * strideOut + r * sizeMapsOut_.width() + c];
                    }
                }
            }
        }
    }
}

template<typename T, class SquFnc> void
ConvLayer<T, SquFnc>::trainableParam(typename Layer<T>::TrainableParam & param)
{
    // Weights
    param.weights.val        = weights_;
    param.weights.dVal       = dWeights_;
    param.weights.mask       = weightsMask_;
    param.weights.strideVal  = strideWeights_;
    param.weights.strideDVal = strideDWeights_;
    param.weights.strideMask = strideWeightsMask_;
    param.weights.rows       = numMapsOut_;
    param.weights.cols       = numMapsIn_ * sizeKernel_.area();

    // Biases
    param.biases.val         = biases_;
    param.biases.dVal        = dBiases_;
    param.biases.len         = numMapsOut_;
}

template<typename T, class SquFnc> std::string
ConvLayer<T, SquFnc>::toString() const
{
    std::stringstream ss;
    ss << "ConvLayer"
       << "<" << numMapsIn_
       << "x" << sizeMapsIn_.toString()
       << "," << numMapsOut_
       << "x" << sizeMapsOut_.toString()
       << "," << conTbl_.numConn()
       << "x" << sizeKernel_.toString()
       << "," << stepV_
       << "," << stepH_
       << ";" << squasher_.toString()
       << ">";
    return ss.str();
}

template<typename T, class SquFnc> size_t
ConvLayer<T, SquFnc>::numTrainableParam() const
{
    return (conTbl_.numConn() * sizeKernel_.area() + numMapsOut_);
}

template<typename T, class SquFnc> size_t
ConvLayer<T, SquFnc>::numConnections() const
{
    return (conTbl_.numConn() * sizeMapsOut_.area() * sizeKernel_.area() + numMapsOut_ * sizeMapsOut_.area());
}

template<typename T, class SquFnc> void
ConvLayer<T, SquFnc>::setConTbl(ConTbl const & conTbl)
{
    if (conTbl.rows() != numMapsOut_ || conTbl.cols() != numMapsIn_)
        throw ParameterError("conTbl", "number of rows/columns doesn't match.");

    conTbl_ = conTbl; // store connection table

    // Initialize weights mask for connection table
    for (size_t r = 0; r < numMapsOut_; ++r) {
        for (size_t i = 0; i < numMapsIn_; ++i) {
            matvecli::setv<T>(
                weightsMask_ + r * strideWeights_ + i * sizeKernel_.area(),
                sizeKernel_.area(), conTbl_.at(r, i) ? T(1) : T(0));
        }
    }
}

/*! \addtogroup eti_grp Explicit Template Instantiation
 @{
 */
template class ConvLayer< float,  Tanh<float>        >;
template class ConvLayer< float,  StdSigmoid<float>  >;
template class ConvLayer< float,  LogSigmoid<float>  >;
template class ConvLayer< float,  Identity<float>    >;

template class ConvLayer< double, Tanh<double>       >;
template class ConvLayer< double, StdSigmoid<double> >;
template class ConvLayer< double, LogSigmoid<double> >;
template class ConvLayer< double, Identity<double>   >;
/*! @} */

CNNPLUS_NS_END
