/**************************************************************************//**
 *
 * \file   cufulllayer.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jun 05 2009
 *
 * $Id: cufulllayer.cc 1904 2009-07-30 19:29:56Z dast $
 *
 * \brief  Implementation of cnnplus::CuFullLayer.
 *
 *****************************************************************************/

#include "cufulllayer.hh"
#include "cumvli.hh"
#include "matvecli.hh"
#include "mathli.hh"
#include <sstream>

CNNPLUS_NS_BEGIN

template<typename T, class SquFnc>
CuFullLayer<T, SquFnc>::CuFullLayer(Size const & sizeIn, Size const & sizeOut)
    : sizeIn_(sizeIn), sizeOut_(sizeOut), squasher_(sizeOut_)
{
    // Allocate memory (GPU)
    d_in_ = cumvli::allocv<T>(sizeIn_.area());
    d_weights_ = cumvli::allocm<T>(sizeOut_.area(), sizeIn_.area(), d_strideWeights_);
    d_dWeights_ = cumvli::allocm<T>(sizeOut_.area(), sizeIn_.area(), d_strideDWeights_);
    d_biases_ = cumvli::allocv<T>(sizeOut_.area());
    d_dBiases_ = cumvli::allocv<T>(sizeOut_.area());
    d_sum_ = cumvli::allocv<T>(sizeOut_.area());
    d_delta_ = cumvli::allocv<T>(sizeOut_.area());
    d_tmp_ = cumvli::allocv<T>(sizeIn_.area());

    // Allocate memory (CPU)
    h_weights_ = matvecli::allocm<T>(sizeOut_.area(), sizeIn_.area(), h_strideWeights_);
    h_biases_ = matvecli::allocv<T>(sizeOut_.area());

    reset(); // Reset gradients to zero
}

template<typename T, class SquFnc>
CuFullLayer<T, SquFnc>::~CuFullLayer()
{
    // Deallocate memory (GPU)
    cumvli::free<T>(d_in_);
    cumvli::free<T>(d_weights_);
    cumvli::free<T>(d_dWeights_);
    cumvli::free<T>(d_biases_);
    cumvli::free<T>(d_dBiases_);
    cumvli::free<T>(d_sum_);
    cumvli::free<T>(d_delta_);
    cumvli::free<T>(d_tmp_);

    // Deallocate memory (CPU)
    matvecli::free<T>(h_weights_);
    matvecli::free<T>(h_biases_);
}

template<typename T, class SquFnc> void
CuFullLayer<T, SquFnc>::forget(T const sigma, bool scale)
{
    // Initialize weights with random values
    matvecli::randm<T>(h_weights_, h_strideWeights_,
                       sizeOut_.area(), sizeIn_.area(),
                       sigma);
    if (scale) {
        matvecli::divmc<T>(h_weights_, h_strideWeights_,
                           static_cast<T>(sizeIn_.area()),
                           sizeOut_.area(), sizeIn_.area());
    }
    cumvli::copym_h2d<T>(h_weights_, h_strideWeights_,
                         d_weights_, d_strideWeights_,
                         sizeOut_.area(), sizeIn_.area());

    // Initialize biases with random values
    matvecli::randv<T>(h_biases_, sizeOut_.area(), sigma);
    cumvli::copyv_h2d<T>(h_biases_, d_biases_, sizeOut_.area());
}

//! Initializes weights
/*! Initializes weights with random values as recommended by Yann LeCun in
    "Efficient BackProp" in Neural Networks: Tricks of the Trade, 1998,
    1.4.6 Initializing the Weights.
    \see http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
 */
template<typename T, class SquFnc> void
CuFullLayer<T, SquFnc>::forget()
{
    // [...] weights should be randomly drawn from a distribution (e.g. uniform)
    // with mean zero and standard deviation 'sigma = m^(-1/2)', where 'm' is the
    // fan-in (the number of connections feeding into the node).
    //T const sigma = mathli::pow(static_cast<T>(sizeIn_.area()), T(-.5));
    T const sigma = 1 / mathli::sqrt(static_cast<T>(sizeIn_.area()));

    // Initialize weights with random values
    matvecli::randm<T>(h_weights_, h_strideWeights_,
                       sizeOut_.area(), sizeIn_.area(),
                       sigma);
    cumvli::copym_h2d<T>(h_weights_, h_strideWeights_,
                         d_weights_, d_strideWeights_,
                         sizeOut_.area(), sizeIn_.area());

    // Set biases to zero (as in lush 1.2.1, \gblearn2\gb-modules-nn.lsh)
    matvecli::zerov<T>(h_biases_, sizeOut_.area());
    cumvli::copyv_h2d<T>(h_biases_, d_biases_, sizeOut_.area());
}

template<typename T, class SquFnc> void
CuFullLayer<T, SquFnc>::reset()
{
    cumvli::zerom<T>(d_dWeights_, d_strideDWeights_, sizeOut_.area(), sizeIn_.area());
    cumvli::zerov<T>(d_dBiases_, sizeOut_.area());
}

template<typename T, class SquFnc> void
CuFullLayer<T, SquFnc>::update(T const eta)
{
    // Compute: weights_ += eta * dWeights_
    cumvli::axpy<T>(d_dWeights_, d_strideDWeights_,
                    sizeOut_.area(), sizeIn_.area(),
                    d_weights_, d_strideWeights_,
                    eta);

    // Compute: biases_ += eta * dBiases_
    cumvli::axpy<T>(d_dBiases_, sizeOut_.area(), d_biases_, eta);
}

template<typename T, class SquFnc> void
CuFullLayer<T, SquFnc>::fprop(T const * in, size_t const strideIn,
                              T * out, size_t const strideOut)
{
    CNNPLUS_ASSERT(in && strideIn >= sizeIn_.width());
    CNNPLUS_ASSERT(!out || (out && strideOut >= sizeOut_.width()));

    // Store matrix 'in' in vector 'in_'
    cumvli::copymv<T>(in, strideIn, d_in_, sizeIn_.height(), sizeIn_.width());

    // Compute: sum_ = weights_ * in_ + biases_
    cumvli::gemv<T>(d_weights_, d_strideWeights_,
                    sizeOut_.area(), sizeIn_.area(),
                    d_in_, sizeIn_.area(),
                    d_biases_, sizeOut_.area(),
                    d_sum_);
    if (out) {
        // Compute: out = f(sum_)
        squasher_.fprop(d_sum_, sizeOut_.width(), out, strideOut);
    }
}

template<typename T, class SquFnc> void
CuFullLayer<T, SquFnc>::bprop(T * in, size_t const strideIn,
                              T const * out, size_t const strideOut,
                              bool accumGradients)
{
    CNNPLUS_ASSERT(!in || (in && strideIn >= sizeIn_.width()));
    CNNPLUS_ASSERT(out && strideOut >= sizeOut_.width());

    // Compute: delta_ = f'(sum_) .* out
    cumvli::copyvv<T>(d_sum_, d_delta_, sizeOut_.area());
    squasher_.bprop(d_delta_, sizeOut_.width(), out, strideOut);

    if (accumGradients) {
        // Compute: dWeights_ += delta_ * in_'
        cumvli::ger<T>(d_delta_, sizeOut_.area(),
                       d_in_, sizeIn_.area(),
                       d_dWeights_, d_strideDWeights_);

        // Compute: dBiases_ += delta_
        cumvli::addv<T>(d_dBiases_, d_delta_, sizeOut_.area());
    }
    else {
        // Compute: dWeights_ = delta_ * in_'
        cumvli::mulmm<T,'n','n'>(d_delta_, 1, sizeOut_.area(), 1,
                                 d_in_, sizeIn_.area(), 1, sizeIn_.area(),
                                 d_dWeights_, d_strideDWeights_);

        // Compute: dBiases_ = delta_
        cumvli::copyvv<T>(d_delta_, d_dBiases_, sizeOut_.area());
    }

    if (in) {
        // Compute: tmp_ = weights_' * delta_
        cumvli::gemv<T,'t'>(d_weights_, d_strideWeights_,
                            sizeOut_.area(), sizeIn_.area(),
                            d_delta_, sizeOut_.area(), d_tmp_, 1, 0);

        // Copy vector 'tmp_' to matrix 'in'
        cumvli::copyvm<T>(d_tmp_, in, strideIn, sizeIn_.height(), sizeIn_.width());
    }
}

#ifdef CNNPLUS_MATLAB_FOUND
template<typename T, class SquFnc> void
CuFullLayer<T, SquFnc>::load(mxArray const * arr)
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
            h_weights_[r * h_strideWeights_ + c] =
                static_cast<T>(pArrW[r + c * sizeOut_.area()]);
        }
    }
    cumvli::copym_h2d<T>(h_weights_, h_strideWeights_,
                         d_weights_, d_strideWeights_,
                         sizeOut_.area(), sizeIn_.area());

    // Read bias values from Matlab array
    for (size_t i = 0; i < sizeOut_.area(); ++i)
        h_biases_[i] = static_cast<T>(pArrB[i]);
    cumvli::copyv_h2d<T>(h_biases_, d_biases_, sizeOut_.area());
}

template<typename T, class SquFnc> mxArray *
CuFullLayer<T, SquFnc>::save() const
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
    cumvli::copym_d2h<T>(d_weights_, d_strideWeights_,
        h_weights_, h_strideWeights_,
        sizeOut_.area(), sizeIn_.area());
    for (size_t r = 0; r < sizeOut_.area(); ++r) {
        for (size_t c = 0; c < sizeIn_.area(); ++c) {
            pArrW[r + c * sizeOut_.area()] = h_weights_[r * h_strideWeights_ + c];
        }
    }

    // Copy bias values to Matlab array
    cumvli::copyv_d2h<T>(d_biases_, h_biases_, sizeOut_.area());
    for (size_t i = 0; i < sizeOut_.area(); ++i)
        pArrB[i] = h_biases_[i];

    // Write Matlab arrays to Matlab structure
    mxSetField(arr, 0, "type",    mxCreateString("f"));
    mxSetField(arr, 0, "weights", arrW);
    mxSetField(arr, 0, "biases",  arrB);

    return arr;
}
#endif // CNNPLUS_MATLAB_FOUND

template<typename T, class SquFnc> void
CuFullLayer<T, SquFnc>::trainableParam(typename Layer<T>::TrainableParam & param)
{
    // Weights
    param.weights.val        = d_weights_;
    param.weights.dVal       = d_dWeights_;
    param.weights.mask       = NULL;
    param.weights.strideVal  = d_strideWeights_;
    param.weights.strideDVal = d_strideDWeights_;
    param.weights.strideMask = 0;
    param.weights.rows       = sizeOut_.area();
    param.weights.cols       = sizeIn_.area();

    // Biases
    param.biases.val         = d_biases_;
    param.biases.dVal        = d_dBiases_;
    param.biases.len         = sizeOut_.area();
}

template<typename T, class SquFnc> std::string
CuFullLayer<T, SquFnc>::toString() const
{
    std::stringstream ss;
    ss << "CuFullLayer"
       << "<" << sizeIn_.area()
       << "," << sizeOut_.area()
       << ";" << squasher_.toString()
       << ">";
    return ss.str();
}

template<typename T, class SquFnc> size_t
CuFullLayer<T, SquFnc>::numTrainableParam() const
{
    return (sizeIn_.area() * sizeOut_.area() + sizeOut_.area());
}

template<typename T, class SquFnc> size_t
CuFullLayer<T, SquFnc>::numConnections() const
{
    return (sizeIn_.area() * sizeOut_.area() + sizeOut_.area());
}

/*! \addtogroup eti_grp Explicit Template Instantiation
 @{
 */
template class CuFullLayer< float,  CuTanh<float>        >;
template class CuFullLayer< float,  CuStdSigmoid<float>  >;
template class CuFullLayer< float,  CuLogSigmoid<float>  >;
template class CuFullLayer< float,  CuIdentity<float>    >;

template class CuFullLayer< double, CuTanh<double>       >;
template class CuFullLayer< double, CuStdSigmoid<double> >;
template class CuFullLayer< double, CuLogSigmoid<double> >;
template class CuFullLayer< double, CuIdentity<double>   >;
/*! @} */

CNNPLUS_NS_END
