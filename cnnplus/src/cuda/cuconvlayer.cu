/**************************************************************************//**
 *
 * \file   cuconvlayer.cu
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jun 23 2009
 *
 * $Id: cuconvlayer.cu 3558 2010-11-22 11:04:51Z klaus $
 *
 * \brief  Implementation of cnnplus::CuConvLayer.
 *
 *****************************************************************************/

#include "cudautils.hh"

///////////////////////////////////////////////////////////////////////////////
// CUDA kernels

__global__ void
wupdate_kernel(float const * src, size_t const strideSrc,
               float const * mask, size_t const strideMask,
               size_t const rows, size_t const cols,
               float * srcDst, size_t const strideSrcDst,
               float const alpha)
{
    size_t const r = CNN_UIMUL(blockIdx.y, blockDim.y) + threadIdx.y;
    size_t const c = CNN_UIMUL(blockIdx.x, blockDim.x) + threadIdx.x;

    if (r >= rows || c >= cols)
        return;

    // Compute: srcDst += alpha * src .* mask
    srcDst[CNN_UIMUL(r, strideSrcDst) + c] +=
        alpha * src[CNN_UIMUL(r, strideSrc) + c] * mask[CNN_UIMUL(r, strideMask) + c];
}

__global__ void
unfold_kernel1(float const * in, size_t const strideIn,
               float * out, size_t const strideOut,
               size_t const mapsInH, size_t const mapsInW,
               size_t const mapsOutH, size_t const mapsOutW,
               size_t const kernelH, size_t const kernelW,
               size_t const stepV, size_t const stepH,
               size_t const numMapsIn)
{
    size_t const K = CNN_UIMUL(kernelH, kernelW); // kernel area
    size_t const r = blockIdx.y;  // row in the output map
    size_t const c = blockIdx.x;  // column in the output map
    size_t const y = threadIdx.y; // row in the filter kernel
    size_t const x = threadIdx.x; // column in the filter kernel

    for (size_t j = 0; j < numMapsIn; ++j) {
        out[CNN_UIMUL((CNN_UIMUL(j, K) + CNN_UIMUL(y, kernelW) + x), strideOut) + CNN_UIMUL(r, mapsOutW) + c] =
            in[CNN_UIMUL(j, strideIn) + CNN_UIMUL((CNN_UIMUL(r, stepV) + y), mapsInW) + CNN_UIMUL(c, stepH) + x];
    }
}

__global__ void
unfold_kernel2(float const * in, size_t const strideIn,
               float * out, size_t const strideOut,
               size_t const mapsInH, size_t const mapsInW,
               size_t const mapsOutH, size_t const mapsOutW,
               size_t const kernelH, size_t const kernelW,
               size_t const stepV, size_t const stepH,
               size_t const numMapsIn)
{
    size_t const v = CNN_UIMUL(blockIdx.y, blockDim.y) + threadIdx.y;
    size_t const h = CNN_UIMUL(blockIdx.x, blockDim.x) + threadIdx.x;
    size_t const K = CNN_UIMUL(kernelH, kernelW); // kernel area

    if (v >= CNN_UIMUL(numMapsIn, K) || h >= CNN_UIMUL(mapsOutH, mapsOutW))
        return;

    size_t const r = h / mapsOutW; // row in the output map
    size_t const c = h % mapsOutW; // column in the output map
    size_t const j = v / K;        // number of input map
    size_t const t = v % K;        // temporal variable
    size_t const y = t / kernelW;  // row in the filter kernel
    size_t const x = t % kernelW;  // column in the filter kernel

    out[CNN_UIMUL(v, strideOut) + h] =
        in[CNN_UIMUL(j, strideIn) + CNN_UIMUL((CNN_UIMUL(r, stepV) + y), mapsInW) + CNN_UIMUL(c, stepH) + x];
}

__global__ void
foldback_kernel(float * in, size_t const strideIn,
                float const * out, size_t const strideOut,
                size_t const mapsInH, size_t const mapsInW,
                size_t const mapsOutH, size_t const mapsOutW,
                size_t const kernelH, size_t const kernelW,
                size_t const stepV, size_t const stepH,
                size_t const numMapsIn)
{
    size_t const numMap = CNN_UIMUL(blockIdx.y, blockDim.y) + threadIdx.y;
    size_t const i      = CNN_UIMUL(blockIdx.x, blockDim.x) + threadIdx.x;

    if (numMap >= numMapsIn || i >= CNN_UIMUL(mapsInH, mapsInW))
        return;

    size_t const K      = CNN_UIMUL(kernelH, kernelW);
    size_t const r      = i / mapsInW;
    size_t const c      = i % mapsInW;
    size_t const yStart = (size_t) max(0, (int)(r) - (int)(kernelH - stepV)) / stepV;
    size_t const yEnd   = min((unsigned int)(r / stepV + 1), (unsigned int)mapsOutH);
    size_t const xStart = (size_t) max(0, (int)(c) - (int)(kernelW - stepH)) / stepH;
    size_t const xEnd   = min((unsigned int)(c / stepH + 1), (unsigned int)mapsOutW);
    float        tmp    = 0;

    for (size_t y = yStart; y < yEnd; ++y)
    {
        for (size_t x = xStart; x < xEnd; ++x)
        {
            tmp += out[CNN_UIMUL(CNN_UIMUL(numMap, K)
                       + (CNN_UIMUL((r - CNN_UIMUL(y, stepV)), kernelW)
                       + (c - CNN_UIMUL(x, stepH))), strideOut)
                       + CNN_UIMUL(y, mapsOutW) + x];
        }
    }

    in[CNN_UIMUL(numMap, strideIn) + i] = tmp;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#include "cuconvlayer.hh"
#include "cumvli.hh"
#include "matvecli.hh"
#include "mathli.hh"
#include <sstream>

CNNPLUS_NS_BEGIN

///////////////////////////////////////////////////////////////////////////////
// CUDA kernel calls

template<typename T> void
wupdate(T const * src, size_t const strideSrc,
        T const * mask, size_t const strideMask,
        size_t const rows, size_t const cols,
        T * srcDst, size_t const strideSrcDst,
        T const alpha);

template<> void
wupdate<float>(float const * src, size_t const strideSrc,
               float const * mask, size_t const strideMask,
               size_t const rows, size_t const cols,
               float * srcDst, size_t const strideSrcDst,
               float const alpha)
{
    CNNPLUS_ASSERT(src && strideSrc >= cols);
    CNNPLUS_ASSERT(mask && strideMask >= cols);
    CNNPLUS_ASSERT(srcDst && strideSrcDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    dim3 const dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 const dimGrid((cols + dimBlock.x - 1) / dimBlock.x,
                       (rows + dimBlock.y - 1) / dimBlock.y);
    wupdate_kernel<<<dimGrid, dimBlock>>>
        (src, strideSrc, mask, strideMask, rows, cols, srcDst, strideSrcDst, alpha);
    CUDA_CHECK_ERROR("Kernel call 'wupdate_kernel' failed");
}

template<> void
wupdate<double>(double const * src, size_t const strideSrc,
                double const * mask, size_t const strideMask,
                size_t const rows, size_t const cols,
                double * srcDst, size_t const strideSrcDst,
                double const alpha)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

template<typename T> void
unfold(T const * in, size_t const strideIn,
       T * out, size_t const strideOut,
       Size const & sizeMapsIn,
       Size const & sizeMapsOut,
       Size const & sizeKernel,
       size_t const stepV, size_t const stepH,
       size_t const numMapsIn);

template<> void
unfold<float>(float const * in, size_t const strideIn,
              float * out, size_t const strideOut,
              Size const & sizeMapsIn,
              Size const & sizeMapsOut,
              Size const & sizeKernel,
              size_t const stepV, size_t const stepH,
              size_t const numMapsIn)
{
    CNNPLUS_ASSERT(in  && strideIn  >= sizeMapsIn.area());
    CNNPLUS_ASSERT(out && strideOut >= sizeMapsOut.area());

#if 0
    if (sizeKernel.area() <= MAX_THREADS)
    {
        dim3 const dimBlock(sizeKernel.width(), sizeKernel.height());
        dim3 const dimGrid(sizeMapsOut.width(), sizeMapsOut.height());
        unfold_kernel1<<<dimGrid, dimBlock>>>(
            in, strideIn, out, strideOut,
            sizeMapsIn.height(), sizeMapsIn.width(),
            sizeMapsOut.height(), sizeMapsOut.width(),
            sizeKernel.height(), sizeKernel.width(),
            stepV, stepH, numMapsIn);
        CUDA_CHECK_ERROR("Kernel call 'unfold_kernel1' failed");
        return;
    }
#endif

    dim3 const dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 const dimGrid((sizeMapsOut.area()            + dimBlock.x - 1) / dimBlock.x,
                       (numMapsIn * sizeKernel.area() + dimBlock.y - 1) / dimBlock.y);

    unfold_kernel2<<<dimGrid, dimBlock>>>(
        in, strideIn, out, strideOut,
        sizeMapsIn.height(), sizeMapsIn.width(),
        sizeMapsOut.height(), sizeMapsOut.width(),
        sizeKernel.height(), sizeKernel.width(),
        stepV, stepH, numMapsIn);
    CUDA_CHECK_ERROR("Kernel call 'unfold_kernel2' failed");
}

template<> void
unfold<double>(double const * in, size_t const strideIn,
               double * out, size_t const strideOut,
               Size const & sizeMapsIn,
               Size const & sizeMapsOut,
               Size const & sizeKernel,
               size_t const stepV, size_t const stepH,
               size_t const numMapsIn)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

template<typename T> void
foldback(T * in, size_t const strideIn,
         T const * out, size_t const strideOut,
         Size const & sizeMapsIn,
         Size const & sizeMapsOut,
         Size const & sizeKernel,
         size_t const stepV, size_t const stepH,
         size_t const numMapsIn);

template<> void
foldback<float>(float * in, size_t const strideIn,
                float const * out, size_t const strideOut,
                Size const & sizeMapsIn,
                Size const & sizeMapsOut,
                Size const & sizeKernel,
                size_t const stepV, size_t const stepH,
                size_t const numMapsIn)
{
    CNNPLUS_ASSERT(in  && strideIn  >= sizeMapsIn.area());
    CNNPLUS_ASSERT(out && strideOut >= sizeMapsOut.area());

    dim3 const dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 const dimGrid((sizeMapsIn.area() + dimBlock.x - 1) / dimBlock.x,
                       (numMapsIn         + dimBlock.y - 1) / dimBlock.y);

    foldback_kernel<<<dimGrid, dimBlock>>>(
        in, strideIn, out, strideOut,
        sizeMapsIn.height(), sizeMapsIn.width(),
        sizeMapsOut.height(), sizeMapsOut.width(),
        sizeKernel.height(), sizeKernel.width(),
        stepV, stepH, numMapsIn);
    CUDA_CHECK_ERROR("Kernel call 'foldback_kernel' failed");
}

template<> void
foldback<double>(double * in, size_t const strideIn,
                 double const * out, size_t const strideOut,
                 Size const & sizeMapsIn,
                 Size const & sizeMapsOut,
                 Size const & sizeKernel,
                 size_t const stepV, size_t const stepH,
                 size_t const numMapsIn)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

///////////////////////////////////////////////////////////////////////////////
// CuConvLayer implementation

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
CuConvLayer<T, SquFnc>::CuConvLayer(Size const & sizeMapsIn, size_t const numMapsIn,
                                    size_t const numMapsOut, Size const & sizeKernel,
                                    size_t const stepV, size_t const stepH,
                                    double const connProp)
    : sizeMapsIn_(sizeMapsIn), numMapsIn_(numMapsIn), numMapsOut_(numMapsOut),
    sizeKernel_(sizeKernel), stepV_(stepV), stepH_(stepH),
    sizeMapsOut_(outputMapsSize(sizeMapsIn_, sizeKernel_, stepV_, stepH_)),
    conTbl_(numMapsOut_, numMapsIn_, connProp), squasher_(Size(numMapsOut_, sizeMapsOut_.area()))
{
    // Allocate memory (GPU)
    d_inUnfolded_ = cumvli::allocm<T>(numMapsIn_ * sizeKernel_.area(), sizeMapsOut_.area(), d_strideInUnfolded_);
    d_weights_ = cumvli::allocm<T>(numMapsOut_, numMapsIn_ * sizeKernel_.area(), d_strideWeights_);
    d_dWeights_ = cumvli::allocm<T>(numMapsOut_, numMapsIn_ * sizeKernel_.area(), d_strideDWeights_);
    d_weightsMask_ = cumvli::allocm<T>(numMapsOut_, numMapsIn_ * sizeKernel_.area(), d_strideWeightsMask_);
    d_biases_ = cumvli::allocv<T>(numMapsOut_);
    d_dBiases_ = cumvli::allocv<T>(numMapsOut_);
    d_sum_ = cumvli::allocm<T>(numMapsOut_, sizeMapsOut_.area(), d_strideSum_);
    d_delta_ = cumvli::allocm<T>(numMapsOut_, sizeMapsOut_.area(), d_strideDelta_);
    d_tmp_ = cumvli::allocm<T>(numMapsIn_ * sizeKernel_.area(), sizeMapsOut_.area(), d_strideTmp_);

    // Allocate memory (CPU)
    h_weights_ = matvecli::allocm<T>(numMapsOut_, numMapsIn_ * sizeKernel_.area(), h_strideWeights_);
    h_biases_ = matvecli::allocv<T>(numMapsOut_);

    setConTbl(conTbl_); // Initialize weights mask for connection table
    reset();            // Reset gradients to zero
}

template<typename T, class SquFnc>
CuConvLayer<T, SquFnc>::~CuConvLayer()
{
    // Deallocate memory (GPU)
    cumvli::free<T>(d_inUnfolded_);
    cumvli::free<T>(d_weights_);
    cumvli::free<T>(d_dWeights_);
    cumvli::free<T>(d_weightsMask_);
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
CuConvLayer<T, SquFnc>::forget(T const sigma, bool scale)
{
    // Initialize weights with random values
    for (size_t r = 0; r < numMapsOut_; ++r)
    {
        matvecli::randv<T>(h_weights_ + r * h_strideWeights_,
                           numMapsIn_ * sizeKernel_.area(),
                           sigma);
        if (scale) {
            size_t const m = conTbl_.numInConn(r) * sizeKernel_.area();
            CNNPLUS_ASSERT(m > 0);
            matvecli::divvc<T>(h_weights_ + r * h_strideWeights_,
                               static_cast<T>(m),
                               numMapsIn_ * sizeKernel_.area());
        }
    }
    cumvli::copym_h2d<T>(h_weights_, h_strideWeights_, d_weights_, d_strideWeights_,
                         numMapsOut_, numMapsIn_ * sizeKernel_.area());
    cumvli::pmulmm<T>(d_weights_, d_strideWeights_, d_weightsMask_, d_strideWeightsMask_,
                      numMapsOut_, numMapsIn_ * sizeKernel_.area());
    cumvli::copym_d2h<T>(d_weights_, d_strideWeights_, h_weights_, h_strideWeights_,
                         numMapsOut_, numMapsIn_ * sizeKernel_.area());

    // Initialize biases with random values
    matvecli::randv<T>(h_biases_, numMapsOut_, sigma);
    cumvli::copyv_h2d<T>(h_biases_, d_biases_, numMapsOut_);
}

//! Initializes weights
/*! Initializes weights with random values as recommended by Yann LeCun in
    "Efficient BackProp" in Neural Networks: Tricks of the Trade, 1998,
    1.4.6 Initializing the Weights.
    \see http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
 */
template<typename T, class SquFnc> void
CuConvLayer<T, SquFnc>::forget()
{
    // [...] weights should be randomly drawn from a distribution (e.g. uniform)
    // with mean zero and standard deviation 'sigma = m^(-1/2)', where 'm' is the
    // fan-in (the number of connections feeding into the node).
    for (size_t r = 0; r < numMapsOut_; ++r)
    {
        size_t const m = conTbl_.numInConn(r) * sizeKernel_.area();
        //T const sigma = mathli::pow(static_cast<T>(m), T(-.5));
        T const sigma = 1 / mathli::sqrt(static_cast<T>(m));

        matvecli::randv<T>(h_weights_ + r * h_strideWeights_,
                           numMapsIn_ * sizeKernel_.area(),
                           sigma);
    }
    cumvli::copym_h2d<T>(h_weights_, h_strideWeights_, d_weights_, d_strideWeights_,
                         numMapsOut_, numMapsIn_ * sizeKernel_.area());
    cumvli::pmulmm<T>(d_weights_, d_strideWeights_, d_weightsMask_, d_strideWeightsMask_,
                      numMapsOut_, numMapsIn_ * sizeKernel_.area());
    cumvli::copym_d2h<T>(d_weights_, d_strideWeights_, h_weights_, h_strideWeights_,
                         numMapsOut_, numMapsIn_ * sizeKernel_.area());

    // Set biases to zero (as in lush 1.2.1, \gblearn2\gb-modules-nn.lsh)
    matvecli::zerov<T>(h_biases_, numMapsOut_);
    cumvli::copyv_h2d<T>(h_biases_, d_biases_, numMapsOut_);
}

template<typename T, class SquFnc> void
CuConvLayer<T, SquFnc>::reset()
{
    cumvli::zerom<T>(d_dWeights_, d_strideDWeights_, numMapsOut_, numMapsIn_ * sizeKernel_.area());
    cumvli::zerov<T>(d_dBiases_, numMapsOut_);
}

template<typename T, class SquFnc> void
CuConvLayer<T, SquFnc>::update(T const eta)
{
    // Compute: weights_ += eta * dWeights_ .* weightsMask_
    wupdate<T>(d_dWeights_, d_strideDWeights_, d_weightsMask_, d_strideWeightsMask_,
               numMapsOut_, numMapsIn_ * sizeKernel_.area(), d_weights_, d_strideWeights_,
               eta);

    // Compute: biases_ += eta * dBiases_
    cumvli::axpy<T>(d_dBiases_, numMapsOut_, d_biases_, eta);
}

template<typename T, class SquFnc> void
CuConvLayer<T, SquFnc>::fprop(T const * in, size_t const strideIn,
                              T * out, size_t const strideOut)
{
    CNNPLUS_ASSERT(in && strideIn >= sizeMapsIn_.area());
    CNNPLUS_ASSERT(!out || (out && strideOut >= sizeMapsOut_.area()));

    // Create a unfolded 'convolution matrix' for 'in'
    unfold(in, strideIn, d_inUnfolded_, d_strideInUnfolded_,
           sizeMapsIn_, sizeMapsOut_, sizeKernel_, stepV_, stepH_,
           numMapsIn_);

#ifndef NDEBUG
    for (size_t r = 0; r < numMapsOut_; ++r) {
        for (size_t i = 0; i < numMapsIn_; ++i) {
            if (!conTbl_.at(r, i)) {
                T const * ptr = d_weights_ + r * d_strideWeights_ + i * sizeKernel_.area();
                CNNPLUS_ASSERT(cumvli::absmaxv<T>(ptr, sizeKernel_.area()) == 0);
            }
        }
    }
#endif

    // Copy vector 'biases_' in each column of matrix 'sum_'
    cumvli::setcol<T>(d_sum_, d_strideSum_, numMapsOut_, sizeMapsOut_.area(), d_biases_);

    // Compute: sum_ = weights_ * inUnfolded_ + sum_
    cumvli::gemm<T,'n','n'>(d_weights_, d_strideWeights_,
                            numMapsOut_, numMapsIn_ * sizeKernel_.area(),
                            d_inUnfolded_, d_strideInUnfolded_,
                            numMapsIn_ * sizeKernel_.area(), sizeMapsOut_.area(),
                            d_sum_, d_strideSum_);
    if (out) {
        // Compute: out = f(sum_)
        squasher_.fprop(d_sum_, d_strideSum_, out, strideOut);
    }
}

template<typename T, class SquFnc> void
CuConvLayer<T, SquFnc>::bprop(T * in, size_t const strideIn,
                              T const * out, size_t const strideOut,
                              bool accumGradients)
{
    CNNPLUS_ASSERT(!in || (in && strideIn >= sizeMapsIn_.area()));
    CNNPLUS_ASSERT(out && strideOut >= sizeMapsOut_.area());

    // Compute: delta_ = f'(sum_) .* out
    cumvli::copymm<T>(d_sum_, d_strideSum_, d_delta_, d_strideDelta_,
                      numMapsOut_, sizeMapsOut_.area());
    squasher_.bprop(d_delta_, d_strideDelta_, out, strideOut);

    if (accumGradients) {
        // Compute: dWeights_ += delta_ * inUnfolded_'
        cumvli::gemm<T,'n','t'>(d_delta_, d_strideDelta_,
                                numMapsOut_, sizeMapsOut_.area(),
                                d_inUnfolded_, d_strideInUnfolded_,
                                numMapsIn_ * sizeKernel_.area(), sizeMapsOut_.area(),
                                d_dWeights_, d_strideDWeights_, 1, 1);

        // Compute: dBiases_ += sums of row vectors in delta_
        cumvli::sumrowacc<T>(d_delta_, d_strideDelta_, d_dBiases_,
                             numMapsOut_, sizeMapsOut_.area());
    }
    else {
        // Compute: dWeights_ = delta_ * inUnfolded_'
        cumvli::gemm<T,'n','t'>(d_delta_, d_strideDelta_,
                                numMapsOut_, sizeMapsOut_.area(),
                                d_inUnfolded_, d_strideInUnfolded_,
                                numMapsIn_ * sizeKernel_.area(), sizeMapsOut_.area(),
                                d_dWeights_, d_strideDWeights_, 1, 0);

        // Compute: dBiases_ = sums of row vectors in delta_
        cumvli::sumrow<T>(d_delta_, d_strideDelta_, d_dBiases_,
                          numMapsOut_, sizeMapsOut_.area());
    }

#ifndef NDEBUG
    for (size_t r = 0; r < numMapsOut_; ++r) {
        for (size_t i = 0; i < numMapsIn_; ++i) {
            if (!conTbl_.at(r, i)) {
                T const * ptr = d_weights_ + r * d_strideWeights_ + i * sizeKernel_.area();
                CNNPLUS_ASSERT(cumvli::absmaxv<T>(ptr, sizeKernel_.area()) == 0);
            }
        }
    }
#endif

    if (in) {
        // Compute: tmp_ = weights_' * delta_
        cumvli::gemm<T,'t','n'>(d_weights_, d_strideWeights_,
                                numMapsOut_, numMapsIn_ * sizeKernel_.area(),
                                d_delta_, d_strideDelta_,
                                numMapsOut_, sizeMapsOut_.area(),
                                d_tmp_, d_strideTmp_, 1, 0);

        // Fold the convolution matrix 'tmp' back to matrix 'in'
        foldback(in, strideIn, d_tmp_, d_strideTmp_,
                 sizeMapsIn_, sizeMapsOut_, sizeKernel_,
                 stepV_, stepH_, numMapsIn_);
    }
}

#ifdef CNNPLUS_MATLAB_FOUND
template<typename T, class SquFnc> void
CuConvLayer<T, SquFnc>::load(mxArray const * arr)
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
                    h_weights_[j * h_strideWeights_ + i * sizeKernel_.area() + r * sizeKernel_.width() + c] =
                        static_cast<T>(pArrW[(j * numMapsIn_ + i) * sizeKernel_.area() + r + c * sizeKernel_.height()]);
                }
            }
        }
    }
    cumvli::copym_h2d<T>(h_weights_, h_strideWeights_,
        d_weights_, d_strideWeights_,
        numMapsOut_, numMapsIn_ * sizeKernel_.area());

    // Read bias values from Matlab array
    for (size_t i = 0; i < numMapsOut_; ++i)
        h_biases_[i] = static_cast<T>(pArrB[i]);
    cumvli::copyv_h2d<T>(h_biases_, d_biases_, numMapsOut_);
}

template<typename T, class SquFnc> mxArray *
CuConvLayer<T, SquFnc>::save() const
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
    cumvli::copym_d2h<T>(d_weights_, d_strideWeights_,
        h_weights_, h_strideWeights_,
        numMapsOut_, numMapsIn_ * sizeKernel_.area());
    for (size_t j = 0; j < numMapsOut_; ++j)
    {
        for (size_t i = 0; i < numMapsIn_; ++i)
        {
            for (size_t r = 0; r < sizeKernel_.height(); ++r)
            {
                for (size_t c = 0; c < sizeKernel_.width(); ++c)
                {
                    pArrW[(j * numMapsIn_ + i) * sizeKernel_.area() + r + c * sizeKernel_.height()] =
                        h_weights_[j * h_strideWeights_ + i * sizeKernel_.area() + r * sizeKernel_.width() + c];
                }
            }
        }
    }

    // Copy bias values to Matlab array
    cumvli::copyv_d2h<T>(d_biases_, h_biases_, numMapsOut_);
    for (size_t i = 0; i < numMapsOut_; ++i)
        pArrB[i] = h_biases_[i];

    // Write Matlab arrays to Matlab structure
    mxSetField(arr, 0, "type",    mxCreateString("c"));
    mxSetField(arr, 0, "weights", arrW);
    mxSetField(arr, 0, "biases",  arrB);
    conTbl_.save(arr);

    return arr;
}
#endif // CNNPLUS_MATLAB_FOUND

template<typename T, class SquFnc> void
CuConvLayer<T, SquFnc>::trainableParam(typename Layer<T>::TrainableParam & param)
{
    // Weights
    param.weights.val        = d_weights_;
    param.weights.dVal       = d_dWeights_;
    param.weights.mask       = d_weightsMask_;
    param.weights.strideVal  = d_strideWeights_;
    param.weights.strideDVal = d_strideDWeights_;
    param.weights.strideMask = d_strideWeightsMask_;
    param.weights.rows       = numMapsOut_;
    param.weights.cols       = numMapsIn_ * sizeKernel_.area();

    // Biases
    param.biases.val         = d_biases_;
    param.biases.dVal        = d_dBiases_;
    param.biases.len         = numMapsOut_;
}

template<typename T, class SquFnc> std::string
CuConvLayer<T, SquFnc>::toString() const
{
    std::stringstream ss;
    ss << "CuConvLayer"
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
CuConvLayer<T, SquFnc>::numTrainableParam() const
{
    return (conTbl_.numConn() * sizeKernel_.area() + numMapsOut_);
}

template<typename T, class SquFnc> size_t
CuConvLayer<T, SquFnc>::numConnections() const
{
    return (conTbl_.numConn() * sizeMapsOut_.area() * sizeKernel_.area() + numMapsOut_ * sizeMapsOut_.area());
}

template<typename T, class SquFnc> void
CuConvLayer<T, SquFnc>::setConTbl(ConTbl const & conTbl)
{
    if (conTbl.rows() != numMapsOut_ || conTbl.cols() != numMapsIn_)
        throw ParameterError("conTbl", "number of rows/columns doesn't match.");

    conTbl_ = conTbl; // store connection table

    // Initialize weights mask for connection table
    for (size_t r = 0; r < numMapsOut_; ++r) {
        for (size_t i = 0; i < numMapsIn_; ++i) {
            cumvli::setv<T>(
                d_weightsMask_ + r * d_strideWeights_ + i * sizeKernel_.area(),
                sizeKernel_.area(), conTbl_.at(r, i) ? T(1) : T(0));
        }
    }
}

/*! \addtogroup eti_grp Explicit Template Instantiation
 @{
 */
template class CuConvLayer< float,  CuTanh<float>        >;
template class CuConvLayer< float,  CuStdSigmoid<float>  >;
template class CuConvLayer< float,  CuLogSigmoid<float>  >;
template class CuConvLayer< float,  CuIdentity<float>    >;

template class CuConvLayer< double, CuTanh<double>       >;
template class CuConvLayer< double, CuStdSigmoid<double> >;
template class CuConvLayer< double, CuLogSigmoid<double> >;
template class CuConvLayer< double, CuIdentity<double>   >;
/*! @} */

CNNPLUS_NS_END
