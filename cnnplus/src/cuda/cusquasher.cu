/**************************************************************************//**
 *
 * \file   cusquasher.cu
 * \author Daniel Strigl, Klaus Kofler
 * \date   May 25 2009
 *
 * $Id: cusquasher.cu 3558 2010-11-22 11:04:51Z klaus $
 *
 * \brief  Implementation of the neural-net squasher functions.
 *
 *****************************************************************************/

#include "cudautils.hh"

///////////////////////////////////////////////////////////////////////////////
// CuLogSigmoid

__global__ void
logsigmoid_fprop_kernel(float const * in, size_t const strideIn,
                        float * out, size_t const strideOut,
                        size_t const rows, size_t const cols)
{
    size_t const r = CNN_UIMUL(blockIdx.y, blockDim.y) + threadIdx.y;
    size_t const c = CNN_UIMUL(blockIdx.x, blockDim.x) + threadIdx.x;

    if (r >= rows || c >= cols)
        return;

    // Compute: out = logsigmoid(in)
    out[CNN_UIMUL(r, strideOut) + c] =
        1.0f / (1.0f + expf(-in[CNN_UIMUL(r, strideIn) + c]));
}

__global__ void
logsigmoid_bprop_kernel(float * in, size_t const strideIn,
                        float const * out, size_t const strideOut,
                        size_t const rows, size_t const cols)
{
    size_t const r = CNN_UIMUL(blockIdx.y, blockDim.y) + threadIdx.y;
    size_t const c = CNN_UIMUL(blockIdx.x, blockDim.x) + threadIdx.x;

    if (r >= rows || c >= cols)
        return;

    // Compute: in = dlogsigmoid(in) .* out
    float const tmp = 1.0f / (1.0f + expf(-in[CNN_UIMUL(r, strideIn) + c]));
    in[CNN_UIMUL(r, strideIn) + c] =
        (tmp * (1.0f - tmp)) * out[CNN_UIMUL(r, strideOut) + c];
}

///////////////////////////////////////////////////////////////////////////////
// CuTanh

__global__ void
tanh_fprop_kernel(float const * in, size_t const strideIn,
                  float * out, size_t const strideOut,
                  size_t const rows, size_t const cols)
{
    size_t const r = CNN_UIMUL(blockIdx.y, blockDim.y) + threadIdx.y;
    size_t const c = CNN_UIMUL(blockIdx.x, blockDim.x) + threadIdx.x;

    if (r >= rows || c >= cols)
        return;

    // Compute: out = tanh(in)
    out[CNN_UIMUL(r, strideOut) + c] = tanhf(in[CNN_UIMUL(r, strideIn) + c]);
}

__global__ void
tanh_bprop_kernel(float * in, size_t const strideIn,
                  float const * out, size_t const strideOut,
                  size_t const rows, size_t const cols)
{
    size_t const r = CNN_UIMUL(blockIdx.y, blockDim.y) + threadIdx.y;
    size_t const c = CNN_UIMUL(blockIdx.x, blockDim.x) + threadIdx.x;

    if (r >= rows || c >= cols)
        return;

    // Compute: in = dtanh(in) .* out
    in[CNN_UIMUL(r, strideIn) + c] =
        (1.0f - powf(tanhf(in[CNN_UIMUL(r, strideIn) + c]), 2)) *
        out[CNN_UIMUL(r, strideOut) + c];
}

///////////////////////////////////////////////////////////////////////////////
// CuStdSigmoid

__global__ void
stdsigmoid_fprop_kernel(float const * in, size_t const strideIn,
                        float * out, size_t const strideOut,
                        size_t const rows, size_t const cols)
{
    size_t const r = CNN_UIMUL(blockIdx.y, blockDim.y) + threadIdx.y;
    size_t const c = CNN_UIMUL(blockIdx.x, blockDim.x) + threadIdx.x;

    if (r >= rows || c >= cols)
        return;

    // Compute: out = stdsigmoid(in)
    out[CNN_UIMUL(r, strideOut) + c] =
        1.7159f * tanhf(0.66666669f * in[CNN_UIMUL(r, strideIn) + c]);
}

__global__ void
stdsigmoid_bprop_kernel(float * in, size_t const strideIn,
                        float const * out, size_t const strideOut,
                        size_t const rows, size_t const cols)
{
    size_t const r = CNN_UIMUL(blockIdx.y, blockDim.y) + threadIdx.y;
    size_t const c = CNN_UIMUL(blockIdx.x, blockDim.x) + threadIdx.x;

    if (r >= rows || c >= cols)
        return;

    // Compute: in = dstdsigmoid(in) .* out
    in[CNN_UIMUL(r, strideIn) + c] =
        (1.1439333f - 1.1439333f *
        powf(tanhf(0.66666669f * in[CNN_UIMUL(r, strideIn) + c]), 2)) *
        out[CNN_UIMUL(r, strideOut) + c];
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#include "cusquasher.hh"
#include "cumvli.hh"
#include "error.hh"

CNNPLUS_NS_BEGIN

///////////////////////////////////////////////////////////////////////////////
// CuLogSigmoid

template<> void
CuLogSigmoid<float>::fprop(float const * in, size_t const strideIn,
                           float * out, size_t const strideOut)
{
    CNNPLUS_ASSERT(in  && strideIn  >= this->size_.width());
    CNNPLUS_ASSERT(out && strideOut >= this->size_.width());

    dim3 const dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 const dimGrid((this->size_.width() + dimBlock.x - 1) / dimBlock.x,
                       (this->size_.height() + dimBlock.y - 1) / dimBlock.y);

    logsigmoid_fprop_kernel<<<dimGrid, dimBlock>>>
        (in, strideIn, out, strideOut, this->size_.height(), this->size_.width());
    CUDA_CHECK_ERROR("Kernel call 'logsigmoid_fprop_kernel' failed");
}

template<> void
CuLogSigmoid<double>::fprop(double const * in, size_t const strideIn,
                            double * out, size_t const strideOut)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

template<> void
CuLogSigmoid<float>::bprop(float * in, size_t const strideIn,
                           float const * out, size_t const strideOut)
{
    CNNPLUS_ASSERT(in  && strideIn  >= this->size_.width());
    CNNPLUS_ASSERT(out && strideOut >= this->size_.width());

    dim3 const dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 const dimGrid((this->size_.width() + dimBlock.x - 1) / dimBlock.x,
                       (this->size_.height() + dimBlock.y - 1) / dimBlock.y);

    logsigmoid_bprop_kernel<<<dimGrid, dimBlock>>>
        (in, strideIn, out, strideOut, this->size_.height(), this->size_.width());
    CUDA_CHECK_ERROR("Kernel call 'logsigmoid_bprop_kernel' failed");
}

template<> void
CuLogSigmoid<double>::bprop(double * in, size_t const strideIn,
                            double const * out, size_t const strideOut)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

///////////////////////////////////////////////////////////////////////////////
// CuTanh

template<> void
CuTanh<float>::fprop(float const * in, size_t const strideIn,
                     float * out, size_t const strideOut)
{
    CNNPLUS_ASSERT(in  && strideIn  >= this->size_.width());
    CNNPLUS_ASSERT(out && strideOut >= this->size_.width());

    dim3 const dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 const dimGrid((this->size_.width() + dimBlock.x - 1) / dimBlock.x,
                       (this->size_.height() + dimBlock.y - 1) / dimBlock.y);

    tanh_fprop_kernel<<<dimGrid, dimBlock>>>
        (in, strideIn, out, strideOut, this->size_.height(), this->size_.width());
    CUDA_CHECK_ERROR("Kernel call 'tanh_fprop_kernel' failed");
}

template<> void
CuTanh<double>::fprop(double const * in, size_t const strideIn,
                      double * out, size_t const strideOut)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

template<> void
CuTanh<float>::bprop(float * in, size_t const strideIn,
                     float const * out, size_t const strideOut)
{
    CNNPLUS_ASSERT(in  && strideIn  >= this->size_.width());
    CNNPLUS_ASSERT(out && strideOut >= this->size_.width());

    dim3 const dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 const dimGrid((this->size_.width() + dimBlock.x - 1) / dimBlock.x,
                       (this->size_.height() + dimBlock.y - 1) / dimBlock.y);

    tanh_bprop_kernel<<<dimGrid, dimBlock>>>
        (in, strideIn, out, strideOut, this->size_.height(), this->size_.width());
    CUDA_CHECK_ERROR("Kernel call 'tanh_bprop_kernel' failed");
}

template<> void
CuTanh<double>::bprop(double * in, size_t const strideIn,
                      double const * out, size_t const strideOut)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

///////////////////////////////////////////////////////////////////////////////
// CuStdSigmoid

template<> void
CuStdSigmoid<float>::fprop(float const * in, size_t const strideIn,
                           float * out, size_t const strideOut)
{
    CNNPLUS_ASSERT(in  && strideIn  >= this->size_.width());
    CNNPLUS_ASSERT(out && strideOut >= this->size_.width());

    dim3 const dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 const dimGrid((this->size_.width() + dimBlock.x - 1) / dimBlock.x,
                       (this->size_.height() + dimBlock.y - 1) / dimBlock.y);

    stdsigmoid_fprop_kernel<<<dimGrid, dimBlock>>>
        (in, strideIn, out, strideOut, this->size_.height(), this->size_.width());
    CUDA_CHECK_ERROR("Kernel call 'stdsigmoid_fprop_kernel' failed");
}

template<> void
CuStdSigmoid<double>::fprop(double const * in, size_t const strideIn,
                            double * out, size_t const strideOut)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

template<> void
CuStdSigmoid<float>::bprop(float * in, size_t const strideIn,
                           float const * out, size_t const strideOut)
{
    CNNPLUS_ASSERT(in  && strideIn  >= this->size_.width());
    CNNPLUS_ASSERT(out && strideOut >= this->size_.width());

    dim3 const dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 const dimGrid((this->size_.width() + dimBlock.x - 1) / dimBlock.x,
                       (this->size_.height() + dimBlock.y - 1) / dimBlock.y);

    stdsigmoid_bprop_kernel<<<dimGrid, dimBlock>>>
        (in, strideIn, out, strideOut, this->size_.height(), this->size_.width());
    CUDA_CHECK_ERROR("Kernel call 'stdsigmoid_bprop_kernel' failed");
}

template<> void
CuStdSigmoid<double>::bprop(double * in, size_t const strideIn,
                            double const * out, size_t const strideOut)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

///////////////////////////////////////////////////////////////////////////////
// CuIdentity

template<> void
CuIdentity<float>::fprop(float const * in, size_t const strideIn,
                         float * out, size_t const strideOut)
{
    CNNPLUS_ASSERT(in  && strideIn  >= this->size_.width());
    CNNPLUS_ASSERT(out && strideOut >= this->size_.width());

    // Copy matrix 'in' to 'out'
    cumvli::copymm<float>(in, strideIn, out, strideOut,
                          this->size_.height(), this->size_.width());
}

template<> void
CuIdentity<double>::fprop(double const * in, size_t const strideIn,
                          double * out, size_t const strideOut)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

template<> void
CuIdentity<float>::bprop(float * in, size_t const strideIn,
                         float const * out, size_t const strideOut)
{
    CNNPLUS_ASSERT(in  && strideIn  >= this->size_.width());
    CNNPLUS_ASSERT(out && strideOut >= this->size_.width());

    // Copy matrix 'out' to 'in'
    cumvli::copymm<float>(out, strideOut, in, strideIn,
                          this->size_.height(), this->size_.width());
}

template<> void
CuIdentity<double>::bprop(double * in, size_t const strideIn,
                          double const * out, size_t const strideOut)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

///////////////////////////////////////////////////////////////////////////////

/*! \addtogroup eti_grp Explicit Template Instantiation
 @{
 */
template class CuLogSigmoid<float>;
template class CuTanh<float>;
template class CuStdSigmoid<float>;
template class CuIdentity<float>;

template class CuLogSigmoid<double>;
template class CuTanh<double>;
template class CuStdSigmoid<double>;
template class CuIdentity<double>;
/*! @} */

CNNPLUS_NS_END
