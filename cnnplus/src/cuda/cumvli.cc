/**************************************************************************//**
 *
 * \file   cumvli.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   May 26 2009
 *
 * $Id: cumvli.cc 3560 2010-11-22 20:47:19Z klaus $
 *
 * \brief  Implementation of cnnplus::cumvli.
 *
 *****************************************************************************/

#include "cumvli.hh"
#include "error.hh"
#include "cudautils.hh"
#include "mathli.hh"
#include <cuda.h>
#include <cublas.h>

#ifdef NDEBUG
#define CUBLAS_SAFE_CALL(call) call
#define CUBLAS_CHECK_ERROR(message)
#else
#define CUBLAS_SAFE_CALL(call) CNNPLUS_ASSERT(call == CUBLAS_STATUS_SUCCESS)
#define CUBLAS_CHECK_ERROR(message) \
    CNNPLUS_ASSERT(cublasGetError() == CUBLAS_STATUS_SUCCESS && message)
#endif

//! CUDA initialization structur
static struct CudaInit {
    CudaInit()
    {
        int deviceCount = 0;
        if (cudaGetDeviceCount(&deviceCount) != cudaSuccess ||
            deviceCount < 1) {
            printf("ERROR: No CUDA device available!\n");
            exit(EXIT_FAILURE);
        }
#if 1
        if (deviceCount == 1)
            printf("There is 1 device supporting CUDA.\n");
        else
            printf("There are %d devices supporting CUDA.\n", deviceCount);

        for (int i = 0; i < deviceCount; ++i)
        {
            cudaDeviceProp deviceProp = { 0 };
            CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, i));

            printf("\nDevice %d: \"%s\"\n", i, deviceProp.name);
            printf("  Major revision number:                         %d\n",
                   deviceProp.major);
            printf("  Minor revision number:                         %d\n",
                   deviceProp.minor);
            printf("  Total amount of global memory:                 %u bytes\n",
                   deviceProp.totalGlobalMem);
        #if CUDART_VERSION >= 2000
            printf("  Number of multiprocessors:                     %d\n",
                   deviceProp.multiProcessorCount);
            printf("  Number of cores:                               %d\n",
                   8 * deviceProp.multiProcessorCount);
        #endif
            printf("  Total amount of constant memory:               %u bytes\n",
                   deviceProp.totalConstMem);
            printf("  Total amount of shared memory per block:       %u bytes\n",
                   deviceProp.sharedMemPerBlock);
            printf("  Total number of registers available per block: %d\n",
                   deviceProp.regsPerBlock);
            printf("  Warp size:                                     %d\n",
                   deviceProp.warpSize);
            printf("  Maximum number of threads per block:           %d\n",
                   deviceProp.maxThreadsPerBlock);
            printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
                   deviceProp.maxThreadsDim[0],
                   deviceProp.maxThreadsDim[1],
                   deviceProp.maxThreadsDim[2]);
            printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
                   deviceProp.maxGridSize[0],
                   deviceProp.maxGridSize[1],
                   deviceProp.maxGridSize[2]);
            printf("  Maximum memory pitch:                          %u bytes\n",
                   deviceProp.memPitch);
            printf("  Texture alignment:                             %u bytes\n",
                   deviceProp.textureAlignment);
            printf("  Clock rate:                                    %.2f GHz\n",
                   deviceProp.clockRate * 1e-6f);
        #if CUDART_VERSION >= 2000
            printf("  Concurrent copy and execution:                 %s\n",
                   deviceProp.deviceOverlap ? "Yes" : "No");
        #endif
        }
        printf("\n"); fflush(stdout);
#endif
        // CUDA Initialization
        cuInit(0);
        // Initializes the CUBLAS library
        CUBLAS_SAFE_CALL(cublasInit());
    }

    ~CudaInit()
    {
        // Releases CPU-side resources used by the CUBLAS library
        CUBLAS_SAFE_CALL(cublasShutdown());
    }
} s_CudaInit;

CNNPLUS_NS_BEGIN

namespace cumvli {

//
// === Forward declarations ===
//
void cu_sumrow(float const * src, size_t stride, float * dst, size_t rows, size_t cols);
void cu_sumrowacc(float const * src, size_t stride, float * dst, size_t rows, size_t cols);
void cu_pmulmm(float const * src1, size_t strideSrc1, float const * src2, size_t strideSrc2,
               float * dst, size_t strideDst, size_t rows, size_t cols);
void cu_axpy(float const * src, size_t strideSrc, size_t rows, size_t cols,
             float * srcDst, size_t strideSrcDst, float alpha);
void cu_gemv(float const * src1, size_t strideSrc1, size_t rowsSrc1, size_t colsSrc1,
             float const * src2, size_t lenSrc2, float * srcDst, float alpha, float beta);
void cu_gemv(float const * src1, size_t strideSrc1, size_t rowsSrc1, size_t colsSrc1,
             float const * src2, size_t lenSrc2, float const * src3, size_t lenSrc3, float * dst);
void cu_setv(float * dst, size_t len, float val);
void cu_setm(float * dst, size_t stride, size_t rows, size_t cols, float val);
void cu_mulv(float const * src1, float const * src2, float * dst, size_t len);
void cu_setcol(float * dst, size_t stride, size_t rows, size_t cols, float const * src);

//! Returns the free GPU memory in bytes
size_t freemem()
{
    unsigned int memFree = 0, memTotal = 0;

    CUcontext cuContext;
    CUdevice cuDevice = 0;
    cuCtxCreate(&cuContext, 0, cuDevice);
    cuMemGetInfo(&memFree, &memTotal);
    cuCtxDetach (cuContext);

    return memFree;
}

//! Allocates GPU memory for a vector of length \a len (single precision)
template<>
float * allocv<float>(size_t len)
{
    CNNPLUS_ASSERT(len > 0);

    void * ptr = NULL;
    CUDA_SAFE_CALL(cudaMalloc(&ptr, len * sizeof(float)));
    return static_cast<float*>(ptr);
}

//! Allocates GPU memory for a vector of length \a len (double precision)
template<>
double * allocv<double>(size_t len)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Allocates GPU memory for a matrix of size \a rows x \a cols (single precision)
template<>
float * allocm<float>(size_t rows, size_t cols, size_t & stride)
{
    CNNPLUS_ASSERT(rows > 0 && cols > 0);
#if 1
    void * ptr = NULL;
    CUDA_SAFE_CALL(cudaMallocPitch(&ptr, &stride, cols * sizeof(float), rows));
    CNNPLUS_ASSERT(stride % sizeof(float) == 0);
    stride /= sizeof(float);
    return static_cast<float*>(ptr);
#else
    stride = cols;
    return allocv<float>(rows * cols);
#endif
}

//! Allocates GPU memory for a matrix of size \a rows x \a cols (double precision)
template<>
double * allocm<double>(size_t rows, size_t cols, size_t & stride)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Frees the allocated GPU memory (single precision)
template<>
void free<float>(float * ptr)
{
    CUDA_SAFE_CALL(cudaFree(ptr));
}

//! Frees the allocated GPU memory (double precision)
template<>
void free<double>(double * ptr)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Returns a GPU vector element (single precision)
template<>
float getelv<float>(float const * src, size_t i)
{
    CNNPLUS_ASSERT(src);

    float val = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&val, src + i, sizeof(float), cudaMemcpyDeviceToHost));
    return val;
}

//! Returns a GPU vector element (double precision)
template<>
double getelv<double>(double const * src, size_t i)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Returns a GPU matrix element (single precision)
template<>
float getelm<float>(float const * src, size_t stride, size_t r, size_t c)
{
    CNNPLUS_ASSERT(src && stride > 0);

    float val = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&val, src + r * stride + c, sizeof(float), cudaMemcpyDeviceToHost));
    return val;
}

//! Returns a GPU matrix element (double precision)
template<>
double getelm<double>(double const * src, size_t stride, size_t r, size_t c)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Sets a GPU vector element (single precision)
template<>
void setelv<float>(float * dst, size_t i, float val)
{
    CNNPLUS_ASSERT(dst);

    CUDA_SAFE_CALL(cudaMemcpy(dst + i, &val, sizeof(float), cudaMemcpyHostToDevice));
}

//! Sets a GPU vector element (double precision)
template<>
void setelv<double>(double * dst, size_t i, double val)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Sets a GPU matrix element (single precision)
template<>
void setelm<float>(float * dst, size_t stride, size_t r, size_t c, float val)
{
    CNNPLUS_ASSERT(dst && stride > 0);

    CUDA_SAFE_CALL(cudaMemcpy(dst + r * stride + c, &val, sizeof(float), cudaMemcpyHostToDevice));
}

//! Sets a GPU matrix element (double precision)
template<>
void setelm<double>(double * dst, size_t stride, size_t r, size_t c, double val)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Copies the contents of one GPU vector into another (single precision)
template<>
void copyvv<float>(float const * src, float * dst, size_t len)
{
    CNNPLUS_ASSERT(src && dst && len > 0);

    CUDA_SAFE_CALL(cudaMemcpy(dst, src, len * sizeof(float), cudaMemcpyDeviceToDevice));
}

//! Copies the contents of one GPU vector into another (double precision)
template<>
void copyvv<double>(double const * src, double * dst, size_t len)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Copies the contents of one GPU matrix into another (single precision)
template<>
void copymm<float>(float const * src, size_t strideSrc, float * dst, size_t strideDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && strideSrc >= cols);
    CNNPLUS_ASSERT(dst && strideDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    if (strideSrc == strideDst) {
        copyvv(src, dst, rows * strideSrc);
    }
    else {
        //for (size_t r = 0; r < rows; ++r)
        //    copyvv(src + r * strideSrc, dst + r * strideDst, cols);
        CUDA_SAFE_CALL(
            cudaMemcpy2D(dst, strideDst * sizeof(float), src, strideSrc * sizeof(float),
                         cols * sizeof(float), rows, cudaMemcpyDeviceToDevice));
    }
}

//! Copies the contents of one GPU matrix into another (double precision)
template<>
void copymm<double>(double const * src, size_t strideSrc, double * dst, size_t strideDst, size_t rows, size_t cols)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Copies the contents of a GPU matrix into a GPU vector (single precision)
template<>
void copymv<float>(float const * src, size_t strideSrc, float * dst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && dst && strideSrc >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    if (strideSrc == cols) copyvv(src, dst, rows * cols);
    else copymm(src, strideSrc, dst, cols, rows, cols);
}

//! Copies the contents of a GPU matrix into a GPU vector (double precision)
template<>
void copymv<double>(double const * src, size_t strideSrc, double * dst, size_t rows, size_t cols)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Copies the contents of a GPU vector into a GPU matrix (single precision)
template<>
void copyvm<float>(float const * src, float * dst, size_t strideDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && dst && strideDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    if (strideDst == cols) copyvv(src, dst, rows * cols);
    else copymm(src, cols, dst, strideDst, rows, cols);
}

//! Copies the contents of a GPU vector into a GPU matrix (double precision)
template<>
void copyvm<double>(double const * src, double * dst, size_t strideDst, size_t rows, size_t cols)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Copies a vector from CPU memory to GPU memory (single precision)
template<>
void copyv_h2d<float>(float const * src, float * dst, size_t len)
{
    CNNPLUS_ASSERT(src && dst && len > 0);

    CUDA_SAFE_CALL(cudaMemcpy(dst, src, len * sizeof(float), cudaMemcpyHostToDevice));
}

//! Copies a vector from CPU memory to GPU memory (double precision)
template<>
void copyv_h2d<double>(double const * src, double * dst, size_t len)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Copies a vector from GPU memory to CPU memory (single precision)
template<>
void copyv_d2h<float>(float const * src, float * dst, size_t len)
{
    CNNPLUS_ASSERT(src && dst && len > 0);

    CUDA_SAFE_CALL(cudaMemcpy(dst, src, len * sizeof(float), cudaMemcpyDeviceToHost));
}

//! Copies a vector from GPU memory to CPU memory (double precision)
template<>
void copyv_d2h<double>(double const * src, double * dst, size_t len)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Copies a matrix from CPU memory to GPU memory (single precision)
template<>
void copym_h2d<float>(float const * src, size_t strideSrc, float * dst, size_t strideDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && strideSrc >= cols);
    CNNPLUS_ASSERT(dst && strideDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    if (strideSrc == strideDst) {
        copyv_h2d(src, dst, rows * strideSrc);
    }
    else {
        //for (size_t r = 0; r < rows; ++r)
        //    copyv_h2d(src + r * strideSrc, dst + r * strideDst, cols);
        CUDA_SAFE_CALL(
            cudaMemcpy2D(dst, strideDst * sizeof(float), src, strideSrc * sizeof(float),
                         cols * sizeof(float), rows, cudaMemcpyHostToDevice));
    }
}

//! Copies a matrix from CPU memory to GPU memory (double precision)
template<>
void copym_h2d<double>(double const * src, size_t strideSrc, double * dst, size_t strideDst, size_t rows, size_t cols)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Copies a matrix from GPU memory to CPU memory (single precision)
template<>
void copym_d2h<float>(float const * src, size_t strideSrc, float * dst, size_t strideDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && strideSrc >= cols);
    CNNPLUS_ASSERT(dst && strideDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    if (strideSrc == strideDst) {
        copyv_d2h(src, dst, rows * strideSrc);
    }
    else {
        //for (size_t r = 0; r < rows; ++r)
        //    copyv_d2h(src + r * strideSrc, dst + r * strideDst, cols);
        CUDA_SAFE_CALL(
            cudaMemcpy2D(dst, strideDst * sizeof(float), src, strideSrc * sizeof(float),
                         cols * sizeof(float), rows, cudaMemcpyDeviceToHost));
    }
}

//! Copies a matrix from GPU memory to CPU memory (double precision)
template<>
void copym_d2h<double>(double const * src, size_t strideSrc, double * dst, size_t strideDst, size_t rows, size_t cols)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Initializes a GPU vector to zero (single precision)
template<>
void zerov<float>(float * dst, size_t len)
{
    CNNPLUS_ASSERT(dst && len > 0);

    CUDA_SAFE_CALL(cudaMemset(dst, 0, len * sizeof(float)));
}

//! Initializes a GPU vector to zero (double precision)
template<>
void zerov<double>(double * dst, size_t len)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Initializes a GPU matrix to zero (single precision)
template<>
void zerom<float>(float * dst, size_t stride, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(dst && stride >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    if (stride == cols) {
        zerov(dst, rows * stride);
    }
    else {
        CUDA_SAFE_CALL(
            cudaMemset2D(dst, stride * sizeof(float), 0, cols * sizeof(float), rows));
    }
}

//! Initializes a GPU matrix to zero (double precision)
template<>
void zerom<double>(double * dst, size_t stride, size_t rows, size_t cols)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Computes a vector-scalar product and adds the result to a vector (single precision)
template<>
void axpy<float>(float const * src, size_t len, float * srcDst, float alpha)
{
    CNNPLUS_ASSERT(src && srcDst && len > 0);

    cublasSaxpy(static_cast<int>(len), alpha, src, 1, srcDst, 1);
    CUBLAS_CHECK_ERROR("Function 'cublasSaxpy' failed.");
}

//! Computes a vector-scalar product and adds the result to a vector (double precision)
template<>
void axpy<double>(double const * src, size_t len, double * srcDst, double alpha)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Computes a matrix-vector product using a general matrix (single precision)
template<>
void gemv<float,'n'>(float const * src1, size_t strideSrc1,
                     size_t rowsSrc1, size_t colsSrc1,
                     float const * src2, size_t lenSrc2,
                     float * srcDst, float alpha, float beta)
{
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && lenSrc2 == colsSrc1);
    CNNPLUS_ASSERT(srcDst);

#if 0
    // CUBLAS uses column-major storage, so we have to use the transposed version
    cublasSgemv('t', colsSrc1, rowsSrc1, alpha, src1, strideSrc1, src2, 1, beta, srcDst, 1);
    CUBLAS_CHECK_ERROR("Function 'cublasSgemv' failed.");
#else
    cu_gemv(src1, strideSrc1, rowsSrc1, colsSrc1, src2, lenSrc2, srcDst, alpha, beta);
#endif
}

//! Computes a matrix-vector product using a general matrix (double precision)
template<>
void gemv<double,'n'>(double const * src1, size_t strideSrc1,
                      size_t rowsSrc1, size_t colsSrc1,
                      double const * src2, size_t lenSrc2,
                      double * srcDst, double alpha, double beta)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Computes a transposed-matrix-vector product using a general matrix (single precision)
template<>
void gemv<float,'t'>(float const * src1, size_t strideSrc1,
                     size_t rowsSrc1, size_t colsSrc1,
                     float const * src2, size_t lenSrc2,
                     float * srcDst, float alpha, float beta)
{
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && lenSrc2 == rowsSrc1);
    CNNPLUS_ASSERT(srcDst);

    // CUBLAS uses column-major storage, so we have to use the non-transposed version
    cublasSgemv('n', colsSrc1, rowsSrc1, alpha, src1, strideSrc1, src2, 1, beta, srcDst, 1);
    CUBLAS_CHECK_ERROR("Function 'cublasSgemv' failed.");
}

//! Computes a transposed-matrix-vector product using a general matrix (double precision)
template<>
void gemv<double,'t'>(double const * src1, size_t strideSrc1,
                      size_t rowsSrc1, size_t colsSrc1,
                      double const * src2, size_t lenSrc2,
                      double * srcDst, double alpha, double beta)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Computes a matrix-vector product using a general matrix (single precision)
template<>
void gemv<float>(float const * src1, size_t strideSrc1,
                 size_t rowsSrc1, size_t colsSrc1,
                 float const * src2, size_t lenSrc2,
                 float const * src3, size_t lenSrc3,
                 float * dst)
{
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && lenSrc2 == colsSrc1);
    CNNPLUS_ASSERT(src3 && lenSrc3 == rowsSrc1);
    CNNPLUS_ASSERT(dst);

#if 0
    copyvv<float>(src3, dst, lenSrc3);
    // CUBLAS uses column-major storage, so we have to use the transposed version
    cublasSgemv('t', colsSrc1, rowsSrc1, 1, src1, strideSrc1, src2, 1, 1, dst, 1);
    CUBLAS_CHECK_ERROR("Function 'cublasSgemv' failed.");
#else
    cu_gemv(src1, strideSrc1, rowsSrc1, colsSrc1, src2, lenSrc2, src3, lenSrc3, dst);
#endif
}

//! Computes a matrix-vector product using a general matrix (double precision)
template<>
void gemv<double>(double const * src1, size_t strideSrc1,
                  size_t rowsSrc1, size_t colsSrc1,
                  double const * src2, size_t lenSrc2,
                  double const * src3, size_t lenSrc3,
                  double * dst)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Performs a rank-1 update of a general matrix (single precision)
template<>
void ger<float>(float const * src1, size_t lenSrc1,
                float const * src2, size_t lenSrc2,
                float * srcDst, size_t strideSrcDst,
                float alpha)
{
    CNNPLUS_ASSERT(src1 && lenSrc1 > 0);
    CNNPLUS_ASSERT(src2 && lenSrc2 > 0);
    CNNPLUS_ASSERT(srcDst && strideSrcDst >= lenSrc2);

    // CUBLAS uses column-major storage, so we have to swap 'src1' and 'src2'
    cublasSger(lenSrc2, lenSrc1, alpha, src2, 1, src1, 1, srcDst, strideSrcDst);
    CUBLAS_CHECK_ERROR("Function 'cublasSger' failed.");
}

//! Performs a rank-1 update of a general matrix (double precision)
template<>
void ger<double>(double const * src1, size_t lenSrc1,
                 double const * src2, size_t lenSrc2,
                 double * srcDst, size_t strideSrcDst,
                 double alpha)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Adds the elements of two vectors (in-place, single precision)
template<>
void addv<float>(float * srcDst, float const * src, size_t len)
{
    CNNPLUS_ASSERT(srcDst && src && len > 0);

    axpy<float>(src, len, srcDst, 1);
}

//! Adds the elements of two vectors (in-place, double precision)
template<>
void addv<double>(double * srcDst, double const * src, size_t len)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Computes a scalar-matrix-matrix product and adds the result to a scalar-matrix product (single precision)
template<>
void gemm<float,'n','n'>(float const * src1, size_t strideSrc1,
                         size_t rowsSrc1, size_t colsSrc1,
                         float const * src2, size_t strideSrc2,
                         size_t rowsSrc2, size_t colsSrc2,
                         float * srcDst, size_t strideSrcDst,
                         float alpha, float beta)
{
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && strideSrc2 >= colsSrc2);
    CNNPLUS_ASSERT(rowsSrc2 > 0 && colsSrc2 > 0);
    CNNPLUS_ASSERT(rowsSrc2 == colsSrc1);
    CNNPLUS_ASSERT(srcDst && strideSrcDst >= colsSrc2);

    cublasSgemm('n', 'n', static_cast<int>(colsSrc2),
        static_cast<int>(rowsSrc1), static_cast<int>(colsSrc1),
        alpha, src2, static_cast<int>(strideSrc2),
        src1, static_cast<int>(strideSrc1),
        beta, srcDst, static_cast<int>(strideSrcDst));
    CUBLAS_CHECK_ERROR("Function 'cublasSgemm' failed.");
}

//! Computes a scalar-matrix-matrix product and adds the result to a scalar-matrix product (double precision)
template<>
void gemm<double,'n','n'>(double const * src1, size_t strideSrc1,
                          size_t rowsSrc1, size_t colsSrc1,
                          double const * src2, size_t strideSrc2,
                          size_t rowsSrc2, size_t colsSrc2,
                          double * srcDst, size_t strideSrcDst,
                          double alpha, double beta)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Computes a scalar-matrix-transposed-matrix product and adds the result to a scalar-matrix product (single precision)
template<>
void gemm<float,'n','t'>(float const * src1, size_t strideSrc1,
                         size_t rowsSrc1, size_t colsSrc1,
                         float const * src2, size_t strideSrc2,
                         size_t rowsSrc2, size_t colsSrc2,
                         float * srcDst, size_t strideSrcDst,
                         float alpha, float beta)
{
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && strideSrc2 >= colsSrc2);
    CNNPLUS_ASSERT(rowsSrc2 > 0 && colsSrc2 > 0);
    CNNPLUS_ASSERT(colsSrc2 == colsSrc1);
    CNNPLUS_ASSERT(srcDst && strideSrcDst >= rowsSrc2);

    cublasSgemm('t', 'n', static_cast<int>(rowsSrc2),
        static_cast<int>(rowsSrc1), static_cast<int>(colsSrc1),
        alpha, src2, static_cast<int>(strideSrc2),
        src1, static_cast<int>(strideSrc1),
        beta, srcDst, static_cast<int>(strideSrcDst));
    CUBLAS_CHECK_ERROR("Function 'cublasSgemm' failed.");
}

//! Computes a scalar-matrix-transposed-matrix product and adds the result to a scalar-matrix product (double precision)
template<>
void gemm<double,'n','t'>(double const * src1, size_t strideSrc1,
                          size_t rowsSrc1, size_t colsSrc1,
                          double const * src2, size_t strideSrc2,
                          size_t rowsSrc2, size_t colsSrc2,
                          double * srcDst, size_t strideSrcDst,
                          double alpha, double beta)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Computes a scalar-transposed-matrix-matrix product and adds the result to a scalar-matrix product (single precision)
template<>
void gemm<float,'t','n'>(float const * src1, size_t strideSrc1,
                         size_t rowsSrc1, size_t colsSrc1,
                         float const * src2, size_t strideSrc2,
                         size_t rowsSrc2, size_t colsSrc2,
                         float * srcDst, size_t strideSrcDst,
                         float alpha, float beta)
{
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && strideSrc2 >= colsSrc2);
    CNNPLUS_ASSERT(rowsSrc2 > 0 && colsSrc2 > 0);
    CNNPLUS_ASSERT(rowsSrc2 == rowsSrc1);
    CNNPLUS_ASSERT(srcDst && strideSrcDst >= colsSrc2);

    cublasSgemm('n', 't', static_cast<int>(colsSrc2),
        static_cast<int>(colsSrc1), static_cast<int>(rowsSrc1),
        alpha, src2, static_cast<int>(strideSrc2),
        src1, static_cast<int>(strideSrc1),
        beta, srcDst, static_cast<int>(strideSrcDst));
    CUBLAS_CHECK_ERROR("Function 'cublasSgemm' failed.");
}

//! Computes a scalar-transposed-matrix-matrix product and adds the result to a scalar-matrix product (double precision)
template<>
void gemm<double,'t','n'>(double const * src1, size_t strideSrc1,
                          size_t rowsSrc1, size_t colsSrc1,
                          double const * src2, size_t strideSrc2,
                          size_t rowsSrc2, size_t colsSrc2,
                          double * srcDst, size_t strideSrcDst,
                          double alpha, double beta)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Computes a scalar-transposed-matrix-transposed-matrix product and adds the result to a scalar-matrix product (single precision)
template<>
void gemm<float,'t','t'>(float const * src1, size_t strideSrc1,
                         size_t rowsSrc1, size_t colsSrc1,
                         float const * src2, size_t strideSrc2,
                         size_t rowsSrc2, size_t colsSrc2,
                         float * srcDst, size_t strideSrcDst,
                         float alpha, float beta)
{
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && strideSrc2 >= colsSrc2);
    CNNPLUS_ASSERT(rowsSrc2 > 0 && colsSrc2 > 0);
    CNNPLUS_ASSERT(colsSrc2 == rowsSrc1);
    CNNPLUS_ASSERT(srcDst && strideSrcDst >= rowsSrc2);

    cublasSgemm('t', 't', static_cast<int>(rowsSrc2),
        static_cast<int>(colsSrc1), static_cast<int>(rowsSrc1),
        alpha, src2, static_cast<int>(strideSrc2),
        src1, static_cast<int>(strideSrc1),
        beta, srcDst, static_cast<int>(strideSrcDst));
    CUBLAS_CHECK_ERROR("Function 'cublasSgemm' failed.");
}

//! Computes a scalar-transposed-matrix-transposed-matrix product and adds the result to a scalar-matrix product (double precision)
template<>
void gemm<double,'t','t'>(double const * src1, size_t strideSrc1,
                          size_t rowsSrc1, size_t colsSrc1,
                          double const * src2, size_t strideSrc2,
                          size_t rowsSrc2, size_t colsSrc2,
                          double * srcDst, size_t strideSrcDst,
                          double alpha, double beta)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Computes a matrix-matrix product (single precision)
template<>
void mulmm<float,'n','n'>(float const * src1, size_t strideSrc1,
                          size_t rowsSrc1, size_t colsSrc1,
                          float const * src2, size_t strideSrc2,
                          size_t rowsSrc2, size_t colsSrc2,
                          float * dst, size_t strideDst)
{
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && strideSrc2 >= colsSrc2);
    CNNPLUS_ASSERT(rowsSrc2 > 0 && colsSrc2 > 0);
    CNNPLUS_ASSERT(rowsSrc2 == colsSrc1);
    CNNPLUS_ASSERT(dst && strideDst >= colsSrc2);

    gemm<float,'n','n'>(src1, strideSrc1, rowsSrc1, colsSrc1,
        src2, strideSrc2, rowsSrc2, colsSrc2, dst, strideDst, 1, 0);
}

//! Computes a matrix-matrix product (double precision)
template<>
void mulmm<double,'n','n'>(double const * src1, size_t strideSrc1,
                           size_t rowsSrc1, size_t colsSrc1,
                           double const * src2, size_t strideSrc2,
                           size_t rowsSrc2, size_t colsSrc2,
                           double * dst, size_t strideDst)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Computes a matrix-transposed-matrix product (single precision)
template<>
void mulmm<float,'n','t'>(float const * src1, size_t strideSrc1,
                          size_t rowsSrc1, size_t colsSrc1,
                          float const * src2, size_t strideSrc2,
                          size_t rowsSrc2, size_t colsSrc2,
                          float * dst, size_t strideDst)
{
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && strideSrc2 >= colsSrc2);
    CNNPLUS_ASSERT(rowsSrc2 > 0 && colsSrc2 > 0);
    CNNPLUS_ASSERT(colsSrc2 == colsSrc1);
    CNNPLUS_ASSERT(dst && strideDst >= rowsSrc2);

    gemm<float,'n','t'>(src1, strideSrc1, rowsSrc1, colsSrc1,
        src2, strideSrc2, rowsSrc2, colsSrc2, dst, strideDst, 1, 0);
}

//! Computes a matrix-transposed-matrix product (double precision)
template<>
void mulmm<double,'n','t'>(double const * src1, size_t strideSrc1,
                           size_t rowsSrc1, size_t colsSrc1,
                           double const * src2, size_t strideSrc2,
                           size_t rowsSrc2, size_t colsSrc2,
                           double * dst, size_t strideDst)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Computes a transposed-matrix-matrix product (single precision)
template<>
void mulmm<float,'t','n'>(float const * src1, size_t strideSrc1,
                          size_t rowsSrc1, size_t colsSrc1,
                          float const * src2, size_t strideSrc2,
                          size_t rowsSrc2, size_t colsSrc2,
                          float * dst, size_t strideDst)
{
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && strideSrc2 >= colsSrc2);
    CNNPLUS_ASSERT(rowsSrc2 > 0 && colsSrc2 > 0);
    CNNPLUS_ASSERT(rowsSrc2 == rowsSrc1);
    CNNPLUS_ASSERT(dst && strideDst >= colsSrc2);

    gemm<float,'t','n'>(src1, strideSrc1, rowsSrc1, colsSrc1,
        src2, strideSrc2, rowsSrc2, colsSrc2, dst, strideDst, 1, 0);
}

//! Computes a transposed-matrix-matrix product (double precision)
template<>
void mulmm<double,'t','n'>(double const * src1, size_t strideSrc1,
                           size_t rowsSrc1, size_t colsSrc1,
                           double const * src2, size_t strideSrc2,
                           size_t rowsSrc2, size_t colsSrc2,
                           double * dst, size_t strideDst)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Computes a transposed-matrix-transposed-matrix product (single precision)
template<>
void mulmm<float,'t','t'>(float const * src1, size_t strideSrc1,
                          size_t rowsSrc1, size_t colsSrc1,
                          float const * src2, size_t strideSrc2,
                          size_t rowsSrc2, size_t colsSrc2,
                          float * dst, size_t strideDst)
{
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && strideSrc2 >= colsSrc2);
    CNNPLUS_ASSERT(rowsSrc2 > 0 && colsSrc2 > 0);
    CNNPLUS_ASSERT(colsSrc2 == rowsSrc1);
    CNNPLUS_ASSERT(dst && strideDst >= rowsSrc2);

    gemm<float,'t','t'>(src1, strideSrc1, rowsSrc1, colsSrc1,
        src2, strideSrc2, rowsSrc2, colsSrc2, dst, strideDst, 1, 0);
}

//! Computes a transposed-matrix-transposed-matrix product (double precision)
template<>
void mulmm<double,'t','t'>(double const * src1, size_t strideSrc1,
                           size_t rowsSrc1, size_t colsSrc1,
                           double const * src2, size_t strideSrc2,
                           size_t rowsSrc2, size_t colsSrc2,
                           double * dst, size_t strideDst)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Calculates the sums of row vectors in a matrix (single precision)
template<>
void sumrow<float>(float const * src, size_t stride, float * dst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && dst && stride >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    cu_sumrow(src, stride, dst, rows, cols);
}

//! Calculates the sums of row vectors in a matrix (double precision)
template<>
void sumrow<double>(double const * src, size_t stride, double * dst, size_t rows, size_t cols)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Calculates the sums of row vectors in a matrix and accumulates it to a vector (single precision)
template<>
void sumrowacc<float>(float const * src, size_t stride, float * srcDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && srcDst && stride >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    cu_sumrowacc(src, stride, srcDst, rows, cols);
}

//! Calculates the sums of row vectors in a matrix and accumulates it to a vector (double precision)
template<>
void sumrowacc<double>(double const * src, size_t stride, double * srcDst, size_t rows, size_t cols)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Computes the pointwise multiplication of two matrices (single precision)
template<>
void pmulmm<float>(float const * src1, size_t strideSrc1,
                   float const * src2, size_t strideSrc2,
                   float * dst, size_t strideDst,
                   size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src1 && strideSrc1 >= cols);
    CNNPLUS_ASSERT(src2 && strideSrc2 >= cols);
    CNNPLUS_ASSERT(dst && strideDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    cu_pmulmm(src1, strideSrc1, src2, strideSrc2, dst, strideDst, rows, cols);
}

//! Computes the pointwise multiplication of two matrices (double precision)
template<>
void pmulmm<double>(double const * src1, size_t strideSrc1,
                    double const * src2, size_t strideSrc2,
                    double * dst, size_t strideDst,
                    size_t rows, size_t cols)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Computes the pointwise multiplication of two matrices (in-place, single precision)
template<>
void pmulmm<float>(float * srcDst, size_t strideSrcDst,
                   float const * src, size_t strideSrc,
                   size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(srcDst && strideSrcDst >= cols);
    CNNPLUS_ASSERT(src && strideSrc >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    cu_pmulmm(srcDst, strideSrcDst, src, strideSrc, srcDst, strideSrcDst, rows, cols);
}

//! Computes the pointwise multiplication of two matrices (in-place, double precision)
template<>
void pmulmm<double>(double * srcDst, size_t strideSrcDst,
                    double const * src, size_t strideSrc,
                    size_t rows, size_t cols)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Computes a matrix-scalar product and adds the result to a matrix (single precision)
template<>
void axpy<float>(float const * src, size_t strideSrc, size_t rows, size_t cols,
                 float * srcDst, size_t strideSrcDst, float alpha)
{
    CNNPLUS_ASSERT(src && strideSrc >= cols);
    CNNPLUS_ASSERT(srcDst && strideSrcDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    cu_axpy(src, strideSrc, rows, cols, srcDst, strideSrcDst, alpha);
}

//! Computes a matrix-scalar product and adds the result to a matrix (double precision)
template<>
void axpy<double>(double const * src, size_t strideSrc, size_t rows, size_t cols,
                  double * srcDst, size_t strideSrcDst, double alpha)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Initializes the vector elements to a specified common value (single precision)
template<>
void setv<float>(float * dst, size_t len, float val)
{
    CNNPLUS_ASSERT(dst && len > 0);

    cu_setv(dst, len, val);
}

//! Initializes the vector elements to a specified common value (double precision)
template<>
void setv<double>(double * dst, size_t len, double val)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Initializes the matrix elements to a specified common value (single precision)
template<>
void setm<float>(float * dst, size_t stride, size_t rows, size_t cols, float val)
{
    CNNPLUS_ASSERT(dst && stride >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    cu_setm(dst, stride, rows, cols, val);
}

//! Initializes the matrix elements to a specified common value (double precision)
template<>
void setm<double>(double * dst, size_t stride, size_t rows, size_t cols, double val)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Multiplies the elements of two vectors (single precision)
template<>
void mulv<float>(float const * src1, float const * src2, float * dst, size_t len)
{
    CNNPLUS_ASSERT(src1 && src2);
    CNNPLUS_ASSERT(dst && len > 0);

    cu_mulv(src1, src2, dst, len);
}

//! Multiplies the elements of two vectors (double precision)
template<>
void mulv<double>(double const * src1, double const * src2, double * dst, size_t len)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Multiplies the elements of two vectors (in-place, single precision)
template<>
void mulv<float>(float * srcDst, float const * src, size_t len)
{
    CNNPLUS_ASSERT(srcDst && src && len > 0);

    cu_mulv(srcDst, src, srcDst, len);
}

//! Multiplies the elements of two vectors (in-place, double precision)
template<>
void mulv<double>(double * srcDst, double const * src, size_t len)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Returns the maximum absolute value of a vector (single precision)
template<>
float absmaxv<float>(float const * src, size_t len)
{
    CNNPLUS_ASSERT(src && len > 0);

    return mathli::abs(getelv<float>(
        src, cublasIsamax(static_cast<int>(len), src, 1) - 1));
}

//! Returns the maximum absolute value of a vector (double precision)
template<>
double absmaxv<double>(double const * src, size_t len)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

//! Initializes the columns of a matrix (single precision)
template<>
void setcol<float>(float * dst, size_t stride, size_t rows, size_t cols, float const * src)
{
    CNNPLUS_ASSERT(dst && stride >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);
    CNNPLUS_ASSERT(src);

    cu_setcol(dst, stride, rows, cols, src);
}

//! Initializes the columns of a matrix (double precision)
template<>
void setcol<double>(double * dst, size_t stride, size_t rows, size_t cols, double const * src)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

}; // namespace cumvli

CNNPLUS_NS_END
