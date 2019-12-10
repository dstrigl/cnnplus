/**************************************************************************//**
 *
 * \file   matvecli.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jan 14 2009
 *
 * $Id: matvecli.cc 2467 2009-11-04 10:30:41Z dast $
 *
 * \brief  Implementation of cnnplus::matvecli.
 *
 *****************************************************************************/

#include "matvecli.hh"
#include "mathli.hh"
#include "randutil.h"
#include <stdio.h>

#ifdef CNNPLUS_USE_INTEL_LIBS
    // Intel Performance Libraries
    #include <mkl_service.h>
    #include <mkl_cblas.h>
    #include <ippcore.h>
    #include <ippm.h>
    #include <ipps.h>
    #include <ippsr.h>
    #include <ippvm.h>
    // Helper macro
    #ifdef NDEBUG
    #    define IPP_SAFE_CALL(call) call
    #else
    #    define IPP_SAFE_CALL(call) CNNPLUS_VERIFY(call == ippStsNoErr)
    #endif
#else
    #include <cstring>
#endif // CNNPLUS_USE_INTEL_LIBS

#ifdef CNNPLUS_USE_INTEL_LIBS
static struct IntelLibsInfo {
    IntelLibsInfo()
    {
        printf("Intel Integrated Performance Primitives (IPP):\n");
        IppLibraryVersion const * ippver = ippGetLibVersion();
        printf("  %s %s\n", ippver->Name, ippver->Version);
        printf("\n"); fflush(stdout);

        MKLVersion mklver;
        MKLGetVersion(&mklver);
        printf("Intel Math Kernel Library (MKL):\n");
        printf("  Major version: %d\n", mklver.MajorVersion);
        printf("  Minor version: %d\n", mklver.MinorVersion);
        //printf("  Update number: %d\n", mklver.BuildNumber);
        printf("  Product status: %s\n", mklver.ProductStatus);
        printf("  Build: %s\n", mklver.Build);
        printf("  Processor optimization: %s\n", mklver.Processor);
        printf("\n"); fflush(stdout);
    }
} s_IntelLibsInfo;
#endif

CNNPLUS_NS_BEGIN

namespace matvecli {

//! Seed value used by the pseudo-random number generation algorithm
static unsigned int s_seed = randreseed();

//! Reseeds the random number generator and return the seed used
unsigned int randreseed()
{
    return (s_seed = ::randreseed());
}

//! Allocates memory for a vector of length \a len
template<typename T>
T * allocv(size_t len)
{
    CNNPLUS_ASSERT(len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    return static_cast<T*>(ippMalloc(static_cast<int>(len * sizeof(T))));
#else
    return static_cast<T*>(malloc(len * sizeof(T)));
#endif
}

//! Allocates memory for a matrix of size \a rows x \a cols
template<typename T>
T * allocm(size_t rows, size_t cols, size_t & stride)
{
    CNNPLUS_ASSERT(rows > 0 && cols > 0);
    stride = (((cols * sizeof(T) + 31) / 32) * 32) / sizeof(T);
    return allocv<T>(rows * stride);
}

//! Frees the allocated memory
template<typename T>
void free(T * ptr)
{
#ifdef CNNPLUS_USE_INTEL_LIBS
    ippFree(ptr);
#else
    ::free(ptr);
#endif
}

//! Returns a vector element
template<typename T>
T getelv(T const * src, size_t i)
{
    CNNPLUS_ASSERT(src);
    return src[i];
}

//! Returns a matrix element
template<typename T>
T getelm(T const * src, size_t stride, size_t r, size_t c)
{
    CNNPLUS_ASSERT(src && stride > 0);
    return src[r * stride + c];
}

//! Sets a vector element
template<typename T>
void setelv(T * dst, size_t i, T val)
{
    CNNPLUS_ASSERT(dst);
    dst[i] = val;
}

//! Sets a matrix element
template<typename T>
void setelm(T * dst, size_t stride, size_t r, size_t c, T val)
{
    CNNPLUS_ASSERT(dst && stride > 0);
    dst[r * stride + c] = val;
}

//! Initializes a vector to zero (single precision)
template<>
void zerov<float>(float * dst, size_t len)
{
    CNNPLUS_ASSERT(dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsZero_32f(dst, static_cast<int>(len)));
#else
    memset(dst, 0, len * sizeof(float));
#endif
}

//! Initializes a vector to zero (double precision)
template<>
void zerov<double>(double * dst, size_t len)
{
    CNNPLUS_ASSERT(dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsZero_64f(dst, static_cast<int>(len)));
#else
    memset(dst, 0, len * sizeof(double));
#endif
}

//! Initializes a matrix to zero
template<typename T>
void zerom(T * dst, size_t stride, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(dst && stride >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);
    zerov<T>(dst, rows * stride);
}

//! Initializes the vector elements to a specified common value (single precision)
template<>
void setv<float>(float * dst, size_t len, float val)
{
    CNNPLUS_ASSERT(dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsSet_32f(val, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = val;
#endif
}

//! Initializes the vector elements to a specified common value (double precision)
template<>
void setv<double>(double * dst, size_t len, double val)
{
    CNNPLUS_ASSERT(dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsSet_64f(val, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = val;
#endif
}

//! Initializes the matrix elements to a specified common value
template<typename T>
void setm(T * dst, size_t stride, size_t rows, size_t cols, T val)
{
    CNNPLUS_ASSERT(dst && stride >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);
    setv<T>(dst, rows * stride, val);
}

//! Initializes a vector with pseudo-random samples (single precision)
template<>
void randv<float>(float * dst, size_t len, float sigma)
{
    CNNPLUS_ASSERT(dst && len > 0);
    sigma = mathli::abs(sigma);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippsRandUniform_Direct_32f(
            dst, static_cast<int>(len), -sigma, sigma, &s_seed));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = static_cast<float>(drand() * 2 - 1) * sigma;
#endif
}

//! Initializes a vector with pseudo-random samples (double precision)
template<>
void randv<double>(double * dst, size_t len, double sigma)
{
    CNNPLUS_ASSERT(dst && len > 0);
    sigma = mathli::abs(sigma);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippsRandUniform_Direct_64f(
            dst, static_cast<int>(len), -sigma, sigma, &s_seed));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = (drand() * 2 - 1) * sigma;
#endif
}

//! Initializes a vector with pseudo-random samples (single precision)
template<>
void randv<float>(float * dst, size_t len, float low, float high)
{
    CNNPLUS_ASSERT(dst && len > 0);
    CNNPLUS_ASSERT(low <= high);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippsRandUniform_Direct_32f(
            dst, static_cast<int>(len), low, high, &s_seed));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = static_cast<float>(drand()) * (high - low) + low;
#endif
}

//! Initializes a vector with pseudo-random samples (double precision)
template<>
void randv<double>(double * dst, size_t len, double low, double high)
{
    CNNPLUS_ASSERT(dst && len > 0);
    CNNPLUS_ASSERT(low <= high);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippsRandUniform_Direct_64f(
            dst, static_cast<int>(len), low, high, &s_seed));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = drand() * (high - low) + low;
#endif
}

//! Initializes a matrix with pseudo-random samples
template<typename T>
void randm(T * dst, size_t stride, size_t rows, size_t cols, T sigma)
{
    CNNPLUS_ASSERT(dst && stride >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);
    randv<T>(dst, rows * stride, sigma);
}

//! Initializes a matrix with pseudo-random samples
template<typename T>
void randm(T * dst, size_t stride, size_t rows, size_t cols, T low, T high)
{
    CNNPLUS_ASSERT(dst && stride >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);
    randv<T>(dst, rows * stride, low, high);
}

//! Adds the elements of two vectors (single precision)
template<>
void addv<float>(float const * src1, float const * src2, float * dst, size_t len)
{
    CNNPLUS_ASSERT(src1 && src2);
    CNNPLUS_ASSERT(dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsAdd_32f(src1, src2, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = src1[i] + src2[i];
#endif
}

//! Adds the elements of two vectors (double precision)
template<>
void addv<double>(double const * src1, double const * src2, double * dst, size_t len)
{
    CNNPLUS_ASSERT(src1 && src2);
    CNNPLUS_ASSERT(dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsAdd_64f(src1, src2, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = src1[i] + src2[i];
#endif
}

//! Adds the elements of two vectors (in-place, single precision)
template<>
void addv<float>(float * srcDst, float const * src, size_t len)
{
    CNNPLUS_ASSERT(srcDst && src && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsAdd_32f_I(src, srcDst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        srcDst[i] += src[i];
#endif
}

//! Adds the elements of two vectors (in-place, double precision)
template<>
void addv<double>(double * srcDst, double const * src, size_t len)
{
    CNNPLUS_ASSERT(srcDst && src && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsAdd_64f_I(src, srcDst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        srcDst[i] += src[i];
#endif
}

//! Adds a constant value to each element of a vector (single precision)
template<>
void addvc<float>(float const * src, float val, float * dst, size_t len)
{
    CNNPLUS_ASSERT(src && dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsAddC_32f(src, val, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = src[i] + val;
#endif
}

//! Adds a constant value to each element of a vector (double precision)
template<>
void addvc<double>(double const * src, double val, double * dst, size_t len)
{
    CNNPLUS_ASSERT(src && dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsAddC_64f(src, val, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = src[i] + val;
#endif
}

//! Adds a constant value to each element of a vector (in-place, single precision)
template<>
void addvc<float>(float * srcDst, float val, size_t len)
{
    CNNPLUS_ASSERT(srcDst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsAddC_32f_I(val, srcDst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        srcDst[i] += val;
#endif
}

//! Adds a constant value to each element of a vector (in-place, double precision)
template<>
void addvc<double>(double * srcDst, double val, size_t len)
{
    CNNPLUS_ASSERT(srcDst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsAddC_64f_I(val, srcDst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        srcDst[i] += val;
#endif
}

//! Subtracts the elements of two vectors (single precision)
template<>
void subv<float>(float const * src1, float const * src2, float * dst, size_t len)
{
    CNNPLUS_ASSERT(src1 && src2);
    CNNPLUS_ASSERT(dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsSub_32f(src2, src1, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = src1[i] - src2[i];
#endif
}

//! Subtracts the elements of two vectors (double precision)
template<>
void subv<double>(double const * src1, double const * src2, double * dst, size_t len)
{
    CNNPLUS_ASSERT(src1 && src2);
    CNNPLUS_ASSERT(dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsSub_64f(src2, src1, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = src1[i] - src2[i];
#endif
}

//! Subtracts the elements of two vectors (in-place, single precision)
template<>
void subv<float>(float * srcDst, float const * src, size_t len)
{
    CNNPLUS_ASSERT(srcDst && src && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsSub_32f_I(src, srcDst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        srcDst[i] -= src[i];
#endif
}

//! Subtracts the elements of two vectors (in-place, double precision)
template<>
void subv<double>(double * srcDst, double const * src, size_t len)
{
    CNNPLUS_ASSERT(srcDst && src && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsSub_64f_I(src, srcDst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        srcDst[i] -= src[i];
#endif
}

//! Subtracts a constant value from each element of a vector (single precision)
template<>
void subvc<float>(float const * src, float val, float * dst, size_t len)
{
    CNNPLUS_ASSERT(src && dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsSubC_32f(src, val, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = src[i] - val;
#endif
}

//! Subtracts a constant value from each element of a vector (double precision)
template<>
void subvc<double>(double const * src, double val, double * dst, size_t len)
{
    CNNPLUS_ASSERT(src && dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsSubC_64f(src, val, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = src[i] - val;
#endif
}

//! Subtracts a constant value from each element of a vector (in-place, single precision)
template<>
void subvc<float>(float * srcDst, float val, size_t len)
{
    CNNPLUS_ASSERT(srcDst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsSubC_32f_I(val, srcDst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        srcDst[i] -= val;
#endif
}

//! Subtracts a constant value from each element of a vector (in-place, double precision)
template<>
void subvc<double>(double * srcDst, double val, size_t len)
{
    CNNPLUS_ASSERT(srcDst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsSubC_64f_I(val, srcDst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        srcDst[i] -= val;
#endif
}

//! Subtracts each element of a vector from a constant value (single precision)
template<>
void subcv<float>(float val, float const * src, float * dst, size_t len)
{
    CNNPLUS_ASSERT(src && dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsSubCRev_32f(src, val, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = val - src[i];
#endif
}

//! Subtracts each element of a vector from a constant value (double precision)
template<>
void subcv<double>(double val, double const * src, double * dst, size_t len)
{
    CNNPLUS_ASSERT(src && dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsSubCRev_64f(src, val, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = val - src[i];
#endif
}

//! Subtracts each element of a vector from a constant value (in-place, single precision)
template<>
void subcv<float>(float val, float * srcDst, size_t len)
{
    CNNPLUS_ASSERT(srcDst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsSubCRev_32f_I(val, srcDst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        srcDst[i] = val - srcDst[i];
#endif
}

//! Subtracts each element of a vector from a constant value (in-place, double precision)
template<>
void subcv<double>(double val, double * srcDst, size_t len)
{
    CNNPLUS_ASSERT(srcDst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsSubCRev_64f_I(val, srcDst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        srcDst[i] = val - srcDst[i];
#endif
}

//! Multiplies the elements of two vectors (single precision)
template<>
void mulv<float>(float const * src1, float const * src2, float * dst, size_t len)
{
    CNNPLUS_ASSERT(src1 && src2);
    CNNPLUS_ASSERT(dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsMul_32f(src1, src2, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = src1[i] * src2[i];
#endif
}

//! Multiplies the elements of two vectors (double precision)
template<>
void mulv<double>(double const * src1, double const * src2, double * dst, size_t len)
{
    CNNPLUS_ASSERT(src1 && src2);
    CNNPLUS_ASSERT(dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsMul_64f(src1, src2, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = src1[i] * src2[i];
#endif
}

//! Multiplies the elements of two vectors (in-place, single precision)
template<>
void mulv<float>(float * srcDst, float const * src, size_t len)
{
    CNNPLUS_ASSERT(srcDst && src && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsMul_32f_I(src, srcDst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        srcDst[i] *= src[i];
#endif
}

//! Multiplies the elements of two vectors (in-place, double precision)
template<>
void mulv<double>(double * srcDst, double const * src, size_t len)
{
    CNNPLUS_ASSERT(srcDst && src && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsMul_64f_I(src, srcDst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        srcDst[i] *= src[i];
#endif
}

//! Multiplies each elements of a vector by a constant value (single precision)
template<>
void mulvc<float>(float const * src, float val, float * dst, size_t len)
{
    CNNPLUS_ASSERT(src && dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsMulC_32f(src, val, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = src[i] * val;
#endif
}

//! Multiplies each elements of a vector by a constant value (double precision)
template<>
void mulvc<double>(double const * src, double val, double * dst, size_t len)
{
    CNNPLUS_ASSERT(src && dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsMulC_64f(src, val, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = src[i] * val;
#endif
}

//! Multiplies each elements of a vector by a constant value (in-place, single precision)
template<>
void mulvc<float>(float * srcDst, float val, size_t len)
{
    CNNPLUS_ASSERT(srcDst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsMulC_32f_I(val, srcDst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        srcDst[i] *= val;
#endif
}

//! Multiplies each elements of a vector by a constant value (in-place, double precision)
template<>
void mulvc<double>(double * srcDst, double val, size_t len)
{
    CNNPLUS_ASSERT(srcDst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsMulC_64f_I(val, srcDst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        srcDst[i] *= val;
#endif
}

//! Divides the elements of two vectors (single precision)
template<>
void divv<float>(float const * src1, float const * src2, float * dst, size_t len)
{
    CNNPLUS_ASSERT(src1 && src2);
    CNNPLUS_ASSERT(dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsDiv_32f(src2, src1, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = src1[i] / src2[i];
#endif
}

//! Divides the elements of two vectors (double precision)
template<>
void divv<double>(double const * src1, double const * src2, double * dst, size_t len)
{
    CNNPLUS_ASSERT(src1 && src2);
    CNNPLUS_ASSERT(dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsDiv_64f(src2, src1, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = src1[i] / src2[i];
#endif
}

//! Divides the elements of two vectors (in-place, single precision)
template<>
void divv<float>(float * srcDst, float const * src, size_t len)
{
    CNNPLUS_ASSERT(srcDst && src && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsDiv_32f_I(src, srcDst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        srcDst[i] /= src[i];
#endif
}

//! Divides the elements of two vectors (in-place, double precision)
template<>
void divv<double>(double * srcDst, double const * src, size_t len)
{
    CNNPLUS_ASSERT(srcDst && src && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsDiv_64f_I(src, srcDst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        srcDst[i] /= src[i];
#endif
}

//! Divides each element of a vector by a constant value (single precision)
template<>
void divvc<float>(float const * src, float val, float * dst, size_t len)
{
    CNNPLUS_ASSERT(src && dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsDivC_32f(src, val, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = src[i] / val;
#endif
}

//! Divides each element of a vector by a constant value (double precision)
template<>
void divvc<double>(double const * src, double val, double * dst, size_t len)
{
    CNNPLUS_ASSERT(src && dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsDivC_64f(src, val, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = src[i] / val;
#endif
}

//! Divides each element of a vector by a constant value (in-place, single precision)
template<>
void divvc<float>(float * srcDst, float val, size_t len)
{
    CNNPLUS_ASSERT(srcDst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsDivC_32f_I(val, srcDst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        srcDst[i] /= val;
#endif
}

//! Divides each element of a vector by a constant value (in-place, double precision)
template<>
void divvc<double>(double * srcDst, double val, size_t len)
{
    CNNPLUS_ASSERT(srcDst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsDivC_64f_I(val, srcDst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        srcDst[i] /= val;
#endif
}

//! Divides each element of a matrix by a constant value
template<typename T>
void divmc(T const * src, size_t strideSrc, T val, T * dst, size_t strideDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && strideSrc >= cols);
    CNNPLUS_ASSERT(dst && strideDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    for (size_t r = 0; r < rows; ++r)
        divvc<T>(src + r * strideSrc, val, dst + r * strideDst, cols);
}

//! Divides each element of a matrix by a constant value (in-place)
template<typename T>
void divmc(T * srcDst, size_t strideSrcDst, T val, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(srcDst && strideSrcDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    for (size_t r = 0; r < rows; ++r)
        divvc<T>(srcDst + r * strideSrcDst, val, cols);
}

//! Computes hyperbolic tangent of each vector element (single precision)
template<>
void tanhv<float>(float const * src, float * dst, size_t len)
{
    CNNPLUS_ASSERT(src && dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsTanh_32f_A24(src, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = mathli::tanh(src[i]);
#endif
}

//! Computes hyperbolic tangent of each vector element (double precision)
template<>
void tanhv<double>(double const * src, double * dst, size_t len)
{
    CNNPLUS_ASSERT(src && dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsTanh_64f_A53(src, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = mathli::tanh(src[i]);
#endif
}

//! Computes \e e to the power of each element of a vector (single precision)
template<>
void expv<float>(float const * src, float * dst, size_t len)
{
    CNNPLUS_ASSERT(src && dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsExp_32f(src, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = mathli::exp(src[i]);
#endif
}

//! Computes \e e to the power of each element of a vector (double precision)
template<>
void expv<double>(double const * src, double * dst, size_t len)
{
    CNNPLUS_ASSERT(src && dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsExp_64f(src, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = mathli::exp(src[i]);
#endif
}

//! Computes \e e to the power of each element of a vector (in-place, single precision)
template<>
void expv<float>(float * srcDst, size_t len)
{
    CNNPLUS_ASSERT(srcDst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsExp_32f_I(srcDst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        srcDst[i] = mathli::exp(srcDst[i]);
#endif
}

//! Computes \e e to the power of each element of a vector (in-place, double precision)
template<>
void expv<double>(double * srcDst, size_t len)
{
    CNNPLUS_ASSERT(srcDst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsExp_64f_I(srcDst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        srcDst[i] = mathli::exp(srcDst[i]);
#endif
}

//! Computes \e e to the power of each element of a matrix
template<typename T>
void expm(T const * src, size_t strideSrc, T * dst, size_t strideDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && strideSrc >= cols);
    CNNPLUS_ASSERT(dst && strideDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    for (size_t r = 0; r < rows; ++r)
        expv<T>(src + r * strideSrc, dst + r * strideDst, cols);
}

//! Computes \e e to the power of each element of a matrix (in-place)
template<typename T>
void expm(T * srcDst, size_t strideSrcDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(srcDst && strideSrcDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    for (size_t r = 0; r < rows; ++r)
        expv<T>(srcDst + r * strideSrcDst, cols);
}

//! Computes the natural logarithm of each element of a vector (single precision)
template<>
void lnv<float>(float const * src, float * dst, size_t len)
{
    CNNPLUS_ASSERT(src && dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsLn_32f(src, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = mathli::log(src[i]);
#endif
}

//! Computes the natural logarithm of each element of a vector (double precision)
template<>
void lnv<double>(double const * src, double * dst, size_t len)
{
    CNNPLUS_ASSERT(src && dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsLn_64f(src, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = mathli::log(src[i]);
#endif
}

//! Computes the natural logarithm of each element of a vector (in-place, single precision)
template<>
void lnv<float>(float * srcDst, size_t len)
{
    CNNPLUS_ASSERT(srcDst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsLn_32f_I(srcDst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        srcDst[i] = mathli::log(srcDst[i]);
#endif
}

//! Computes the natural logarithm of each element of a vector (in-place, double precision)
template<>
void lnv<double>(double * srcDst, size_t len)
{
    CNNPLUS_ASSERT(srcDst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsLn_64f_I(srcDst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        srcDst[i] = mathli::log(srcDst[i]);
#endif
}

//! Computes a square of each element of a vector (single precision)
template<>
void sqrv<float>(float const * src, float * dst, size_t len)
{
    CNNPLUS_ASSERT(src && dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsSqr_32f(src, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = src[i] * src[i];
#endif
}

//! Computes a square of each element of a vector (double precision)
template<>
void sqrv<double>(double const * src, double * dst, size_t len)
{
    CNNPLUS_ASSERT(src && dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsSqr_64f(src, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = src[i] * src[i];
#endif
}

//! Computes a square of each element of a vector (in-place, single precision)
template<>
void sqrv<float>(float * srcDst, size_t len)
{
    CNNPLUS_ASSERT(srcDst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsSqr_32f_I(srcDst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        srcDst[i] *= srcDst[i];
#endif
}

//! Computes a square of each element of a vector (in-place, double precision)
template<>
void sqrv<double>(double * srcDst, size_t len)
{
    CNNPLUS_ASSERT(srcDst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsSqr_64f_I(srcDst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        srcDst[i] *= srcDst[i];
#endif
}

//! Computes a square root of each element of a vector (single precision)
template<>
void sqrtv<float>(float const * src, float * dst, size_t len)
{
    CNNPLUS_ASSERT(src && dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsSqrt_32f(src, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = mathli::sqrt(src[i]);
#endif
}

//! Computes a square root of each element of a vector (double precision)
template<>
void sqrtv<double>(double const * src, double * dst, size_t len)
{
    CNNPLUS_ASSERT(src && dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsSqrt_64f(src, dst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        dst[i] = mathli::sqrt(src[i]);
#endif
}

//! Computes a square root of each element of a vector (in-place, single precision)
template<>
void sqrtv<float>(float * srcDst, size_t len)
{
    CNNPLUS_ASSERT(srcDst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsSqrt_32f_I(srcDst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        srcDst[i] = mathli::sqrt(srcDst[i]);
#endif
}

//! Computes a square root of each element of a vector (in-place, double precision)
template<>
void sqrtv<double>(double * srcDst, size_t len)
{
    CNNPLUS_ASSERT(srcDst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsSqrt_64f_I(srcDst, static_cast<int>(len)));
#else
    for (size_t i = 0; i < len; ++i)
        srcDst[i] = mathli::sqrt(srcDst[i]);
#endif
}

//! Computes the sum of the elements of a vector (single precision)
template<>
float sumv<float>(float const * src, size_t len)
{
    CNNPLUS_ASSERT(src && len > 0);
    float sum = 0;
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippsSum_32f(src, static_cast<int>(len), &sum, ippAlgHintNone));
#else
    for (size_t i = 0; i < len; ++i)
        sum += src[i];
#endif
    return sum;
}

//! Computes the sum of the elements of a vector (double precision)
template<>
double sumv<double>(double const * src, size_t len)
{
    CNNPLUS_ASSERT(src && len > 0);
    double sum = 0;
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsSum_64f(src, static_cast<int>(len), &sum));
#else
    for (size_t i = 0; i < len; ++i)
        sum += src[i];
#endif
    return sum;
}

//! Computes the sum of the elements of a matrix
template<typename T>
T summ(T const * src, size_t stride, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && stride >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    if (stride == cols)
        return sumv<T>(src, rows * stride);

    T sum = 0;
    for (size_t r = 0; r < rows; ++r)
        sum += sumv<T>(src + r * stride, cols);
    return sum;
}

//! Computes the dot product of two vectors (single precision)
template<>
float dotprod<float>(float const * src1, float const * src2, size_t len)
{
    CNNPLUS_ASSERT(src1 && src2 && len > 0);
    float dp = 0;
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsDotProd_32f(src1, src2, static_cast<int>(len), &dp));
#else
    for (size_t i = 0; i < len; ++i)
        dp += src1[i] * src2[i];
#endif
    return dp;
}

//! Computes the dot product of two vectors (double precision)
template<>
double dotprod<double>(double const * src1, double const * src2, size_t len)
{
    CNNPLUS_ASSERT(src1 && src2 && len > 0);
    double dp = 0;
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsDotProd_64f(src1, src2, static_cast<int>(len), &dp));
#else
    for (size_t i = 0; i < len; ++i)
        dp += src1[i] * src2[i];
#endif
    return dp;
}

//! Computes the sum of square values of a vector
template<typename T>
T sumsqrv(T const * src, size_t len)
{
    CNNPLUS_ASSERT(src && len > 0);
    return dotprod<T>(src, src, len);
}

//! Copies the contents of one vector into another (single precision)
template<>
void copyvv<float>(float const * src, float * dst, size_t len)
{
    CNNPLUS_ASSERT(src && dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsCopy_32f(src, dst, static_cast<int>(len)));
#else
    memcpy(dst, src, len * sizeof(float));
#endif
}

//! Copies the contents of one vector into another (double precision)
template<>
void copyvv<double>(double const * src, double * dst, size_t len)
{
    CNNPLUS_ASSERT(src && dst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsCopy_64f(src, dst, static_cast<int>(len)));
#else
    memcpy(dst, src, len * sizeof(double));
#endif
}

//! Copies the contents of one matrix into another (single precision)
template<>
void copymm<float>(float const * src, size_t strideSrc, float * dst, size_t strideDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && strideSrc >= cols);
    CNNPLUS_ASSERT(dst && strideDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippmCopy_ma_32f_SS(src, 0,
                           static_cast<int>(strideSrc * sizeof(float)),
                           sizeof(float),
                           dst, 0,
                           static_cast<int>(strideDst * sizeof(float)),
                           sizeof(float),
                           static_cast<int>(cols),
                           static_cast<int>(rows),
                           1));
#else
    if (strideSrc == strideDst) {
        copyvv<float>(src, dst, rows * strideSrc);
    }
    else {
        for (size_t r = 0; r < rows; ++r)
            copyvv<float>(src + r * strideSrc, dst + r * strideDst, cols);
    }
#endif
}

//! Copies the contents of one matrix into another (double precision)
template<>
void copymm<double>(double const * src, size_t strideSrc, double * dst, size_t strideDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && strideSrc >= cols);
    CNNPLUS_ASSERT(dst && strideDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippmCopy_ma_64f_SS(src, 0,
                           static_cast<int>(strideSrc * sizeof(double)),
                           sizeof(double),
                           dst, 0,
                           static_cast<int>(strideDst * sizeof(double)),
                           sizeof(double),
                           static_cast<int>(cols),
                           static_cast<int>(rows),
                           1));
#else
    if (strideSrc == strideDst) {
        copyvv<double>(src, dst, rows * strideSrc);
    }
    else {
        for (size_t r = 0; r < rows; ++r)
            copyvv<double>(src + r * strideSrc, dst + r * strideDst, cols);
    }
#endif
}

//! Copies the contents of a matrix into a vector
template<typename T>
void copymv(T const * src, size_t strideSrc, T * dst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && dst && strideSrc >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    if (strideSrc == cols)
        copyvv<T>(src, dst, rows * cols);
    else
        copymm<T>(src, strideSrc, dst, cols, rows, cols);
}

//! Copies the contents of a vector into a matrix
template<typename T>
void copyvm(T const * src, T * dst, size_t strideDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && dst && strideDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    if (strideDst == cols)
        copyvv<T>(src, dst, rows * cols);
    else
        copymm<T>(src, cols, dst, strideDst, rows, cols);
}

//! Computes the L2 norm of a vector (single precision)
template<>
float nrm2<float>(float const * src, size_t len)
{
    CNNPLUS_ASSERT(src && len > 0);
    float norm = 0;
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsNorm_L2_32f(src, static_cast<int>(len), &norm));
#else
    if (len == 0) return 0;
    else if (len == 1) return mathli::abs(src[0]);
    float scale = 0, ssq = 1;
    for (size_t i = 0; i < len; ++i) {
        if (src[i] != 0) {
            float const absxi = mathli::abs(src[i]);
            if (scale < absxi) {
                /* Computing 2nd power */
                float const tmp = scale / absxi;
                ssq = ssq * (tmp * tmp) + 1;
                scale = absxi;
            }
            else {
                /* Computing 2nd power */
                float const tmp = absxi / scale;
                ssq += tmp * tmp;
            }
        }
    }
    norm = scale * mathli::sqrt(ssq);
#endif
    return norm;
}

//! Computes the L2 norm of a vector (double precision)
template<>
double nrm2<double>(double const * src, size_t len)
{
    CNNPLUS_ASSERT(src && len > 0);
    double norm = 0;
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(ippsNorm_L2_64f(src, static_cast<int>(len), &norm));
#else
    if (len == 0) return 0;
    else if (len == 1) return mathli::abs(src[0]);
    double scale = 0, ssq = 1;
    for (size_t i = 0; i < len; ++i) {
        if (src[i] != 0) {
            double const absxi = mathli::abs(src[i]);
            if (scale < absxi) {
                /* Computing 2nd power */
                double const tmp = scale / absxi;
                ssq = ssq * (tmp * tmp) + 1;
                scale = absxi;
            }
            else {
                /* Computing 2nd power */
                double const tmp = absxi / scale;
                ssq += tmp * tmp;
            }
        }
    }
    norm = scale * mathli::sqrt(ssq);
#endif
    return norm;
}

//! Computes the L2 norm of two vectors' difference (single precision)
template<>
float nrm2<float>(float const * src1, float const * src2, size_t len)
{
    CNNPLUS_ASSERT(src1 && src2 && len > 0);
    float norm = 0;
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippsNormDiff_L2_32f(src1, src2, static_cast<int>(len), &norm));
#else
    if (len == 0) return 0;
    else if (len == 1) return mathli::abs(src1[0] - src2[0]);
    float scale = 0, ssq = 1;
    for (size_t i = 0; i < len; ++i) {
        float const diff = src1[i] - src2[i];
        if (diff != 0) {
            float const absxi = mathli::abs(diff);
            if (scale < absxi) {
                /* Computing 2nd power */
                float const tmp = scale / absxi;
                ssq = ssq * (tmp * tmp) + 1;
                scale = absxi;
            }
            else {
                /* Computing 2nd power */
                float const tmp = absxi / scale;
                ssq += tmp * tmp;
            }
        }
    }
    norm = scale * mathli::sqrt(ssq);
#endif
    return norm;
}

//! Computes the L2 norm of two vectors' difference (double precision)
template<>
double nrm2<double>(double const * src1, double const * src2, size_t len)
{
    CNNPLUS_ASSERT(src1 && src2 && len > 0);
    double norm = 0;
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippsNormDiff_L2_64f(src1, src2, static_cast<int>(len), &norm));
#else
    if (len == 0) return 0;
    else if (len == 1) return mathli::abs(src1[0] - src2[0]);
    double scale = 0, ssq = 1;
    for (size_t i = 0; i < len; ++i) {
        double const diff = src1[i] - src2[i];
        if (diff != 0) {
            double const absxi = mathli::abs(diff);
            if (scale < absxi) {
                /* Computing 2nd power */
                double const tmp = scale / absxi;
                ssq = ssq * (tmp * tmp) + 1;
                scale = absxi;
            }
            else {
                /* Computing 2nd power */
                double const tmp = absxi / scale;
                ssq += tmp * tmp;
            }
        }
    }
    norm = scale * mathli::sqrt(ssq);
#endif
    return norm;
}

//! Computes the Frobenius norm of a matrix (single precision)
template<>
float nrmf<float>(float const * src, size_t stride, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && stride >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);
    float norm = 0;
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippmFrobNorm_m_32f(src, static_cast<int>(stride * sizeof(float)),
                           sizeof(float),
                           static_cast<int>(cols),
                           static_cast<int>(rows),
                           &norm));
#else
    if (rows * cols == 0) return 0;
    else if (rows * cols == 1) return mathli::abs(src[0]);
    float scale = 0, ssq = 1;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            if (src[r * stride + c] != 0) {
                float const absxi = mathli::abs(src[r * stride + c]);
                if (scale < absxi) {
                    /* Computing 2nd power */
                    float const tmp = scale / absxi;
                    ssq = ssq * (tmp * tmp) + 1;
                    scale = absxi;
                }
                else {
                    /* Computing 2nd power */
                    float const tmp = absxi / scale;
                    ssq += tmp * tmp;
                }
            }
        }
    }
    norm = scale * mathli::sqrt(ssq);
#endif
    return norm;
}

//! Computes the Frobenius norm of a matrix (double precision)
template<>
double nrmf<double>(double const * src, size_t stride, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && stride >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);
    double norm = 0;
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippmFrobNorm_m_64f(src, static_cast<int>(stride * sizeof(double)),
                           sizeof(double),
                           static_cast<int>(cols),
                           static_cast<int>(rows),
                           &norm));
#else
    if (rows * cols == 0) return 0;
    else if (rows * cols == 1) return mathli::abs(src[0]);
    double scale = 0, ssq = 1;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            if (src[r * stride + c] != 0) {
                double const absxi = mathli::abs(src[r * stride + c]);
                if (scale < absxi) {
                    /* Computing 2nd power */
                    double const tmp = scale / absxi;
                    ssq = ssq * (tmp * tmp) + 1;
                    scale = absxi;
                }
                else {
                    /* Computing 2nd power */
                    double const tmp = absxi / scale;
                    ssq += tmp * tmp;
                }
            }
        }
    }
    norm = scale * mathli::sqrt(ssq);
#endif
    return norm;
}

//! Returns the maximum absolute value of a vector (single precision)
template<>
float absmaxv<float>(float const * src, size_t len)
{
    CNNPLUS_ASSERT(src && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    return mathli::abs(src[cblas_isamax(static_cast<int>(len), src, 1)]);
#else
    if (len == 0) return 0;
    else if (len == 1) return mathli::abs(src[0]);
    float maxval = mathli::abs(src[0]);
    for (size_t i = 1; i < len; ++i) {
        float const tmp = mathli::abs(src[i]);
        if (tmp > maxval)
            maxval = tmp;
    }
    return maxval;
#endif
}

//! Returns the maximum absolute value of a vector (double precision)
template<>
double absmaxv<double>(double const * src, size_t len)
{
    CNNPLUS_ASSERT(src && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    return mathli::abs(src[cblas_idamax(static_cast<int>(len), src, 1)]);
#else
    if (len == 0) return 0;
    else if (len == 1) return mathli::abs(src[0]);
    double maxval = mathli::abs(src[0]);
    for (size_t i = 1; i < len; ++i) {
        double const tmp = mathli::abs(src[i]);
        if (tmp > maxval)
            maxval = tmp;
    }
    return maxval;
#endif
}

//! Returns the minimum absolute value of a vector (single precision)
template<>
float absminv<float>(float const * src, size_t len)
{
    CNNPLUS_ASSERT(src && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    return mathli::abs(src[cblas_isamin(static_cast<int>(len), src, 1)]);
#else
    if (len == 0) return 0;
    else if (len == 1) return mathli::abs(src[0]);
    float minval = mathli::abs(src[0]);
    for (size_t i = 1; i < len; ++i) {
        float const tmp = mathli::abs(src[i]);
        if (tmp < minval)
            minval = tmp;
    }
    return minval;
#endif
}

//! Returns the minimum absolute value of a vector (double precision)
template<>
double absminv<double>(double const * src, size_t len)
{
    CNNPLUS_ASSERT(src && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    return mathli::abs(src[cblas_idamin(static_cast<int>(len), src, 1)]);
#else
    if (len == 0) return 0;
    else if (len == 1) return mathli::abs(src[0]);
    double minval = mathli::abs(src[0]);
    for (size_t i = 1; i < len; ++i) {
        double const tmp = mathli::abs(src[i]);
        if (tmp < minval)
            minval = tmp;
    }
    return minval;
#endif
}

//! Returns the maximum value of a vector (single precision)
template<>
float maxv<float>(float const * src, size_t len)
{
    CNNPLUS_ASSERT(src && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    float val = 0;
    IPP_SAFE_CALL(ippsMax_32f(src, static_cast<int>(len), &val));
    return val;
#else
    if (len == 0) return 0;
    else if (len == 1) return src[0];
    float maxval = src[0];
    for (size_t i = 1; i < len; ++i) {
        float const tmp = src[i];
        if (tmp > maxval)
            maxval = tmp;
    }
    return maxval;
#endif
}

//! Returns the maximum value of a vector (double precision)
template<>
double maxv<double>(double const * src, size_t len)
{
    CNNPLUS_ASSERT(src && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    double val = 0;
    IPP_SAFE_CALL(ippsMax_64f(src, static_cast<int>(len), &val));
    return val;
#else
    if (len == 0) return 0;
    else if (len == 1) return src[0];
    double maxval = src[0];
    for (size_t i = 1; i < len; ++i) {
        double const tmp = src[i];
        if (tmp > maxval)
            maxval = tmp;
    }
    return maxval;
#endif
}

//! Returns the minimum value of a vector (single precision)
template<>
float minv<float>(float const * src, size_t len)
{
    CNNPLUS_ASSERT(src && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    float val = 0;
    IPP_SAFE_CALL(ippsMin_32f(src, static_cast<int>(len), &val));
    return val;
#else
    if (len == 0) return 0;
    else if (len == 1) return src[0];
    float minval = src[0];
    for (size_t i = 1; i < len; ++i) {
        float const tmp = src[i];
        if (tmp < minval)
            minval = tmp;
    }
    return minval;
#endif
}

//! Returns the minimum value of a vector (double precision)
template<>
double minv<double>(double const * src, size_t len)
{
    CNNPLUS_ASSERT(src && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    double val = 0;
    IPP_SAFE_CALL(ippsMin_64f(src, static_cast<int>(len), &val));
    return val;
#else
    if (len == 0) return 0;
    else if (len == 1) return src[0];
    double minval = src[0];
    for (size_t i = 1; i < len; ++i) {
        double const tmp = src[i];
        if (tmp < minval)
            minval = tmp;
    }
    return minval;
#endif
}

//! Returns the index of the maximum element of a vector (single precision)
template<>
int maxidxv<float>(float const * src, size_t len)
{
    CNNPLUS_ASSERT(src && len > 0);
    int index = 0;
#ifdef CNNPLUS_USE_INTEL_LIBS
    float val = 0;
    IPP_SAFE_CALL(ippsMaxIndx_32f(src, static_cast<int>(len), &val, &index));
    val = val; // warning #177: variable "val" was declared but never referenced
#else
    if (len == 0) return -1;
    else if (len == 1) return 0;
    float maxval = src[0];
    for (size_t i = 1; i < len; ++i) {
        float const tmp = src[i];
        if (tmp > maxval) {
            maxval = tmp;
            index = i;
        }
    }
#endif
    return index;
}

//! Returns the index of the maximum element of a vector (double precision)
template<>
int maxidxv<double>(double const * src, size_t len)
{
    CNNPLUS_ASSERT(src && len > 0);
    int index = 0;
#ifdef CNNPLUS_USE_INTEL_LIBS
    double val = 0;
    IPP_SAFE_CALL(ippsMaxIndx_64f(src, static_cast<int>(len), &val, &index));
    val = val; // warning #177: variable "val" was declared but never referenced
#else
    if (len == 0) return -1;
    else if (len == 1) return 0;
    double maxval = src[0];
    for (size_t i = 1; i < len; ++i) {
        double const tmp = src[i];
        if (tmp > maxval) {
            maxval = tmp;
            index = i;
        }
    }
#endif
    return index;
}

//! Returns the index of the minimum element of a vector (single precision)
template<>
int minidxv<float>(float const * src, size_t len)
{
    CNNPLUS_ASSERT(src && len > 0);
    int index = 0;
#ifdef CNNPLUS_USE_INTEL_LIBS
    float val = 0;
    IPP_SAFE_CALL(ippsMinIndx_32f(src, static_cast<int>(len), &val, &index));
    val = val; // warning #177: variable "val" was declared but never referenced
#else
    if (len == 0) return -1;
    else if (len == 1) return 0;
    float minval = src[0];
    for (size_t i = 1; i < len; ++i) {
        float const tmp = src[i];
        if (tmp < minval) {
            minval = tmp;
            index = i;
        }
    }
#endif
    return index;
}

//! Returns the index of the minimum element of a vector (double precision)
template<>
int minidxv<double>(double const * src, size_t len)
{
    CNNPLUS_ASSERT(src && len > 0);
    int index = 0;
#ifdef CNNPLUS_USE_INTEL_LIBS
    double val = 0;
    IPP_SAFE_CALL(ippsMinIndx_64f(src, static_cast<int>(len), &val, &index));
    val = val; // warning #177: variable "val" was declared but never referenced
#else
    if (len == 0) return -1;
    else if (len == 1) return 0;
    double minval = src[0];
    for (size_t i = 1; i < len; ++i) {
        double const tmp = src[i];
        if (tmp < minval) {
            minval = tmp;
            index = i;
        }
    }
#endif
    return index;
}

//! Calculates the sums of row vectors in a matrix (single precision)
template<>
void sumrow<float>(float const * src, size_t stride, float * dst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && dst && stride >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippsSumRow_32f_D2(src,
                          static_cast<int>(cols),
                          static_cast<int>(stride),
                          dst,
                          static_cast<int>(rows)));
#else
    for (size_t r = 0; r < rows; ++r)
        dst[r] = sumv<float>(src + r * stride, cols);
#endif
}

//! Calculates the sums of row vectors in a matrix (double precision)
template<>
void sumrow<double>(double const * src, size_t stride, double * dst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && dst && stride >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippsSumRow_64f_D2(src,
                          static_cast<int>(cols),
                          static_cast<int>(stride),
                          dst,
                          static_cast<int>(rows)));
#else
    for (size_t r = 0; r < rows; ++r)
        dst[r] = sumv<double>(src + r * stride, cols);
#endif
}

//! Calculates the sums of row vectors in a matrix and accumulates it to a vector
template<typename T>
void sumrowacc(T const * src, size_t stride, T * srcDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && srcDst && stride >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    for (size_t r = 0; r < rows; ++r)
        srcDst[r] += sumv<T>(src + r * stride, cols);
}

//! Calculates the sums of column vectors in a matrix (single precision)
template<>
void sumcol<float>(float const * src, size_t stride, float * dst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && dst && stride >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippsSumColumn_32f_D2(src,
                             static_cast<int>(stride),
                             static_cast<int>(rows),
                             dst,
                             static_cast<int>(cols)));
#else
    copyvv<float>(src, dst, cols);
    for (size_t r = 1; r < rows; ++r)
        addv<float>(dst, src + r * stride, cols);
#endif
}

//! Calculates the sums of column vectors in a matrix (double precision)
template<>
void sumcol<double>(double const * src, size_t stride, double * dst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && dst && stride >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippsSumColumn_64f_D2(src,
                             static_cast<int>(stride),
                             static_cast<int>(rows),
                             dst,
                             static_cast<int>(cols)));
#else
    copyvv<double>(src, dst, cols);
    for (size_t r = 1; r < rows; ++r)
        addv<double>(dst, src + r * stride, cols);
#endif
}

//! Calculates the sums of column vectors in a matrix and accumulates it to a vector
template<typename T>
void sumcolacc(T const * src, size_t stride, T * srcDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && srcDst && stride >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    for (size_t r = 0; r < rows; ++r)
        addv<T>(srcDst, src + r * stride, cols);
}

//! Multiplies each elements of a matrix by a constant value (single precision)
template<>
void mulmc<float>(float const * src, size_t strideSrc, float val,
                  float * dst, size_t strideDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && strideSrc >= cols);
    CNNPLUS_ASSERT(dst && strideDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippmMul_mc_32f(src,
                       static_cast<int>(strideSrc * sizeof(float)),
                       sizeof(float),
                       val,
                       dst,
                       static_cast<int>(strideDst * sizeof(float)),
                       sizeof(float),
                       static_cast<int>(cols),
                       static_cast<int>(rows)));
#else
    if (strideSrc == strideDst) {
        mulvc<float>(src, val, dst, rows * strideSrc);
    }
    else {
        for (size_t r = 0; r < rows; ++r)
            mulvc<float>(src + r * strideSrc, val, dst + r * strideDst, cols);
    }
#endif
}

//! Multiplies each elements of a matrix by a constant value (double precision)
template<>
void mulmc<double>(double const * src, size_t strideSrc, double val,
                   double * dst, size_t strideDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && strideSrc >= cols);
    CNNPLUS_ASSERT(dst && strideDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippmMul_mc_64f(src,
                       static_cast<int>(strideSrc * sizeof(double)),
                       sizeof(double),
                       val,
                       dst,
                       static_cast<int>(strideDst * sizeof(double)),
                       sizeof(double),
                       static_cast<int>(cols),
                       static_cast<int>(rows)));
#else
    if (strideSrc == strideDst) {
        mulvc<double>(src, val, dst, rows * strideSrc);
    }
    else {
        for (size_t r = 0; r < rows; ++r)
            mulvc<double>(src + r * strideSrc, val, dst + r * strideDst, cols);
    }
#endif
}

//! Multiplies each elements of a matrix by a constant value (in-place)
template<typename T>
void mulmc(T * srcDst, size_t strideSrcDst, T val, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(srcDst && strideSrcDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    mulvc<T>(srcDst, val, rows * strideSrcDst);
}

//! Computes the pointwise multiplication of two matrices
template<typename T>
void pmulmm(T const * src1, size_t strideSrc1,
            T const * src2, size_t strideSrc2,
            T * dst, size_t strideDst,
            size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src1 && strideSrc1 >= cols);
    CNNPLUS_ASSERT(src2 && strideSrc2 >= cols);
    CNNPLUS_ASSERT(dst && strideDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    for (size_t r = 0; r < rows; ++r)
        mulv<T>(src1 + r * strideSrc1, src2 + r * strideSrc2, dst + r * strideDst, cols);
}

//! Computes the pointwise multiplication of two matrices (in-place)
template<typename T>
void pmulmm(T * srcDst, size_t strideSrcDst,
            T const * src, size_t strideSrc,
            size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(srcDst && strideSrcDst >= cols);
    CNNPLUS_ASSERT(src && strideSrc >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    for (size_t r = 0; r < rows; ++r)
        mulv<T>(srcDst + r * strideSrcDst, src + r * strideSrc, cols);
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
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippmMul_mm_32f(src1,
                       static_cast<int>(strideSrc1 * sizeof(float)),
                       sizeof(float),
                       static_cast<int>(colsSrc1),
                       static_cast<int>(rowsSrc1),
                       src2,
                       static_cast<int>(strideSrc2 * sizeof(float)),
                       sizeof(float),
                       static_cast<int>(colsSrc2),
                       static_cast<int>(rowsSrc2),
                       dst,
                       static_cast<int>(strideDst * sizeof(float)),
                       sizeof(float)));
#else
    for (size_t r = 0; r < rowsSrc1; ++r) {
        for (size_t c = 0; c < colsSrc2; ++c) {
            dst[r * strideDst + c] = 0;
        }
        for (size_t i = 0; i < colsSrc1; ++i) {
            if (src1[r * strideSrc1 + i] != 0) {
                float const tmp = src1[r * strideSrc1 + i];
                for (size_t c = 0; c < colsSrc2; ++c) {
                    dst[r * strideDst + c] += tmp * src2[i * strideSrc2 + c];
                }
            }
        }
    }
#endif
}

//! Computes a matrix-matrix product (double precision)
template<>
void mulmm<double,'n','n'>(double const * src1, size_t strideSrc1,
                           size_t rowsSrc1, size_t colsSrc1,
                           double const * src2, size_t strideSrc2,
                           size_t rowsSrc2, size_t colsSrc2,
                           double * dst, size_t strideDst)
{
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && strideSrc2 >= colsSrc2);
    CNNPLUS_ASSERT(rowsSrc2 > 0 && colsSrc2 > 0);
    CNNPLUS_ASSERT(rowsSrc2 == colsSrc1);
    CNNPLUS_ASSERT(dst && strideDst >= colsSrc2);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippmMul_mm_64f(src1,
                       static_cast<int>(strideSrc1 * sizeof(double)),
                       sizeof(double),
                       static_cast<int>(colsSrc1),
                       static_cast<int>(rowsSrc1),
                       src2,
                       static_cast<int>(strideSrc2 * sizeof(double)),
                       sizeof(double),
                       static_cast<int>(colsSrc2),
                       static_cast<int>(rowsSrc2),
                       dst,
                       static_cast<int>(strideDst * sizeof(double)),
                       sizeof(double)));
#else
    for (size_t r = 0; r < rowsSrc1; ++r) {
        for (size_t c = 0; c < colsSrc2; ++c) {
            dst[r * strideDst + c] = 0;
        }
        for (size_t i = 0; i < colsSrc1; ++i) {
            if (src1[r * strideSrc1 + i] != 0) {
                double const tmp = src1[r * strideSrc1 + i];
                for (size_t c = 0; c < colsSrc2; ++c) {
                    dst[r * strideDst + c] += tmp * src2[i * strideSrc2 + c];
                }
            }
        }
    }
#endif
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
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippmMul_mt_32f(src1,
                       static_cast<int>(strideSrc1 * sizeof(float)),
                       sizeof(float),
                       static_cast<int>(colsSrc1),
                       static_cast<int>(rowsSrc1),
                       src2,
                       static_cast<int>(strideSrc2 * sizeof(float)),
                       sizeof(float),
                       static_cast<int>(colsSrc2),
                       static_cast<int>(rowsSrc2),
                       dst,
                       static_cast<int>(strideDst * sizeof(float)),
                       sizeof(float)));
#else
    for (size_t r = 0; r < rowsSrc1; ++r) {
        for (size_t c = 0; c < rowsSrc2; ++c) {
            float tmp = 0;
            for (size_t i = 0; i < colsSrc1; ++i) {
                tmp += src1[r * strideSrc1 + i] * src2[c * strideSrc2 + i];
            }
            dst[r * strideDst + c] = tmp;
        }
    }
#endif
}

//! Computes a matrix-transposed-matrix product (double precision)
template<>
void mulmm<double,'n','t'>(double const * src1, size_t strideSrc1,
                           size_t rowsSrc1, size_t colsSrc1,
                           double const * src2, size_t strideSrc2,
                           size_t rowsSrc2, size_t colsSrc2,
                           double * dst, size_t strideDst)
{
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && strideSrc2 >= colsSrc2);
    CNNPLUS_ASSERT(rowsSrc2 > 0 && colsSrc2 > 0);
    CNNPLUS_ASSERT(colsSrc2 == colsSrc1);
    CNNPLUS_ASSERT(dst && strideDst >= rowsSrc2);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippmMul_mt_64f(src1,
                       static_cast<int>(strideSrc1 * sizeof(double)),
                       sizeof(double),
                       static_cast<int>(colsSrc1),
                       static_cast<int>(rowsSrc1),
                       src2,
                       static_cast<int>(strideSrc2 * sizeof(double)),
                       sizeof(double),
                       static_cast<int>(colsSrc2),
                       static_cast<int>(rowsSrc2),
                       dst,
                       static_cast<int>(strideDst * sizeof(double)),
                       sizeof(double)));
#else
    for (size_t r = 0; r < rowsSrc1; ++r) {
        for (size_t c = 0; c < rowsSrc2; ++c) {
            double tmp = 0;
            for (size_t i = 0; i < colsSrc1; ++i) {
                tmp += src1[r * strideSrc1 + i] * src2[c * strideSrc2 + i];
            }
            dst[r * strideDst + c] = tmp;
        }
    }
#endif
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
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippmMul_tm_32f(src1,
                       static_cast<int>(strideSrc1 * sizeof(float)),
                       sizeof(float),
                       static_cast<int>(colsSrc1),
                       static_cast<int>(rowsSrc1),
                       src2,
                       static_cast<int>(strideSrc2 * sizeof(float)),
                       sizeof(float),
                       static_cast<int>(colsSrc2),
                       static_cast<int>(rowsSrc2),
                       dst,
                       static_cast<int>(strideDst * sizeof(float)),
                       sizeof(float)));
#else
    for (size_t r = 0; r < colsSrc1; ++r) {
        for (size_t c = 0; c < colsSrc2; ++c) {
            dst[r * strideDst + c] = 0;
        }
        for (size_t i = 0; i < rowsSrc1; ++i) {
            if (src1[i * strideSrc1 + r] != 0) {
                float const tmp = src1[i * strideSrc1 + r];
                for (size_t c = 0; c < colsSrc2; ++c) {
                    dst[r * strideDst + c] += tmp * src2[i * strideSrc2 + c];
                }
            }
        }
    }
#endif
}

//! Computes a transposed-matrix-matrix product (double precision)
template<>
void mulmm<double,'t','n'>(double const * src1, size_t strideSrc1,
                           size_t rowsSrc1, size_t colsSrc1,
                           double const * src2, size_t strideSrc2,
                           size_t rowsSrc2, size_t colsSrc2,
                           double * dst, size_t strideDst)
{
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && strideSrc2 >= colsSrc2);
    CNNPLUS_ASSERT(rowsSrc2 > 0 && colsSrc2 > 0);
    CNNPLUS_ASSERT(rowsSrc2 == rowsSrc1);
    CNNPLUS_ASSERT(dst && strideDst >= colsSrc2);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippmMul_tm_64f(src1,
                       static_cast<int>(strideSrc1 * sizeof(double)),
                       sizeof(double),
                       static_cast<int>(colsSrc1),
                       static_cast<int>(rowsSrc1),
                       src2,
                       static_cast<int>(strideSrc2 * sizeof(double)),
                       sizeof(double),
                       static_cast<int>(colsSrc2),
                       static_cast<int>(rowsSrc2),
                       dst,
                       static_cast<int>(strideDst * sizeof(double)),
                       sizeof(double)));
#else
    for (size_t r = 0; r < colsSrc1; ++r) {
        for (size_t c = 0; c < colsSrc2; ++c) {
            dst[r * strideDst + c] = 0;
        }
        for (size_t i = 0; i < rowsSrc1; ++i) {
            if (src1[i * strideSrc1 + r] != 0) {
                double const tmp = src1[i * strideSrc1 + r];
                for (size_t c = 0; c < colsSrc2; ++c) {
                    dst[r * strideDst + c] += tmp * src2[i * strideSrc2 + c];
                }
            }
        }
    }
#endif
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
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippmMul_tt_32f(src1,
                       static_cast<int>(strideSrc1 * sizeof(float)),
                       sizeof(float),
                       static_cast<int>(colsSrc1),
                       static_cast<int>(rowsSrc1),
                       src2,
                       static_cast<int>(strideSrc2 * sizeof(float)),
                       sizeof(float),
                       static_cast<int>(colsSrc2),
                       static_cast<int>(rowsSrc2),
                       dst,
                       static_cast<int>(strideDst * sizeof(float)),
                       sizeof(float)));
#else
    for (size_t r = 0; r < colsSrc1; ++r) {
        for (size_t c = 0; c < rowsSrc2; ++c) {
            float tmp = 0;
            for (size_t i = 0; i < rowsSrc1; ++i) {
                tmp += src1[i * strideSrc1 + r] * src2[c * strideSrc2 + i];
            }
            dst[r * strideDst + c] = tmp;
        }
    }
#endif
}

//! Computes a transposed-matrix-transposed-matrix product (double precision)
template<>
void mulmm<double,'t','t'>(double const * src1, size_t strideSrc1,
                           size_t rowsSrc1, size_t colsSrc1,
                           double const * src2, size_t strideSrc2,
                           size_t rowsSrc2, size_t colsSrc2,
                           double * dst, size_t strideDst)
{
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && strideSrc2 >= colsSrc2);
    CNNPLUS_ASSERT(rowsSrc2 > 0 && colsSrc2 > 0);
    CNNPLUS_ASSERT(colsSrc2 == rowsSrc1);
    CNNPLUS_ASSERT(dst && strideDst >= rowsSrc2);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippmMul_tt_64f(src1,
                       static_cast<int>(strideSrc1 * sizeof(double)),
                       sizeof(double),
                       static_cast<int>(colsSrc1),
                       static_cast<int>(rowsSrc1),
                       src2,
                       static_cast<int>(strideSrc2 * sizeof(double)),
                       sizeof(double),
                       static_cast<int>(colsSrc2),
                       static_cast<int>(rowsSrc2),
                       dst,
                       static_cast<int>(strideDst * sizeof(double)),
                       sizeof(double)));
#else
    for (size_t r = 0; r < colsSrc1; ++r) {
        for (size_t c = 0; c < rowsSrc2; ++c) {
            double tmp = 0;
            for (size_t i = 0; i < rowsSrc1; ++i) {
                tmp += src1[i * strideSrc1 + r] * src2[c * strideSrc2 + i];
            }
            dst[r * strideDst + c] = tmp;
        }
    }
#endif
}

//! Computes a matrix-vector product (single precision)
template<>
void mulmv<float,'n'>(float const * src1, size_t strideSrc1,
                      size_t rowsSrc1, size_t colsSrc1,
                      float const * src2, size_t lenSrc2,
                      float * dst)
{
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && lenSrc2 == colsSrc1);
    CNNPLUS_ASSERT(dst);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippmMul_mv_32f(src1,
                       static_cast<int>(strideSrc1 * sizeof(float)),
                       sizeof(float),
                       static_cast<int>(colsSrc1),
                       static_cast<int>(rowsSrc1),
                       src2,
                       sizeof(float),
                       static_cast<int>(lenSrc2),
                       dst,
                       sizeof(float)));
#else
    for (size_t r = 0; r < rowsSrc1; ++r) {
        float tmp = 0;
        for (size_t c = 0; c < colsSrc1; ++c) {
            tmp += src1[r * strideSrc1 + c] * src2[c];
        }
        dst[r] = tmp;
    }
#endif
}

//! Computes a matrix-vector product (double precision)
template<>
void mulmv<double,'n'>(double const * src1, size_t strideSrc1,
                       size_t rowsSrc1, size_t colsSrc1,
                       double const * src2, size_t lenSrc2,
                       double * dst)
{
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && lenSrc2 == colsSrc1);
    CNNPLUS_ASSERT(dst);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippmMul_mv_64f(src1,
                       static_cast<int>(strideSrc1 * sizeof(double)),
                       sizeof(double),
                       static_cast<int>(colsSrc1),
                       static_cast<int>(rowsSrc1),
                       src2,
                       sizeof(double),
                       static_cast<int>(lenSrc2),
                       dst,
                       sizeof(double)));
#else
    for (size_t r = 0; r < rowsSrc1; ++r) {
        double tmp = 0;
        for (size_t c = 0; c < colsSrc1; ++c) {
            tmp += src1[r * strideSrc1 + c] * src2[c];
        }
        dst[r] = tmp;
    }
#endif
}

//! Computes a transposed-matrix-vector product (single precision)
template<>
void mulmv<float,'t'>(float const * src1, size_t strideSrc1,
                      size_t rowsSrc1, size_t colsSrc1,
                      float const * src2, size_t lenSrc2,
                      float * dst)
{
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && lenSrc2 == rowsSrc1);
    CNNPLUS_ASSERT(dst);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippmMul_tv_32f(src1,
                       static_cast<int>(strideSrc1 * sizeof(float)),
                       sizeof(float),
                       static_cast<int>(colsSrc1),
                       static_cast<int>(rowsSrc1),
                       src2,
                       sizeof(float),
                       static_cast<int>(lenSrc2),
                       dst,
                       sizeof(float)));
#else
    for (size_t r = 0; r < rowsSrc1; ++r) {
        for (size_t c = 0; c < colsSrc1; ++c) {
            dst[c] = 0;
        }
        if (src2[r] != 0) {
            float const tmp = src2[r];
            for (size_t c = 0; c < colsSrc1; ++c) {
                dst[c] += tmp * src1[r * strideSrc1 + c];
            }
        }
    }
#endif
}

//! Computes a transposed-matrix-vector product (double precision)
template<>
void mulmv<double,'t'>(double const * src1, size_t strideSrc1,
                       size_t rowsSrc1, size_t colsSrc1,
                       double const * src2, size_t lenSrc2,
                       double * dst)
{
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && lenSrc2 == rowsSrc1);
    CNNPLUS_ASSERT(dst);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippmMul_tv_64f(src1,
                       static_cast<int>(strideSrc1 * sizeof(double)),
                       sizeof(double),
                       static_cast<int>(colsSrc1),
                       static_cast<int>(rowsSrc1),
                       src2,
                       sizeof(double),
                       static_cast<int>(lenSrc2),
                       dst,
                       sizeof(double)));
#else
    for (size_t r = 0; r < rowsSrc1; ++r) {
        for (size_t c = 0; c < colsSrc1; ++c) {
            dst[c] = 0;
        }
        if (src2[r] != 0) {
            double const tmp = src2[r];
            for (size_t c = 0; c < colsSrc1; ++c) {
                dst[c] += tmp * src1[r * strideSrc1 + c];
            }
        }
    }
#endif
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
#ifdef CNNPLUS_USE_INTEL_LIBS
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                static_cast<int>(rowsSrc1),
                static_cast<int>(colsSrc2),
                static_cast<int>(colsSrc1),
                alpha,
                src1, static_cast<int>(strideSrc1),
                src2, static_cast<int>(strideSrc2),
                beta,
                srcDst, static_cast<int>(strideSrcDst));
#else
    if (alpha == 0) {
        if (beta == 0) {
            for (size_t r = 0; r < rowsSrc1; ++r) {
                for (size_t c = 0; c < colsSrc2; ++c) {
                    srcDst[r * strideSrcDst + c] = 0;
                }
            }
        }
        else {
            for (size_t r = 0; r < rowsSrc1; ++r) {
                for (size_t c = 0; c < colsSrc2; ++c) {
                    srcDst[r * strideSrcDst + c] *= beta;
                }
            }
        }
    }
    else { // if (alpha != 0)
        for (size_t r = 0; r < rowsSrc1; ++r) {
            if (beta == 0) {
                for (size_t c = 0; c < colsSrc2; ++c) {
                    srcDst[r * strideSrcDst + c] = 0;
                }
            }
            else if (beta != 1) {
                for (size_t c = 0; c < colsSrc2; ++c) {
                    srcDst[r * strideSrcDst + c] *= beta;
                }
            }
            for (size_t i = 0; i < colsSrc1; ++i) {
                if (src1[r * strideSrc1 + i] != 0) {
                    float const tmp = alpha * src1[r * strideSrc1 + i];
                    for (size_t c = 0; c < colsSrc2; ++c) {
                        srcDst[r * strideSrcDst + c] += tmp * src2[i * strideSrc2 + c];
                    }
                }
            }
        }
    }
#endif
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
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && strideSrc2 >= colsSrc2);
    CNNPLUS_ASSERT(rowsSrc2 > 0 && colsSrc2 > 0);
    CNNPLUS_ASSERT(rowsSrc2 == colsSrc1);
    CNNPLUS_ASSERT(srcDst && strideSrcDst >= colsSrc2);
#ifdef CNNPLUS_USE_INTEL_LIBS
    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                static_cast<int>(rowsSrc1),
                static_cast<int>(colsSrc2),
                static_cast<int>(colsSrc1),
                alpha,
                src1, static_cast<int>(strideSrc1),
                src2, static_cast<int>(strideSrc2),
                beta,
                srcDst, static_cast<int>(strideSrcDst));
#else
    if (alpha == 0) {
        if (beta == 0) {
            for (size_t r = 0; r < rowsSrc1; ++r) {
                for (size_t c = 0; c < colsSrc2; ++c) {
                    srcDst[r * strideSrcDst + c] = 0;
                }
            }
        }
        else {
            for (size_t r = 0; r < rowsSrc1; ++r) {
                for (size_t c = 0; c < colsSrc2; ++c) {
                    srcDst[r * strideSrcDst + c] *= beta;
                }
            }
        }
    }
    else { // if (alpha != 0)
        for (size_t r = 0; r < rowsSrc1; ++r) {
            if (beta == 0) {
                for (size_t c = 0; c < colsSrc2; ++c) {
                    srcDst[r * strideSrcDst + c] = 0;
                }
            }
            else if (beta != 1) {
                for (size_t c = 0; c < colsSrc2; ++c) {
                    srcDst[r * strideSrcDst + c] *= beta;
                }
            }
            for (size_t i = 0; i < colsSrc1; ++i) {
                if (src1[r * strideSrc1 + i] != 0) {
                    double const tmp = alpha * src1[r * strideSrc1 + i];
                    for (size_t c = 0; c < colsSrc2; ++c) {
                        srcDst[r * strideSrcDst + c] += tmp * src2[i * strideSrc2 + c];
                    }
                }
            }
        }
    }
#endif
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
#ifdef CNNPLUS_USE_INTEL_LIBS
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                static_cast<int>(rowsSrc1),
                static_cast<int>(rowsSrc2),
                static_cast<int>(colsSrc1),
                alpha,
                src1, static_cast<int>(strideSrc1),
                src2, static_cast<int>(strideSrc2),
                beta,
                srcDst, static_cast<int>(strideSrcDst));
#else
    if (alpha == 0) {
        if (beta == 0) {
            for (size_t r = 0; r < rowsSrc1; ++r) {
                for (size_t c = 0; c < rowsSrc2; ++c) {
                    srcDst[r * strideSrcDst + c] = 0;
                }
            }
        }
        else {
            for (size_t r = 0; r < rowsSrc1; ++r) {
                for (size_t c = 0; c < rowsSrc2; ++c) {
                    srcDst[r * strideSrcDst + c] *= beta;
                }
            }
        }
    }
    else { // if (alpha != 0)
        for (size_t r = 0; r < rowsSrc1; ++r) {
            for (size_t c = 0; c < rowsSrc2; ++c) {
                float tmp = 0;
                for (size_t i = 0; i < colsSrc1; ++i) {
                    tmp += src1[r * strideSrc1 + i] * src2[c * strideSrc2 + i];
                }
                if (beta == 0) {
                    srcDst[r * strideSrcDst + c] = alpha * tmp;
                }
                else {
                    srcDst[r * strideSrcDst + c] = alpha * tmp + beta * srcDst[r * strideSrcDst + c];
                }
            }
        }
    }
#endif
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
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && strideSrc2 >= colsSrc2);
    CNNPLUS_ASSERT(rowsSrc2 > 0 && colsSrc2 > 0);
    CNNPLUS_ASSERT(colsSrc2 == colsSrc1);
    CNNPLUS_ASSERT(srcDst && strideSrcDst >= rowsSrc2);
#ifdef CNNPLUS_USE_INTEL_LIBS
    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                static_cast<int>(rowsSrc1),
                static_cast<int>(rowsSrc2),
                static_cast<int>(colsSrc1),
                alpha,
                src1, static_cast<int>(strideSrc1),
                src2, static_cast<int>(strideSrc2),
                beta,
                srcDst, static_cast<int>(strideSrcDst));
#else
    if (alpha == 0) {
        if (beta == 0) {
            for (size_t r = 0; r < rowsSrc1; ++r) {
                for (size_t c = 0; c < rowsSrc2; ++c) {
                    srcDst[r * strideSrcDst + c] = 0;
                }
            }
        }
        else {
            for (size_t r = 0; r < rowsSrc1; ++r) {
                for (size_t c = 0; c < rowsSrc2; ++c) {
                    srcDst[r * strideSrcDst + c] *= beta;
                }
            }
        }
    }
    else { // if (alpha != 0)
        for (size_t r = 0; r < rowsSrc1; ++r) {
            for (size_t c = 0; c < rowsSrc2; ++c) {
                double tmp = 0;
                for (size_t i = 0; i < colsSrc1; ++i) {
                    tmp += src1[r * strideSrc1 + i] * src2[c * strideSrc2 + i];
                }
                if (beta == 0) {
                    srcDst[r * strideSrcDst + c] = alpha * tmp;
                }
                else {
                    srcDst[r * strideSrcDst + c] = alpha * tmp + beta * srcDst[r * strideSrcDst + c];
                }
            }
        }
    }
#endif
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
#ifdef CNNPLUS_USE_INTEL_LIBS
    cblas_sgemm(CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                static_cast<int>(colsSrc1),
                static_cast<int>(colsSrc2),
                static_cast<int>(rowsSrc1),
                alpha,
                src1, static_cast<int>(strideSrc1),
                src2, static_cast<int>(strideSrc2),
                beta,
                srcDst, static_cast<int>(strideSrcDst));
#else
    if (alpha == 0) {
        if (beta == 0) {
            for (size_t r = 0; r < colsSrc1; ++r) {
                for (size_t c = 0; c < colsSrc2; ++c) {
                    srcDst[r * strideSrcDst + c] = 0;
                }
            }
        }
        else {
            for (size_t r = 0; r < colsSrc1; ++r) {
                for (size_t c = 0; c < colsSrc2; ++c) {
                    srcDst[r * strideSrcDst + c] *= beta;
                }
            }
        }
    }
    else { // if (alpha != 0)
        for (size_t r = 0; r < colsSrc1; ++r) {
            if (beta == 0) {
                for (size_t c = 0; c < colsSrc2; ++c) {
                    srcDst[r * strideSrcDst + c] = 0;
                }
            }
            else if (beta != 1) {
                for (size_t c = 0; c < colsSrc2; ++c) {
                    srcDst[r * strideSrcDst + c] *= beta;
                }
            }
            for (size_t i = 0; i < rowsSrc1; ++i) {
                if (src1[i * strideSrc1 + r] != 0) {
                    float const tmp = alpha * src1[i * strideSrc1 + r];
                    for (size_t c = 0; c < colsSrc2; ++c) {
                        srcDst[r * strideSrcDst + c] += tmp * src2[i * strideSrc2 + c];
                    }
                }
            }
        }
    }
#endif
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
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && strideSrc2 >= colsSrc2);
    CNNPLUS_ASSERT(rowsSrc2 > 0 && colsSrc2 > 0);
    CNNPLUS_ASSERT(rowsSrc2 == rowsSrc1);
    CNNPLUS_ASSERT(srcDst && strideSrcDst >= colsSrc2);
#ifdef CNNPLUS_USE_INTEL_LIBS
    cblas_dgemm(CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                static_cast<int>(colsSrc1),
                static_cast<int>(colsSrc2),
                static_cast<int>(rowsSrc1),
                alpha,
                src1, static_cast<int>(strideSrc1),
                src2, static_cast<int>(strideSrc2),
                beta,
                srcDst, static_cast<int>(strideSrcDst));
#else
    if (alpha == 0) {
        if (beta == 0) {
            for (size_t r = 0; r < colsSrc1; ++r) {
                for (size_t c = 0; c < colsSrc2; ++c) {
                    srcDst[r * strideSrcDst + c] = 0;
                }
            }
        }
        else {
            for (size_t r = 0; r < colsSrc1; ++r) {
                for (size_t c = 0; c < colsSrc2; ++c) {
                    srcDst[r * strideSrcDst + c] *= beta;
                }
            }
        }
    }
    else { // if (alpha != 0)
        for (size_t r = 0; r < colsSrc1; ++r) {
            if (beta == 0) {
                for (size_t c = 0; c < colsSrc2; ++c) {
                    srcDst[r * strideSrcDst + c] = 0;
                }
            }
            else if (beta != 1) {
                for (size_t c = 0; c < colsSrc2; ++c) {
                    srcDst[r * strideSrcDst + c] *= beta;
                }
            }
            for (size_t i = 0; i < rowsSrc1; ++i) {
                if (src1[i * strideSrc1 + r] != 0) {
                    double const tmp = alpha * src1[i * strideSrc1 + r];
                    for (size_t c = 0; c < colsSrc2; ++c) {
                        srcDst[r * strideSrcDst + c] += tmp * src2[i * strideSrc2 + c];
                    }
                }
            }
        }
    }
#endif
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
#ifdef CNNPLUS_USE_INTEL_LIBS
    cblas_sgemm(CblasRowMajor,
                CblasTrans,
                CblasTrans,
                static_cast<int>(colsSrc1),
                static_cast<int>(rowsSrc2),
                static_cast<int>(rowsSrc1),
                alpha,
                src1, static_cast<int>(strideSrc1),
                src2, static_cast<int>(strideSrc2),
                beta,
                srcDst, static_cast<int>(strideSrcDst));
#else
    if (alpha == 0) {
        if (beta == 0) {
            for (size_t r = 0; r < colsSrc1; ++r) {
                for (size_t c = 0; c < rowsSrc2; ++c) {
                    srcDst[r * strideSrcDst + c] = 0;
                }
            }
        }
        else {
            for (size_t r = 0; r < colsSrc1; ++r) {
                for (size_t c = 0; c < rowsSrc2; ++c) {
                    srcDst[r * strideSrcDst + c] *= beta;
                }
            }
        }
    }
    else { // if (alpha != 0)
        for (size_t r = 0; r < colsSrc1; ++r) {
            for (size_t c = 0; c < rowsSrc2; ++c) {
                float tmp = 0;
                for (size_t i = 0; i < rowsSrc1; ++i) {
                    tmp += src1[i * strideSrc1 + r] * src2[c * strideSrc2 + i];
                }
                if (beta == 0) {
                    srcDst[r * strideSrcDst + c] = alpha * tmp;
                }
                else {
                    srcDst[r * strideSrcDst + c] = alpha * tmp + beta * srcDst[r * strideSrcDst + c];
                }
            }
        }
    }
#endif
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
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && strideSrc2 >= colsSrc2);
    CNNPLUS_ASSERT(rowsSrc2 > 0 && colsSrc2 > 0);
    CNNPLUS_ASSERT(colsSrc2 == rowsSrc1);
    CNNPLUS_ASSERT(srcDst && strideSrcDst >= rowsSrc2);
#ifdef CNNPLUS_USE_INTEL_LIBS
    cblas_dgemm(CblasRowMajor,
                CblasTrans,
                CblasTrans,
                static_cast<int>(colsSrc1),
                static_cast<int>(rowsSrc2),
                static_cast<int>(rowsSrc1),
                alpha,
                src1, static_cast<int>(strideSrc1),
                src2, static_cast<int>(strideSrc2),
                beta,
                srcDst, static_cast<int>(strideSrcDst));
#else
    if (alpha == 0) {
        if (beta == 0) {
            for (size_t r = 0; r < colsSrc1; ++r) {
                for (size_t c = 0; c < rowsSrc2; ++c) {
                    srcDst[r * strideSrcDst + c] = 0;
                }
            }
        }
        else {
            for (size_t r = 0; r < colsSrc1; ++r) {
                for (size_t c = 0; c < rowsSrc2; ++c) {
                    srcDst[r * strideSrcDst + c] *= beta;
                }
            }
        }
    }
    else { // if (alpha != 0)
        for (size_t r = 0; r < colsSrc1; ++r) {
            for (size_t c = 0; c < rowsSrc2; ++c) {
                double tmp = 0;
                for (size_t i = 0; i < rowsSrc1; ++i) {
                    tmp += src1[i * strideSrc1 + r] * src2[c * strideSrc2 + i];
                }
                if (beta == 0) {
                    srcDst[r * strideSrcDst + c] = alpha * tmp;
                }
                else {
                    srcDst[r * strideSrcDst + c] = alpha * tmp + beta * srcDst[r * strideSrcDst + c];
                }
            }
        }
    }
#endif
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
#ifdef CNNPLUS_USE_INTEL_LIBS
    cblas_sgemv(CblasRowMajor,
                CblasNoTrans,
                static_cast<int>(rowsSrc1),
                static_cast<int>(colsSrc1),
                alpha,
                src1, static_cast<int>(strideSrc1),
                src2, 1,
                beta,
                srcDst, 1);
#else
    if (beta == 0) {
        for (size_t r = 0; r < rowsSrc1; ++r)
            srcDst[r] = 0;
    }
    else {
        for (size_t r = 0; r < rowsSrc1; ++r)
            srcDst[r] *= beta;
    }
    if (alpha != 0) {
        for (size_t r = 0; r < rowsSrc1; ++r) {
            float tmp = 0;
            for (size_t c = 0; c < colsSrc1; ++c)
                tmp += src1[r * strideSrc1 + c] * src2[c];
            srcDst[r] += alpha * tmp;
        }
    }
#endif
}

//! Computes a matrix-vector product using a general matrix (double precision)
template<>
void gemv<double,'n'>(double const * src1, size_t strideSrc1,
                      size_t rowsSrc1, size_t colsSrc1,
                      double const * src2, size_t lenSrc2,
                      double * srcDst, double alpha, double beta)
{
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && lenSrc2 == colsSrc1);
    CNNPLUS_ASSERT(srcDst);
#ifdef CNNPLUS_USE_INTEL_LIBS
    cblas_dgemv(CblasRowMajor,
                CblasNoTrans,
                static_cast<int>(rowsSrc1),
                static_cast<int>(colsSrc1),
                alpha,
                src1, static_cast<int>(strideSrc1),
                src2, 1,
                beta,
                srcDst, 1);
#else
    if (beta == 0) {
        for (size_t r = 0; r < rowsSrc1; ++r)
            srcDst[r] = 0;
    }
    else {
        for (size_t r = 0; r < rowsSrc1; ++r)
            srcDst[r] *= beta;
    }
    if (alpha != 0) {
        for (size_t r = 0; r < rowsSrc1; ++r) {
            double tmp = 0;
            for (size_t c = 0; c < colsSrc1; ++c)
                tmp += src1[r * strideSrc1 + c] * src2[c];
            srcDst[r] += alpha * tmp;
        }
    }
#endif
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
#ifdef CNNPLUS_USE_INTEL_LIBS
    cblas_sgemv(CblasRowMajor,
                CblasTrans,
                static_cast<int>(rowsSrc1),
                static_cast<int>(colsSrc1),
                alpha,
                src1, static_cast<int>(strideSrc1),
                src2, 1,
                beta,
                srcDst, 1);
#else
    if (beta == 0) {
        for (size_t c = 0; c < colsSrc1; ++c)
            srcDst[c] = 0;
    }
    else {
        for (size_t c = 0; c < colsSrc1; ++c)
            srcDst[c] *= beta;
    }
    if (alpha != 0) {
        for (size_t r = 0; r < rowsSrc1; ++r) {
            if (src2[r] != 0) {
                float const tmp = alpha * src2[r];
                for (size_t c = 0; c < colsSrc1; ++c)
                    srcDst[c] += tmp * src1[r * strideSrc1 + c];
            }
        }
    }
#endif
}

//! Computes a transposed-matrix-vector product using a general matrix (double precision)
template<>
void gemv<double,'t'>(double const * src1, size_t strideSrc1,
                      size_t rowsSrc1, size_t colsSrc1,
                      double const * src2, size_t lenSrc2,
                      double * srcDst, double alpha, double beta)
{
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && lenSrc2 == rowsSrc1);
    CNNPLUS_ASSERT(srcDst);
#ifdef CNNPLUS_USE_INTEL_LIBS
    cblas_dgemv(CblasRowMajor,
                CblasTrans,
                static_cast<int>(rowsSrc1),
                static_cast<int>(colsSrc1),
                alpha,
                src1, static_cast<int>(strideSrc1),
                src2, 1,
                beta,
                srcDst, 1);
#else
    if (beta == 0) {
        for (size_t c = 0; c < colsSrc1; ++c)
            srcDst[c] = 0;
    }
    else {
        for (size_t c = 0; c < colsSrc1; ++c)
            srcDst[c] *= beta;
    }
    if (alpha != 0) {
        for (size_t r = 0; r < rowsSrc1; ++r) {
            if (src2[r] != 0) {
                double const tmp = alpha * src2[r];
                for (size_t c = 0; c < colsSrc1; ++c)
                    srcDst[c] += tmp * src1[r * strideSrc1 + c];
            }
        }
    }
#endif
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
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippmGaxpy_mv_32f(src1,
                         static_cast<int>(strideSrc1 * sizeof(float)),
                         sizeof(float),
                         static_cast<int>(colsSrc1),
                         static_cast<int>(rowsSrc1),
                         src2,
                         sizeof(float),
                         static_cast<int>(lenSrc2),
                         src3,
                         sizeof(float),
                         static_cast<int>(lenSrc3),
                         dst,
                         sizeof(float)));
#else
    for (size_t r = 0; r < rowsSrc1; ++r) {
        float tmp = 0;
        for (size_t c = 0; c < colsSrc1; ++c)
            tmp += src1[r * strideSrc1 + c] * src2[c];
        dst[r] = tmp + src3[r];
    }
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
    CNNPLUS_ASSERT(src1 && strideSrc1 >= colsSrc1);
    CNNPLUS_ASSERT(rowsSrc1 > 0 && colsSrc1 > 0);
    CNNPLUS_ASSERT(src2 && lenSrc2 == colsSrc1);
    CNNPLUS_ASSERT(src3 && lenSrc3 == rowsSrc1);
    CNNPLUS_ASSERT(dst);
#ifdef CNNPLUS_USE_INTEL_LIBS
    IPP_SAFE_CALL(
        ippmGaxpy_mv_64f(src1,
                         static_cast<int>(strideSrc1 * sizeof(double)),
                         sizeof(double),
                         static_cast<int>(colsSrc1),
                         static_cast<int>(rowsSrc1),
                         src2,
                         sizeof(double),
                         static_cast<int>(lenSrc2),
                         src3,
                         sizeof(double),
                         static_cast<int>(lenSrc3),
                         dst,
                         sizeof(double)));
#else
    for (size_t r = 0; r < rowsSrc1; ++r) {
        double tmp = 0;
        for (size_t c = 0; c < colsSrc1; ++c)
            tmp += src1[r * strideSrc1 + c] * src2[c];
        dst[r] = tmp + src3[r];
    }
#endif
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
#ifdef CNNPLUS_USE_INTEL_LIBS
    cblas_sger(CblasRowMajor,
               static_cast<int>(lenSrc1),
               static_cast<int>(lenSrc2),
               alpha,
               src1, 1,
               src2, 1,
               srcDst, static_cast<int>(strideSrcDst));
#else
    if (lenSrc1 == 0 || lenSrc2 == 0 || alpha == 0) return;
    for (size_t r = 0; r < lenSrc1; ++r) {
        if (src1[r] != 0) {
            float const tmp = alpha * src1[r];
            for (size_t c = 0; c < lenSrc2; ++c)
                srcDst[r * strideSrcDst + c] += tmp * src2[c];
        }
    }
#endif
}

//! Performs a rank-1 update of a general matrix (double precision)
template<>
void ger<double>(double const * src1, size_t lenSrc1,
                 double const * src2, size_t lenSrc2,
                 double * srcDst, size_t strideSrcDst,
                 double alpha)
{
    CNNPLUS_ASSERT(src1 && lenSrc1 > 0);
    CNNPLUS_ASSERT(src2 && lenSrc2 > 0);
    CNNPLUS_ASSERT(srcDst && strideSrcDst >= lenSrc2);
#ifdef CNNPLUS_USE_INTEL_LIBS
    cblas_dger(CblasRowMajor,
               static_cast<int>(lenSrc1),
               static_cast<int>(lenSrc2),
               alpha,
               src1, 1,
               src2, 1,
               srcDst, static_cast<int>(strideSrcDst));
#else
    if (lenSrc1 == 0 || lenSrc2 == 0 || alpha == 0) return;
    for (size_t r = 0; r < lenSrc1; ++r) {
        if (src1[r] != 0) {
            double const tmp = alpha * src1[r];
            for (size_t c = 0; c < lenSrc2; ++c)
                srcDst[r * strideSrcDst + c] += tmp * src2[c];
        }
    }
#endif
}

//! Computes a vector-scalar product and adds the result to a vector (single precision)
template<>
void axpy<float>(float const * src, size_t len, float * srcDst, float alpha)
{
    CNNPLUS_ASSERT(src && srcDst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    cblas_saxpy(static_cast<int>(len), alpha, src, 1, srcDst, 1);
#else
    if (len == 0 || alpha == 0) return;
    for (size_t i = 0; i < len; ++i)
        srcDst[i] += alpha * src[i];
#endif
}

//! Computes a vector-scalar product and adds the result to a vector (double precision)
template<>
void axpy<double>(double const * src, size_t len, double * srcDst, double alpha)
{
    CNNPLUS_ASSERT(src && srcDst && len > 0);
#ifdef CNNPLUS_USE_INTEL_LIBS
    cblas_daxpy(static_cast<int>(len), alpha, src, 1, srcDst, 1);
#else
    if (len == 0 || alpha == 0) return;
    for (size_t i = 0; i < len; ++i)
        srcDst[i] += alpha * src[i];
#endif
}

//! The \e logistic neural-net sigmoid function
template<typename T>
void logsigm(T const * src, size_t strideSrc, T * dst, size_t strideDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && strideSrc >= cols);
    CNNPLUS_ASSERT(dst && strideDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            dst[r * strideDst + c] = mathli::logsigmoid(src[r * strideSrc + c]);
        }
    }
}

//! The \e logistic neural-net sigmoid function (in-place)
template<typename T>
void logsigm(T * srcDst, size_t strideSrcDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(srcDst && strideSrcDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            srcDst[r * strideSrcDst + c] = mathli::logsigmoid(srcDst[r * strideSrcDst + c]);
        }
    }
}

//! The derivative of #logsigm
template<typename T>
void dlogsigm(T const * src, size_t strideSrc, T * dst, size_t strideDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && strideSrc >= cols);
    CNNPLUS_ASSERT(dst && strideDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            dst[r * strideDst + c] = mathli::dlogsigmoid(src[r * strideSrc + c]);
        }
    }
}

//! The derivative of #logsigm (in-place)
template<typename T>
void dlogsigm(T * srcDst, size_t strideSrcDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(srcDst && strideSrcDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            srcDst[r * strideSrcDst + c] = mathli::dlogsigmoid(srcDst[r * strideSrcDst + c]);
        }
    }
}

//! The \e standard neural-net sigmoid function
template<typename T>
void stdsigm(T const * src, size_t strideSrc, T * dst, size_t strideDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && strideSrc >= cols);
    CNNPLUS_ASSERT(dst && strideDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            dst[r * strideDst + c] = mathli::stdsigmoid(src[r * strideSrc + c]);
        }
    }
}

//! The \e standard neural-net sigmoid function (in-place)
template<typename T>
void stdsigm(T * srcDst, size_t strideSrcDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(srcDst && strideSrcDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            srcDst[r * strideSrcDst + c] = mathli::stdsigmoid(srcDst[r * strideSrcDst + c]);
        }
    }
}

//! The derivative of #stdsigm
template<typename T>
void dstdsigm(T const * src, size_t strideSrc, T * dst, size_t strideDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && strideSrc >= cols);
    CNNPLUS_ASSERT(dst && strideDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            dst[r * strideDst + c] = mathli::dstdsigmoid(src[r * strideSrc + c]);
        }
    }
}

//! The derivative of #stdsigm (in-place)
template<typename T>
void dstdsigm(T * srcDst, size_t strideSrcDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(srcDst && strideSrcDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            srcDst[r * strideSrcDst + c] = mathli::dstdsigmoid(srcDst[r * strideSrcDst + c]);
        }
    }
}

//! The hyperbolic tangent function
template<typename T>
void tanhm(T const * src, size_t strideSrc, T * dst, size_t strideDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && strideSrc >= cols);
    CNNPLUS_ASSERT(dst && strideDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            dst[r * strideDst + c] = mathli::tanh(src[r * strideSrc + c]);
        }
    }
}

//! The hyperbolic tangent function (in-place)
template<typename T>
void tanhm(T * srcDst, size_t strideSrcDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(srcDst && strideSrcDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            srcDst[r * strideSrcDst + c] = mathli::tanh(srcDst[r * strideSrcDst + c]);
        }
    }
}

//! The derivative of #tanhm
template<typename T>
void dtanhm(T const * src, size_t strideSrc, T * dst, size_t strideDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && strideSrc >= cols);
    CNNPLUS_ASSERT(dst && strideDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            dst[r * strideDst + c] = mathli::dtanh(src[r * strideSrc + c]);
        }
    }
}

//! The derivative of #tanhm (in-place)
template<typename T>
void dtanhm(T * srcDst, size_t strideSrcDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(srcDst && strideSrcDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            srcDst[r * strideSrcDst + c] = mathli::dtanh(srcDst[r * strideSrcDst + c]);
        }
    }
}

//! Computes the summed-area table (also known as integral image)
/*! \see Crow, Franklin (1984). "Summed-area tables for texture mapping".
         SIGGRAPH '84: Proceedings of the 11th annual conference on Computer
         graphics and interactive techniques: 207-212.
 */
template<typename T>
void sat(T const * src, size_t strideSrc, T * dst, size_t strideDst, size_t rows, size_t cols)
{
    CNNPLUS_ASSERT(src && strideSrc >= cols);
    CNNPLUS_ASSERT(dst && strideDst >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    // Compute horizontal sum
    for (size_t r = 0; r < rows; ++r)
    {
        T const * pSrc = src + r * strideSrc;
        T       * pDst = dst + r * strideDst;

        pDst[0] = pSrc[0];

        for (size_t c = 1; c < cols; ++c)
            pDst[c] = pDst[c-1] + pSrc[c];
    }

    // Compute vertical sum
    for (size_t r = 1; r < rows; ++r) {
        addv<T>(dst +  r    * strideDst,
                dst + (r-1) * strideDst,
                cols);
    }
}

//! Initializes the columns of a matrix
template<typename T>
void setcol(T * dst, size_t stride, size_t rows, size_t cols, T const * src)
{
    CNNPLUS_ASSERT(dst && stride >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);
    CNNPLUS_ASSERT(src);

    for (size_t r = 0; r < rows; ++r)
        setv<T>(dst + r * stride, cols, src[r]);
}

/*! \addtogroup eti_grp Explicit Template Instantiation
 @{
 */
template float  * allocv<float >(size_t len);
template double * allocv<double>(size_t len);

template float  * allocm<float >(size_t rows, size_t cols, size_t & stride);
template double * allocm<double>(size_t rows, size_t cols, size_t & stride);

template void free<float >(float  * ptr);
template void free<double>(double * ptr);

template float  getelv<float >(float  const * src, size_t i);
template double getelv<double>(double const * src, size_t i);

template float  getelm<float >(float  const * src, size_t stride, size_t r, size_t c);
template double getelm<double>(double const * src, size_t stride, size_t r, size_t c);

template void setelv<float >(float  * dst, size_t i, float  val);
template void setelv<double>(double * dst, size_t i, double val);

template void setelm<float >(float  * dst, size_t stride, size_t r, size_t c, float  val);
template void setelm<double>(double * dst, size_t stride, size_t r, size_t c, double val);

template void zerom<float >(float  * dst, size_t stride, size_t rows, size_t cols);
template void zerom<double>(double * dst, size_t stride, size_t rows, size_t cols);

template void setm<float >(float  * dst, size_t stride, size_t rows, size_t cols, float  val);
template void setm<double>(double * dst, size_t stride, size_t rows, size_t cols, double val);

template void randm<float >(float  * dst, size_t stride, size_t rows, size_t cols, float  sigma);
template void randm<double>(double * dst, size_t stride, size_t rows, size_t cols, double sigma);

template void randm<float >(float  * dst, size_t stride, size_t rows, size_t cols, float  low, float  high);
template void randm<double>(double * dst, size_t stride, size_t rows, size_t cols, double low, double high);

template void divmc<float >(float  const * src, size_t strideSrc, float  val, float  * dst, size_t strideDst, size_t rows, size_t cols);
template void divmc<double>(double const * src, size_t strideSrc, double val, double * dst, size_t strideDst, size_t rows, size_t cols);

template void divmc<float >(float  * srcDst, size_t strideSrcDst, float  val, size_t rows, size_t cols);
template void divmc<double>(double * srcDst, size_t strideSrcDst, double val, size_t rows, size_t cols);

template void expm<float >(float  const * src, size_t strideSrc, float  * dst, size_t strideDst, size_t rows, size_t cols);
template void expm<double>(double const * src, size_t strideSrc, double * dst, size_t strideDst, size_t rows, size_t cols);

template void expm<float >(float  * srcDst, size_t strideSrcDst, size_t rows, size_t cols);
template void expm<double>(double * srcDst, size_t strideSrcDst, size_t rows, size_t cols);

template float  summ<float >(float  const * src, size_t strideSrc, size_t rows, size_t cols);
template double summ<double>(double const * src, size_t strideSrc, size_t rows, size_t cols);

template float  sumsqrv<float >(float  const * src, size_t len);
template double sumsqrv<double>(double const * src, size_t len);

template void copymv<float >(float  const * src, size_t strideSrc, float  * dst, size_t rows, size_t cols);
template void copymv<double>(double const * src, size_t strideSrc, double * dst, size_t rows, size_t cols);

template void copyvm<float >(float  const * src, float  * dst, size_t strideDst, size_t rows, size_t cols);
template void copyvm<double>(double const * src, double * dst, size_t strideDst, size_t rows, size_t cols);

template void sumrowacc<float >(float  const * src, size_t stride, float  * srcDst, size_t rows, size_t cols);
template void sumrowacc<double>(double const * src, size_t stride, double * srcDst, size_t rows, size_t cols);

template void sumcolacc<float >(float  const * src, size_t stride, float  * srcDst, size_t rows, size_t cols);
template void sumcolacc<double>(double const * src, size_t stride, double * srcDst, size_t rows, size_t cols);

template void mulmc<float >(float  * srcDst, size_t strideSrcDst, float  val, size_t rows, size_t cols);
template void mulmc<double>(double * srcDst, size_t strideSrcDst, double val, size_t rows, size_t cols);

template void pmulmm<float >(float  const * src1, size_t strideSrc1, float  const * src2, size_t strideSrc2, float  * dst, size_t strideDst, size_t rows, size_t cols);
template void pmulmm<double>(double const * src1, size_t strideSrc1, double const * src2, size_t strideSrc2, double * dst, size_t strideDst, size_t rows, size_t cols);

template void pmulmm<float >(float  * srcDst, size_t strideSrcDst, float  const * src, size_t strideSrc, size_t rows, size_t cols);
template void pmulmm<double>(double * srcDst, size_t strideSrcDst, double const * src, size_t strideSrc, size_t rows, size_t cols);

template void logsigm<float >(float  const * src, size_t strideSrc, float  * dst, size_t strideDst, size_t rows, size_t cols);
template void logsigm<double>(double const * src, size_t strideSrc, double * dst, size_t strideDst, size_t rows, size_t cols);

template void logsigm<float >(float  * srcDst, size_t strideSrcDst, size_t rows, size_t cols);
template void logsigm<double>(double * srcDst, size_t strideSrcDst, size_t rows, size_t cols);

template void dlogsigm<float >(float  const * src, size_t strideSrc, float  * dst, size_t strideDst, size_t rows, size_t cols);
template void dlogsigm<double>(double const * src, size_t strideSrc, double * dst, size_t strideDst, size_t rows, size_t cols);

template void dlogsigm<float >(float  * srcDst, size_t strideSrcDst, size_t rows, size_t cols);
template void dlogsigm<double>(double * srcDst, size_t strideSrcDst, size_t rows, size_t cols);

template void stdsigm<float >(float  const * src, size_t strideSrc, float  * dst, size_t strideDst, size_t rows, size_t cols);
template void stdsigm<double>(double const * src, size_t strideSrc, double * dst, size_t strideDst, size_t rows, size_t cols);

template void stdsigm<float >(float  * srcDst, size_t strideSrcDst, size_t rows, size_t cols);
template void stdsigm<double>(double * srcDst, size_t strideSrcDst, size_t rows, size_t cols);

template void dstdsigm<float >(float  const * src, size_t strideSrc, float  * dst, size_t strideDst, size_t rows, size_t cols);
template void dstdsigm<double>(double const * src, size_t strideSrc, double * dst, size_t strideDst, size_t rows, size_t cols);

template void dstdsigm<float >(float  * srcDst, size_t strideSrcDst, size_t rows, size_t cols);
template void dstdsigm<double>(double * srcDst, size_t strideSrcDst, size_t rows, size_t cols);

template void tanhm<float >(float  const * src, size_t strideSrc, float  * dst, size_t strideDst, size_t rows, size_t cols);
template void tanhm<double>(double const * src, size_t strideSrc, double * dst, size_t strideDst, size_t rows, size_t cols);

template void tanhm<float >(float  * srcDst, size_t strideSrcDst, size_t rows, size_t cols);
template void tanhm<double>(double * srcDst, size_t strideSrcDst, size_t rows, size_t cols);

template void dtanhm<float >(float  const * src, size_t strideSrc, float  * dst, size_t strideDst, size_t rows, size_t cols);
template void dtanhm<double>(double const * src, size_t strideSrc, double * dst, size_t strideDst, size_t rows, size_t cols);

template void dtanhm<float >(float  * srcDst, size_t strideSrcDst, size_t rows, size_t cols);
template void dtanhm<double>(double * srcDst, size_t strideSrcDst, size_t rows, size_t cols);

template void sat<float >(float  const * src, size_t strideSrc, float  * dst, size_t strideDst, size_t rows, size_t cols);
template void sat<double>(double const * src, size_t strideSrc, double * dst, size_t strideDst, size_t rows, size_t cols);

template void setcol<float >(float  * dst, size_t stride, size_t rows, size_t cols, float  const * src);
template void setcol<double>(double * dst, size_t stride, size_t rows, size_t cols, double const * src);

/*! @} */

}; // namespace matvecli

CNNPLUS_NS_END
