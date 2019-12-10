/**************************************************************************//**
 *
 * \file   cudautils.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   May 25 2009
 *
 * $Id: cudautils.hh 3561 2010-11-22 21:14:16Z klaus $
 *
 * \brief  Helper tools for CUDA.
 *
 *****************************************************************************/

#ifndef CNNPLUS_CUDAUTILS_HH
#define CNNPLUS_CUDAUTILS_HH

#include <cuda_runtime.h>
#include <cstdio>

#if defined(_MSC_VER)
// warning C4231: nonstandard extension used :
//   'extern' before template explicit instantiation
#pragma warning(disable: 4231)
// warning C4355: 'this' : used in base member initializer list
#pragma warning(disable: 4355)
#endif // _MSC_VER

#ifdef NDEBUG

#define CUDA_SAFE_CALL_NO_SYNC(call) call
#define CUDA_SAFE_CALL(call) call
#define CUDA_CHECK_ERROR_NO_SYNC(message)
#define CUDA_CHECK_ERROR(message)

#else

#define CUDA_SAFE_CALL_NO_SYNC(call) do {                                     \
    cudaError const err = call;                                               \
    if (err != cudaSuccess) {                                                 \
        printf("Cuda error in file '%s' at line %d: %s.\n",                   \
            __FILE__, __LINE__, cudaGetErrorString(err));                     \
        exit(EXIT_FAILURE);                                                   \
    } } while (0)

#define CUDA_SAFE_CALL(call) do {                                             \
    CUDA_SAFE_CALL_NO_SYNC(call);                                             \
    cudaError const err = cudaThreadSynchronize();                            \
    if (err != cudaSuccess) {                                                 \
        printf("Cuda error in file '%s' at line %d: %s.\n",                   \
            __FILE__, __LINE__, cudaGetErrorString(err));                     \
        exit(EXIT_FAILURE);                                                   \
    } } while (0)

#define CUDA_CHECK_ERROR_NO_SYNC(message) do {                                \
    cudaError const err = cudaGetLastError();                                 \
    if (err != cudaSuccess) {                                                 \
        printf("Cuda error: %s [file '%s', line %d; %s].\n",                  \
            message, __FILE__, __LINE__, cudaGetErrorString(err));            \
        exit(EXIT_FAILURE);                                                   \
    } } while (0)

#define CUDA_CHECK_ERROR(message) do {                                        \
    CUDA_CHECK_ERROR_NO_SYNC(message);                                        \
    cudaError const err = cudaThreadSynchronize();                            \
    if (err != cudaSuccess) {                                                 \
        printf("Cuda error: %s [file '%s', line %d; %s].\n",                  \
            message, __FILE__, __LINE__, cudaGetErrorString(err));            \
        exit(EXIT_FAILURE);                                                   \
    } } while (0)

#endif

#define WARP_SIZE        32
#define MAX_THREADS      512
#define THREADS          128
#define BLOCK_WIDTH      16
#define BLOCK_HEIGHT     16

#if defined(__DEVICE_EMULATION__) || defined(__GF100__)
    #define CNN_IMUL(a, b)   (((int)(a)) * ((int)(b)))
    #define CNN_UIMUL(a, b)  (((size_t)(a)) * ((size_t)(b)))
    #define EMUSYNC      __syncthreads()
#else
    // __mul24 and __umul24 take only 4 cycles to compute, in the future however
    // this may change and direct 32-bit operations (that take 16 cycles) can
    // be faster. Use IMUL and UIMUL instead of __[u]mul24 so that the code can
    // be easilty adapted in the future.
    #define CNN_IMUL(a, b)   __mul24(a, b)
    #define CNN_UIMUL(a, b)  __umul24((size_t)a, (size_t)b)
    #define EMUSYNC
#endif

#endif // CNNPLUS_CUDAUTILS_HH
