/**************************************************************************//**
 *
 * \file   cumvli.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   May 26 2009
 *
 * $Id: cumvli.hh 1586 2009-06-23 19:25:24Z dast $
 *
 * \brief  Header for cnnplus::cumvli.
 *
 *****************************************************************************/

#ifndef CNNPLUS_CUMVLI_HH
#define CNNPLUS_CUMVLI_HH

#include "common.hh"

CNNPLUS_NS_BEGIN

//! A set of functions to compute common vector- and matrix-operations on the GPU
namespace cumvli {

//! Returns the free GPU memory in bytes
size_t freemem();

//! Allocates GPU memory for a vector of length \a len
template<typename T>
T * allocv(size_t len);

//! Allocates GPU memory for a matrix of size \a rows x \a cols
template<typename T>
T * allocm(size_t rows, size_t cols, size_t & stride);

//! Frees the allocated GPU memory
template<typename T>
void free(T * ptr);

//! Returns a GPU vector element
template<typename T>
T getelv(T const * src, size_t i);

//! Returns a GPU matrix element
template<typename T>
T getelm(T const * src, size_t stride, size_t r, size_t c);

//! Sets a GPU vector element
template<typename T>
void setelv(T * dst, size_t i, T val);

//! Sets a GPU matrix element
template<typename T>
void setelm(T * dst, size_t stride, size_t r, size_t c, T val);

//! Copies the contents of one GPU vector into another
template<typename T>
void copyvv(T const * src, T * dst, size_t len);

//! Copies the contents of one GPU matrix into another
template<typename T>
void copymm(T const * src, size_t strideSrc, T * dst, size_t strideDst, size_t rows, size_t cols);

//! Copies the contents of a GPU matrix into a GPU vector
template<typename T>
void copymv(T const * src, size_t strideSrc, T * dst, size_t rows, size_t cols);

//! Copies the contents of a GPU vector into a GPU matrix
template<typename T>
void copyvm(T const * src, T * dst, size_t strideDst, size_t rows, size_t cols);

//! Copies a vector from CPU memory to GPU memory
template<typename T>
void copyv_h2d(T const * src, T * dst, size_t len);

//! Copies a vector from GPU memory to CPU memory
template<typename T>
void copyv_d2h(T const * src, T * dst, size_t len);

//! Copies a matrix from CPU memory to GPU memory
template<typename T>
void copym_h2d(T const * src, size_t strideSrc, T * dst, size_t strideDst, size_t rows, size_t cols);

//! Copies a matrix from GPU memory to CPU memory
template<typename T>
void copym_d2h(T const * src, size_t strideSrc, T * dst, size_t strideDst, size_t rows, size_t cols);

//! Initializes a GPU vector to zero
template<typename T>
void zerov(T * dst, size_t len);

//! Initializes a GPU matrix to zero
template<typename T>
void zerom(T * dst, size_t stride, size_t rows, size_t cols);

//! Computes a vector-scalar product and adds the result to a vector
template<typename T>
void axpy(T const * src, size_t len, T * srcDst, T alpha = 1);

//! Computes a matrix-vector product using a general matrix
template<typename T, char tSrc1>
void gemv(T const * src1, size_t strideSrc1,
          size_t rowsSrc1, size_t colsSrc1,
          T const * src2, size_t lenSrc2,
          T * srcDst, T alpha = 1, T beta = 1);

//! Computes a matrix-vector product using a general matrix
template<typename T>
void gemv(T const * src1, size_t strideSrc1,
          size_t rowsSrc1, size_t colsSrc1,
          T const * src2, size_t lenSrc2,
          T const * src3, size_t lenSrc3,
          T * dst);

//! Performs a rank-1 update of a general matrix
template<typename T>
void ger(T const * src1, size_t lenSrc1,
         T const * src2, size_t lenSrc2,
         T * srcDst, size_t strideSrcDst,
         T alpha = 1);

//! Adds the elements of two vectors (in-place)
template<typename T>
void addv(T * srcDst, T const * src, size_t len);

//! Computes a scalar-matrix-matrix product and adds the result to a scalar-matrix product
template<typename T, char tSrc1, char tSrc2>
void gemm(T const * src1, size_t strideSrc1,
          size_t rowsSrc1, size_t colsSrc1,
          T const * src2, size_t strideSrc2,
          size_t rowsSrc2, size_t colsSrc2,
          T * srcDst, size_t strideSrcDst,
          T alpha = 1, T beta = 1);

//! Computes a matrix-matrix product
template<typename T, char tSrc1, char tSrc2>
void mulmm(T const * src1, size_t strideSrc1,
           size_t rowsSrc1, size_t colsSrc1,
           T const * src2, size_t strideSrc2,
           size_t rowsSrc2, size_t colsSrc2,
           T * dst, size_t strideDst);

//! Calculates the sums of row vectors in a matrix
template<typename T>
void sumrow(T const * src, size_t stride, T * dst, size_t rows, size_t cols);

//! Calculates the sums of row vectors in a matrix and accumulates it to a vector
template<typename T>
void sumrowacc(T const * src, size_t stride, T * srcDst, size_t rows, size_t cols);

//! Computes the pointwise multiplication of two matrices
template<typename T>
void pmulmm(T const * src1, size_t strideSrc1,
            T const * src2, size_t strideSrc2,
            T * dst, size_t strideDst,
            size_t rows, size_t cols);

//! Computes the pointwise multiplication of two matrices (in-place)
template<typename T>
void pmulmm(T * srcDst, size_t strideSrcDst,
            T const * src, size_t strideSrc,
            size_t rows, size_t cols);

//! Computes a matrix-scalar product and adds the result to a matrix
template<typename T>
void axpy(T const * src, size_t strideSrc, size_t rows, size_t cols,
          T * srcDst, size_t strideSrcDst, T alpha = 1);

//! Initializes the vector elements to a specified common value
template<typename T>
void setv(T * dst, size_t len, T val);

//! Initializes the matrix elements to a specified common value
template<typename T>
void setm(T * dst, size_t stride, size_t rows, size_t cols, T val);

//! Multiplies the elements of two vectors
template<typename T>
void mulv(T const * src1, T const * src2, T * dst, size_t len);

//! Multiplies the elements of two vectors (in-place)
template<typename T>
void mulv(T * srcDst, T const * src, size_t len);

//! Returns the maximum absolute value of a vector
template<typename T>
T absmaxv(T const * src, size_t len);

//! Initializes the columns of a matrix
template<typename T>
void setcol(T * dst, size_t stride, size_t rows, size_t cols, T const * src);

}; // namespace cumvli

CNNPLUS_NS_END

#endif // CNNPLUS_CUMVLI_HH
