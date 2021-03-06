/**************************************************************************//**
 *
 * \file   matvecli.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jan 14 2009
 *
 * $Id: matvecli.hh 1638 2009-06-30 19:50:23Z dast $
 *
 * \brief  Header for cnnplus::matvecli.
 *
 *****************************************************************************/

#ifndef CNNPLUS_MATVECLI_HH
#define CNNPLUS_MATVECLI_HH

#include "common.hh"

CNNPLUS_NS_BEGIN

//! A set of functions to compute common vector- and matrix-operations
namespace matvecli {

//! Reseeds the random number generator and return the seed used
unsigned int randreseed();

//! Allocates memory for a vector of length \a len
template<typename T>
T * allocv(size_t len);

//! Allocates memory for a matrix of size \a rows x \a cols
template<typename T>
T * allocm(size_t rows, size_t cols, size_t & stride);

//! Frees the allocated memory
template<typename T>
void free(T * ptr);

//! Returns a vector element
template<typename T>
T getelv(T const * src, size_t i);

//! Returns a matrix element
template<typename T>
T getelm(T const * src, size_t stride, size_t r, size_t c);

//! Sets a vector element
template<typename T>
void setelv(T * dst, size_t i, T val);

//! Sets a matrix element
template<typename T>
void setelm(T * dst, size_t stride, size_t r, size_t c, T val);

//! Initializes a vector to zero
template<typename T>
void zerov(T * dst, size_t len);

//! Initializes a matrix to zero
template<typename T>
void zerom(T * dst, size_t stride, size_t rows, size_t cols);

//! Initializes the vector elements to a specified common value
template<typename T>
void setv(T * dst, size_t len, T val);

//! Initializes the matrix elements to a specified common value
template<typename T>
void setm(T * dst, size_t stride, size_t rows, size_t cols, T val);

//! Initializes a vector with pseudo-random samples
template<typename T>
void randv(T * dst, size_t len, T sigma = 1);

//! Initializes a vector with pseudo-random samples
template<typename T>
void randv(T * dst, size_t len, T low, T high);

//! Initializes a matrix with pseudo-random samples
template<typename T>
void randm(T * dst, size_t stride, size_t rows, size_t cols, T sigma = 1);

//! Initializes a matrix with pseudo-random samples
template<typename T>
void randm(T * dst, size_t stride, size_t rows, size_t cols, T low, T high);

//! Adds the elements of two vectors
template<typename T>
void addv(T const * src1, T const * src2, T * dst, size_t len);

//! Adds the elements of two vectors (in-place)
template<typename T>
void addv(T * srcDst, T const * src, size_t len);

//! Adds a constant value to each element of a vector
template<typename T>
void addvc(T const * src, T val, T * dst, size_t len);

//! Adds a constant value to each element of a vector (in-place)
template<typename T>
void addvc(T * srcDst, T val, size_t len);

//! Subtracts the elements of two vectors
template<typename T>
void subv(T const * src1, T const * src2, T * dst, size_t len);

//! Subtracts the elements of two vectors (in-place)
template<typename T>
void subv(T * srcDst, T const * src, size_t len);

//! Subtracts a constant value from each element of a vector
template<typename T>
void subvc(T const * src, T val, T * dst, size_t len);

//! Subtracts a constant value from each element of a vector (in-place)
template<typename T>
void subvc(T * srcDst, T val, size_t len);

//! Subtracts each element of a vector from a constant value
template<typename T>
void subcv(T val, T const * src, T * dst, size_t len);

//! Subtracts each element of a vector from a constant value (in-place)
template<typename T>
void subcv(T val, T * srcDst, size_t len);

//! Multiplies the elements of two vectors
template<typename T>
void mulv(T const * src1, T const * src2, T * dst, size_t len);

//! Multiplies the elements of two vectors (in-place)
template<typename T>
void mulv(T * srcDst, T const * src, size_t len);

//! Multiplies each elements of a vector by a constant value
template<typename T>
void mulvc(T const * src, T val, T * dst, size_t len);

//! Multiplies each elements of a vector by a constant value (in-place)
template<typename T>
void mulvc(T * srcDst, T val, size_t len);

//! Divides the elements of two vectors
template<typename T>
void divv(T const * src1, T const * src2, T * dst, size_t len);

//! Divides the elements of two vectors (in-place)
template<typename T>
void divv(T * srcDst, T const * src, size_t len);

//! Divides each element of a vector by a constant value
template<typename T>
void divvc(T const * src, T val, T * dst, size_t len);

//! Divides each element of a vector by a constant value (in-place)
template<typename T>
void divvc(T * srcDst, T val, size_t len);

//! Divides each element of a matrix by a constant value
template<typename T>
void divmc(T const * src, size_t strideSrc, T val, T * dst, size_t strideDst, size_t rows, size_t cols);

//! Divides each element of a matrix by a constant value (in-place)
template<typename T>
void divmc(T * srcDst, size_t strideSrcDst, T val, size_t rows, size_t cols);

//! Computes hyperbolic tangent of each vector element
template<typename T>
void tanhv(T const * src, T * dst, size_t len);

//! Computes \e e to the power of each element of a vector
template<typename T>
void expv(T const * src, T * dst, size_t len);

//! Computes \e e to the power of each element of a vector (in-place)
template<typename T>
void expv(T * srcDst, size_t len);

//! Computes \e e to the power of each element of a matrix
template<typename T>
void expm(T const * src, size_t strideSrc, T * dst, size_t strideDst, size_t rows, size_t cols);

//! Computes \e e to the power of each element of a matrix (in-place)
template<typename T>
void expm(T * srcDst, size_t strideSrcDst, size_t rows, size_t cols);

//! Computes the natural logarithm of each element of a vector
template<typename T>
void lnv(T const * src, T * dst, size_t len);

//! Computes the natural logarithm of each element of a vector (in-place)
template<typename T>
void lnv(T * srcDst, size_t len);

//! Computes a square of each element of a vector
template<typename T>
void sqrv(T const * src, T * dst, size_t len);

//! Computes a square of each element of a vector (in-place)
template<typename T>
void sqrv(T * srcDst, size_t len);

//! Computes a square root of each element of a vector
template<typename T>
void sqrtv(T const * src, T * dst, size_t len);

//! Computes a square root of each element of a vector (in-place)
template<typename T>
void sqrtv(T * srcDst, size_t len);

//! Computes the sum of the elements of a vector
template<typename T>
T sumv(T const * src, size_t len);

//! Computes the sum of the elements of a matrix
template<typename T>
T summ(T const * src, size_t strideSrc, size_t rows, size_t cols);

//! Computes the dot product of two vectors
template<typename T>
T dotprod(T const * src1, T const * src2, size_t len);

//! Computes the sum of square values of a vector
template<typename T>
T sumsqrv(T const * src, size_t len);

//! Copies the contents of one vector into another
template<typename T>
void copyvv(T const * src, T * dst, size_t len);

//! Copies the contents of one matrix into another
template<typename T>
void copymm(T const * src, size_t strideSrc, T * dst, size_t strideDst, size_t rows, size_t cols);

//! Copies the contents of a matrix into a vector
template<typename T>
void copymv(T const * src, size_t strideSrc, T * dst, size_t rows, size_t cols);

//! Copies the contents of a vector into a matrix
template<typename T>
void copyvm(T const * src, T * dst, size_t strideDst, size_t rows, size_t cols);

//! Computes the L2 norm of a vector
template<typename T>
T nrm2(T const * src, size_t len);

//! Computes the L2 norm of two vectors' difference
template<typename T>
T nrm2(T const * src1, T const * src2, size_t len);

//! Computes the Frobenius norm of a matrix
template<typename T>
T nrmf(T const * src, size_t stride, size_t rows, size_t cols);

//! Returns the maximum absolute value of a vector
template<typename T>
T absmaxv(T const * src, size_t len);

//! Returns the minimum absolute value of a vector
template<typename T>
T absminv(T const * src, size_t len);

//! Returns the maximum value of a vector
template<typename T>
T maxv(T const * src, size_t len);

//! Returns the minimum value of a vector
template<typename T>
T minv(T const * src, size_t len);

//! Returns the index of the maximum element of a vector
template<typename T>
int maxidxv(T const * src, size_t len);

//! Returns the index of the minimum element of a vector
template<typename T>
int minidxv(T const * src, size_t len);

//! Calculates the sums of row vectors in a matrix
template<typename T>
void sumrow(T const * src, size_t stride, T * dst, size_t rows, size_t cols);

//! Calculates the sums of row vectors in a matrix and accumulates it to a vector
template<typename T>
void sumrowacc(T const * src, size_t stride, T * srcDst, size_t rows, size_t cols);

//! Calculates the sums of column vectors in a matrix
template<typename T>
void sumcol(T const * src, size_t stride, T * dst, size_t rows, size_t cols);

//! Calculates the sums of column vectors in a matrix and accumulates it to a vector
template<typename T>
void sumcolacc(T const * src, size_t stride, T * srcDst, size_t rows, size_t cols);

//! Multiplies each elements of a matrix by a constant value
template<typename T>
void mulmc(T const * src, size_t strideSrc, T val, T * dst, size_t strideDst, size_t rows, size_t cols);

//! Multiplies each elements of a matrix by a constant value (in-place)
template<typename T>
void mulmc(T * srcDst, size_t strideSrcDst, T val, size_t rows, size_t cols);

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

//! Computes a matrix-matrix product
template<typename T, char tSrc1, char tSrc2>
void mulmm(T const * src1, size_t strideSrc1,
           size_t rowsSrc1, size_t colsSrc1,
           T const * src2, size_t strideSrc2,
           size_t rowsSrc2, size_t colsSrc2,
           T * dst, size_t strideDst);

//! Computes a matrix-vector product
template<typename T, char tSrc1>
void mulmv(T const * src1, size_t strideSrc1,
           size_t rowsSrc1, size_t colsSrc1,
           T const * src2, size_t lenSrc2,
           T * dst);

//! Computes a scalar-matrix-matrix product and adds the result to a scalar-matrix product
template<typename T, char tSrc1, char tSrc2>
void gemm(T const * src1, size_t strideSrc1,
          size_t rowsSrc1, size_t colsSrc1,
          T const * src2, size_t strideSrc2,
          size_t rowsSrc2, size_t colsSrc2,
          T * srcDst, size_t strideSrcDst,
          T alpha = 1, T beta = 1);

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

//! Computes a vector-scalar product and adds the result to a vector
template<typename T>
void axpy(T const * src, size_t len, T * srcDst, T alpha = 1);

//! The \e logistic neural-net sigmoid function
template<typename T>
void logsigm(T const * src, size_t strideSrc, T * dst, size_t strideDst, size_t rows, size_t cols);

//! The \e logistic neural-net sigmoid function (in-place)
template<typename T>
void logsigm(T * srcDst, size_t strideSrcDst, size_t rows, size_t cols);

//! The derivative of #logsigm
template<typename T>
void dlogsigm(T const * src, size_t strideSrc, T * dst, size_t strideDst, size_t rows, size_t cols);

//! The derivative of #logsigm (in-place)
template<typename T>
void dlogsigm(T * srcDst, size_t strideSrcDst, size_t rows, size_t cols);

//! The \e standard neural-net sigmoid function
template<typename T>
void stdsigm(T const * src, size_t strideSrc, T * dst, size_t strideDst, size_t rows, size_t cols);

//! The \e standard neural-net sigmoid function (in-place)
template<typename T>
void stdsigm(T * srcDst, size_t strideSrcDst, size_t rows, size_t cols);

//! The derivative of #stdsigm
template<typename T>
void dstdsigm(T const * src, size_t strideSrc, T * dst, size_t strideDst, size_t rows, size_t cols);

//! The derivative of #stdsigm (in-place)
template<typename T>
void dstdsigm(T * srcDst, size_t strideSrcDst, size_t rows, size_t cols);

//! The hyperbolic tangent function
template<typename T>
void tanhm(T const * src, size_t strideSrc, T * dst, size_t strideDst, size_t rows, size_t cols);

//! The hyperbolic tangent function (in-place)
template<typename T>
void tanhm(T * srcDst, size_t strideSrcDst, size_t rows, size_t cols);

//! The derivative of #tanhm
template<typename T>
void dtanhm(T const * src, size_t strideSrc, T * dst, size_t strideDst, size_t rows, size_t cols);

//! The derivative of #tanhm (in-place)
template<typename T>
void dtanhm(T * srcDst, size_t strideSrcDst, size_t rows, size_t cols);

//! Computes the summed-area table (also known as integral image)
template<typename T>
void sat(T const * src, size_t strideSrc, T * dst, size_t strideDst, size_t rows, size_t cols);

//! Initializes the columns of a matrix
template<typename T>
void setcol(T * dst, size_t stride, size_t rows, size_t cols, T const * src);

}; // namespace matvecli

CNNPLUS_NS_END

#endif // CNNPLUS_MATVECLI_HH
