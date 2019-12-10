/**************************************************************************//**
 *
 * \file   mathli.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Dec 07 2008
 *
 * $Id: mathli.hh 1771 2009-07-15 21:16:17Z dast $
 *
 * \brief  Header for cnnplus::mathli.
 *
 *****************************************************************************/

#ifndef CNNPLUS_MATHLI_HH
#define CNNPLUS_MATHLI_HH

#include "common.hh"
#include <cmath>
#include <algorithm> // need for 'std::min' and 'std::max'
#include <limits>

CNNPLUS_NS_BEGIN

//! A set of functions to compute common mathematical operations
namespace mathli {

//! \todo doc
template<typename T>
T eps()
{
    return std::numeric_limits<T>::epsilon();
}

//! Returns a value indicating the sign of a number
template<typename T>
int sign(T const x)
{
    return (x < 0) ? -1 : ((x > 0) ? 1 : 0);
}

//! Returns the lesser of \a a and \a b
template<typename T>
T const & min(T const & a, T const & b)
{
    return std::min<T>(a, b);
}

//! Returns the greater of \a a and \a b
template<typename T>
T const & max(T const & a, T const & b)
{
    return std::max<T>(a, b);
}

//! Returns the absolute value of \a x (double precision)
inline double abs(double const x)
{
    return std::abs(x);
}

//! Returns the absolute value of \a x (single precision)
inline float abs(float const x)
{
    return std::abs(x);
}

//! Returns the square root of \a x (double precision)
inline double sqrt(double const x)
{
    return std::sqrt(x);
}

//! Returns the square root of \a x (single precision)
inline float sqrt(float const x)
{
    return std::sqrt(x);
}

//! Returns the exponential value of \a x (double precision)
inline double exp(double const x)
{
    return std::exp(x);
}

//! Returns the exponential value of \a x (single precision)
inline float exp(float const x)
{
    return std::exp(x);
}

//! Returns the natural logarithm value of \a x (double precision)
inline double log(double const x)
{
    return std::log(x);
}

//! Returns the natural logarithm value of \a x (single precision)
inline float log(float const x)
{
    return std::log(x);
}

//! Computes \a x raised to the power of \a y (double precision)
inline double pow(double const x, int const y)
{
    return std::pow(x, y);
}

//! Computes \a x raised to the power of \a y (single precision)
inline float pow(float const x, int const y)
{
    return std::pow(x, y);
}

//! Computes \a x raised to the power of \a y (double precision)
inline double pow(double const x, double const y)
{
    return std::pow(x, y);
}

//! Computes \a x raised to the power of \a y (single precision)
inline float pow(float const x, float const y)
{
    return std::pow(x, y);
}

//! The hyperbolic tangent function (double precision)
inline double tanh(double const x)
{
    return std::tanh(x);
}

//! The hyperbolic tangent function (single precision)
inline float tanh(float const x)
{
    return std::tanh(x);
}

//! The derivative of #tanh (double precision)
inline double dtanh(double const x)
{
    return (1.0 - pow(tanh(x), 2));
}

//! The derivative of #tanh (single precision)
inline float dtanh(float const x)
{
    return (1.0f - pow(tanh(x), 2));
}

//! The \e standard neural-net sigmoid function (double precision)
/*! Recommended by Yann LeCun in "Efficient BackProp" in Neural Networks:
    Tricks of the Trade, 1998, 1.4.4 The Sigmoid.
    \see http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
 */
inline double stdsigmoid(double const x)
{
    //  f(x) = 1.7159 * tanh(2/3 * x)
    return 1.7159 * tanh(0.66666666666666663 * x);
}

//! The \e standard neural-net sigmoid function (single precision)
/*! Recommended by Yann LeCun in "Efficient BackProp" in Neural Networks:
    Tricks of the Trade, 1998, 1.4.4 The Sigmoid.
    \see http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
 */
inline float stdsigmoid(float const x)
{
    //  f(x) = 1.7159 * tanh(2/3 * x)
    return 1.7159f * tanh(0.66666669f * x);
}

//! The derivative of #stdsigmoid (double precision)
inline double dstdsigmoid(double const x)
{
    //  f'(x) = 1.7159 * 2/3 * (1 - tanh(2/3 * x)^2)
    return 1.1439333333333332 - 1.1439333333333332
        * pow(tanh(0.66666666666666663 * x), 2);
}

//! The derivative of #stdsigmoid (single precision)
inline float dstdsigmoid(float const x)
{
    //  f'(x) = 1.7159 * 2/3 * (1 - tanh(2/3 * x)^2)
    return 1.1439333f - 1.1439333f * pow(tanh(0.66666669f * x), 2);
}

//! The \e logistic neural-net sigmoid function (double precision)
inline double logsigmoid(double const x)
{
    return 1.0 / (1.0 + exp(-x));
}

//! The \e logistic neural-net sigmoid function (single precision)
inline float logsigmoid(float const x)
{
    return 1.0f / (1.0f + exp(-x));
}

//! The derivative of #logsigmoid (double precision)
inline double dlogsigmoid(double const x)
{
    return logsigmoid(x) * (1.0 - logsigmoid(x));
}

//! The derivative of #logsigmoid (single precision)
inline float dlogsigmoid(float const x)
{
    return logsigmoid(x) * (1.0f - logsigmoid(x));
}

//! Round \a a divided by \a b to the nearest higher integer value
template<typename T>
T divup(T const a, T const b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//! Round \a a to the nearest multiple of \a b
template<typename T>
T roundup(T const a, T const b)
{
    return divup<T>(a, b) * b;
}

}; // namespace mathli

CNNPLUS_NS_END

#endif // CNNPLUS_MATHLI_HH
