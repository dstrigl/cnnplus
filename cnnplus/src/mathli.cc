/**************************************************************************//**
 *
 * \file   mathli.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jan 07 2009
 *
 * $Id: mathli.cc 1396 2009-06-04 06:33:11Z dast $
 *
 * \brief  Implementation of cnnplus::mathli.
 *
 *****************************************************************************/

#include "mathli.hh"

CNNPLUS_NS_BEGIN

namespace mathli {

#if 0

/*! Implement the hyperbolic tangent using the Cody-Waite algoritm.
    \par
    The computation is divided into 4 regions:
    \li <tt>0       <= x <  XSMALL  </tt> tanh(x) = x
    \li <tt>XSMALL  <= x <  XMEDIUM </tt> tanh(x) = rational polynomial
    \li <tt>XMEDIUM <= x <  XLARGE  </tt> tanh(x) = 1 - 2/(1 + exp(2x))
    \li <tt>XLARGE  <= x <= infinity</tt> tanh(x) = 1
    \par
    In this implementation, <tt>t = 53</tt> and <tt>B = 2</tt>.
    \see http://www.math.utah.edu/~beebe/software/ieee/tanh.pdf
 */
double tanh(double const x)
{
    /* XLARGE = (ln(2) + (t + 1)ln(B))/2
     *        = smallest value for which tanh(x) = 1
     */
    double const XLARGE = 19.06154746539849600897e+00;

    /* XMED = ln(3)/2
     *      = cutoff for second approximation
     */
    double const XMED   =  0.54930614433405484570e+00;

    /* XSMALL = sqrt(3) * B^((-t-1)/2)
     *        = cutoff for third approximation
     */
    double const XSMALL =  1.29047841397589243466e-08;

    double const p[] = {
        -0.16134119023996228053e+04,
        -0.99225929672236083313e+02,
        -0.96437492777225469787e+00
    };

    double const q[] = {
         0.48402357071988688686e+04,
         0.22337720718962312926e+04,
         0.11274474380534949335e+03,
         1.00000000000000000000e+00,
    };

    double const f = abs(x);
    double result;

    if (f != f) {
        /* x is a NaN, so tanh is too--generate a run-time trap */
        result = f + f;
    }
    else if (f >= XLARGE) {
        result = 1.0;
    }
    else if (f >= XMED) {
        result = 0.5 - 1.0 / (1.0 + exp(f + f));
        result += result;
    }
    else if (f >= XSMALL) {
        double const g = f * f;
        double const r = g * ((p[2] * g + p[1]) * g + p[0]) /
                        (((g + q[2])* g + q[1]) * g + q[0]);
        result = f + f * r;
    }
    else {
        result = f;
    }

    if (x < 0.0) result = -result;

    return result;
}

/*! Implement the hyperbolic tangent using the Cody-Waite algoritm.
    \par
    The computation is divided into 4 regions:
    \li <tt>0       <= x <  XSMALL  </tt> tanh(x) = x
    \li <tt>XSMALL  <= x <  XMEDIUM </tt> tanh(x) = rational polynomial
    \li <tt>XMEDIUM <= x <  XLARGE  </tt> tanh(x) = 1 - 2/(1 + exp(2x))
    \li <tt>XLARGE  <= x <= infinity</tt> tanh(x) = 1
    \par
    In this implementation, <tt>t = 53</tt> and <tt>B = 2</tt>.
    \see http://www.math.utah.edu/~beebe/software/ieee/tanh.pdf
 */
float tanh(float const x)
{
    /* XLARGE = (ln(2) + (t + 1)ln(B))/2
     *        = smallest value for which tanh(x) = 1
     */
    float const XLARGE = 8.66433975699931636772e+00f;

    /* XMED = ln(3)/2
     *      = cutoff for second approximation
     */
    float const XMED   = 0.54930614433405484570e+00f;

    /* XSMALL = sqrt(3) * B^((-t-1)/2)
     *        = cutoff for third approximation
     */
    float const XSMALL = 4.22863966691620432990e-04f;

    float const p[] = {
        -0.8237728127e+00f,
        -0.3831010665e-02f
    };

    float const q[] = {
         0.2471319654e+01f,
         1.0000000000e+00f
    };

    float const f = abs(x);
    float result;

    if (f != f) {
        /* x is a NaN, so tanh is too--generate a run-time trap */
        result = f + f;
    }
    else if (f >= XLARGE) {
        result = 1.0f;
    }
    else if (f >= XMED) {
        result = 0.5f - 1.0f / (1.0f + exp(f + f));
        result += result;
    }
    else if (f >= XSMALL) {
        float const g = f * f;
        float const r = g * (p[1] * g + p[0]) / (g + q[0]);
        result = f + f * r;
    }
    else {
        result = f;
    }

    if (x < 0.0f) result = -result;

    return result;
}

#endif

}; // namespace mathli

CNNPLUS_NS_END
