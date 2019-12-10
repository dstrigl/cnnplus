/**************************************************************************//**
 *
 * \file   cusquasher.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   May 25 2009
 *
 * $Id: cusquasher.hh 1429 2009-06-06 08:49:44Z dast $
 *
 * \brief  Neural-net squasher functions (also known as activation functions).
 *
 *****************************************************************************/

#ifndef CNNPLUS_CUSQUASHER_HH
#define CNNPLUS_CUSQUASHER_HH

#include "common.hh"
#include "squasher.hh"

CNNPLUS_NS_BEGIN

//! The \e logistic neural-net sigmoid squasher function
template<typename T>
class CuLogSigmoid : public Squasher<T>
{
public:
    //! Ctr
    explicit CuLogSigmoid(Size const & size) : Squasher<T>(size)
    {}
    virtual void fprop(T const * in, size_t strideIn,
                       T * out, size_t strideOut);
    virtual void bprop(T * in, size_t strideIn,
                       T const * out, size_t strideOut);
    virtual std::string toString() const { return "CuLogSigmoid"; }
};

//! The hyperbolic tangent sigmoid squasher function
template<typename T>
class CuTanh : public Squasher<T>
{
public:
    //! Ctr
    explicit CuTanh(Size const & size) : Squasher<T>(size)
    {}
    virtual void fprop(T const * in, size_t strideIn,
                       T * out, size_t strideOut);
    virtual void bprop(T * in, size_t strideIn,
                       T const * out, size_t strideOut);
    virtual std::string toString() const { return "CuTanh"; }
};

//! The \e standard neural-net sigmoid squasher function
template<typename T>
class CuStdSigmoid : public Squasher<T>
{
public:
    //! Ctr
    explicit CuStdSigmoid(Size const & size) : Squasher<T>(size)
    {}
    virtual void fprop(T const * in, size_t strideIn,
                       T * out, size_t strideOut);
    virtual void bprop(T * in, size_t strideIn,
                       T const * out, size_t strideOut);
    virtual std::string toString() const { return "CuStdSigmoid"; }
};

//! The \e identity activation function
template<typename T>
class CuIdentity : public Squasher<T>
{
public:
    //! Ctr
    explicit CuIdentity(Size const & size) : Squasher<T>(size)
    {}
    virtual void fprop(T const * in, size_t strideIn,
                       T * out, size_t strideOut);
    virtual void bprop(T * in, size_t strideIn,
                       T const * out, size_t strideOut);
    virtual std::string toString() const { return "CuIdentity"; }
};

CNNPLUS_NS_END

#endif // CNNPLUS_CUSQUASHER_HH
