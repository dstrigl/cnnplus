/**************************************************************************//**
 *
 * \file   common.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Dec 06 2008
 *
 * $Id: common.hh 1492 2009-06-15 09:24:02Z dast $
 *
 * \brief  Common helper functions and macros.
 *
 *****************************************************************************/

#ifndef CNNPLUS_COMMON_HH
#define CNNPLUS_COMMON_HH

#include <cstdlib> // need for 'size_t'
#include <cassert>

#if defined(_MSC_VER)
//! Returns the undecorated name of the enclosing function
#define __func__ __FUNCTION__
#endif

//! \name Defines for the cnnplus namespace
/// @{
#define CNNPLUS_NS_BEGIN namespace cnnplus {
#define CNNPLUS_NS_END }
/// @}

//! \name Defines for a anonymous namespace
/// @{
#define CNNPLUS_ANON_NS_BEGIN namespace {
#define CNNPLUS_ANON_NS_END }
/// @}

CNNPLUS_NS_BEGIN

//! \name Classical ASSERT and VERIFY macros
/// @{
#ifdef NDEBUG
#    define CNNPLUS_ASSERT(A)
#    define CNNPLUS_VERIFY(A) A
#else
#    define CNNPLUS_ASSERT(A) assert(A)
#    define CNNPLUS_VERIFY(A) assert(A)
#endif
/// @}

template<bool> struct CompileTimeError;
//! Helper struct for #CNNPLUS_STATIC_ASSERT
template<> struct CompileTimeError<true> {};

//! Static ASSERT macro (compile time check)
/*!
\par Invocation:
\code
CNNPLUS_STATIC_ASSERT(expr, id)
\endcode
\par where:
\li \a expr is a compile-time integral or pointer expression.
\li \a id is a C++ identifier that does not need to be defined.
If \a expr is zero, \a id will appear in a compile-time error message.
\sa http://loki-lib.sourceforge.net/
*/
#define CNNPLUS_STATIC_ASSERT(expr, msg) \
    { cnnplus::CompileTimeError<((expr) != 0)> ERROR_##msg; (void)ERROR_##msg; }

/*! \def countof(arr)
    Returns the number of elements in array \a arr.
 */
template <typename T, size_t N>
char ( &_ArraySizeHelper( T (&arr)[N] ))[N];
#define countof(arr) (sizeof(_ArraySizeHelper(arr)))

CNNPLUS_NS_END

#endif // CNNPLUS_COMMON_HH
