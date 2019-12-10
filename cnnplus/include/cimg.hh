/**************************************************************************//**
 *
 * \file   cimg.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Feb 03 2009
 *
 * $Id: cimg.hh 1396 2009-06-04 06:33:11Z dast $
 *
 * \brief  Wrapper for "The C++ Template Image Processing Library".
 *
 *****************************************************************************/

#ifndef CNNPLUS_CIMG_HH
#define CNNPLUS_CIMG_HH

#ifdef __INTEL_COMPILER
// warning #1684: conversion from pointer to same-sized integral type
//   (potential portability problem)
#    pragma warning(disable: 1684)
#endif // __INTEL_COMPILER

#ifdef WIN32
#    define _WIN32_WINNT 0x0400
#    define NOMINMAX // disable the "ugly" Windows macros MIN and MAX
#endif // WIN32

#include "CImg.h"

#ifdef __INTEL_COMPILER
#    pragma warning(default: 1684)
#endif // __INTEL_COMPILER

using namespace cimg_library;

#endif // CNNPLUS_CIMG_HH