/**************************************************************************//**
 *
 * \file   timer.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Feb 02 2009
 *
 * $Id: timer.cc 1396 2009-06-04 06:33:11Z dast $
 *
 * \brief  Implementation of cnnplus::Timer.
 *
 *****************************************************************************/

#include "timer.hh"
#include <sstream>

#if defined(_MSC_VER)
#    include <windows.h>
#elif defined(__GNUC__)
#    include <sys/time.h>
#    include <time.h>
#endif

CNNPLUS_NS_BEGIN

#if defined(_MSC_VER)
//
// Implementation for Windows
//
class TimerImpl
{
public:
    TimerImpl() {
        CNNPLUS_VERIFY(QueryPerformanceFrequency(&frequency_));
        CNNPLUS_VERIFY(QueryPerformanceCounter(&start_));
    }
    void tic() {
        CNNPLUS_VERIFY(QueryPerformanceCounter(&start_));
    }
    double toc() {
        LARGE_INTEGER now = {0};
        CNNPLUS_VERIFY(QueryPerformanceCounter(&now));
        return static_cast<double>(now.QuadPart - start_.QuadPart)
            / static_cast<double>(frequency_.QuadPart);
    }
private:
    LARGE_INTEGER frequency_, start_;
};
#elif defined(__GNUC__)
//
// Implementation for Linux
//
class TimerImpl
{
public:
    TimerImpl() {
        gettimeofday(&start_, 0);
    }
    void tic() {
        gettimeofday(&start_, 0);
    }
    double toc() {
        struct timeval now = {0};
        gettimeofday(&now, 0);
        return ((now.tv_sec - start_.tv_sec) * 1000000.0
           + (now.tv_usec - start_.tv_usec)) / 1000000.0;
    }
private:
    struct timeval start_;
};
#endif

Timer::Timer()
    : pImpl_(new TimerImpl), elapsed_(0)
{}

Timer::~Timer()
{
    delete pImpl_;
}

void Timer::tic()
{
    pImpl_->tic();
}

double Timer::toc()
{
    return (elapsed_ = pImpl_->toc());
}

std::string Timer::report(char const * head)
{
    std::stringstream ss;
    ss << head << ": took " << toc() << " sec";
    return ss.str();
}

CNNPLUS_NS_END
