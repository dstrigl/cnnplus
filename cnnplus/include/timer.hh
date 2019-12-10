/**************************************************************************//**
 *
 * \file   timer.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Dec 06 2008
 *
 * $Id: timer.hh 1396 2009-06-04 06:33:11Z dast $
 *
 * \brief  Header for cnnplus::Timer.
 *
 *****************************************************************************/

#ifndef CNNPLUS_TIMER_HH
#define CNNPLUS_TIMER_HH

#include "common.hh"
#include <string>

CNNPLUS_NS_BEGIN

class TimerImpl;

//! Stopwatch timer for performance measurements
/*!
\par Usage:
\code
Timer tm;
foo1();
double const et = tm.toc();
cout << "foo1: took " << et << " sec" << endl;

tm.tic();
foo2();
cout << tm.report("foo2") << endl;
\endcode
*/
class Timer
{
public:
    //! Ctr
    Timer();
    //! Dtr
    ~Timer();
    //! Starts the timer
    void tic();
    //! Stops the timer and returns the elapsed time (in sec)
    double toc();
    //! Returns a string that reports an elapsed time
    std::string report(char const * head);
    //! Returns the elapsed time (in sec)
    double elapsed() const { return elapsed_; }
private:
    //! Cpy-Ctr, disabled
    Timer(Timer const & rhs);
    //! Assignment, disabled
    Timer & operator=(Timer const & rhs);
    //! Pointer to implementation
    TimerImpl * pImpl_;
    //! Elapsed time (in sec)
    double elapsed_;
};

CNNPLUS_NS_END

#endif // CNNPLUS_TIMER_HH
