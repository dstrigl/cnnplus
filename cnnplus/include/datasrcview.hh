/**************************************************************************//**
 *
 * \file   datasrcview.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Apr 25 2009
 *
 * $Id: datasrcview.hh 1396 2009-06-04 06:33:11Z dast $
 *
 * \brief  Header for cnnplus::DataSrcView.
 *
 *****************************************************************************/

#ifndef CNNPLUS_DATASRCVIEW_HH
#define CNNPLUS_DATASRCVIEW_HH

#include "common.hh"

CNNPLUS_NS_BEGIN

template<typename T> class DataSource;

//! \todo doc
class DataSrcView
{
public:
    //! Ctr
    DataSrcView(double const zoomFactor = 1, unsigned int delay = 0)
        : zoomFactor_(zoomFactor), delay_(delay)
    {}
    //! \todo doc
    template<typename T>
    void run(DataSource<T> & ds);
    //! \todo doc
    template<typename T>
    void run(DataSource<T> & dsA, DataSource<T> & dsB);

private:
    double const zoomFactor_;
    unsigned int const delay_;
};

CNNPLUS_NS_END

#endif // CNNPLUS_DATASRCVIEW_HH
