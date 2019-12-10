/**************************************************************************//**
 *
 * \file   subsetdatasrc.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   May 03 2009
 *
 * $Id: subsetdatasrc.hh 1873 2009-07-24 19:31:35Z dast $
 *
 * \brief  Header for cnnplus::SubsetDataSrc.
 *
 *****************************************************************************/

#ifndef CNNPLUS_SUBSETDATASRC_HH
#define CNNPLUS_SUBSETDATASRC_HH

#include "common.hh"
#include "datasource.hh"
#include <vector>

CNNPLUS_NS_BEGIN

//! \todo doc
template<typename T>
class SubsetDataSrc : public DataSource<T>
{
public:
    //! Ctr
    SubsetDataSrc(DataSource<T> & ds, int start, int size);

    virtual int fprop(T * out);
    virtual int seek(int pos);
    virtual void shuffle();
    virtual void shift(int dist);
    virtual void reset();
    virtual size_t sizeOut() const;
    virtual Size sizePattern() const;
    virtual int idx(int pos = -1) const;
    virtual std::string toString() const;

    int start() const { return start_; }

private:
    DataSource<T> & ds_;
    int const start_;
    std::vector<int> idx_;
};

CNNPLUS_NS_END

#endif // CNNPLUS_SUBSETDATASRC_HH
