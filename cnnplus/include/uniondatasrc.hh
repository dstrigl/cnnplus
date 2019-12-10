/**************************************************************************//**
 *
 * \file   uniondatasrc.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jun 11 2009
 *
 * $Id: uniondatasrc.hh 1827 2009-07-20 18:17:58Z dast $
 *
 * \brief  Header for cnnplus::UnionDataSrc.
 *
 *****************************************************************************/

#ifndef CNNPLUS_UNIONDATASRC_HH
#define CNNPLUS_UNIONDATASRC_HH

#include "common.hh"
#include "datasource.hh"
#include <vector>

CNNPLUS_NS_BEGIN

//! \todo doc
template<typename T>
class UnionDataSrc : public DataSource<T>
{
public:
    //! Ctr
    UnionDataSrc(DataSource<T> & ds);

    virtual int fprop(T * out);
    virtual int seek(int pos);
    virtual void shuffle();
    virtual void shift(int dist);
    virtual void reset();
    virtual size_t sizeOut() const;
    virtual Size sizePattern() const;
    virtual int idx(int pos = -1) const;
    virtual std::string toString() const;

    //! Adds a data source
    UnionDataSrc & add(DataSource<T> & ds);

private:
    std::vector<DataSource<T>*> ds_;
    std::vector<int> idx_;
};

CNNPLUS_NS_END

#endif // CNNPLUS_UNIONDATASRC_HH
