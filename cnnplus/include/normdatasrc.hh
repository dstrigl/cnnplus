/**************************************************************************//**
 *
 * \file   normdatasrc.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Mar 03 2009
 *
 * $Id: normdatasrc.hh 1827 2009-07-20 18:17:58Z dast $
 *
 * \brief  Header for cnnplus::NormDataSrc.
 *
 *****************************************************************************/

#ifndef CNNPLUS_NORMDATASRC_HH
#define CNNPLUS_NORMDATASRC_HH

#include "common.hh"
#include "datasource.hh"

CNNPLUS_NS_BEGIN

//! A normalized data source
/*! \see
    \li Y. LeCun, L. Bottou, G. Orr, and K. Muller. "Efficient BackProp", in
        Neural Networks: Tricks of the trade, G. Orr and Muller K., eds.,
        Springer, 1998.
    \li Alexander Graves. Supervised Sequence Labelling with Recurrent Neural
        Networks. Dissertation, Technische Universit�t M�nchen, M�nchen,
        July 2008.
 */
template<typename T>
class NormDataSrc : public DataSource<T>
{
public:
    //! Ctr
    NormDataSrc(DataSource<T> & ds);
    //! Ctr
    /*! \param mean mean value
        \param stdev standard deviation value
     */
    NormDataSrc(DataSource<T> & ds, T const mean, T const stdev)
        : DataSource<T>(ds.dataRangeMin(), ds.dataRangeMax()),
        ds_(ds), mean_(mean), stdev_(stdev)
    {}
    //! Ctr
    /*! \param nds normalized data source
     */
    NormDataSrc(DataSource<T> & ds, NormDataSrc<T> const & nds)
        : DataSource<T>(ds.dataRangeMin(), ds.dataRangeMax()),
        ds_(ds), mean_(nds.mean_), stdev_(nds.stdev_)
    {}

    virtual int fprop(T * out);
    virtual int seek(int pos) { return ds_.seek(pos); }
    virtual void shuffle() { return ds_.shuffle(); }
    virtual void shift(int dist) { return ds_.shift(dist); }
    virtual void reset() { return ds_.reset(); }
    virtual size_t sizeOut() const { return ds_.sizeOut(); }
    virtual Size sizePattern() const { return ds_.sizePattern(); }
    virtual int idx(int pos = -1) const { return ds_.idx(pos); }
    virtual T dataRangeMin() const { return (this->dataRangeMin_ - mean_) / stdev_; }
    virtual T dataRangeMax() const { return (this->dataRangeMax_ - mean_) / stdev_; }
    virtual int size() const { return ds_.size(); }
    virtual int tell() const { return ds_.tell(); }
    virtual void rewind() { ds_.rewind(); }
    virtual bool next() { return ds_.next(); }
    virtual std::string toString() const;

    //! Returns the mean value
    T mean() const { return mean_; }
    //! Returns the standard deviation value
    T stdev() const { return stdev_; }
    //! Computes the covariance of the data source
    T covariance(bool normalized = true);

private:
    DataSource<T> & ds_;
    T mean_;
    T stdev_;
};

CNNPLUS_NS_END

#endif // CNNPLUS_NORMDATASRC_HH
