/**************************************************************************//**
 *
 * \file   datasource.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Dec 15 2008
 *
 * $Id: datasource.hh 1827 2009-07-20 18:17:58Z dast $
 *
 * \brief  Header for cnnplus::DataSource.
 *
 *****************************************************************************/

#ifndef CNNPLUS_DATASOURCE_HH
#define CNNPLUS_DATASOURCE_HH

#include "common.hh"
#include "types.hh"
#include "error.hh"
#include <string>

CNNPLUS_NS_BEGIN

//! Specifies a data source for a neural network
template<typename T>
class DataSource
{
public:
    //! Ctr
    DataSource(T dataRangeMin, T dataRangeMax)
        : dataRangeMin_(dataRangeMin), dataRangeMax_(dataRangeMax),
        size_(-1), curPos_(-1) {
        if (dataRangeMin >= dataRangeMax)
            throw ParameterError("'dataRangeMin' must be lower than 'dataRangeMax'.");
    }
    //! Ctr
    DataSource(T dataRangeMin, T dataRangeMax, int size, int curPos)
        : dataRangeMin_(dataRangeMin), dataRangeMax_(dataRangeMax),
        size_(size), curPos_(curPos) {
        if (dataRangeMin >= dataRangeMax)
            throw ParameterError("'dataRangeMin' must be lower than 'dataRangeMax'.");
    }
    //! Dtr
    virtual ~DataSource() {}
    //! Returns the current pattern and corresponding label
    virtual int fprop(T * out) = 0;
    //! Moves to pattern at position \a pos
    virtual int seek(int pos) = 0;
    //! Shuffles the order of patterns
    virtual void shuffle() = 0;
    //! Shift the order of patterns by \a dist
    virtual void shift(int dist) = 0;
    //! Resets to the original order of patterns
    virtual void reset() = 0;
    //! Returns the output size
    virtual size_t sizeOut() const = 0;
    //! Returns the pattern size
    virtual Size sizePattern() const = 0;
    //! Returns the true index of a position
    virtual int idx(int pos = -1) const = 0;
    //! Returns the minimum of the data range
    virtual T dataRangeMin() const { return dataRangeMin_; }
    //! Returns the maximum of the data range
    virtual T dataRangeMax() const { return dataRangeMax_; }
    //! Returns the number of patterns
    virtual int size() const { return size_; }
    //! Returns the current position
    virtual int tell() const { return curPos_; }
    //! Moves to the first pattern
    virtual void rewind() { seek(0); }
    //! Moves to the next pattern
    /*! \return \a false if there are no more patterns, otherwise \a true
     */
    virtual bool next() {
        if (curPos_ < size() - 1) {
            ++curPos_;
            return true;
        }
        return false;
    }
    //! Returns a string that describes the data source
    virtual std::string toString() const = 0;

private:
    //! Cpy-Ctr, disabled
    DataSource(DataSource const & rhs);
    //! Assignment, disabled
    DataSource & operator=(DataSource const & rhs);

protected:
    T const dataRangeMin_;
    T const dataRangeMax_;
    int size_, curPos_;
};

CNNPLUS_NS_END

#endif // CNNPLUS_DATASOURCE_HH
