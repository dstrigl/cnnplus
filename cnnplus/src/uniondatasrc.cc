/**************************************************************************//**
 *
 * \file   uniondatasrc.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jun 11 2009
 *
 * $Id: uniondatasrc.cc 1827 2009-07-20 18:17:58Z dast $
 *
 * \brief  Implementation of cnnplus::UnionDataSrc.
 *
 *****************************************************************************/

#include "uniondatasrc.hh"
#include "error.hh"
#include <algorithm>
#include <sstream>

CNNPLUS_NS_BEGIN

template<typename T>
UnionDataSrc<T>::UnionDataSrc(DataSource<T> & ds)
    : DataSource<T>(ds.dataRangeMin(), ds.dataRangeMax(), ds.size(), 0),
    ds_(), idx_()
{
    ds_.push_back(&ds);
    reset();
}

template<typename T> UnionDataSrc<T> &
UnionDataSrc<T>::add(DataSource<T> & ds)
{
    // Check if data source is compatible
    if (ds.sizeOut() != sizeOut()) {
        throw ParameterError("ds", "output size doesn't match.");
    }
    else if (ds.dataRangeMin() != this->dataRangeMin() ||
             ds.dataRangeMax() != this->dataRangeMax()) {
        throw ParameterError("ds", "data range doesn't match.");
    }

    ds_.push_back(&ds);
    this->size_ += ds.size();
    reset();
    return *this;
}

template<typename T> int
UnionDataSrc<T>::fprop(T * out)
{
    CNNPLUS_ASSERT(!ds_.empty());
    int pos = idx_[this->curPos_], i = 0;
    for (; pos >= ds_[i]->size(); ++i)
        pos -= ds_[i]->size();
    CNNPLUS_ASSERT(i < static_cast<int>(ds_.size()));
    CNNPLUS_ASSERT(pos >= 0 && pos < ds_[i]->size());
    ds_[i]->seek(pos);
    return ds_[i]->fprop(out);
}

template<typename T> int
UnionDataSrc<T>::seek(int pos)
{
    if (pos < 0 || pos >= this->size_) {
        std::stringstream ss;
        ss << "must be between 0 and " << (this->size_ - 1) << ".";
        throw ParameterError("pos", ss.str());
    }

    int const oldPos = this->curPos_;
    this->curPos_ = pos;
    return oldPos;
}

template<typename T> void
UnionDataSrc<T>::shuffle()
{
    std::random_shuffle(idx_.begin(), idx_.end());
}

template<typename T> void
UnionDataSrc<T>::shift(int dist)
{
    if (this->size_ > 0) {
        while (dist < 0) dist += this->size_;
        for (int i = 0; i < this->size_; ++i)
            idx_[i] = (idx_[i] + dist) % this->size_;
    }
}

template<typename T> void
UnionDataSrc<T>::reset()
{
    if (this->size_ > 0) {
        idx_.resize(this->size_);
        for (int i = 0; i < this->size_; ++i)
            idx_[i] = i;
    }
}

template<typename T> size_t
UnionDataSrc<T>::sizeOut() const
{
    CNNPLUS_ASSERT(ds_.front());
    return ds_.front()->sizeOut();
}

template<typename T> Size
UnionDataSrc<T>::sizePattern() const
{
    CNNPLUS_ASSERT(ds_.front());
    return ds_.front()->sizePattern();
}

template<typename T> int
UnionDataSrc<T>::idx(int pos) const
{
    pos = (pos < 0) ? this->curPos_ : pos;

    if (pos < 0 || pos >= this->size_) {
        std::stringstream ss;
        ss << "must be between 0 and " << (this->size_ - 1) << ".";
        throw ParameterError("pos", ss.str());
    }

    int i = 0;
    for (; pos >= ds_[i]->size(); ++i)
        pos -= ds_[i]->size();
    return ds_[i]->idx(pos);
}

template<typename T> std::string
UnionDataSrc<T>::toString() const
{
    std::stringstream ss;
    ss << "UnionDataSrc[{";
    for (size_t i = 0; i < ds_.size(); ++i) {
        CNNPLUS_ASSERT(ds_[i]);
        ss << ds_[i]->toString();
        ss << (i < ds_.size() - 1 ? "," : "}");
    }
    ss << "; size=" << this->size_ << "]";
    return ss.str();
}

/*! \addtogroup eti_grp Explicit Template Instantiation
 @{
 */
template class UnionDataSrc<float>;
template class UnionDataSrc<double>;
/*! @} */

CNNPLUS_NS_END
