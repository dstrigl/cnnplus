/**************************************************************************//**
 *
 * \file   types.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Dec 13 2008
 *
 * $Id: types.hh 1396 2009-06-04 06:33:11Z dast $
 *
 * \brief  Some common types.
 *
 *****************************************************************************/

#ifndef CNNPLUS_TYPES_HH
#define CNNPLUS_TYPES_HH

#include "common.hh"
#include <string>
#include <sstream>

CNNPLUS_NS_BEGIN

//! Specifies the width and height of a rectangle
class Size
{
public:
    //! Default Ctr
    Size() : height_(0), width_(0)
    {}
    //! Ctr
    Size(size_t const height, size_t const width)
        : height_(height), width_(width)
    {}
    //! Cpy-Ctr
    Size(Size const & other)
        : height_(other.height_), width_(other.width_)
    {}
    //! Assignment
    Size const & operator=(Size const & rhs) {
        if (this != &rhs) {
            height_ = rhs.height_;
            width_  = rhs.width_;
        }
        return *this;
    }
    //! Adds \a rhs
    Size const & operator+=(Size const & rhs) {
        height_ += rhs.height_;
        width_  += rhs.width_;
        return *this;
    }
    //! Subtracts \a rhs
    Size const & operator-=(Size const & rhs) {
        height_ -= rhs.height_;
        width_  -= rhs.width_;
        return *this;
    }
    //! Scales by scalar \a v
    Size const & operator*=(size_t const v) {
        height_ *= v;
        width_  *= v;
        return *this;
    }
    //! Divide by scalar \a v
    Size const & operator/=(size_t const v) {
        height_ /= v;
        width_  /= v;
        return *this;
    }
    //! Divide by scalar \a v
    Size operator/(size_t const v) const {
        Size tmp(*this);
        tmp /= v;
        return tmp;
    }
    //! Sets the height and the width of the rectangle to zero
    void clear() { height_ = width_ = 0; }
    //! Sets the height and the width of the rectangle
    void set(size_t const height, size_t const width) {
        height_ = height;
        width_ = width;
    }
    //! Returns the height of the rectangle
    size_t height() const { return height_; }
    //! Sets the height of the rectangle
    void setHeight(size_t const height) { height_ = height; }
    //! Returns the width of the rectangle
    size_t width() const { return width_; }
    //! Sets the width of the rectangle
    void setWidth(size_t const width) { width_ = width; }
    //! Returns the area of the rectangle
    size_t area() const { return height_ * width_; }
    //! Returns a string representing the rectangle
    std::string toString() const {
        std::stringstream ss;
        ss << "(" << height_ << "," << width_ << ")";
        return ss.str();
    }
private:
    size_t height_, width_;
};

//! Addition
inline Size const
operator+(Size const & lhs, Size const & rhs) {
    Size r(lhs);
    return r += rhs;
}

//! Subtraction
inline Size const
operator-(Size const & lhs, Size const & rhs) {
    Size r(lhs);
    return r -= rhs;
}

//! Scaling
inline Size const
operator*(size_t const v, Size const & s) {
    Size r(s);
    return r *= v;
}

//! Comparison of equality
inline bool
operator==(Size const & lhs, Size const & rhs) {
    return lhs.height() == rhs.height() && lhs.width() == rhs.width();
}

//! Comparison of inequality
inline bool
operator!=(Size const & lhs, Size const & rhs) {
    return !(lhs == rhs);
}

CNNPLUS_NS_END

#endif // CNNPLUS_TYPES_HH
