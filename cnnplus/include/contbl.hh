/**************************************************************************//**
 *
 * \file   contbl.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Feb 24 2009
 *
 * $Id: contbl.hh 1904 2009-07-30 19:29:56Z dast $
 *
 * \brief  Header for cnnplus::ConTbl.
 *
 *****************************************************************************/

#ifndef CNNPLUS_CONTBL_HH
#define CNNPLUS_CONTBL_HH

#include "common.hh"
#include <valarray>
#include <string>
#ifdef CNNPLUS_MATLAB_FOUND
#include <mat.h>
#endif // CNNPLUS_MATLAB_FOUND

CNNPLUS_NS_BEGIN

//! Connection table for convolutional layers
class ConTbl
{
    size_t rows_, cols_;
    std::valarray<bool> tbl_;

    bool & ref(size_t r, size_t c) { return tbl_[r * cols_ + c]; }
    bool ref(size_t r, size_t c) const { return tbl_[r * cols_ + c]; }

public:
    //! Ctr
    ConTbl(size_t rows, size_t cols, bool val = true);
    //! Ctr
    ConTbl(size_t rows, size_t cols, double p);
    //! Ctr
    ConTbl(size_t rows, size_t cols, bool const * tbl);
    //! Ctr
    ConTbl(size_t rows, size_t cols, int const * tbl);

    //! Returns the size of the connection table
    size_t size() const { return tbl_.size(); }
    //! Returns the number of rows of the connection table
    size_t rows() const { return rows_; }
    //! Returns the number of columns of the connection table
    size_t cols() const { return cols_; }
    //! Returns a reference to the cell at row \a r and column \a c
    bool & at(size_t r, size_t c) { return ref(r, c); }
    //! Returns a copy of the cell at row \a r and column \a c
    bool at(size_t r, size_t c) const { return ref(r, c); }
    //! Fills the connection table
    void fill(bool val = true) { tbl_ = val; }
    //! Clears the connection table
    void clear() { fill(false); }
    //! Fills the connection table with random connections
    void fillRandom(double p);
    //! Fills all defects in the connection table
    void fillDefects();
    //! Returns the number of connections
    size_t numConn() const;
    //! Returns the number of input-connections in row \a r
    size_t numInConn(size_t r) const;
    //! Returns the number of output-connections in column \a c
    size_t numOutConn(size_t c) const;
    //! Returns whether fully connected or not
    bool fullConnected() const { return numConn() == size(); }
    //! Returns the connection table as a string
    std::string toString() const;
#ifdef CNNPLUS_MATLAB_FOUND
    //! Loads the connection table
    void load(mxArray const * arr);
    //! Saves the connection table
    void save(mxArray * arr) const;
#endif // CNNPLUS_MATLAB_FOUND
};

CNNPLUS_NS_END

#endif // CNNPLUS_CONTBL_HH
