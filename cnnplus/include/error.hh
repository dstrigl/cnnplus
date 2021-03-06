/**************************************************************************//**
 *
 * \file   error.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Dec 06 2008
 *
 * $Id: error.hh 1797 2009-07-18 12:28:02Z dast $
 *
 * \brief  Exception classes for cnnplus-errors.
 *
 *****************************************************************************/

#ifndef CNNPLUS_ERROR_HH
#define CNNPLUS_ERROR_HH

#include "common.hh"
#include <stdexcept>
#include <string>
#include <sstream>

CNNPLUS_NS_BEGIN

//! Basic error class
class CnnPlusError : public std::exception
{
public:
    //! Ctr with error message
    explicit CnnPlusError(std::string const & msg)
        : msg_(msg)
    {}

    //! Dtr
    virtual ~CnnPlusError() throw() {}

    //! Returns a C-style character string with the error message
    virtual const char * what() const throw() {
        return msg_.c_str();
    }

protected:
    //! String with error message
    std::string msg_;
};

//! Functionality not implemented
class NotImplementedError : public CnnPlusError
{
public:
    //! Ctr with error message
    explicit NotImplementedError(std::string const & msg)
        : CnnPlusError(msg)
    {}

    //! Dtr
    virtual ~NotImplementedError() throw() {}
};

//! Invalid parameter in function call
/*!
\par Example:
\code
void foo(int aParam)
{
    if (aParam < 0)
        throw ParameterError("aParam", "must be greater or equal zero.");
    ...
}
\endcode
*/
class ParameterError : public CnnPlusError
{
public:
    //! Ctr with error message
    ParameterError(std::string const & msg) : CnnPlusError(msg) {}

    //! Ctr with parameter name and error message
    /*! \param par parameter name
        \param msg error message
     */
    ParameterError(std::string const & par, std::string const & msg)
        : CnnPlusError("")
    {
        std::stringstream ss;
        ss << "Invalid parameter '" << par << "'";
        if (msg.empty()) ss << ".";
        else ss << ": " << msg;
        msg_ = ss.str();
    }

    //! Dtr
    virtual ~ParameterError() throw() {}
};

//! File IO error
class IoError : public CnnPlusError
{
public:
    //! Ctr with error message
    explicit IoError(std::string const & msg)
        : CnnPlusError(msg)
    {}

    //! Dtr
    virtual ~IoError() throw() {}
};

//! Data source error
class DataSourceError : public CnnPlusError
{
public:
    //! Ctr with error message
    explicit DataSourceError(std::string const & msg)
        : CnnPlusError(msg)
    {}

    //! Dtr
    virtual ~DataSourceError() throw() {}
};

//! Matlab error
class MatlabError : public CnnPlusError
{
public:
    //! Ctr with error message
    explicit MatlabError(std::string const & msg)
        : CnnPlusError(msg)
    {}

    //! Dtr
    virtual ~MatlabError() throw() {}
};

CNNPLUS_NS_END

#endif // CNNPLUS_ERROR_HH
