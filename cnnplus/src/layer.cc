/**************************************************************************//**
 *
 * \file   layer.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Dec 15 2008
 *
 * $Id: layer.cc 2161 2009-09-09 11:49:16Z dast $
 *
 * \brief  Implementation of cnnplus::Layer.
 *
 *****************************************************************************/

#include "layer.hh"

CNNPLUS_NS_BEGIN

#ifdef CNNPLUS_MATLAB_FOUND
/*! \par Usage:
\code
mxArray * arr = ...
mwSize const dims[] = { 5, 5, 6, 50 };
if (!validateArr(arr, countof(dims), dims)) {
    ...
}
\endcode
 */
template<typename T> bool
Layer<T>::checkArr(mxArray const * arr, mwSize const ndims, mwSize const * dims) const
{
    if (!arr) return false;
    if (!mxIsDouble(arr)) return false;
    if (mxIsComplex(arr)) return false;
    if (mxGetNumberOfDimensions(arr) > ndims) return false;
    for (mwSize i = 0; i < ndims; ++i) {
        if (i < mxGetNumberOfDimensions(arr)) {
            if (mxGetDimensions(arr)[i] != dims[i])
                return false;
        }
        else {
            if (dims[i] != 1)
                return false;
        }
    }
    return true;
}

template<typename T> bool
Layer<T>::checkType(mxArray const * arr, std::string const & type) const
{
    if (!arr) return false;
    mxArray const * arrType = mxGetField(arr, 0, "type");
    if (!arrType) return false;
    char * pszType = mxArrayToString(arrType);
    if (!pszType) return false;
    std::string const strType(pszType);
    mxFree(pszType);
    return (strType == type);
}
#endif // CNNPLUS_MATLAB_FOUND

/*! \addtogroup eti_grp Explicit Template Instantiation
 @{
 */
template class Layer<float>;
template class Layer<double>;
/*! @} */

CNNPLUS_NS_END
