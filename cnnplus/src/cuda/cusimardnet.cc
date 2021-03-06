/**************************************************************************//**
 *
 * \file   cusimardnet.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jun 30 2009
 *
 * $Id: cusimardnet.cc 1904 2009-07-30 19:29:56Z dast $
 *
 * \brief  Implementation of cnnplus::CuSimardNet.
 *
 *****************************************************************************/

#include "cusimardnet.hh"
#include "matvecli.hh"
#include "cumvli.hh"
#include <sstream>
#include <typeinfo>
#ifdef CNNPLUS_MATLAB_FOUND
#include <mat.h>
#endif // CNNPLUS_MATLAB_FOUND

CNNPLUS_NS_BEGIN

template<typename T>
CuSimardNet<T>::CuSimardNet(Size const & sizeImgIn,
                            size_t const numMapsL1,
                            size_t const numMapsL2,
                            size_t const sizeL3)
    : layer1_(sizeImgIn, 1, numMapsL1, Size(5, 5), 2, 2),
    layer2_(layer1_.sizeMapsOut(), numMapsL1, numMapsL2, Size(5, 5), 2, 2),
    layer3_(layer2_.sizeOut(), Size(1, sizeL3)),
    layer4_(layer3_.sizeOut(), Size(1, 10))
{
    // Allocate GPU memory for the states
    stateL1_ = cumvli::allocm<T>(layer1_.sizeOut().height(),
                                 layer1_.sizeOut().width(),
                                 strideL1_);
    stateL2_ = cumvli::allocm<T>(layer2_.sizeOut().height(),
                                 layer2_.sizeOut().width(),
                                 strideL2_);
    stateL3_ = cumvli::allocm<T>(layer3_.sizeOut().height(),
                                 layer3_.sizeOut().width(),
                                 strideL3_);

    // Allocate GPU memory for 'in' and 'out'
    in_  = cumvli::allocv<T>(this->sizeIn());
    out_ = cumvli::allocv<T>(this->sizeOut());
}

template<typename T>
CuSimardNet<T>::~CuSimardNet()
{
    // Deallocate GPU memory
    cumvli::free<T>(stateL1_);
    cumvli::free<T>(stateL2_);
    cumvli::free<T>(stateL3_);
    cumvli::free<T>(in_);
    cumvli::free<T>(out_);
}

template<typename T> void
CuSimardNet<T>::forget(T const sigma, bool scale)
{
    matvecli::randreseed();

    layer1_.forget(sigma, scale);
    layer2_.forget(sigma, scale);
    layer3_.forget(sigma, scale);
    layer4_.forget(sigma, scale);
}

template<typename T> void
CuSimardNet<T>::forget()
{
    matvecli::randreseed();

    layer1_.forget();
    layer2_.forget();
    layer3_.forget();
    layer4_.forget();
}

template<typename T> void
CuSimardNet<T>::reset()
{
    layer1_.reset();
    layer2_.reset();
    layer3_.reset();
    layer4_.reset();
}

template<typename T> void
CuSimardNet<T>::update(T const eta)
{
    layer1_.update(eta);
    layer2_.update(eta);
    layer3_.update(eta);
    layer4_.update(eta);
}

template<typename T> void
CuSimardNet<T>::fprop(T const * in, T * out)
{
    CNNPLUS_ASSERT(in); // 'out' can be NULL

    // Copy vector 'in' from CPU to GPU
    cumvli::copyv_h2d<T>(in, in_, sizeIn());

    layer1_.fprop(in_,      sizeIn(),  stateL1_, strideL1_);
    layer2_.fprop(stateL1_, strideL1_, stateL2_, strideL2_);
    layer3_.fprop(stateL2_, strideL2_, stateL3_, strideL3_);
    layer4_.fprop(stateL3_, strideL3_, out_,     sizeOut());

    if (out) {
        // Copy vector 'out' from GPU to CPU
        cumvli::copyv_d2h<T>(out_, out, sizeOut());
    }
}

template<typename T> void
CuSimardNet<T>::bprop(T * in, T const * out, bool accumGradients)
{
    CNNPLUS_ASSERT(out); // 'in' can be NULL

    // Copy vector 'out' from CPU to GPU
    cumvli::copyv_h2d<T>(out, out_, sizeOut());

    layer4_.bprop(stateL3_, strideL3_, out_,     sizeOut(), accumGradients);
    layer3_.bprop(stateL2_, strideL2_, stateL3_, strideL3_, accumGradients);
    layer2_.bprop(stateL1_, strideL1_, stateL2_, strideL2_, accumGradients);
    layer1_.bprop(in_,      sizeIn(),  stateL1_, strideL1_, accumGradients);

    if (in) {
        // Copy vector 'in' from GPU to CPU
        cumvli::copyv_d2h<T>(in_, in, sizeIn());
    }
}

template<typename T> void
CuSimardNet<T>::load(std::string const & filename)
{
#ifdef CNNPLUS_MATLAB_FOUND
    MATFile * file = matOpen(filename.c_str(), "r");
    if (!file)
        throw MatlabError("Failed to open '" + filename + "'.");

    mxArray * arrLayer = NULL;

    try {
        if (!(arrLayer = matGetVariable(file, "layer")) || !mxIsCell(arrLayer))
            throw MatlabError("Failed to read 'layer'.");

        layer1_.load(mxGetCell(arrLayer, 0));
        layer2_.load(mxGetCell(arrLayer, 1));
        layer3_.load(mxGetCell(arrLayer, 2));
        layer4_.load(mxGetCell(arrLayer, 3));
    }
    catch (...) {
        if (arrLayer) mxDestroyArray(arrLayer);
        matClose(file);
        throw;
    }

    mxDestroyArray(arrLayer);
    matClose(file);
#else
    throw NotImplementedError("MATLAB not found, function 'load' not supported.");
#endif // CNNPLUS_MATLAB_FOUND
}

template<typename T> void
CuSimardNet<T>::save(std::string const & filename) const
{
#ifdef CNNPLUS_MATLAB_FOUND
    MATFile * file = matOpen(filename.c_str(), "w");
    if (!file)
        throw MatlabError("Failed to create '" + filename + "'.");

    mxArray * arrLayer = NULL;

    try {
        if (!(arrLayer = mxCreateCellMatrix(4, 1)))
            throw MatlabError("Failed to create array.");

        mxSetCell(arrLayer, 0, layer1_.save());
        mxSetCell(arrLayer, 1, layer2_.save());
        mxSetCell(arrLayer, 2, layer3_.save());
        mxSetCell(arrLayer, 3, layer4_.save());

        if (matPutVariable(file, "layer", arrLayer))
            throw MatlabError("Failed to write 'layer'.");
    }
    catch (...) {
        if (arrLayer) mxDestroyArray(arrLayer);
        matClose(file);
        throw;
    }

    mxDestroyArray(arrLayer);
    matClose(file);
#else
    throw NotImplementedError("MATLAB not found, function 'save' not supported.");
#endif // CNNPLUS_MATLAB_FOUND
}

template<typename T> void
CuSimardNet<T>::trainableParam(typename NeuralNet<T>::TrainableParam & param)
{
    param.resize(4);
    layer1_.trainableParam(param[0]);
    layer2_.trainableParam(param[1]);
    layer3_.trainableParam(param[2]);
    layer4_.trainableParam(param[3]);
}

template<typename T> std::string
CuSimardNet<T>::toString() const
{
    std::stringstream ss;
    ss << "CuSimardNet<" << typeid(T).name() << ">"
       << "[L1:" << layer1_.toString()
       << ",L2:" << layer2_.toString()
       << ",L3:" << layer3_.toString()
       << ",L4:" << layer4_.toString()
       << "]";
    return ss.str();
}

template<typename T> size_t
CuSimardNet<T>::numTrainableParam() const
{
    return (layer1_.numTrainableParam() +
            layer2_.numTrainableParam() +
            layer3_.numTrainableParam() +
            layer4_.numTrainableParam());
}

template<typename T> size_t
CuSimardNet<T>::numConnections() const
{
    return (layer1_.numConnections() +
            layer2_.numConnections() +
            layer3_.numConnections() +
            layer4_.numConnections());
}

/*! \addtogroup eti_grp Explicit Template Instantiation
 @{
 */
template class CuSimardNet<float>;
template class CuSimardNet<double>;
/*! @} */

CNNPLUS_NS_END
