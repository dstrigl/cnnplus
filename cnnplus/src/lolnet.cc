/**************************************************************************//**
 *
 * \file   lolnet.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Mar 12 2009
 *
 * $Id: lolnet.cc 1942 2009-08-05 18:25:28Z dast $
 *
 * \brief  Implementation of cnnplus::LolNet.
 *
 *****************************************************************************/

#include "lolnet.hh"
#include "error.hh"
#include "matvecli.hh"
#include <sstream>
#include <typeinfo>
#ifdef CNNPLUS_MATLAB_FOUND
#include <mat.h>
#endif // CNNPLUS_MATLAB_FOUND

CNNPLUS_NS_BEGIN

template<typename T>
LolNet<T>::LolNet(size_t const numMapsL1,
                  size_t const numMapsL2, double const connPropL2,
                  size_t const sizeL3)
    : // Convolutional layer
      layer1_(Size(29, 29), 1, numMapsL1, Size(5, 5), 2, 2),
      // Convolutional layer
      layer2_(layer1_.sizeMapsOut(), numMapsL1, numMapsL2, Size(5, 5), 2, 2, connPropL2),
      // Fully connected layer
      layer3_(layer2_.sizeOut(), Size(1, sizeL3)),
      // Fully connected layer
      layer4_(layer3_.sizeOut(), Size(1, 10))
{
    // Allocate memory
    stateL1_ = matvecli::allocm<T>(layer1_.sizeOut().height(),
                                   layer1_.sizeOut().width(),
                                   strideL1_);
    stateL2_ = matvecli::allocm<T>(layer2_.sizeOut().height(),
                                   layer2_.sizeOut().width(),
                                   strideL2_);
    stateL3_ = matvecli::allocm<T>(layer3_.sizeOut().height(),
                                   layer3_.sizeOut().width(),
                                   strideL3_);
}

template<typename T>
LolNet<T>::~LolNet()
{
    // Deallocate memory
    matvecli::free<T>(stateL1_);
    matvecli::free<T>(stateL2_);
    matvecli::free<T>(stateL3_);
}

template<typename T> void
LolNet<T>::forget(T const sigma, bool scale)
{
    matvecli::randreseed();

    layer1_.forget(sigma, scale);
    layer2_.forget(sigma, scale);
    layer3_.forget(sigma, scale);
    layer4_.forget(sigma, scale);
}

template<typename T> void
LolNet<T>::forget()
{
    matvecli::randreseed();

    layer1_.forget();
    layer2_.forget();
    layer3_.forget();
    layer4_.forget();
}

template<typename T> void
LolNet<T>::reset()
{
    layer1_.reset();
    layer2_.reset();
    layer3_.reset();
    layer4_.reset();
}

template<typename T> void
LolNet<T>::update(T const eta)
{
    layer1_.update(eta);
    layer2_.update(eta);
    layer3_.update(eta);
    layer4_.update(eta);
}

template<typename T> void
LolNet<T>::fprop(T const * in, T * out)
{
    CNNPLUS_ASSERT(in); // 'out' can be NULL

    layer1_.fprop(in,       sizeIn(),  stateL1_, strideL1_);
    layer2_.fprop(stateL1_, strideL1_, stateL2_, strideL2_);
    layer3_.fprop(stateL2_, strideL2_, stateL3_, strideL3_);
    layer4_.fprop(stateL3_, strideL3_, out,      sizeOut());
}

template<typename T> void
LolNet<T>::bprop(T * in, T const * out, bool accumGradients)
{
    CNNPLUS_ASSERT(out); // 'in' can be NULL

    layer4_.bprop(stateL3_, strideL3_, out,      sizeOut(), accumGradients);
    layer3_.bprop(stateL2_, strideL2_, stateL3_, strideL3_, accumGradients);
    layer2_.bprop(stateL1_, strideL1_, stateL2_, strideL2_, accumGradients);
    layer1_.bprop(in,       sizeIn(),  stateL1_, strideL1_, accumGradients);
}

template<typename T> void
LolNet<T>::load(std::string const & filename)
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
LolNet<T>::save(std::string const & filename) const
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
LolNet<T>::writeOut(std::string const & filename, T * const out) const
{
#ifdef CNNPLUS_MATLAB_FOUND
    MATFile * file = matOpen(filename.c_str(), "w");
    if (!file)
        throw MatlabError("Failed to create '" + filename + "'.");

    mxArray * arrLayer = NULL;

    try {
        if (!(arrLayer = mxCreateCellMatrix(out ? 4 : 3, 1)))
            throw MatlabError("Failed to create array.");

        mxSetCell(arrLayer, 0, layer1_.writeOut(stateL1_, strideL1_));
        mxSetCell(arrLayer, 1, layer2_.writeOut(stateL2_, strideL2_));
        mxSetCell(arrLayer, 2, layer3_.writeOut(stateL3_, strideL3_));
        if (out) mxSetCell(arrLayer, 3, layer4_.writeOut(out, sizeOut()));

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
    throw NotImplementedError("MATLAB not found, function 'writeOut' not supported.");
#endif // CNNPLUS_MATLAB_FOUND
}

template<typename T> void
LolNet<T>::trainableParam(typename NeuralNet<T>::TrainableParam & param)
{
    param.resize(4);
    layer1_.trainableParam(param[0]);
    layer2_.trainableParam(param[1]);
    layer3_.trainableParam(param[2]);
    layer4_.trainableParam(param[3]);
}

template<typename T> std::string
LolNet<T>::toString() const
{
    std::stringstream ss;
    ss << "LolNet<" << typeid(T).name() << ">"
       << "[L1:" << layer1_.toString()
       << ",L2:" << layer2_.toString()
       << ",L3:" << layer3_.toString()
       << ",L4:" << layer4_.toString()
       << "]";
    return ss.str();
}

template<typename T> size_t
LolNet<T>::numTrainableParam() const
{
    return (layer1_.numTrainableParam() +
            layer2_.numTrainableParam() +
            layer3_.numTrainableParam() +
            layer4_.numTrainableParam());
}

template<typename T> size_t
LolNet<T>::numConnections() const
{
    return (layer1_.numConnections() +
            layer2_.numConnections() +
            layer3_.numConnections() +
            layer4_.numConnections());
}

/*! \addtogroup eti_grp Explicit Template Instantiation
 @{
 */
template class LolNet<float>;
template class LolNet<double>;
/*! @} */

CNNPLUS_NS_END
