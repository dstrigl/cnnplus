/**************************************************************************//**
 *
 * \file   phungnet.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jul 15 2009
 *
 * $Id: phungnet.cc 1942 2009-08-05 18:25:28Z dast $
 *
 * \brief  Implementation of cnnplus::PhungNet.
 *
 *****************************************************************************/

#include "phungnet.hh"
#include "error.hh"
#include "matvecli.hh"
#include <sstream>
#include <typeinfo>
#ifdef CNNPLUS_MATLAB_FOUND
#include <mat.h>
#endif // CNNPLUS_MATLAB_FOUND

CNNPLUS_NS_BEGIN

template<typename T>
PhungNet<T>::PhungNet()
    : // C1: Convolutional layer
      layer1_(Size(32, 32), 1, 2, Size(5, 5)),
      // S2: Subsampling layer
      layer2_(layer1_.sizeMapsOut(), 2, Size(2, 2)),
      // C3: Convolutional layer
      layer3_(layer2_.sizeMapsOut(), 2, 5, Size(3, 3)),
      // S4: Subsampling layer
      layer4_(layer3_.sizeMapsOut(), 5, Size(2, 2)),
      // C5: Convolutional layer
      layer5_(layer4_.sizeMapsOut(), 5, 5, Size(6, 6)),
      // F6: Fully connected layer
      layer6_(layer5_.sizeOut(), Size(1, 1))
{
    // Set connection table for C3 (layer 3)
    int const conTblC3[] = {
        1, 0,
        1, 0,
        0, 1,
        0, 1,
        1, 1
    };
    layer3_.setConTbl(ConTbl(5, 2, conTblC3));
    CNNPLUS_ASSERT(layer3_.conTbl().numConn() == 6);

    // Set connection table for C5 (layer 5)
    int const conTblC5[] = {
        1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1
    };
    layer5_.setConTbl(ConTbl(5, 5, conTblC5));
    CNNPLUS_ASSERT(layer5_.conTbl().numConn() == 5);

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
    stateL4_ = matvecli::allocm<T>(layer4_.sizeOut().height(),
                                   layer4_.sizeOut().width(),
                                   strideL4_);
    stateL5_ = matvecli::allocm<T>(layer5_.sizeOut().height(),
                                   layer5_.sizeOut().width(),
                                   strideL5_);
}

template<typename T>
PhungNet<T>::~PhungNet()
{
    // Deallocate memory
    matvecli::free<T>(stateL1_);
    matvecli::free<T>(stateL2_);
    matvecli::free<T>(stateL3_);
    matvecli::free<T>(stateL4_);
    matvecli::free<T>(stateL5_);
}

template<typename T> void
PhungNet<T>::forget(T const sigma, bool scale)
{
    matvecli::randreseed();

    layer1_.forget(sigma, scale);
    layer2_.forget(sigma, scale);
    layer3_.forget(sigma, scale);
    layer4_.forget(sigma, scale);
    layer5_.forget(sigma, scale);
    layer6_.forget(sigma, scale);
}

template<typename T> void
PhungNet<T>::forget()
{
    matvecli::randreseed();

    layer1_.forget();
    layer2_.forget();
    layer3_.forget();
    layer4_.forget();
    layer5_.forget();
    layer6_.forget();
}

template<typename T> void
PhungNet<T>::reset()
{
    layer1_.reset();
    layer2_.reset();
    layer3_.reset();
    layer4_.reset();
    layer5_.reset();
    layer6_.reset();
}

template<typename T> void
PhungNet<T>::update(T const eta)
{
    layer1_.update(eta);
    layer2_.update(eta);
    layer3_.update(eta);
    layer4_.update(eta);
    layer5_.update(eta);
    layer6_.update(eta);
}

template<typename T> void
PhungNet<T>::fprop(T const * in, T * out)
{
    CNNPLUS_ASSERT(in); // 'out' can be NULL

    layer1_.fprop(in,       sizeIn(),  stateL1_, strideL1_);
    layer2_.fprop(stateL1_, strideL1_, stateL2_, strideL2_);
    layer3_.fprop(stateL2_, strideL2_, stateL3_, strideL3_);
    layer4_.fprop(stateL3_, strideL3_, stateL4_, strideL4_);
    layer5_.fprop(stateL4_, strideL4_, stateL5_, strideL5_);
    layer6_.fprop(stateL5_, strideL5_, out,      sizeOut());
}

template<typename T> void
PhungNet<T>::bprop(T * in, T const * out, bool accumGradients)
{
    CNNPLUS_ASSERT(out); // 'in' can be NULL

    layer6_.bprop(stateL5_, strideL5_, out,      sizeOut(), accumGradients);
    layer5_.bprop(stateL4_, strideL4_, stateL5_, strideL5_, accumGradients);
    layer4_.bprop(stateL3_, strideL3_, stateL4_, strideL4_, accumGradients);
    layer3_.bprop(stateL2_, strideL2_, stateL3_, strideL3_, accumGradients);
    layer2_.bprop(stateL1_, strideL1_, stateL2_, strideL2_, accumGradients);
    layer1_.bprop(in,       sizeIn(),  stateL1_, strideL1_, accumGradients);
}

template<typename T> void
PhungNet<T>::load(std::string const & filename)
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
        layer5_.load(mxGetCell(arrLayer, 4));
        layer6_.load(mxGetCell(arrLayer, 5));
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
PhungNet<T>::save(std::string const & filename) const
{
#ifdef CNNPLUS_MATLAB_FOUND
    MATFile * file = matOpen(filename.c_str(), "w");
    if (!file)
        throw MatlabError("Failed to create '" + filename + "'.");

    mxArray * arrLayer = NULL;

    try {
        if (!(arrLayer = mxCreateCellMatrix(6, 1)))
            throw MatlabError("Failed to create array.");

        mxSetCell(arrLayer, 0, layer1_.save());
        mxSetCell(arrLayer, 1, layer2_.save());
        mxSetCell(arrLayer, 2, layer3_.save());
        mxSetCell(arrLayer, 3, layer4_.save());
        mxSetCell(arrLayer, 4, layer5_.save());
        mxSetCell(arrLayer, 5, layer6_.save());

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
PhungNet<T>::writeOut(std::string const & filename, T * const out) const
{
#ifdef CNNPLUS_MATLAB_FOUND
    MATFile * file = matOpen(filename.c_str(), "w");
    if (!file)
        throw MatlabError("Failed to create '" + filename + "'.");

    mxArray * arrLayer = NULL;

    try {
        if (!(arrLayer = mxCreateCellMatrix(out ? 6 : 5, 1)))
            throw MatlabError("Failed to create array.");

        mxSetCell(arrLayer, 0, layer1_.writeOut(stateL1_, strideL1_));
        mxSetCell(arrLayer, 1, layer2_.writeOut(stateL2_, strideL2_));
        mxSetCell(arrLayer, 2, layer3_.writeOut(stateL3_, strideL3_));
        mxSetCell(arrLayer, 3, layer4_.writeOut(stateL4_, strideL4_));
        mxSetCell(arrLayer, 4, layer5_.writeOut(stateL5_, strideL5_));
        if (out) mxSetCell(arrLayer, 5, layer6_.writeOut(out, sizeOut()));

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
PhungNet<T>::trainableParam(typename NeuralNet<T>::TrainableParam & param)
{
    param.resize(6);
    layer1_.trainableParam(param[0]);
    layer2_.trainableParam(param[1]);
    layer3_.trainableParam(param[2]);
    layer4_.trainableParam(param[3]);
    layer5_.trainableParam(param[4]);
    layer6_.trainableParam(param[5]);
}

template<typename T> std::string
PhungNet<T>::toString() const
{
    std::stringstream ss;
    ss << "PhungNet<" << typeid(T).name() << ">"
       << "[L1:" << layer1_.toString()
       << ",L2:" << layer2_.toString()
       << ",L3:" << layer3_.toString()
       << ",L4:" << layer4_.toString()
       << ",L5:" << layer5_.toString()
       << ",L6:" << layer6_.toString()
       << "]";
    return ss.str();
}

template<typename T> size_t
PhungNet<T>::numTrainableParam() const
{
    return (layer1_.numTrainableParam() +
            layer2_.numTrainableParam() +
            layer3_.numTrainableParam() +
            layer4_.numTrainableParam() +
            layer5_.numTrainableParam() +
            layer6_.numTrainableParam());
}

template<typename T> size_t
PhungNet<T>::numConnections() const
{
    return (layer1_.numConnections() +
            layer2_.numConnections() +
            layer3_.numConnections() +
            layer4_.numConnections() +
            layer5_.numConnections() +
            layer6_.numConnections());
}

/*! \addtogroup eti_grp Explicit Template Instantiation
 @{
 */
template class PhungNet<float>;
template class PhungNet<double>;
/*! @} */

CNNPLUS_NS_END
