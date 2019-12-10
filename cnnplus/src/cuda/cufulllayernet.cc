/**************************************************************************//**
 *
 * \file   cufulllayernet.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jun 05 2009
 *
 * $Id: cufulllayernet.cc 1904 2009-07-30 19:29:56Z dast $
 *
 * \brief  Implementation of cnnplus::CuFullLayerNet.
 *
 *****************************************************************************/

#include "cufulllayernet.hh"
#include "cumvli.hh"
#include "matvecli.hh"
#include <sstream>
#include <cstdarg>
#include <typeinfo>
#ifdef CNNPLUS_MATLAB_FOUND
#include <mat.h>
#endif // CNNPLUS_MATLAB_FOUND

CNNPLUS_NS_BEGIN

template<typename T, class SquFnc>
CuFullLayerNet<T, SquFnc>::CuFullLayerNet(size_t const sizeIn, size_t const numLayers, ...)
{
    if (sizeIn < 1)
        throw ParameterError("sizeIn", "must be greater or equal to one.");
    else if (numLayers < 1)
        throw ParameterError("numLayers", "must be greater or equal to one.");

    // TODO more checks for validity, throw exceptions!
    va_list ap; va_start(ap, numLayers);
    for (size_t i = 0, size = sizeIn; i < numLayers; ++i)
    {
        // Create layer
        layers_.push_back(new CuFullLayer<T, SquFnc>(
            Size(1, size), Size(1, va_arg(ap, size_t))));
        size = layers_.back()->sizeOut().area();
        CNNPLUS_ASSERT(size > 0);
        // Allocate memory
        if (i < numLayers - 1)
            states_.push_back(cumvli::allocv<T>(size));
    }
    va_end(ap);

    // Allocate GPU memory for 'in' and 'out'
    in_  = cumvli::allocv<T>(this->sizeIn());
    out_ = cumvli::allocv<T>(this->sizeOut());
}

template<typename T, class SquFnc>
CuFullLayerNet<T, SquFnc>::~CuFullLayerNet()
{
    // Delete layers
    for (size_t i = 0; i < layers_.size(); ++i)
        delete layers_[i];

    // Deallocate GPU memory
    for (size_t i = 0; i < states_.size(); ++i)
        cumvli::free<T>(states_[i]);

    // Deallocate GPU memory
    cumvli::free<T>(in_);
    cumvli::free<T>(out_);
}

template<typename T, class SquFnc> void
CuFullLayerNet<T, SquFnc>::forget(T const sigma, bool scale)
{
    matvecli::randreseed();

    for (size_t i = 0; i < layers_.size(); ++i)
        layers_[i]->forget(sigma, scale);
}

template<typename T, class SquFnc> void
CuFullLayerNet<T, SquFnc>::forget()
{
    matvecli::randreseed();

    for (size_t i = 0; i < layers_.size(); ++i)
        layers_[i]->forget();
}

template<typename T, class SquFnc> void
CuFullLayerNet<T, SquFnc>::reset()
{
    for (size_t i = 0; i < layers_.size(); ++i)
        layers_[i]->reset();
}

template<typename T, class SquFnc> void
CuFullLayerNet<T, SquFnc>::update(T const eta)
{
    for (size_t i = 0; i < layers_.size(); ++i)
        layers_[i]->update(eta);
}

template<typename T, class SquFnc> void
CuFullLayerNet<T, SquFnc>::fprop(T const * in, T * out)
{
    CNNPLUS_ASSERT(in); // 'out' can be NULL

    // Copy vector 'in' from CPU to GPU
    cumvli::copyv_h2d<T>(in, in_, sizeIn());

    // For networks with only one layer
    if (layers_.size() == 1) {
        layers_[0]->fprop(in_, layers_[0]->sizeIn().area(), out_, layers_[0]->sizeOut().area());
    }
    else {
        // ... and for networks with more than one layer
        for (int i = 0; i < static_cast<int>(layers_.size()); ++i)
        {
            size_t const strideIn = layers_[i]->sizeIn().area();
            size_t const strideOut = layers_[i]->sizeOut().area();

            if (i == 0) {
                // First layer
                layers_[i]->fprop(in_, strideIn, states_[i], strideOut);
            }
            else if (i < static_cast<int>(layers_.size() - 1)) {
                // Layers between
                layers_[i]->fprop(states_[i-1], strideIn, states_[i], strideOut);
            }
            else {
                // Last layer
                layers_[i]->fprop(states_[i-1], strideIn, out_, strideOut);
            }
        }
    }
    if (out) {
        // Copy vector 'out' from GPU to CPU
        cumvli::copyv_d2h<T>(out_, out, sizeOut());
    }
}

template<typename T, class SquFnc> void
CuFullLayerNet<T, SquFnc>::bprop(T * in, T const * out, bool accumGradients)
{
    CNNPLUS_ASSERT(out); // 'in' can be NULL

    // Copy vector 'out' from CPU to GPU
    cumvli::copyv_h2d<T>(out, out_, sizeOut());

    // For networks with only one layer
    if (layers_.size() == 1) {
        layers_[0]->bprop(in_, layers_[0]->sizeIn().area(), out_, layers_[0]->sizeOut().area());
    }
    else {
        // ... and for networks with more than one layer
        for (int i = static_cast<int>(layers_.size() - 1); i >= 0; --i)
        {
            size_t const strideIn = layers_[i]->sizeIn().area();
            size_t const strideOut = layers_[i]->sizeOut().area();

            if (i == static_cast<int>(layers_.size() - 1)) {
                // Last layer
                layers_[i]->bprop(states_[i-1], strideIn, out_, strideOut);
            }
            else if (i > 0) {
                // Layers between
                layers_[i]->bprop(states_[i-1], strideIn, states_[i], strideOut);
            }
            else {
                // First layer
                layers_[i]->bprop(in_, strideIn, states_[i], strideOut);
            }
        }
    }
    if (in) {
        // Copy vector 'in' from GPU to CPU
        cumvli::copyv_d2h<T>(in_, in, sizeIn());
    }
}

template<typename T, class SquFnc> void
CuFullLayerNet<T, SquFnc>::load(std::string const & filename)
{
#ifdef CNNPLUS_MATLAB_FOUND
    MATFile * file = matOpen(filename.c_str(), "r");
    if (!file)
        throw MatlabError("Failed to open '" + filename + "'.");

    mxArray * arrLayer = NULL;

    try {
        if (!(arrLayer = matGetVariable(file, "layer")) || !mxIsCell(arrLayer))
            throw MatlabError("Failed to read 'layer'.");

        for (size_t i = 0; i < layers_.size(); ++i) {
            std::stringstream ss;
            ss << "layer" << (i + 1);
            layers_[i]->load(mxGetCell(arrLayer, i));
        }
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

template<typename T, class SquFnc> void
CuFullLayerNet<T, SquFnc>::save(std::string const & filename) const
{
#ifdef CNNPLUS_MATLAB_FOUND
    MATFile * file = matOpen(filename.c_str(), "w");
    if (!file)
        throw MatlabError("Failed to create '" + filename + "'.");

    mxArray * arrLayer = NULL;

    try {
        if (!(arrLayer = mxCreateCellMatrix(layers_.size(), 1)))
            throw MatlabError("Failed to create array.");

        for (size_t i = 0; i < layers_.size(); ++i) {
            std::stringstream ss;
            ss << "layer" << (i + 1);
            mxSetCell(arrLayer, i, layers_[i]->save());
        }

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

template<typename T, class SquFnc> void
CuFullLayerNet<T, SquFnc>::trainableParam(typename NeuralNet<T>::TrainableParam & param)
{
    param.resize(layers_.size());
    for (size_t i = 0; i < layers_.size(); ++i)
        layers_[i]->trainableParam(param[i]);
}

template<typename T, class SquFnc> std::string
CuFullLayerNet<T, SquFnc>::toString() const
{
    std::stringstream ss;
    ss << "CuFullLayerNet<" << typeid(T).name() << ">[";
    for (size_t i = 0; i < layers_.size(); ++i) {
        ss << "L" << (i + 1) << ":" << layers_[i]->toString();
        ss << (i < (layers_.size() - 1) ? "," : "]");
    }
    return ss.str();
}

template<typename T, class SquFnc> size_t
CuFullLayerNet<T, SquFnc>::numTrainableParam() const
{
    size_t numTrainableParam = 0;
    for (size_t i = 0; i < layers_.size(); ++i)
        numTrainableParam += layers_[i]->numTrainableParam();
    return numTrainableParam;
}

template<typename T, class SquFnc> size_t
CuFullLayerNet<T, SquFnc>::numConnections() const
{
    size_t numConnections = 0;
    for (size_t i = 0; i < layers_.size(); ++i)
        numConnections += layers_[i]->numConnections();
    return numConnections;
}

/*! \addtogroup eti_grp Explicit Template Instantiation
 @{
 */
template class CuFullLayerNet< float,  CuTanh<float>        >;
template class CuFullLayerNet< float,  CuStdSigmoid<float>  >;
template class CuFullLayerNet< float,  CuLogSigmoid<float>  >;
template class CuFullLayerNet< float,  CuIdentity<float>    >;

template class CuFullLayerNet< double, CuTanh<double>       >;
template class CuFullLayerNet< double, CuStdSigmoid<double> >;
template class CuFullLayerNet< double, CuLogSigmoid<double> >;
template class CuFullLayerNet< double, CuIdentity<double>   >;
/*! @} */

CNNPLUS_NS_END
