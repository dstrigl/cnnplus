/**************************************************************************//**
 *
 * \file   cufulllayernet.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jun 05 2009
 *
 * $Id: cufulllayernet.hh 1835 2009-07-21 08:03:24Z dast $
 *
 * \brief  Header for cnnplus::CuFullLayerNet.
 *
 *****************************************************************************/

#ifndef CNNPLUS_CUFULLLAYERNET_HH
#define CNNPLUS_CUFULLLAYERNET_HH

#include "common.hh"
#include "cuneuralnet.hh"
#include "cufulllayer.hh"
#include <vector>

CNNPLUS_NS_BEGIN

//! A neural network of multiple fully connected layers
template< typename T, class SquFnc = CuTanh<T> >
class CuFullLayerNet : public CuNeuralNet<T>
{
public:
    //! Ctr
    /*! \param sizeIn input size
        \param numLayers number of layers
     */
    CuFullLayerNet(size_t sizeIn, size_t numLayers, ...);
    //! Dtr
    virtual ~CuFullLayerNet();
    virtual void forget(T sigma, bool scale = false);
    virtual void forget();
    virtual void reset();
    virtual void update(T eta);
    virtual void fprop(T const * in, T * out);
    virtual void bprop(T * in, T const * out, bool accumGradients = false);
    virtual void load(std::string const & filename);
    virtual void save(std::string const & filename) const;
    virtual size_t sizeIn() const { return layers_.front()->sizeIn().area(); }
    virtual size_t sizeOut() const { return layers_.back()->sizeOut().area(); }
    virtual void trainableParam(typename NeuralNet<T>::TrainableParam & param);
    virtual std::string toString() const;
    virtual size_t numTrainableParam() const;
    virtual size_t numConnections() const;

private:
    std::vector< CuFullLayer< T, SquFnc > * > layers_;
    std::vector< T * > states_;
    T * in_, * out_;
};

CNNPLUS_NS_END

#endif // CNNPLUS_CUFULLLAYERNET_HH
