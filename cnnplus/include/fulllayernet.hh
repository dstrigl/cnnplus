/**************************************************************************//**
 *
 * \file   fulllayernet.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jun 04 2009
 *
 * $Id: fulllayernet.hh 1835 2009-07-21 08:03:24Z dast $
 *
 * \brief  Header for cnnplus::FullLayerNet.
 *
 *****************************************************************************/

#ifndef CNNPLUS_FULLLAYERNET_HH
#define CNNPLUS_FULLLAYERNET_HH

#include "common.hh"
#include "neuralnet.hh"
#include "fulllayer.hh"
#include <vector>

CNNPLUS_NS_BEGIN

//! A neural network of multiple fully connected layers
template< typename T, class SquFnc = Tanh<T> >
class FullLayerNet : public NeuralNet<T>
{
public:
    //! Ctr
    /*! \param sizeIn input size
        \param numLayers number of layers
     */
    FullLayerNet(size_t sizeIn, size_t numLayers, ...);
    //! Dtr
    virtual ~FullLayerNet();
    virtual void forget(T sigma, bool scale = false);
    virtual void forget();
    virtual void reset();
    virtual void update(T eta);
    virtual void fprop(T const * in, T * out);
    virtual void bprop(T * in, T const * out, bool accumGradients = false);
    virtual void load(std::string const & filename);
    virtual void save(std::string const & filename) const;
    virtual void writeOut(std::string const & filename, T * const out = NULL) const;
    virtual size_t sizeIn() const { return layers_.front()->sizeIn().area(); }
    virtual size_t sizeOut() const { return layers_.back()->sizeOut().area(); }
    virtual void trainableParam(typename NeuralNet<T>::TrainableParam & param);
    virtual std::string toString() const;
    virtual size_t numTrainableParam() const;
    virtual size_t numConnections() const;

private:
    std::vector< FullLayer< T, SquFnc > * > layers_;
    std::vector< T * > states_;
};

CNNPLUS_NS_END

#endif // CNNPLUS_FULLLAYERNET_HH
