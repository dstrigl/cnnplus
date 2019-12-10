/**************************************************************************//**
 *
 * \file   lolnet.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Mar 12 2009
 *
 * $Id: lolnet.hh 1835 2009-07-21 08:03:24Z dast $
 *
 * \brief  Header for cnnplus::LolNet.
 *
 *****************************************************************************/

#ifndef CNNPLUS_LOLNET_HH
#define CNNPLUS_LOLNET_HH

#include "common.hh"
#include "neuralnet.hh"
#include "fulllayer.hh"
#include "convlayer.hh"

CNNPLUS_NS_BEGIN

//! \todo doc
template<typename T>
class LolNet : public NeuralNet<T>
{
public:
    //! Ctr
    /*! \param numMapsL1 number of feature maps in layer 1
        \param numMapsL2 number of feature maps in layer 2
        \param connPropL2 probability of the connection ration in layer 2
        \param sizeL3 size of layer 3
     */
    LolNet(size_t numMapsL1 =  10,
           size_t numMapsL2 =  80, double connPropL2 = 0.2,
           size_t sizeL3    = 100);
    //! Dtr
    virtual ~LolNet();
    virtual void forget(T sigma, bool scale = false);
    virtual void forget();
    virtual void reset();
    virtual void update(T eta);
    virtual void fprop(T const * in, T * out);
    virtual void bprop(T * in, T const * out, bool accumGradients = false);
    virtual void load(std::string const & filename);
    virtual void save(std::string const & filename) const;
    virtual void writeOut(std::string const & filename, T * const out = NULL) const;
    virtual size_t sizeIn() const { return layer1_.sizeIn().area(); }
    virtual size_t sizeOut() const { return layer4_.sizeOut().area(); }
    virtual void trainableParam(typename NeuralNet<T>::TrainableParam & param);
    virtual std::string toString() const;
    virtual size_t numTrainableParam() const;
    virtual size_t numConnections() const;

private:
    ConvLayer< T, Tanh<T> > layer1_;
    ConvLayer< T, Tanh<T> > layer2_;
    FullLayer< T, Tanh<T> > layer3_;
    FullLayer< T, Tanh<T> > layer4_;

    T * stateL1_; size_t strideL1_;
    T * stateL2_; size_t strideL2_;
    T * stateL3_; size_t strideL3_;
};

CNNPLUS_NS_END

#endif // CNNPLUS_LOLNET_HH
