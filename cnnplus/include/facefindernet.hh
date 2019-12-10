/**************************************************************************//**
 *
 * \file   facefindernet.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jul 17 2009
 *
 * $Id: facefindernet.hh 1908 2009-07-31 07:00:47Z dast $
 *
 * \brief  Header for cnnplus::FaceFinderNet.
 *
 *****************************************************************************/

#ifndef CNNPLUS_FACEFINDERNET_HH
#define CNNPLUS_FACEFINDERNET_HH

#include "common.hh"
#include "neuralnet.hh"
#include "fulllayer.hh"
#include "convlayer.hh"
#include "sublayer.hh"

CNNPLUS_NS_BEGIN

//! The face finder neural-net
template<typename T>
class FaceFinderNet : public NeuralNet<T>
{
public:
    //! Ctr
    FaceFinderNet();
    //! Dtr
    virtual ~FaceFinderNet();
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
    virtual size_t sizeOut() const { return layer6_.sizeOut().area(); }
    virtual void trainableParam(typename NeuralNet<T>::TrainableParam & param);
    virtual std::string toString() const;
    virtual size_t numTrainableParam() const;
    virtual size_t numConnections() const;

private:
    ConvLayer< T, Tanh<T> > layer1_;
    SubLayer < T, Tanh<T> > layer2_;
    ConvLayer< T, Tanh<T> > layer3_;
    SubLayer < T, Tanh<T> > layer4_;
    ConvLayer< T, Tanh<T> > layer5_;
    FullLayer< T, Tanh<T> > layer6_;

    T * stateL1_; size_t strideL1_;
    T * stateL2_; size_t strideL2_;
    T * stateL3_; size_t strideL3_;
    T * stateL4_; size_t strideL4_;
    T * stateL5_; size_t strideL5_;
};

CNNPLUS_NS_END

#endif // CNNPLUS_FACEFINDERNET_HH
