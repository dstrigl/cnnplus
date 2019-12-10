/**************************************************************************//**
 *
 * \file   garcianet.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Apr 07 2009
 *
 * $Id: garcianet.hh 2006 2009-08-12 19:54:00Z dast $
 *
 * \brief  Header for cnnplus::GarciaNet.
 *
 *****************************************************************************/

#ifndef CNNPLUS_GARCIANET_HH
#define CNNPLUS_GARCIANET_HH

#include "common.hh"
#include "neuralnet.hh"
#include "fulllayer.hh"
#include "convlayer.hh"
#include "sublayer.hh"

CNNPLUS_NS_BEGIN

//! A 6-layer convolutional neural network, as recommended by Christophe Garcia
/*! \see
    \li M. Delakis, C. Garcia. Robust Face Detection based on Convolutional Neural Networks.
        Proc. of 2nd Hellenic Conference on Artificial Intelligence (SETN'02), pages 367-378,
        Thessalonique, Greece, April 2002.
    \li C. Garcia, M. Delakis. Convolutional Face Finder: A Neural Architecture for Fast
        and Robust Face Detection. IEEE Transactions on Pattern Analysis and Machine Intelligence,
        26(11):1408--1423, 2004.
 */
template<typename T>
class GarciaNet : public NeuralNet<T>
{
public:
    //! Ctr
    /*! \param sizeImgIn input image size (default is 20x20)
     */
    GarciaNet(Size const & sizeImgIn = Size(20, 20));
    //! Dtr
    virtual ~GarciaNet();
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

#endif // CNNPLUS_GARCIANET_HH
