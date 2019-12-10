/**************************************************************************//**
 *
 * \file   cugarcianet.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Aug 12 2009
 *
 * $Id: cugarcianet.hh 2010 2009-08-12 20:15:10Z dast $
 *
 * \brief  Header for cnnplus::CuGarciaNet.
 *
 *****************************************************************************/

#ifndef CNNPLUS_CUGARCIANET_HH
#define CNNPLUS_CUGARCIANET_HH

#include "common.hh"
#include "cuneuralnet.hh"
#include "cufulllayer.hh"
#include "cuconvlayer.hh"
#include "cusublayer.hh"

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
class CuGarciaNet : public CuNeuralNet<T>
{
public:
    //! Ctr
    /*! \param sizeImgIn input image size (default is 20x20)
     */
    CuGarciaNet(Size const & sizeImgIn = Size(20, 20));
    //! Dtr
    virtual ~CuGarciaNet();
    virtual void forget(T sigma, bool scale = false);
    virtual void forget();
    virtual void reset();
    virtual void update(T eta);
    virtual void fprop(T const * in, T * out);
    virtual void bprop(T * in, T const * out, bool accumGradients = false);
    virtual void load(std::string const & filename);
    virtual void save(std::string const & filename) const;
    virtual size_t sizeIn() const { return layer1_.sizeIn().area(); }
    virtual size_t sizeOut() const { return layer6_.sizeOut().area(); }
    virtual void trainableParam(typename NeuralNet<T>::TrainableParam & param);
    virtual std::string toString() const;
    virtual size_t numTrainableParam() const;
    virtual size_t numConnections() const;

private:
    CuConvLayer< T, CuTanh<T> > layer1_;
    CuSubLayer < T, CuTanh<T> > layer2_;
    CuConvLayer< T, CuTanh<T> > layer3_;
    CuSubLayer < T, CuTanh<T> > layer4_;
    CuConvLayer< T, CuTanh<T> > layer5_;
    CuFullLayer< T, CuTanh<T> > layer6_;

    T * stateL1_; size_t strideL1_;
    T * stateL2_; size_t strideL2_;
    T * stateL3_; size_t strideL3_;
    T * stateL4_; size_t strideL4_;
    T * stateL5_; size_t strideL5_;
    T * in_, * out_;
};

CNNPLUS_NS_END

#endif // CNNPLUS_CUGARCIANET_HH
