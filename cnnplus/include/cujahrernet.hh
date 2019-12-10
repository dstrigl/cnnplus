/**************************************************************************//**
 *
 * \file   cujahrernet.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jun 30 2009
 *
 * $Id: cujahrernet.hh 1835 2009-07-21 08:03:24Z dast $
 *
 * \brief  Header for cnnplus::CuJahrerNet.
 *
 *****************************************************************************/

#ifndef CNNPLUS_CUJAHRERNET_HH
#define CNNPLUS_CUJAHRERNET_HH

#include "common.hh"
#include "cuneuralnet.hh"
#include "cufulllayer.hh"
#include "cuconvlayer.hh"
#include "cusublayer.hh"

CNNPLUS_NS_BEGIN

//! A 6-layer convolutional neural network, as recommended by Michael Jahrer
/*! \see Handwritten digit recognition using a Convolutional Neural Network,
         Michael Jahrer, University of Technology, Graz, September 22, 2008
 */
template<typename T>
class CuJahrerNet : public CuNeuralNet<T>
{
public:
    //! Ctr
    /*! \param numMapsL1 number of feature maps in layer 1
        \param numMapsL3 number of feature maps in layer 3
        \param connPropL3 probability of the connection ration in layer 3
        \param numMapsL5 number of feature maps in layer 5
        \param connPropL5 probability of the connection ration in layer 5
     */
    CuJahrerNet(size_t numMapsL1 =  20,
                size_t numMapsL3 =  80, double connPropL3 = 0.14,
                size_t numMapsL5 = 350, double connPropL5 = 0.2);
    //! Dtr
    virtual ~CuJahrerNet();
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

#endif // CNNPLUS_CUJAHRERNET_HH
