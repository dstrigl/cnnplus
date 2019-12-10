/**************************************************************************//**
 *
 * \file   culenet5.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jun 30 2009
 *
 * $Id: culenet5.hh 1962 2009-08-09 19:06:16Z dast $
 *
 * \brief  Header for cnnplus::CuLeNet5.
 *
 *****************************************************************************/

#ifndef CNNPLUS_CULENET5_HH
#define CNNPLUS_CULENET5_HH

#include "common.hh"
#include "cuneuralnet.hh"
#include "cufulllayer.hh"
#include "cuconvlayer.hh"
#include "cusublayer.hh"

CNNPLUS_NS_BEGIN

//! LeNet-5 type convolutional neural network, without the extra RBF layer
/*! \see Y. LeCun, L. Bottou, Y. Bengio and P. Haffner: Gradient-Based Learning
         Applied to Document Recognition, Proceedings of the IEEE,
         86(11):2278-2324, November 1998.
 */
template<typename T>
class CuLeNet5 : public CuNeuralNet<T>
{
public:
    //! Ctr
    CuLeNet5(Size const & sizeImgIn = Size(32, 32));
    //! Dtr
    virtual ~CuLeNet5();
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

#endif // CNNPLUS_CULENET5_HH
