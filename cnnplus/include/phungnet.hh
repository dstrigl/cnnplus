/**************************************************************************//**
 *
 * \file   phungnet.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jul 15 2009
 *
 * $Id: phungnet.hh 1835 2009-07-21 08:03:24Z dast $
 *
 * \brief  Header for cnnplus::PhungNet.
 *
 *****************************************************************************/

#ifndef CNNPLUS_PHUNGNET_HH
#define CNNPLUS_PHUNGNET_HH

#include "common.hh"
#include "neuralnet.hh"
#include "fulllayer.hh"
#include "convlayer.hh"
#include "sublayer.hh"

CNNPLUS_NS_BEGIN

//! A 6-layer convolutional neural network, as recommended by Son Lam Phung
/*! \see S. L. Phung and A. Bouzerdoum, "MATLAB Library for Convolutional
         Neural Network", Technical Report, ICT Research Institute, Visual
         and Audio Processing Laboratory, University of Wollongong.
         Available at: http://www.elec.uow.edu.au/staff/sphung.
 */
template<typename T>
class PhungNet : public NeuralNet<T>
{
public:
    //! Ctr
    PhungNet();
    //! Dtr
    virtual ~PhungNet();
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
    ConvLayer< T, Tanh<T>     > layer1_;
    SubLayer < T, Identity<T> > layer2_;
    ConvLayer< T, Tanh<T>     > layer3_;
    SubLayer < T, Identity<T> > layer4_;
    ConvLayer< T, Tanh<T>     > layer5_;
    FullLayer< T, Tanh<T>     > layer6_;

    T * stateL1_; size_t strideL1_;
    T * stateL2_; size_t strideL2_;
    T * stateL3_; size_t strideL3_;
    T * stateL4_; size_t strideL4_;
    T * stateL5_; size_t strideL5_;
};

CNNPLUS_NS_END

#endif // CNNPLUS_PHUNGNET_HH
