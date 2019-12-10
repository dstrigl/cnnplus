/**************************************************************************//**
 *
 * \file   cusimardnet.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jun 30 2009
 *
 * $Id: cusimardnet.hh 1835 2009-07-21 08:03:24Z dast $
 *
 * \brief  Header for cnnplus::CuSimardNet.
 *
 *****************************************************************************/

#ifndef CNNPLUS_CUSIMARDNET_HH
#define CNNPLUS_CUSIMARDNET_HH

#include "common.hh"
#include "cuneuralnet.hh"
#include "cufulllayer.hh"
#include "cuconvlayer.hh"

CNNPLUS_NS_BEGIN

//! A 4-layer convolutional neural network, as recommended by P. Y. Simard et al.
/*! \see
    \li P. Y. Simard, D. Steinkraus, & J. Platt, "Best Practice for
        Convolutional Neural Networks Applied to Visual Document Analysis",
        International Conference on Document Analysis and Recognition (ICDAR),
        IEEE Computer Society, Los Alamitos, 2003, pp. 958-962.
    \li Chellapilla K., Puri S., Simard P., "High Performance Convolutional
        Neural Networks for Document Processing", 10th International Workshop
        on Frontiers in Handwriting Recognition (IWFHR'2006) will be held in
        La Baule, France on October 23-26, 2006.
 */
template<typename T>
class CuSimardNet : public CuNeuralNet<T>
{
public:
    //! Ctr
    /*! \param sizeImgIn input image size (default is 29x29)
        \param numMapsL1 number of feature maps in layer 1
        \param numMapsL2 number of feature maps in layer 2
        \param sizeL3 size of layer 3
     */
    CuSimardNet(Size const & sizeImgIn = Size(29, 29),
                size_t numMapsL1 = 5,
                size_t numMapsL2 = 50,
                size_t sizeL3 = 100);
    //! Dtr
    virtual ~CuSimardNet();
    virtual void forget(T sigma, bool scale = false);
    virtual void forget();
    virtual void reset();
    virtual void update(T eta);
    virtual void fprop(T const * in, T * out);
    virtual void bprop(T * in, T const * out, bool accumGradients = false);
    virtual void load(std::string const & filename);
    virtual void save(std::string const & filename) const;
    virtual size_t sizeIn() const { return layer1_.sizeIn().area(); }
    virtual size_t sizeOut() const { return layer4_.sizeOut().area(); }
    virtual void trainableParam(typename NeuralNet<T>::TrainableParam & param);
    virtual std::string toString() const;
    virtual size_t numTrainableParam() const;
    virtual size_t numConnections() const;

private:
    CuConvLayer< T, CuTanh<T> > layer1_;
    CuConvLayer< T, CuTanh<T> > layer2_;
    CuFullLayer< T, CuTanh<T> > layer3_;
    CuFullLayer< T, CuTanh<T> > layer4_;

    T * stateL1_; size_t strideL1_;
    T * stateL2_; size_t strideL2_;
    T * stateL3_; size_t strideL3_;
    T * in_, * out_;
};

CNNPLUS_NS_END

#endif // CNNPLUS_CUSIMARDNET_HH