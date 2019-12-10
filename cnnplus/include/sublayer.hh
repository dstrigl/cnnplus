/**************************************************************************//**
 *
 * \file   sublayer.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Feb 23 2009
 *
 * $Id: sublayer.hh 1905 2009-07-30 20:21:22Z dast $
 *
 * \brief  Header for cnnplus::SubLayer.
 *
 *****************************************************************************/

#ifndef CNNPLUS_SUBLAYER_HH
#define CNNPLUS_SUBLAYER_HH

#include "common.hh"
#include "layer.hh"
#include "squasher.hh"

CNNPLUS_NS_BEGIN

//! Subsampling layer
template< typename T, class SquFnc = Tanh<T> >
class SubLayer : public Layer<T>
{
public:
    //! Ctr
    /*! \param sizeMapsIn size of input feature maps
        \param numMaps number of feature maps
        \param sizeSample sample size
     */
    SubLayer(Size const & sizeMapsIn, size_t numMaps,
        Size const & sizeSample = Size(2, 2));
    //! Dtr
    virtual ~SubLayer();

    virtual void forget(T sigma, bool scale = false);
    virtual void forget();
    virtual void reset();
    virtual void update(T eta);
    virtual void fprop(T const * in, size_t strideIn,
                       T * out, size_t strideOut);
    virtual void bprop(T * in, size_t strideIn,
                       T const * out, size_t strideOut,
                       bool accumGradients = false);
#ifdef CNNPLUS_MATLAB_FOUND
    virtual void load(mxArray const * arr);
    virtual mxArray * save() const;
    virtual mxArray * writeOut(T * const out, size_t strideOut) const;
#endif // CNNPLUS_MATLAB_FOUND
    virtual Size sizeIn() const { return Size(numMaps_, sizeMapsIn_.area()); }
    virtual Size sizeOut() const { return Size(numMaps_, sizeMapsOut_.area()); }
    virtual void trainableParam(typename Layer<T>::TrainableParam & param);
    virtual std::string toString() const;
    virtual size_t numTrainableParam() const;
    virtual size_t numConnections() const;

    //! Returns the size of the input feature maps
    Size sizeMapsIn() const { return sizeMapsIn_; }
    //! Returns the size of the output feature maps
    Size sizeMapsOut() const { return sizeMapsOut_; }
    //! Returns the number of feature maps
    size_t numMaps() const { return numMaps_; }
    //! Returns the sample size
    Size sizeSample() const { return sizeSample_; }

private:
    Size const sizeMapsIn_;
    size_t const numMaps_;
    Size const sizeSample_;
    Size const sizeMapsOut_;

    SquFnc squasher_;

    T * inSat_;
    T * inSub_;
    size_t strideInSub_;
    T * tmp_;
    T * weights_;
    T * dWeights_;
    T * biases_;
    T * dBiases_;
    T * sum_;
    size_t strideSum_;
    T * delta_;
    size_t strideDelta_;
};

CNNPLUS_NS_END

#endif // CNNPLUS_SUBLAYER_HH
