/**************************************************************************//**
 *
 * \file   cufulllayer.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jun 05 2009
 *
 * $Id: cufulllayer.hh 1904 2009-07-30 19:29:56Z dast $
 *
 * \brief  Header for cnnplus::CuFullLayer.
 *
 *****************************************************************************/

#ifndef CNNPLUS_CUFULLLAYER_HH
#define CNNPLUS_CUFULLLAYER_HH

#include "common.hh"
#include "culayer.hh"
#include "cusquasher.hh"

CNNPLUS_NS_BEGIN

//! Fully connected layer
template< typename T, class SquFnc = CuTanh<T> >
class CuFullLayer : public CuLayer<T>
{
public:
    //! Ctr
    /*! \param sizeIn input size
        \param sizeOut output size
     */
    CuFullLayer(Size const & sizeIn, Size const & sizeOut);
    //! Dtr
    virtual ~CuFullLayer();

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
#endif // CNNPLUS_MATLAB_FOUND
    virtual Size sizeIn() const { return sizeIn_; }
    virtual Size sizeOut() const { return sizeOut_; }
    virtual void trainableParam(typename Layer<T>::TrainableParam & param);
    virtual std::string toString() const;
    virtual size_t numTrainableParam() const;
    virtual size_t numConnections() const;

private:
    Size const sizeIn_;
    Size const sizeOut_;

    SquFnc squasher_;

    // GPU
    T * d_in_;
    T * d_weights_;
    size_t d_strideWeights_;
    T * d_dWeights_;
    size_t d_strideDWeights_;
    T * d_biases_;
    T * d_dBiases_;
    T * d_sum_;
    T * d_delta_;
    T * d_tmp_;

    // CPU
    T * h_weights_;
    size_t h_strideWeights_;
    T * h_biases_;
};

CNNPLUS_NS_END

#endif // CNNPLUS_CUFULLLAYER_HH
