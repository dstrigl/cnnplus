/**************************************************************************//**
 *
 * \file   rproptrainer.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jul 14 2009
 *
 * $Id: rproptrainer.cc 2725 2009-11-27 10:04:45Z dast $
 *
 * \brief  Implementation of cnnplus::RpropTrainer.
 *
 *****************************************************************************/

#include "rproptrainer.hh"
#include "neuralnet.hh"
#include "datasource.hh"
#include "error.hh"
#include "mathli.hh"
#include "matvecli.hh"
#include <sstream>
#include <cstdio>

CNNPLUS_NS_BEGIN

template<typename T, class ErrFnc> bool
RpropTrainer<T, ErrFnc>::Adapter::update(
    RpropTrainer<T, ErrFnc> & trainer, DataSource<T> & ds, size_t const epoch)
{
    double errRate = 0;
    T const error = trainer.test(ds, errRate);

    printf("# epoch: %04d, error: %10.8f, error-rate: %8.4f%%\n",
        epoch, error, errRate);
    fflush(stdout);

    return (epoch < maxEpochs_);
}

template<typename T, class ErrFnc> void
RpropTrainer<T, ErrFnc>::init()
{
    CNNPLUS_ASSERT(trainableParam_.empty());
    this->net_.trainableParam(trainableParam_);
    weightsParam_.resize(trainableParam_.size());
    biasesParam_.resize(trainableParam_.size());

    for (size_t i = 0; i < trainableParam_.size(); ++i)
    {
        //
        // Weights
        //
        size_t const rows = trainableParam_[i].weights.rows;
        size_t const cols = trainableParam_[i].weights.cols;

        weightsParam_[i].stepSize = matvecli::allocm<T>(
            rows, cols, weightsParam_[i].strideStepSize);
        weightsParam_[i].updateVal = matvecli::allocm<T>(
            rows, cols, weightsParam_[i].strideUpdateVal);
        weightsParam_[i].dVal = matvecli::allocm<T>(
            rows, cols, weightsParam_[i].strideDVal);

        matvecli::setm<T>(weightsParam_[i].stepSize,
            weightsParam_[i].strideStepSize, rows, cols, stepSizeInit_);
        matvecli::zerom<T>(weightsParam_[i].updateVal,
            weightsParam_[i].strideUpdateVal, rows, cols);
        matvecli::zerom<T>(weightsParam_[i].dVal,
            weightsParam_[i].strideDVal, rows, cols);

        //
        // Biases
        //
        size_t const len = trainableParam_[i].biases.len;

        biasesParam_[i].stepSize = matvecli::allocv<T>(len);
        biasesParam_[i].updateVal = matvecli::allocv<T>(len);
        biasesParam_[i].dVal = matvecli::allocv<T>(len);

        matvecli::setv<T>(biasesParam_[i].stepSize, len, stepSizeInit_);
        matvecli::zerov<T>(biasesParam_[i].updateVal, len);
        matvecli::zerov<T>(biasesParam_[i].dVal, len);
    }
}

/*! \todo optimize
 */
template<typename T, class ErrFnc>
void RpropTrainer<T, ErrFnc>::updateVal(
    T & val, T & dValCurr, T & dValPrev, T & stepSize, T & updateVal,
    T const errCurr, T const errPrev)
{
#if 0
    /* C. Igel, M. Husken, Empirical evaluation of the improved Rprop
     * learning algorithms, Neurocomputing 50 (2003) 105–123.
     */
    if (dValPrev * dValCurr > 0)
    {
        stepSize = mathli::min<T>(stepSize * etap_, stepSizeMax_);
        updateVal = -mathli::sign<T>(dValCurr) * stepSize;
        val += updateVal;
    }
    else if (dValPrev * dValCurr < 0)
    {
        stepSize = mathli::max<T>(stepSize * etam_, stepSizeMin_);
        if (errCurr > errPrev) val -= updateVal;
        dValCurr = 0;
    }
    else // if (dValPrev * dValCurr == 0)
    {
        updateVal = -mathli::sign<T>(dValCurr) * stepSize;
        val += updateVal;
    }
#else
    /* S. L. Phung and A. Bouzerdoum, "MATLAB Library for Convolutional
     * Neural Network", Technical Report, ICT Research Institute, Visual
     * and Audio Processing Laboratory, University of Wollongong.
     * Available at: http://www.elec.uow.edu.au/staff/sphung.
     */
    int const sign = mathli::sign<T>(dValPrev) * mathli::sign<T>(dValCurr);
    stepSize = stepSize * ((sign ==  1) * etap_ +
                           (sign == -1) * etam_ +
                           (sign ==  0));
    //stepSize = (sign ==  1) * mathli::min<T>(stepSize * etap_, stepSizeMax_)
    //         + (sign == -1) * mathli::max<T>(stepSize * etam_, stepSizeMin_)
    //         + (sign ==  0) * stepSize;
    val = val - mathli::sign<T>(dValCurr) * stepSize;
#endif

    dValPrev = dValCurr;
}

/*! \todo optimize
 */
template<typename T, class ErrFnc> void
RpropTrainer<T, ErrFnc>::update(T const errCur, T & errPrev)
{
    for (size_t i = 0; i < trainableParam_.size(); ++i)
    {
        //
        // Weights
        //
        for (size_t r = 0; r < trainableParam_[i].weights.rows; ++r)
        {
            for (size_t c = 0; c < trainableParam_[i].weights.cols; ++c)
            {
                if (trainableParam_[i].weights.mask &&
                    !trainableParam_[i].weights.mask[r * trainableParam_[i].weights.strideMask + c])
                    continue;

                updateVal(
                    trainableParam_[i].weights.val[r * trainableParam_[i].weights.strideVal + c],
                    trainableParam_[i].weights.dVal[r * trainableParam_[i].weights.strideDVal + c],
                    weightsParam_[i].dVal[r * weightsParam_[i].strideDVal + c],
                    weightsParam_[i].stepSize[r * weightsParam_[i].strideStepSize + c],
                    weightsParam_[i].updateVal[r * weightsParam_[i].strideUpdateVal + c],
                    errCur, errPrev);
            }
        }
        //
        // Biases
        //
        for (size_t j = 0; j < trainableParam_[i].biases.len; ++j)
        {
            updateVal(
                trainableParam_[i].biases.val[j],
                trainableParam_[i].biases.dVal[j],
                biasesParam_[i].dVal[j],
                biasesParam_[i].stepSize[j],
                biasesParam_[i].updateVal[j],
                errCur, errPrev);
        }
    }

    errPrev = errCur;
}

/*! \todo more checks of ctr-args
 */
template<typename T, class ErrFnc>
RpropTrainer<T, ErrFnc>::RpropTrainer(NeuralNet<T> & net,
    T const etap, T const etam, T const stepSizeInit,
    T const stepSizeMin, T const stepSizeMax, Adapter & adapter)
    : Trainer<T, ErrFnc>(net), adapter_(&adapter), delAdapter_(false),
    etap_(etap), etam_(etam), stepSizeInit_(stepSizeInit),
    stepSizeMin_(stepSizeMin), stepSizeMax_(stepSizeMax)
{
    if (etap <= 1)
        throw ParameterError("etap", "must be greater one.");
    else if (etam <= 0 || etam >= 1)
        throw ParameterError("etam", "must be between 0 and 1.");

    init();
}

/*! \todo more checks of ctr-args
 */
template<typename T, class ErrFnc>
RpropTrainer<T, ErrFnc>::RpropTrainer(NeuralNet<T> & net,
    T const etap, T const etam, T const stepSizeInit,
    T const stepSizeMin, T const stepSizeMax, size_t const maxEpochs)
    : Trainer<T, ErrFnc>(net), adapter_(new Adapter(maxEpochs)), delAdapter_(true),
    etap_(etap), etam_(etam), stepSizeInit_(stepSizeInit),
    stepSizeMin_(stepSizeMin), stepSizeMax_(stepSizeMax)
{
    if (etap <= 1)
        throw ParameterError("etap", "must be greater one.");
    else if (etam <= 0 || etam >= 1)
        throw ParameterError("etam", "must be between 0 and 1.");

    init();
}

template<typename T, class ErrFnc>
RpropTrainer<T, ErrFnc>::~RpropTrainer()
{
    if (delAdapter_) delete adapter_;

    for (size_t i = 0; i < trainableParam_.size(); ++i)
    {
        // Weights
        matvecli::free<T>(weightsParam_[i].stepSize);
        matvecli::free<T>(weightsParam_[i].updateVal);
        matvecli::free<T>(weightsParam_[i].dVal);

        // Biases
        matvecli::free<T>(biasesParam_[i].stepSize);
        matvecli::free<T>(biasesParam_[i].updateVal);
        matvecli::free<T>(biasesParam_[i].dVal);
    }
}

template<typename T, class ErrFnc> void
RpropTrainer<T, ErrFnc>::train(DataSource<T> & ds)
{
    if (ds.sizeOut() != this->net_.sizeIn())
        throw ParameterError("ds", "size doesn't match.");

    // Reset gradients to zero
    this->net_.reset();

    // Set vector 'des_' to negative target value
    if (this->net_.sizeOut() > 1)
        matvecli::setv<T>(this->des_, this->net_.sizeOut(), this->targetVal_.NEG());

    // Loop over epochs until 'adapter_->update(...)' returns 'false'
    for (size_t epoch = 0; adapter_->update(*this, ds, epoch); ++epoch)
    {
        ds.shuffle();
        ds.rewind();
        T error = 0, errPrev = 0;

        // Loop over all patterns
        for (int i = 1; i <= ds.size(); ds.next(), ++i)
        {
            // Read pattern and desired label from data source
            int const desLbl = ds.fprop(this->in_);

            // Compute neural-net output
            this->net_.fprop(this->in_, this->out_);

            // Compute error
            if (this->net_.sizeOut() > 1) {
                CNNPLUS_ASSERT(0 <= desLbl && desLbl < static_cast<int>(this->net_.sizeOut()));
                this->des_[desLbl] = this->targetVal_.POS();
                error += this->errFnc_.fprop(this->out_, this->des_);
                this->des_[desLbl] = this->targetVal_.NEG();
            }
            else {
                this->des_[0] = static_cast<T>(desLbl);
                error += this->errFnc_.fprop(this->out_, this->des_);
            }

            // Backpropagate error through network
            this->errFnc_.bprop(this->out_);
            this->net_.bprop(NULL, this->out_, true);

#ifdef CNNPLUS_PRINT_PROGRESS
            printf("train %.2f%%\r", i * 100.0 / ds.size());
            fflush(stdout);
#endif // CNNPLUS_PRINT_PROGRESS
        }

        // Updates weights and biases and reset gradients
        update(normErrVal(error, ds.size()), errPrev);
        this->net_.reset();

#ifdef CNNPLUS_PRINT_PROGRESS
        printf("             \r");
        fflush(stdout);
#endif // CNNPLUS_PRINT_PROGRESS
    }
}

template<typename T, class ErrFnc> std::string
RpropTrainer<T, ErrFnc>::toString() const
{
    std::stringstream ss;
    ss << "RpropTrainer["
        << this->errFnc_.toString()
        << "; targetVal=("
        << this->targetVal_.NEG() << ","
        << this->targetVal_.POS() << ")"
        << ", etap=" << etap_
        << ", etam=" << etam_
        << ", stepSizeInit=" << stepSizeInit_
        << ", stepSizeMin=" << stepSizeMin_
        << ", stepSizeMax=" << stepSizeMax_ << "]";
    return ss.str();
}

/*! \addtogroup eti_grp Explicit Template Instantiation
 @{
 */
template class RpropTrainer< float,  MeanSquaredError<float>  >;
template class RpropTrainer< double, MeanSquaredError<double> >;
template class RpropTrainer< float,  CrossEntropy<float>      >;
template class RpropTrainer< double, CrossEntropy<double>     >;
/*! @} */

CNNPLUS_NS_END
