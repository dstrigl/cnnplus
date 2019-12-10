/**************************************************************************//**
 *
 * \file   momentumtrainer.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   May 12 2009
 *
 * $Id: momentumtrainer.cc 1745 2009-07-14 16:12:08Z dast $
 *
 * \brief  Implementation of cnnplus::MomentumTrainer.
 *
 *****************************************************************************/

#include "momentumtrainer.hh"
#include "neuralnet.hh"
#include "datasource.hh"
#include "error.hh"
#include "matvecli.hh"
#include <sstream>
#include <cstdio>

CNNPLUS_NS_BEGIN

template<typename T, class ErrFnc> bool
MomentumTrainer<T, ErrFnc>::Adapter::update(MomentumTrainer<T, ErrFnc> & trainer,
    DataSource<T> & ds, size_t const epoch, int & batchSize, T & eta, T & alpha)
{
    double errRate = 0;
    T const error = trainer.test(ds, errRate);

    printf("# epoch: %04d, error: %10.8f, error-rate: %8.4f%%\n",
        epoch, error, errRate);
    fflush(stdout);

    return (epoch < maxEpochs_);
}

template<typename T, class ErrFnc> void
MomentumTrainer<T, ErrFnc>::init()
{
    CNNPLUS_ASSERT(trainableParam_.empty());
    this->net_.trainableParam(trainableParam_);
    deltaW_.resize(trainableParam_.size());
    strideDeltaW_.resize(trainableParam_.size());
    deltaB_.resize(trainableParam_.size());

    for (size_t i = 0; i < trainableParam_.size(); ++i)
    {
        // Weights
        deltaW_[i] = matvecli::allocm<T>(
            trainableParam_[i].weights.rows,
            trainableParam_[i].weights.cols,
            strideDeltaW_[i]);
        matvecli::zerom<T>(
            deltaW_[i], strideDeltaW_[i],
            trainableParam_[i].weights.rows,
            trainableParam_[i].weights.cols);

        // Biases
        deltaB_[i] = matvecli::allocv<T>(
            trainableParam_[i].biases.len);
        matvecli::zerov<T>(
            deltaB_[i], trainableParam_[i].biases.len);
    }
}

template<typename T, class ErrFnc> void
MomentumTrainer<T, ErrFnc>::update()
{
    for (size_t i = 0; i < trainableParam_.size(); ++i)
    {
        //
        // Update weights
        //
        size_t const rows = trainableParam_[i].weights.rows;
        size_t const cols = trainableParam_[i].weights.cols;

        T       * const weights  = trainableParam_[i].weights.val;
        T const * const dWeights = trainableParam_[i].weights.dVal;
        T const * const mask     = trainableParam_[i].weights.mask;

        size_t const strideWeights  = trainableParam_[i].weights.strideVal;
        size_t const strideDWeights = trainableParam_[i].weights.strideDVal;
        size_t const strideMask     = trainableParam_[i].weights.strideMask;

        for (size_t r = 0; r < rows; ++r)
        {
            // Compute: deltaW_ = deltaW_ * alpha_
            matvecli::mulvc<T>(deltaW_[i] + r * strideDeltaW_[i], alpha_, cols);
            // Compute: deltaW_ = -eta_ * dWeights + deltaW_
            matvecli::axpy<T>(dWeights + r * strideDWeights, cols, deltaW_[i] + r * strideDeltaW_[i], -eta_);
            // Compute: deltaW_ = deltaW_ .* mask
            if (mask) matvecli::mulv<T>(deltaW_[i] + r * strideDeltaW_[i], mask + r * strideMask, cols);
            // Compute: weights = weights + deltaW_
            matvecli::addv<T>(weights + r * strideWeights, deltaW_[i] + r * strideDeltaW_[i], cols);
        }

        //
        // Update biases
        //
        size_t    const len     = trainableParam_[i].biases.len;
        T       * const biases  = trainableParam_[i].biases.val;
        T const * const dBiases = trainableParam_[i].biases.dVal;

        // Compute: deltaB_ = deltaB_ * alpha_
        matvecli::mulvc<T>(deltaB_[i], alpha_, len);
        // Compute: deltaB_ = -eta_ * dBiases + deltaB_
        matvecli::axpy<T>(dBiases, len, deltaB_[i], -eta_);
        // Compute: biases = biases + deltaB_
        matvecli::addv<T>(biases, deltaB_[i], len);
    }
}

template<typename T, class ErrFnc>
MomentumTrainer<T, ErrFnc>::MomentumTrainer(
    NeuralNet<T> & net, int const batchSize, T const eta, T const alpha, Adapter & adapter)
    : Trainer<T, ErrFnc>(net), adapter_(&adapter), delAdapter_(false),
    batchSize_(batchSize), eta_(eta), alpha_(alpha)
{
    if (batchSize <= 0)
        throw ParameterError("batchSize", "must be greater zero.");
    else if (eta <= 0)
        throw ParameterError("eta", "must be greater zero.");
    else if (alpha < 0 || alpha > 1)
        throw ParameterError("alpha", "must be between 0 and 1.");

    init();
}

template<typename T, class ErrFnc>
MomentumTrainer<T, ErrFnc>::MomentumTrainer(
    NeuralNet<T> & net, int const batchSize, T const eta, T const alpha, size_t const maxEpochs)
    : Trainer<T, ErrFnc>(net), adapter_(new Adapter(maxEpochs)), delAdapter_(true),
    batchSize_(batchSize), eta_(eta), alpha_(alpha)
{
    if (batchSize <= 0)
        throw ParameterError("batchSize", "must be greater zero.");
    else if (eta <= 0)
        throw ParameterError("eta", "must be greater zero.");
    else if (alpha < 0 || alpha > 1)
        throw ParameterError("alpha", "must be between 0 and 1.");

    init();
}

template<typename T, class ErrFnc>
MomentumTrainer<T, ErrFnc>::MomentumTrainer(
    NeuralNet<T> & net, T const eta, T const alpha, Adapter & adapter)
    : Trainer<T, ErrFnc>(net), adapter_(&adapter), delAdapter_(false),
    batchSize_(0), eta_(eta), alpha_(alpha)
{
    if (eta <= 0)
        throw ParameterError("eta", "must be greater zero.");
    else if (alpha < 0 || alpha > 1)
        throw ParameterError("alpha", "must be between 0 and 1.");

    init();
}

template<typename T, class ErrFnc>
MomentumTrainer<T, ErrFnc>::MomentumTrainer(
    NeuralNet<T> & net, T const eta, T const alpha, size_t const maxEpochs)
    : Trainer<T, ErrFnc>(net), adapter_(new Adapter(maxEpochs)), delAdapter_(true),
    batchSize_(0), eta_(eta), alpha_(alpha)
{
    if (eta <= 0)
        throw ParameterError("eta", "must be greater zero.");
    else if (alpha < 0 || alpha > 1)
        throw ParameterError("alpha", "must be between 0 and 1.");

    init();
}

template<typename T, class ErrFnc>
MomentumTrainer<T, ErrFnc>::~MomentumTrainer()
{
    if (delAdapter_) delete adapter_;

    for (size_t i = 0; i < trainableParam_.size(); ++i)
    {
        matvecli::free<T>(deltaW_[i]);
        matvecli::free<T>(deltaB_[i]);
    }
}

template<typename T, class ErrFnc> void
MomentumTrainer<T, ErrFnc>::train(DataSource<T> & ds)
{
    if (ds.sizeOut() != this->net_.sizeIn())
        throw ParameterError("ds", "size doesn't match.");

    // Reset gradients to zero
    this->net_.reset();

    // Set vector 'des_' to negative target value
    if (this->net_.sizeOut() > 1)
        matvecli::setv<T>(this->des_, this->net_.sizeOut(), this->targetVal_.NEG());

    // Loop over epochs until 'adapter_->update(...)' returns 'false'
    for (size_t epoch = 0; adapter_->update(*this, ds, epoch, batchSize_, eta_, alpha_); ++epoch)
    {
        ds.shuffle();
        ds.rewind();

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
                this->errFnc_.fprop(this->out_, this->des_);
                this->des_[desLbl] = this->targetVal_.NEG();
            }
            else {
                this->des_[0] = static_cast<T>(desLbl);
                this->errFnc_.fprop(this->out_, this->des_);
            }

            // Backpropagate error through network
            this->errFnc_.bprop(this->out_);
            this->net_.bprop(NULL, this->out_, true);

            // Updates weights and biases
            if ((batchSize_ > 0 && i % batchSize_ == 0) || (i == ds.size())) {
                update();
                this->net_.reset();
            }

#ifdef CNNPLUS_PRINT_PROGRESS
            printf("train %.2f%%\r", i * 100.0 / ds.size());
            fflush(stdout);
#endif // CNNPLUS_PRINT_PROGRESS
        }
#ifdef CNNPLUS_PRINT_PROGRESS
        printf("             \r");
        fflush(stdout);
#endif // CNNPLUS_PRINT_PROGRESS
    }
}

template<typename T, class ErrFnc> std::string
MomentumTrainer<T, ErrFnc>::toString() const
{
    std::stringstream ss;
    ss << "MomentumTrainer["
        << this->errFnc_.toString()
        << "; targetVal=("
        << this->targetVal_.NEG() << ","
        << this->targetVal_.POS() << ")"
        << ", batchSize=" << batchSize_
        << ", eta=" << eta_
        << ", alpha=" << alpha_ << "]";
    return ss.str();
}

/*! \addtogroup eti_grp Explicit Template Instantiation
 @{
 */
template class MomentumTrainer< float,  MeanSquaredError<float>  >;
template class MomentumTrainer< double, MeanSquaredError<double> >;
template class MomentumTrainer< float,  CrossEntropy<float>      >;
template class MomentumTrainer< double, CrossEntropy<double>     >;
/*! @} */

CNNPLUS_NS_END
