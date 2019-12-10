/**************************************************************************//**
 *
 * \file   cumomentumtrainer.cu
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jul 13 2009
 *
 * $Id: cumomentumtrainer.cu 3558 2010-11-22 11:04:51Z klaus $
 *
 * \brief  Implementation of cnnplus::CuMomentumTrainer.
 *
 *****************************************************************************/

#include "cudautils.hh"

///////////////////////////////////////////////////////////////////////////////
// CUDA kernels

__global__ void
updateW_kernel(float        * const weights,  size_t const strideWeights,
               float  const * const dWeights, size_t const strideDWeights,
               float  const * const mask,     size_t const strideMask,
               float        * const deltaW,   size_t const strideDeltaW,
               size_t const rows, size_t const cols,
               float  const eta,  float  const alpha)
{
    size_t const r = CNN_UIMUL(blockIdx.y, blockDim.y) + threadIdx.y;
    size_t const c = CNN_UIMUL(blockIdx.x, blockDim.x) + threadIdx.x;

    if (r >= rows || c >= cols)
        return;

    // Compute: deltaW = -eta * dWeights + alpha * deltaW
    deltaW[CNN_UIMUL(r, strideDeltaW) + c] =
         -eta * dWeights[CNN_UIMUL(r, strideDWeights) + c] +
        alpha * deltaW  [CNN_UIMUL(r, strideDeltaW  ) + c];
    // Compute: deltaW = deltaW .* mask
    if (mask)
        deltaW[CNN_UIMUL(r, strideDeltaW) + c] *= mask[CNN_UIMUL(r, strideMask) + c];
    // Compute: weights = weights + deltaW
    weights[CNN_UIMUL(r, strideWeights) + c] += deltaW[CNN_UIMUL(r, strideDeltaW) + c];
}

__global__ void
updateB_kernel(float        * const biases,
               float  const * const dBiases,
               float        * const deltaB,
               size_t const len,
               float  const eta, float const alpha)
{
    size_t const i = CNN_UIMUL(blockIdx.x, blockDim.x) + threadIdx.x;

    if (i >= len)
        return;

    // Compute: deltaB = -eta * dBiases + alpha * deltaB
    deltaB[i] = -eta * dBiases[i] + alpha * deltaB[i];
    // Compute: biases = biases + deltaB
    biases[i] += deltaB[i];
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#include "cumomentumtrainer.hh"
#include "cuneuralnet.hh"
#include "datasource.hh"
#include "error.hh"
#include "matvecli.hh"
#include "cumvli.hh"
#include <sstream>
#include <cstdio>

CNNPLUS_NS_BEGIN

///////////////////////////////////////////////////////////////////////////////
// CUDA kernel calls

template<typename T> void
update(typename NeuralNet<T>::TrainableParam const & trainableParam,
       std::vector<T *> const & deltaW,
       std::vector<size_t> const & strideDeltaW,
       std::vector<T *> const & deltaB,
       T const eta, T const alpha);

template<> void
update<float>(NeuralNet<float>::TrainableParam const & trainableParam,
              std::vector<float *> const & deltaW,
              std::vector<size_t> const & strideDeltaW,
              std::vector<float *> const & deltaB,
              float const eta, float const alpha)
{
    for (size_t i = 0; i < trainableParam.size(); ++i)
    {
        //
        // Update weights
        //
        {
            size_t const rows = trainableParam[i].weights.rows;
            size_t const cols = trainableParam[i].weights.cols;

            float       * const weights  = trainableParam[i].weights.val;
            float const * const dWeights = trainableParam[i].weights.dVal;
            float const * const mask     = trainableParam[i].weights.mask;

            size_t const strideWeights  = trainableParam[i].weights.strideVal;
            size_t const strideDWeights = trainableParam[i].weights.strideDVal;
            size_t const strideMask     = trainableParam[i].weights.strideMask;

            dim3 const dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
            dim3 const dimGrid((cols + dimBlock.x - 1) / dimBlock.x,
                               (rows + dimBlock.y - 1) / dimBlock.y);
            updateW_kernel<<<dimGrid, dimBlock>>>(
                weights, strideWeights, dWeights, strideDWeights,
                mask, strideMask, deltaW[i], strideDeltaW[i],
                rows, cols, eta, alpha);
            CUDA_CHECK_ERROR("Kernel call 'updateW_kernel' failed");
        }
        //
        // Update biases
        //
        {
            size_t        const len     = trainableParam[i].biases.len;
            float       * const biases  = trainableParam[i].biases.val;
            float const * const dBiases = trainableParam[i].biases.dVal;

            updateB_kernel<<<(len + THREADS - 1) / THREADS, THREADS>>>
                (biases, dBiases, deltaB[i], len, eta, alpha);
            CUDA_CHECK_ERROR("Kernel call 'updateB_kernel' failed");
        }
    }
}

template<> void
update<double>(NeuralNet<double>::TrainableParam const & trainableParam,
               std::vector<double *> const & deltaW,
               std::vector<size_t> const & strideDeltaW,
               std::vector<double *> const & deltaB,
               double const eta, double const alpha)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

///////////////////////////////////////////////////////////////////////////////
// CuMomentumTrainer implementation

template<typename T, class ErrFnc> bool
CuMomentumTrainer<T, ErrFnc>::Adapter::update(CuMomentumTrainer<T, ErrFnc> & trainer,
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
CuMomentumTrainer<T, ErrFnc>::init()
{
    CNNPLUS_ASSERT(trainableParam_.empty());
    this->net_.trainableParam(trainableParam_);
    deltaW_.resize(trainableParam_.size());
    strideDeltaW_.resize(trainableParam_.size());
    deltaB_.resize(trainableParam_.size());

    for (size_t i = 0; i < trainableParam_.size(); ++i)
    {
        // Weights
        deltaW_[i] = cumvli::allocm<T>(
            trainableParam_[i].weights.rows,
            trainableParam_[i].weights.cols,
            strideDeltaW_[i]);
        cumvli::zerom<T>(
            deltaW_[i], strideDeltaW_[i],
            trainableParam_[i].weights.rows,
            trainableParam_[i].weights.cols);

        // Biases
        deltaB_[i] = cumvli::allocv<T>(
            trainableParam_[i].biases.len);
        cumvli::zerov<T>(
            deltaB_[i], trainableParam_[i].biases.len);
    }
}

template<typename T, class ErrFnc>
CuMomentumTrainer<T, ErrFnc>::CuMomentumTrainer(
    NeuralNet<T> & net, int const batchSize, T const eta, T const alpha, Adapter & adapter)
    : Trainer<T, ErrFnc>(net), adapter_(&adapter), delAdapter_(false),
    batchSize_(batchSize), eta_(eta), alpha_(alpha)
{
    if (!dynamic_cast<CuNeuralNet<T>*>(&net))
        throw ParameterError("net", "no CUDA enabled neural-net.");
    else if (batchSize <= 0)
        throw ParameterError("batchSize", "must be greater zero.");
    else if (eta <= 0)
        throw ParameterError("eta", "must be greater zero.");
    else if (alpha < 0 || alpha > 1)
        throw ParameterError("alpha", "must be between 0 and 1.");

    init();
}

template<typename T, class ErrFnc>
CuMomentumTrainer<T, ErrFnc>::CuMomentumTrainer(
    NeuralNet<T> & net, int const batchSize, T const eta, T const alpha, size_t const maxEpochs)
    : Trainer<T, ErrFnc>(net), adapter_(new Adapter(maxEpochs)), delAdapter_(true),
    batchSize_(batchSize), eta_(eta), alpha_(alpha)
{
    if (!dynamic_cast<CuNeuralNet<T>*>(&net))
        throw ParameterError("net", "no CUDA enabled neural-net.");
    else if (batchSize <= 0)
        throw ParameterError("batchSize", "must be greater zero.");
    else if (eta <= 0)
        throw ParameterError("eta", "must be greater zero.");
    else if (alpha < 0 || alpha > 1)
        throw ParameterError("alpha", "must be between 0 and 1.");

    init();
}

template<typename T, class ErrFnc>
CuMomentumTrainer<T, ErrFnc>::CuMomentumTrainer(
    NeuralNet<T> & net, T const eta, T const alpha, Adapter & adapter)
    : Trainer<T, ErrFnc>(net), adapter_(&adapter), delAdapter_(false),
    batchSize_(0), eta_(eta), alpha_(alpha)
{
    if (!dynamic_cast<CuNeuralNet<T>*>(&net))
        throw ParameterError("net", "no CUDA enabled neural-net.");
    else if (eta <= 0)
        throw ParameterError("eta", "must be greater zero.");
    else if (alpha < 0 || alpha > 1)
        throw ParameterError("alpha", "must be between 0 and 1.");

    init();
}

template<typename T, class ErrFnc>
CuMomentumTrainer<T, ErrFnc>::CuMomentumTrainer(
    NeuralNet<T> & net, T const eta, T const alpha, size_t const maxEpochs)
    : Trainer<T, ErrFnc>(net), adapter_(new Adapter(maxEpochs)), delAdapter_(true),
    batchSize_(0), eta_(eta), alpha_(alpha)
{
    if (!dynamic_cast<CuNeuralNet<T>*>(&net))
        throw ParameterError("net", "no CUDA enabled neural-net.");
    else if (eta <= 0)
        throw ParameterError("eta", "must be greater zero.");
    else if (alpha < 0 || alpha > 1)
        throw ParameterError("alpha", "must be between 0 and 1.");

    init();
}

template<typename T, class ErrFnc>
CuMomentumTrainer<T, ErrFnc>::~CuMomentumTrainer()
{
    if (delAdapter_) delete adapter_;

    for (size_t i = 0; i < trainableParam_.size(); ++i)
    {
        cumvli::free<T>(deltaW_[i]);
        cumvli::free<T>(deltaB_[i]);
    }
}

template<typename T, class ErrFnc> void
CuMomentumTrainer<T, ErrFnc>::train(DataSource<T> & ds)
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
                update<T>(trainableParam_, deltaW_, strideDeltaW_, deltaB_, eta_, alpha_);
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
CuMomentumTrainer<T, ErrFnc>::toString() const
{
    std::stringstream ss;
    ss << "CuMomentumTrainer["
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
template class CuMomentumTrainer< float,  MeanSquaredError<float>  >;
template class CuMomentumTrainer< double, MeanSquaredError<double> >;
template class CuMomentumTrainer< float,  CrossEntropy<float>      >;
template class CuMomentumTrainer< double, CrossEntropy<double>     >;
/*! @} */

CNNPLUS_NS_END
