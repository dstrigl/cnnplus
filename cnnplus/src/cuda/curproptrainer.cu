/**************************************************************************//**
 *
 * \file   curproptrainer.cu
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jul 20 2009
 *
 * $Id: curproptrainer.cu 3558 2010-11-22 11:04:51Z klaus $
 *
 * \brief  Implementation of cnnplus::CuRpropTrainer.
 *
 *****************************************************************************/

#include "cudautils.hh"

///////////////////////////////////////////////////////////////////////////////
// CUDA kernels

__global__ void
updateW_kernel(float        * const weights,      size_t const strideWeights,
               float  const * const mask,         size_t const strideMask,
               float        * const dWeightsCurr, size_t const strideDWeightsCurr,
               float        * const dWeightsPrev, size_t const strideDWeightsPrev,
               float        * const stepSize,     size_t const strideStepSize,
               float        * const updateVal,    size_t const strideUpdateVal,
               float  const etap,                 float  const etam,
               float  const stepSizeMin,          float  const stepSizeMax,
               float  const errCurr,              float  const errPrev,
               size_t const rows,                 size_t const cols)
{
    size_t const r = CNN_UIMUL(blockIdx.y, blockDim.y) + threadIdx.y;
    size_t const c = CNN_UIMUL(blockIdx.x, blockDim.x) + threadIdx.x;

    if (r >= rows || c >= cols)
        return;
    if (mask && !mask[CNN_UIMUL(r, strideMask) + c])
        return;

    /* S. L. Phung and A. Bouzerdoum, "MATLAB Library for Convolutional
     * Neural Network", Technical Report, ICT Research Institute, Visual
     * and Audio Processing Laboratory, University of Wollongong.
     * Available at: http://www.elec.uow.edu.au/staff/sphung.
     */
    float const SIGN[] = { -1, 1 };

    float & dW_prev = dWeightsPrev[CNN_UIMUL(r, strideDWeightsPrev) + c];
    float & dW_curr = dWeightsCurr[CNN_UIMUL(r, strideDWeightsCurr) + c];
    float & step_sz = stepSize[CNN_UIMUL(r, strideStepSize) + c];

    float const sign = dW_prev * dW_curr;
    step_sz = step_sz * ((sign >  0) * etap +
                         (sign <  0) * etam +
                         (sign == 0));
    //step_sz = (sign >  0) * fminf(step_sz * etap, stepSizeMax)
    //        + (sign <  0) * fmaxf(step_sz * etam, stepSizeMin)
    //        + (sign == 0) * step_sz;
    weights[CNN_UIMUL(r, strideWeights) + c] -=
        (dW_curr != 0) * SIGN[dW_curr >= 0] * step_sz;

    dW_prev = dW_curr;
}

__global__ void
updateB_kernel(float * const biases,
               float * const dBiasesCurr,
               float * const dBiasesPrev,
               float * const stepSize,
               float * const updateVal,
               float   const etap,
               float   const etam,
               float   const stepSizeMin,
               float   const stepSizeMax,
               float   const errCurr,
               float   const errPrev,
               size_t  const len)
{
    size_t const i = CNN_UIMUL(blockIdx.x, blockDim.x) + threadIdx.x;

    if (i >= len)
        return;

    /* S. L. Phung and A. Bouzerdoum, "MATLAB Library for Convolutional
     * Neural Network", Technical Report, ICT Research Institute, Visual
     * and Audio Processing Laboratory, University of Wollongong.
     * Available at: http://www.elec.uow.edu.au/staff/sphung.
     */
    float const SIGN[] = { -1, 1 };

    float & dB_prev = dBiasesPrev[i];
    float & dB_curr = dBiasesCurr[i];
    float & step_sz = stepSize[i];

    float const sign = dB_prev * dB_curr;
    step_sz = step_sz * ((sign >  0) * etap +
                         (sign <  0) * etam +
                         (sign == 0));
    //step_sz = (sign >  0) * fminf(step_sz * etap, stepSizeMax)
    //        + (sign <  0) * fmaxf(step_sz * etam, stepSizeMin)
    //        + (sign == 0) * step_sz;
    biases[i] -= (dB_curr != 0) * SIGN[dB_curr >= 0] * step_sz;

    dB_prev = dB_curr;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#include "curproptrainer.hh"
#include "cuneuralnet.hh"
#include "datasource.hh"
#include "error.hh"
#include "mathli.hh"
#include "matvecli.hh"
#include "cumvli.hh"
#include <sstream>
#include <cstdio>

CNNPLUS_NS_BEGIN

///////////////////////////////////////////////////////////////////////////////
// CUDA kernel calls

template<typename T> void
updateW(T            * const weights,      size_t const strideWeights,
        T      const * const mask,         size_t const strideMask,
        T            * const dWeightsCurr, size_t const strideDWeightsCurr,
        T            * const dWeightsPrev, size_t const strideDWeightsPrev,
        T            * const stepSize,     size_t const strideStepSize,
        T            * const updateVal,    size_t const strideUpdateVal,
        T      const etap,                 T      const etam,
        T      const stepSizeMin,          T      const stepSizeMax,
        T      const errCurr,              T      const errPrev,
        size_t const rows,                 size_t const cols);

template<typename T> void
updateB(T     * const biases,
        T     * const dBiasesCurr,
        T     * const dBiasesPrev,
        T     * const stepSize,
        T     * const updateVal,
        T       const etap,
        T       const etam,
        T       const stepSizeMin,
        T       const stepSizeMax,
        T       const errCurr,
        T       const errPrev,
        size_t  const len);

template<> void
updateW<float>(float        * const weights,      size_t const strideWeights,
               float  const * const mask,         size_t const strideMask,
               float        * const dWeightsCurr, size_t const strideDWeightsCurr,
               float        * const dWeightsPrev, size_t const strideDWeightsPrev,
               float        * const stepSize,     size_t const strideStepSize,
               float        * const updateVal,    size_t const strideUpdateVal,
               float  const etap,                 float  const etam,
               float  const stepSizeMin,          float  const stepSizeMax,
               float  const errCurr,              float  const errPrev,
               size_t const rows,                 size_t const cols)
{
    CNNPLUS_ASSERT(weights      && strideWeights      >= cols);
    CNNPLUS_ASSERT(mask         && strideMask         >= cols);
    CNNPLUS_ASSERT(dWeightsCurr && strideDWeightsCurr >= cols);
    CNNPLUS_ASSERT(dWeightsPrev && strideDWeightsPrev >= cols);
    CNNPLUS_ASSERT(stepSize     && strideStepSize     >= cols);
    CNNPLUS_ASSERT(updateVal    && strideUpdateVal    >= cols);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    dim3 const dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 const dimGrid((cols + dimBlock.x - 1) / dimBlock.x,
        (rows + dimBlock.y - 1) / dimBlock.y);
    updateW_kernel<<<dimGrid, dimBlock>>>(
        weights,      strideWeights,
        mask,         strideMask,
        dWeightsCurr, strideDWeightsCurr,
        dWeightsPrev, strideDWeightsPrev,
        stepSize,     strideStepSize,
        updateVal,    strideUpdateVal,
        etap,         etam,
        stepSizeMin,  stepSizeMax,
        errCurr,      errPrev,
        rows,         cols);
    CUDA_CHECK_ERROR("Kernel call 'updateW_kernel' failed");
}

template<> void
updateB<float>(float * const biases,
               float * const dBiasesCurr,
               float * const dBiasesPrev,
               float * const stepSize,
               float * const updateVal,
               float   const etap,
               float   const etam,
               float   const stepSizeMin,
               float   const stepSizeMax,
               float   const errCurr,
               float   const errPrev,
               size_t  const len)
{
    CNNPLUS_ASSERT(biases);
    CNNPLUS_ASSERT(dBiasesCurr);
    CNNPLUS_ASSERT(dBiasesPrev);
    CNNPLUS_ASSERT(stepSize);
    CNNPLUS_ASSERT(updateVal);
    CNNPLUS_ASSERT(len > 0);

    updateB_kernel<<<(len + THREADS - 1) / THREADS, THREADS>>>(
        biases, dBiasesCurr, dBiasesPrev, stepSize, updateVal,
        etap, etam, stepSizeMin, stepSizeMax, errCurr, errPrev, len);
    CUDA_CHECK_ERROR("Kernel call 'updateB_kernel' failed");
}

template<> void
updateW<double>(double       * const weights,      size_t const strideWeights,
                double const * const mask,         size_t const strideMask,
                double       * const dWeightsCurr, size_t const strideDWeightsCurr,
                double       * const dWeightsPrev, size_t const strideDWeightsPrev,
                double       * const stepSize,     size_t const strideStepSize,
                double       * const updateVal,    size_t const strideUpdateVal,
                double const etap,                 double const etam,
                double const stepSizeMin,          double const stepSizeMax,
                double const errCurr,              double const errPrev,
                size_t const rows,                 size_t const cols)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

template<> void
updateB<double>(double * const biases,
                double * const dBiasesCurr,
                double * const dBiasesPrev,
                double * const stepSize,
                double * const updateVal,
                double   const etap,
                double   const etam,
                double   const stepSizeMin,
                double   const stepSizeMax,
                double   const errCurr,
                double   const errPrev,
                size_t   const len)
{
    throw NotImplementedError("Not yet supported [CUDA<double>].");
}

///////////////////////////////////////////////////////////////////////////////
// CuRpropTrainer implementation

template<typename T, class ErrFnc> bool
CuRpropTrainer<T, ErrFnc>::Adapter::update(
    CuRpropTrainer<T, ErrFnc> & trainer, DataSource<T> & ds, size_t const epoch)
{
    double errRate = 0;
    T const error = trainer.test(ds, errRate);

    printf("# epoch: %04d, error: %10.8f, error-rate: %8.4f%%\n",
        epoch, error, errRate);
    fflush(stdout);

    return (epoch < maxEpochs_);
}

template<typename T, class ErrFnc> void
CuRpropTrainer<T, ErrFnc>::init()
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

        weightsParam_[i].stepSize = cumvli::allocm<T>(
            rows, cols, weightsParam_[i].strideStepSize);
        weightsParam_[i].updateVal = cumvli::allocm<T>(
            rows, cols, weightsParam_[i].strideUpdateVal);
        weightsParam_[i].dVal = cumvli::allocm<T>(
            rows, cols, weightsParam_[i].strideDVal);

        cumvli::setm<T>(weightsParam_[i].stepSize,
            weightsParam_[i].strideStepSize, rows, cols, stepSizeInit_);
        cumvli::zerom<T>(weightsParam_[i].updateVal,
            weightsParam_[i].strideUpdateVal, rows, cols);
        cumvli::zerom<T>(weightsParam_[i].dVal,
            weightsParam_[i].strideDVal, rows, cols);

        //
        // Biases
        //
        size_t const len = trainableParam_[i].biases.len;

        biasesParam_[i].stepSize = cumvli::allocv<T>(len);
        biasesParam_[i].updateVal = cumvli::allocv<T>(len);
        biasesParam_[i].dVal = cumvli::allocv<T>(len);

        cumvli::setv<T>(biasesParam_[i].stepSize, len, stepSizeInit_);
        cumvli::zerov<T>(biasesParam_[i].updateVal, len);
        cumvli::zerov<T>(biasesParam_[i].dVal, len);
    }
}

template<typename T, class ErrFnc> void
CuRpropTrainer<T, ErrFnc>::update(T const errCur, T & errPrev)
{
    for (size_t i = 0; i < trainableParam_.size(); ++i)
    {
        // Weights
        updateW<T>(trainableParam_[i].weights.val,  trainableParam_[i].weights.strideVal,
                   trainableParam_[i].weights.mask, trainableParam_[i].weights.strideMask,
                   trainableParam_[i].weights.dVal, trainableParam_[i].weights.strideDVal,
                   weightsParam_[i].dVal,           weightsParam_[i].strideDVal,
                   weightsParam_[i].stepSize,       weightsParam_[i].strideStepSize,
                   weightsParam_[i].updateVal,      weightsParam_[i].strideUpdateVal,
                   etap_,                           etam_,
                   stepSizeMin_,                    stepSizeMax_,
                   errCur,                          errPrev,
                   trainableParam_[i].weights.rows, trainableParam_[i].weights.cols);

        // Biases
        updateB<T>(trainableParam_[i].biases.val,
                   trainableParam_[i].biases.dVal,
                   biasesParam_[i].dVal,
                   biasesParam_[i].stepSize,
                   biasesParam_[i].updateVal,
                   etap_,
                   etam_,
                   stepSizeMin_,
                   stepSizeMax_,
                   errCur,
                   errPrev,
                   trainableParam_[i].biases.len);
    }

    errPrev = errCur;
}

/*! \todo more checks of ctr-args
 */
template<typename T, class ErrFnc>
CuRpropTrainer<T, ErrFnc>::CuRpropTrainer(NeuralNet<T> & net,
    T const etap, T const etam, T const stepSizeInit,
    T const stepSizeMin, T const stepSizeMax, Adapter & adapter)
    : Trainer<T, ErrFnc>(net), adapter_(&adapter), delAdapter_(false),
    etap_(etap), etam_(etam), stepSizeInit_(stepSizeInit),
    stepSizeMin_(stepSizeMin), stepSizeMax_(stepSizeMax)
{
    if (!dynamic_cast<CuNeuralNet<T>*>(&net))
        throw ParameterError("net", "no CUDA enabled neural-net.");
    else if (etap <= 1)
        throw ParameterError("etap", "must be greater one.");
    else if (etam <= 0 || etam >= 1)
        throw ParameterError("etam", "must be between 0 and 1.");

    init();
}

/*! \todo more checks of ctr-args
 */
template<typename T, class ErrFnc>
CuRpropTrainer<T, ErrFnc>::CuRpropTrainer(NeuralNet<T> & net,
    T const etap, T const etam, T const stepSizeInit,
    T const stepSizeMin, T const stepSizeMax, size_t const maxEpochs)
    : Trainer<T, ErrFnc>(net), adapter_(new Adapter(maxEpochs)), delAdapter_(true),
    etap_(etap), etam_(etam), stepSizeInit_(stepSizeInit),
    stepSizeMin_(stepSizeMin), stepSizeMax_(stepSizeMax)
{
    if (!dynamic_cast<CuNeuralNet<T>*>(&net))
        throw ParameterError("net", "no CUDA enabled neural-net.");
    else if (etap <= 1)
        throw ParameterError("etap", "must be greater one.");
    else if (etam <= 0 || etam >= 1)
        throw ParameterError("etam", "must be between 0 and 1.");

    init();
}

template<typename T, class ErrFnc>
CuRpropTrainer<T, ErrFnc>::~CuRpropTrainer()
{
    if (delAdapter_) delete adapter_;

    for (size_t i = 0; i < trainableParam_.size(); ++i)
    {
        // Weights
        cumvli::free<T>(weightsParam_[i].stepSize);
        cumvli::free<T>(weightsParam_[i].updateVal);
        cumvli::free<T>(weightsParam_[i].dVal);

        // Biases
        cumvli::free<T>(biasesParam_[i].stepSize);
        cumvli::free<T>(biasesParam_[i].updateVal);
        cumvli::free<T>(biasesParam_[i].dVal);
    }
}

template<typename T, class ErrFnc> void
CuRpropTrainer<T, ErrFnc>::train(DataSource<T> & ds)
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
CuRpropTrainer<T, ErrFnc>::toString() const
{
    std::stringstream ss;
    ss << "CuRpropTrainer["
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
template class CuRpropTrainer< float,  MeanSquaredError<float>  >;
template class CuRpropTrainer< double, MeanSquaredError<double> >;
template class CuRpropTrainer< float,  CrossEntropy<float>      >;
template class CuRpropTrainer< double, CrossEntropy<double>     >;
/*! @} */

CNNPLUS_NS_END
