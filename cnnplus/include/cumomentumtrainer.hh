/**************************************************************************//**
 *
 * \file   cumomentumtrainer.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jul 13 2009
 *
 * $Id: cumomentumtrainer.hh 1815 2009-07-20 08:20:11Z dast $
 *
 * \brief  Header for cnnplus::CuMomentumTrainer.
 *
 *****************************************************************************/

#ifndef CNNPLUS_CUMOMENTUMTRAINER_HH
#define CNNPLUS_CUMOMENTUMTRAINER_HH

#include "common.hh"
#include "trainer.hh"
#include "neuralnet.hh"
#include <vector>

CNNPLUS_NS_BEGIN

//! Trainer for learning with momentum term for CUDA enabled neural-nets
template< typename T, class ErrFnc = MeanSquaredError<T> >
class CuMomentumTrainer : public Trainer<T, ErrFnc>
{
    void init();

public:
    //! \todo doc
    class Adapter
    {
    public:
        explicit Adapter(size_t maxEpochs = 1000) : maxEpochs_(maxEpochs)
        {}
        virtual ~Adapter() {}
        virtual bool update(
            CuMomentumTrainer<T, ErrFnc> & trainer,
            DataSource<T> & ds, size_t epoch, int & batchSize, T & eta, T & alpha);
    protected:
        size_t const maxEpochs_;
    };

    //! Ctr
    CuMomentumTrainer(NeuralNet<T> & net, int batchSize, T eta, T alpha, Adapter & adapter);
    //! Ctr
    CuMomentumTrainer(NeuralNet<T> & net, int batchSize, T eta, T alpha, size_t maxEpochs = 1000);
    //! Ctr
    CuMomentumTrainer(NeuralNet<T> & net, T eta, T alpha, Adapter & adapter);
    //! Ctr
    CuMomentumTrainer(NeuralNet<T> & net, T eta, T alpha, size_t maxEpochs = 1000);
    //! Dtr
    virtual ~CuMomentumTrainer();

    virtual void train(DataSource<T> & ds);
    virtual std::string toString() const;

    int batchSize() const { return batchSize_; }
    T eta() const { return eta_; }
    T alpha() const { return alpha_; }

private:
    Adapter * adapter_;
    bool const delAdapter_;
    int batchSize_;
    T eta_;
    T alpha_;
    typename NeuralNet<T>::TrainableParam trainableParam_;
    std::vector<T *> deltaW_;
    std::vector<size_t> strideDeltaW_;
    std::vector<T *> deltaB_;
};

CNNPLUS_NS_END

#endif // CNNPLUS_CUMOMENTUMTRAINER_HH
