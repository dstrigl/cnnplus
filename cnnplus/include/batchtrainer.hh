/**************************************************************************//**
 *
 * \file   batchtrainer.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Mar 07 2009
 *
 * $Id: batchtrainer.hh 1396 2009-06-04 06:33:11Z dast $
 *
 * \brief  Header for cnnplus::BatchTrainer.
 *
 *****************************************************************************/

#ifndef CNNPLUS_BATCHTRAINER_HH
#define CNNPLUS_BATCHTRAINER_HH

#include "common.hh"
#include "trainer.hh"

CNNPLUS_NS_BEGIN

//! Supervised batch learning trainer
template< typename T, class ErrFnc = MeanSquaredError<T> >
class BatchTrainer : public Trainer<T, ErrFnc>
{
public:
    //! \todo doc
    class Adapter
    {
    public:
        explicit Adapter(size_t maxEpochs = 1000) : maxEpochs_(maxEpochs)
        {}
        virtual ~Adapter() {}
        virtual bool update(
            BatchTrainer<T, ErrFnc> & trainer,
            DataSource<T> & ds, size_t epoch, int & batchSize, T & eta);
    protected:
        size_t const maxEpochs_;
    };

    //! Ctr
    BatchTrainer(NeuralNet<T> & net, int batchSize, T eta, Adapter & adapter);
    //! Ctr
    BatchTrainer(NeuralNet<T> & net, int batchSize, T eta, size_t maxEpochs = 1000);
    //! Ctr
    BatchTrainer(NeuralNet<T> & net, T eta, Adapter & adapter);
    //! Ctr
    BatchTrainer(NeuralNet<T> & net, T eta, size_t maxEpochs = 1000);
    //! Dtr
    virtual ~BatchTrainer();

    virtual void train(DataSource<T> & ds);
    virtual std::string toString() const;

    int batchSize() const { return batchSize_; }
    T eta() const { return eta_; }

private:
    Adapter * adapter_;
    bool const delAdapter_;
    int batchSize_;
    T eta_;
};

CNNPLUS_NS_END

#endif // CNNPLUS_BATCHTRAINER_HH
