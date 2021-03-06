/**************************************************************************//**
 *
 * \file   rproptrainer.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jul 14 2009
 *
 * $Id: rproptrainer.hh 1775 2009-07-16 19:24:58Z dast $
 *
 * \brief  Header for cnnplus::RpropTrainer.
 *
 *****************************************************************************/

#ifndef CNNPLUS_RPROPTRAINER_HH
#define CNNPLUS_RPROPTRAINER_HH

#include "common.hh"
#include "trainer.hh"
#include "neuralnet.hh"
#include <vector>

CNNPLUS_NS_BEGIN

//! Resilient Propagation (Rprop) trainer
/*! \see
    \li M. Riedmiller and H. Braun. A direct adaptive method for faster
        backpropagation learning: The RPROP algorithm. In E. H. Ruspini,
        editor, Proceedings of the IEEE International Conference on Neural
        Networks, pages 586�591. IEEE Press, 1993.
    \li C. Igel, M. Husken, Empirical evaluation of the improved Rprop
        learning algorithms, Neurocomputing 50 (2003) 105�123.
 */
template< typename T, class ErrFnc = MeanSquaredError<T> >
class RpropTrainer : public Trainer<T, ErrFnc>
{
    struct WeightsParam
    {
        T *    stepSize;
        T *    updateVal;
        T *    dVal;
        size_t strideStepSize;
        size_t strideUpdateVal;
        size_t strideDVal;
    };
    struct BiasesParam
    {
        T *    stepSize;
        T *    updateVal;
        T *    dVal;
    };

    void init();
    void update(T const errCur, T & errPrev);
    void updateVal(T & val, T & dValCurr, T & dValPrev,
        T & stepSize, T & updateVal, T errCurr, T errPrev);

public:
    //! \todo doc
    class Adapter
    {
    public:
        explicit Adapter(size_t maxEpochs = 1000) : maxEpochs_(maxEpochs)
        {}
        virtual ~Adapter() {}
        virtual bool update(
            RpropTrainer<T, ErrFnc> & trainer, DataSource<T> & ds, size_t epoch);
    protected:
        size_t const maxEpochs_;
    };

    //! Ctr
    RpropTrainer(NeuralNet<T> & net, T etap, T etam, T stepSizeInit,
        T stepSizeMin, T stepSizeMax, Adapter & adapter);
    //! Ctr
    RpropTrainer(NeuralNet<T> & net, T etap, T etam, T stepSizeInit,
        T stepSizeMin, T stepSizeMax, size_t maxEpochs = 1000);
    //! Dtr
    virtual ~RpropTrainer();

    T etap()         const { return etap_;         }
    T etam()         const { return etam_;         }
    T stepSizeInit() const { return stepSizeInit_; }
    T stepSizeMin()  const { return stepSizeMin_;  }
    T stepSizeMax()  const { return stepSizeMax_;  }

    virtual void train(DataSource<T> & ds);
    virtual std::string toString() const;

private:
    Adapter * adapter_;
    bool const delAdapter_;
    T const etap_, etam_;
    T const stepSizeInit_, stepSizeMin_, stepSizeMax_;
    typename NeuralNet<T>::TrainableParam trainableParam_;
    std::vector<WeightsParam> weightsParam_;
    std::vector<BiasesParam> biasesParam_;
};

CNNPLUS_NS_END

#endif // CNNPLUS_RPROPTRAINER_HH
