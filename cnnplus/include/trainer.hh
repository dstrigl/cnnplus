/**************************************************************************//**
 *
 * \file   trainer.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Feb 04 2009
 *
 * $Id: trainer.hh 3163 2010-01-06 15:59:13Z dast $
 *
 * \brief  Header for cnnplus::Trainer.
 *
 *****************************************************************************/

#ifndef CNNPLUS_TRAINER_HH
#define CNNPLUS_TRAINER_HH

#include "common.hh"
#include "errorfnc.hh"
#include <string>

CNNPLUS_NS_BEGIN

template<typename T> class NeuralNet;
template<typename T> class DataSource;

//! Abstract base class for a supervised trainer
template< typename T, class ErrFnc = MeanSquaredError<T> >
class Trainer
{
protected:
    //! Target values
    struct TargetValues {
        TargetValues(T const neg, T const pos)
            : NEG_(neg), POS_(pos), THR_(NEG_ + (POS_ - NEG_) / 2)
        {
            CNNPLUS_ASSERT(NEG_ < POS_);
        }
        T NEG() const { return NEG_; }
        T POS() const { return POS_; }
        T THR() const { return THR_; }
    private:
        T NEG_, POS_, THR_;
    };
    //! Returns the default target values for a specified error function
    TargetValues defaultTargetValues();
    //! Returns the normalized error value
    T normErrVal(T err, size_t cnt = 1) const;

public:
    //! Ctr
    explicit Trainer(NeuralNet<T> & net);
    //! Dtr
    virtual ~Trainer();
    //! Measures the performance over all samples of a data source
    T test(DataSource<T> & ds, double & errRate);
    //! Measures the performance over a single sample of a data source
    bool testSample(DataSource<T> & ds, int & label, T & error);
    //! Trains the neural-net with a data source
    virtual void train(DataSource<T> & ds) {}
    //! Returns a string that describes the trainer
    virtual std::string toString() const;
    //! Sets the target values
    void setTargetValues(T valueNeg, T valuePos);
    //! Returns the error percentage
    /*! \see
        L. Prechelt. PROBEN1 — A set of benchmarks and benchmarking rules for
        neural network training algorithms. Technical Report 21/94, Fakultät
        für Informatik, Universität Karlsruhe, 1994.
     */
    double errorPercent(T err) const;

#ifdef CNNPLUS_MATLAB_FOUND
    //! Writes the Receiver Operating Characteristics (ROC) points to a MAT-file
    /*! \see Fawcett T., "ROC Graphs: Notes and Practical Considerations for Researchers",
             Technical report, HP Laboratories, MS 1143, 1501 Page Mill Road, Palo Alto CA
             94304, USA, April 2004.
     */
    void writeRocPoints(DataSource<T> & ds,
        std::string const & filename, std::string const & name = "roc_pts");
    //! Writes the Confusion Matrix (Kohavi and Provost, 1998) to a MAT-file
    void writeConfMatrix(DataSource<T> & ds, size_t numClasses,
        std::string const & filename, std::string const & name = "conf_mtx");
    //! Writes the classification results of a data source to a MAT-file
    void writeClassResult(DataSource<T> & ds, std::string const & filename,
        std::string const & name = "class_res");
    //! Writes the output of the single neural-net layers to a MAT-file
    void writeLayersOut(DataSource<T> & ds, std::string const & filename = "layers_out.mat");
#endif // CNNPLUS_MATLAB_FOUND

private:
    //! Cpy-Ctr, disabled
    Trainer(Trainer const & rhs);
    //! Assignment, disabled
    Trainer & operator=(Trainer const & rhs);

protected:
    NeuralNet<T> & net_;
    ErrFnc errFnc_;
    TargetValues targetVal_;
    T * in_;
    T * out_;
    T * des_;
};

CNNPLUS_NS_END

#endif // CNNPLUS_TRAINER_HH
