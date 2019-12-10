/**************************************************************************//**
 *
 * \file   trainer.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Feb 17 2009
 *
 * $Id: trainer.cc 1942 2009-08-05 18:25:28Z dast $
 *
 * \brief  Implementation of cnnplus::Trainer.
 *
 *****************************************************************************/

#include "trainer.hh"
#include "neuralnet.hh"
#include "datasource.hh"
#include "error.hh"
#include "matvecli.hh"
#include "mathli.hh"
#include <cstdio>
#ifdef CNNPLUS_MATLAB_FOUND
#include <vector>
#include <functional>
#include <mat.h>
#endif // CNNPLUS_MATLAB_FOUND

CNNPLUS_NS_BEGIN

template<> Trainer< float, MeanSquaredError<float> >::TargetValues
Trainer< float, MeanSquaredError<float> >::defaultTargetValues()
{
    return TargetValues(-1, 1);
}

template<> Trainer< double, MeanSquaredError<double> >::TargetValues
Trainer< double, MeanSquaredError<double> >::defaultTargetValues()
{
    return TargetValues(-1, 1);
}

template<> Trainer< float, CrossEntropy<float> >::TargetValues
Trainer< float, CrossEntropy<float> >::defaultTargetValues()
{
    return TargetValues(0, 1);
}

template<> Trainer< double, CrossEntropy<double> >::TargetValues
Trainer< double, CrossEntropy<double> >::defaultTargetValues()
{
    return TargetValues(0, 1);
}

template<> float
Trainer< float, MeanSquaredError<float> >::normErrVal(
    float const err, size_t const cnt) const
{
    CNNPLUS_ASSERT(cnt > 0);
    return err / (net_.sizeOut() * cnt);
}

template<> double
Trainer< double, MeanSquaredError<double> >::normErrVal(
    double const err, size_t const cnt) const
{
    CNNPLUS_ASSERT(cnt > 0);
    return err / (net_.sizeOut() * cnt);
}

template<> float
Trainer< float, CrossEntropy<float> >::normErrVal(
    float const err, size_t const cnt) const
{
    CNNPLUS_ASSERT(cnt > 0);
    return err / cnt;
}

template<> double
Trainer< double, CrossEntropy<double> >::normErrVal(
    double const err, size_t const cnt) const
{
    CNNPLUS_ASSERT(cnt > 0);
    return err / cnt;
}

template<typename T, class ErrFnc>
Trainer<T, ErrFnc>::Trainer(NeuralNet<T> & net)
    : net_(net), errFnc_(net_.sizeOut()), targetVal_(defaultTargetValues())
{
    // Allocate memory
    in_  = matvecli::allocv<T>(net_.sizeIn());
    out_ = matvecli::allocv<T>(net_.sizeOut());
    des_ = matvecli::allocv<T>(net_.sizeOut());
}

template<typename T, class ErrFnc>
Trainer<T, ErrFnc>::~Trainer()
{
    // Deallocate memory
    matvecli::free<T>(in_);
    matvecli::free<T>(out_);
    matvecli::free<T>(des_);
}

template<typename T, class ErrFnc> bool
Trainer<T, ErrFnc>::testSample(DataSource<T> & ds, int & label, T & error)
{
    if (ds.sizeOut() != net_.sizeIn())
        throw ParameterError("ds", "size doesn't match.");

    // Read pattern and desired label from data source
    int const desLbl = ds.fprop(in_);

    // Compute neural-net output
    net_.fprop(in_, out_);

    // Compute error and label
    if (net_.sizeOut() > 1) {
        matvecli::setv<T>(des_, net_.sizeOut(), targetVal_.NEG());
        CNNPLUS_ASSERT(0 <= desLbl && desLbl < static_cast<int>(net_.sizeOut()));
        des_[desLbl] = targetVal_.POS();
        error = errFnc_.fprop(out_, des_);
        label = matvecli::maxidxv<T>(out_, net_.sizeOut());
    }
    else {
        CNNPLUS_ASSERT(desLbl == targetVal_.POS() || desLbl == targetVal_.NEG());
        des_[0] = static_cast<T>(desLbl);
        error = errFnc_.fprop(out_, des_);
        label = static_cast<int>(
            (out_[0] > targetVal_.THR()) ? targetVal_.POS() : targetVal_.NEG());
    }
    error = normErrVal(error);

    // Return if correct classified or not
    return (label == desLbl);
}

template<typename T, class ErrFnc> T
Trainer<T, ErrFnc>::test(DataSource<T> & ds, double & errRate)
{
    if (ds.sizeOut() != net_.sizeIn())
        throw ParameterError("ds", "size doesn't match.");

    T error = 0;
    errRate = 0;
    ds.rewind();

    // Set vector 'des_' to negative target value
    if (net_.sizeOut() > 1)
        matvecli::setv<T>(des_, net_.sizeOut(), targetVal_.NEG());

    // Loop over all patterns
    for (int i = 1; i <= ds.size(); ds.next(), ++i)
    {
        // Read pattern and desired label from data source
        int const desLbl = ds.fprop(in_);

        // Compute neural-net output
        net_.fprop(in_, out_);

        if (net_.sizeOut() > 1) {
            // Compute error
            CNNPLUS_ASSERT(0 <= desLbl && desLbl < static_cast<int>(net_.sizeOut()));
            des_[desLbl] = targetVal_.POS();
            error += errFnc_.fprop(out_, des_);
            des_[desLbl] = targetVal_.NEG();

            // Update number of wrong classified patterns
            errRate += (matvecli::maxidxv<T>(out_, net_.sizeOut()) != desLbl);
        }
        else {
            // Compute error
            CNNPLUS_ASSERT(desLbl == targetVal_.POS() || desLbl == targetVal_.NEG());
            des_[0] = static_cast<T>(desLbl);
            error += errFnc_.fprop(out_, des_);

            // Update number of wrong classified patterns
            int const label = static_cast<int>(
                (out_[0] > targetVal_.THR()) ? targetVal_.POS() : targetVal_.NEG());
            errRate += (label != desLbl);
        }

#ifdef CNNPLUS_PRINT_PROGRESS
        printf("test %.2f%%\r", i * 100.0 / ds.size());
        fflush(stdout);
#endif // CNNPLUS_PRINT_PROGRESS
    }
#ifdef CNNPLUS_PRINT_PROGRESS
    printf("            \r");
    fflush(stdout);
#endif // CNNPLUS_PRINT_PROGRESS

    errRate *= 100.0 / ds.size();
    return normErrVal(error, ds.size());
}

template<typename T, class ErrFnc> void
Trainer<T, ErrFnc>::setTargetValues(T const valueNeg, T const valuePos)
{
    if (valueNeg >= valuePos) throw ParameterError(
        "Parameter 'valueNeg' must be lower than parameter 'valuePos'.");
    targetVal_ = TargetValues(valueNeg, valuePos);
}

template<typename T, class ErrFnc> double
Trainer<T, ErrFnc>::errorPercent(T const err) const
{
    return 100.0 * err / mathli::pow(targetVal_.POS() - targetVal_.NEG(), 2);
}

template<typename T, class ErrFnc> std::string
Trainer<T, ErrFnc>::toString() const
{
    std::stringstream ss;
    ss << "Trainer["
        << errFnc_.toString()
        << "; targetVal=("
        << targetVal_.NEG() << ","
        << targetVal_.POS() << ")]";
    return ss.str();
}

#ifdef CNNPLUS_MATLAB_FOUND
//
// Fawcett T., "ROC Graphs: Notes and Practical Considerations for Researchers",
// Technical report, HP Laboratories, MS 1143, 1501 Page Mill Road, Palo Alto CA
// 94304, USA, April 2004.
//
// Algorithm 2: Efficient method for generating ROC points
//
struct compare_pair_second {
    template<class T1, class T2>
    bool operator()(std::pair<T1, T2> const & left, std::pair<T1, T2> const & right)
    {
        return std::greater<T2>()(left.second, right.second);
    }
};
template<typename T, class ErrFnc> void
Trainer<T, ErrFnc>::writeRocPoints(
    DataSource<T> & ds, std::string const & filename, std::string const & name)
{
    if (net_.sizeOut() != 1)
        throw CnnPlusError("Only supported for neural-nets with one output neuron.");
    else if (ds.sizeOut() != net_.sizeIn())
        throw ParameterError("ds", "size doesn't match.");

    //
    // Generate list of ROC points increasing by fp rate
    //

    int P = 0, N = 0;
    std::vector< std::pair<int, T> > L(ds.size());

    ds.rewind();
    for (int i = 0; i < ds.size(); ds.next(), ++i)
    {
        L[i].first = ds.fprop(in_);
        P += L[i].first > targetVal_.THR();
        net_.fprop(in_, out_);
        L[i].second = out_[0];
#ifdef CNNPLUS_PRINT_PROGRESS
        printf("test %.2f%%\r", (i+1) * 100.0 / ds.size());
        fflush(stdout);
#endif // CNNPLUS_PRINT_PROGRESS
    }
#ifdef CNNPLUS_PRINT_PROGRESS
    printf("            \r");
    fflush(stdout);
#endif // CNNPLUS_PRINT_PROGRESS
    N = ds.size() - P;

    if (P < 1 || N < 1)
        throw ParameterError(
            "ds", "must have at least one positive and one negative example.");

    std::sort(L.begin(), L.end(), compare_pair_second());

    std::vector< std::pair<double, double> > R;
    R.push_back(std::pair<double, double>(0, 0));
    T f_prev = L[0].second;
    int TP = L[0].first > targetVal_.THR(),
        FP = 1 - TP;

    for (int i = 1; i < ds.size(); ++i)
    {
        CNNPLUS_ASSERT(L[i].second <= f_prev);
        if (L[i].second < f_prev) {
            R.push_back(std::pair<double, double>(
                static_cast<double>(FP)/N, static_cast<double>(TP)/P));
            f_prev = L[i].second;
        }
        if (L[i].first > targetVal_.THR())
            TP++;
        else // i is a negative example
            FP++;
    }

    R.push_back(std::pair<double, double>(1, 1));

    //
    // Write list of ROC points to MAT-file
    //

    MATFile * matFile = matOpen(filename.c_str(), "w");
    if (!matFile)
        throw MatlabError("Failed to create '" + filename + "'.");

    try {
        mxArray * arr = mxCreateDoubleMatrix(2, R.size(), mxREAL);
        if (!arr) throw MatlabError("Failed to create array.");
        double * pArr = static_cast<double*>(mxGetData(arr));
        for (size_t i = 0; i < R.size(); ++i) {
            *pArr++ = R[i].first;
            *pArr++ = R[i].second;
        }
        if (matPutVariable(matFile, name.c_str(), arr))
            throw MatlabError("Failed to write '" + name + "'.");
    }
    catch (...) {
        matClose(matFile);
        throw;
    }

    matClose(matFile);
}

template<typename T, class ErrFnc> void
Trainer<T, ErrFnc>::writeConfMatrix(DataSource<T> & ds,
    size_t const numClasses, std::string const & filename, std::string const & name)
{
    if (ds.sizeOut() != net_.sizeIn())
        throw ParameterError("ds", "size doesn't match.");
    else if (numClasses < 2)
        throw ParameterError("numClasses", "must be greater one.");

    MATFile * matFile = matOpen(filename.c_str(), "w");
    if (!matFile)
        throw MatlabError("Failed to create '" + filename + "'.");

    try {
        mxArray * arr = mxCreateDoubleMatrix(numClasses, numClasses, mxREAL);
        if (!arr) throw MatlabError("Failed to create array.");
        double * pArr = static_cast<double*>(mxGetData(arr));
        matvecli::zerov<double>(pArr, numClasses * numClasses);

        // Loop over all patterns
        ds.rewind();
        for (int i = 1; i <= ds.size(); ds.next(), ++i)
        {
            // Read pattern and desired label from data source
            int const desLbl = (net_.sizeOut() > 1) ?
                ds.fprop(in_) : (ds.fprop(in_) > targetVal_.THR());

            // Compute neural-net output
            net_.fprop(in_, out_);

            // Compute predicted label
            int const preLbl = (net_.sizeOut() > 1) ?
                matvecli::maxidxv<T>(out_, net_.sizeOut()) :
                (out_[0] > targetVal_.THR());

            // Write the confusion matrix
            CNNPLUS_ASSERT(desLbl >= 0 && desLbl < static_cast<int>(numClasses));
            CNNPLUS_ASSERT(preLbl >= 0 && preLbl < static_cast<int>(numClasses));
            if (desLbl >= 0 && desLbl < static_cast<int>(numClasses) &&
                preLbl >= 0 && preLbl < static_cast<int>(numClasses)) {
                pArr[preLbl * numClasses + desLbl] += 1;
            }

#ifdef CNNPLUS_PRINT_PROGRESS
            printf("test %.2f%%\r", i * 100.0 / ds.size());
            fflush(stdout);
#endif // CNNPLUS_PRINT_PROGRESS
        }
#ifdef CNNPLUS_PRINT_PROGRESS
        printf("            \r");
        fflush(stdout);
#endif // CNNPLUS_PRINT_PROGRESS

        if (matPutVariable(matFile, name.c_str(), arr))
            throw MatlabError("Failed to write '" + name + "'.");
    }
    catch (...) {
        matClose(matFile);
        throw;
    }

    matClose(matFile);
}

template<typename T, class ErrFnc> void
Trainer<T, ErrFnc>::writeClassResult(
    DataSource<T> & ds, std::string const & filename, std::string const & name)
{
    if (ds.sizeOut() != net_.sizeIn())
        throw ParameterError("ds", "size doesn't match.");

    MATFile * matFile = matOpen(filename.c_str(), "w");
    if (!matFile)
        throw MatlabError("Failed to create '" + filename + "'.");

    try {
        mxArray * arr = mxCreateDoubleMatrix(4, ds.size(), mxREAL);
        if (!arr) throw MatlabError("Failed to create array.");
        double * pArr = static_cast<double*>(mxGetData(arr));

        T error = 0;
        ds.rewind();

        // Set vector 'des_' to negative target value
        if (net_.sizeOut() > 1)
            matvecli::setv<T>(des_, net_.sizeOut(), targetVal_.NEG());

        // Loop over all patterns
        for (int i = 1; i <= ds.size(); ds.next(), ++i)
        {
            // Read pattern and desired label from data source
            int const desLbl = ds.fprop(in_);

            // Compute neural-net output
            net_.fprop(in_, out_);

            // Compute error
            if (net_.sizeOut() > 1) {
                CNNPLUS_ASSERT(0 <= desLbl && desLbl < static_cast<int>(net_.sizeOut()));
                des_[desLbl] = targetVal_.POS();
                error = errFnc_.fprop(out_, des_);
                des_[desLbl] = targetVal_.NEG();
            }
            else {
                CNNPLUS_ASSERT(desLbl == targetVal_.POS() || desLbl == targetVal_.NEG());
                des_[0] = static_cast<T>(desLbl);
                error = errFnc_.fprop(out_, des_);
            }

            // Compute predicted label
            int const preLbl = (net_.sizeOut() > 1) ?
                matvecli::maxidxv<T>(out_, net_.sizeOut()) :
                static_cast<int>((out_[0] > targetVal_.THR()) ?
                    targetVal_.POS() : targetVal_.NEG());

            *pArr++ = ds.idx() + 1; // + 1 for MATLAB indices
            *pArr++ = desLbl;
            *pArr++ = preLbl;
            *pArr++ = error;

#ifdef CNNPLUS_PRINT_PROGRESS
            printf("test %.2f%%\r", i * 100.0 / ds.size());
            fflush(stdout);
#endif // CNNPLUS_PRINT_PROGRESS
        }
#ifdef CNNPLUS_PRINT_PROGRESS
        printf("            \r");
        fflush(stdout);
#endif // CNNPLUS_PRINT_PROGRESS

        if (matPutVariable(matFile, name.c_str(), arr))
            throw MatlabError("Failed to write '" + name + "'.");
    }
    catch (...) {
        matClose(matFile);
        throw;
    }

    matClose(matFile);
}

template<typename T, class ErrFnc> void
Trainer<T, ErrFnc>::writeLayersOut(DataSource<T> & ds, std::string const & filename)
{
    if (ds.sizeOut() != net_.sizeIn())
        throw ParameterError("ds", "size doesn't match.");

    // Read pattern from data source
    ds.fprop(in_);

    // Compute neural-net output
    net_.fprop(in_, out_);

    // Write neural-net outputs to MAT-file
    net_.writeOut(filename, out_);
}
#endif // CNNPLUS_MATLAB_FOUND

/*! \addtogroup eti_grp Explicit Template Instantiation
 @{
 */
template class Trainer< float,  MeanSquaredError<float>  >;
template class Trainer< double, MeanSquaredError<double> >;
template class Trainer< float,  CrossEntropy<float>      >;
template class Trainer< double, CrossEntropy<double>     >;
/*! @} */

CNNPLUS_NS_END
