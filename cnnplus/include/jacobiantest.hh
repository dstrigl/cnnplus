/**************************************************************************//**
 *
 * \file   jacobiantest.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jan 02 2009
 *
 * $Id: jacobiantest.hh 1426 2009-06-05 20:47:23Z dast $
 *
 * \brief  Header for cnnplus::JacobianTest.
 *
 *****************************************************************************/

#ifndef CNNPLUS_JACOBIANTEST_HH
#define CNNPLUS_JACOBIANTEST_HH

#include "common.hh"

CNNPLUS_NS_BEGIN

template<typename T> class NeuralNet;

//! Jacobian Matrix Test
/*! \see
    \li C. M. Bishop, Neural Networks for Pattern Recognition, Oxford
        University Press, (1995), page 148, 4.9 The Jacobian matrix.
    \li P. Y. Simard, D. Steinkraus, & J. Platt, "Best Practice for
        Convolutional Neural Networks Applied to Visual Document Analysis",
        International Conference on Document Analysis and Recognition (ICDAR),
        IEEE Computer Society, Los Alamitos, 2003, pp. 958-962.
 */
template<typename T>
class JacobianTest
{
public:
    //! Default Ctr
    JacobianTest();
    //! Ctr
    /*! \param accThresh accuracy threshold
        \param epsilon epsilon-value for perturbation
     */
    JacobianTest(T accThresh, T epsilon);
    //! Dtr
    virtual ~JacobianTest() {}
    //! Runs the test
    /*! \return \c true if test succeeded, otherwise \c false
     */
    virtual bool run(NeuralNet<T> & net);
    //! Returns the maximal error
    T err() const { return err_; }

protected:
    T const accThresh_;
    T const epsilon_;
    T err_;
};

CNNPLUS_NS_END

#endif // CNNPLUS_JACOBIANTEST_HH
