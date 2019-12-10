/**************************************************************************//**
 *
 * \file   neuralnet.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jan 03 2009
 *
 * $Id: neuralnet.hh 1904 2009-07-30 19:29:56Z dast $
 *
 * \brief  Header for cnnplus::NeuralNet.
 *
 *****************************************************************************/

#ifndef CNNPLUS_NEURALNET_HH
#define CNNPLUS_NEURALNET_HH

#include "common.hh"
#include "types.hh"
#include "layer.hh"
#include <vector>
#include <string>

CNNPLUS_NS_BEGIN

//! Represents a neural network
template<typename T>
class NeuralNet
{
public:
    //! The trainable parameters of a neural network
    typedef std::vector<typename Layer<T>::TrainableParam> TrainableParam;

public:
    //! Ctr
    NeuralNet() {}
    //! Dtr
    virtual ~NeuralNet() {}
    //! Initializes weights with random values
    /*! \param sigma standard deviation
     */
    virtual void forget(T sigma, bool scale = false) = 0;
    //! Initializes weights
    virtual void forget() = 0;
    //! Resets the gradients to zero
    virtual void reset() = 0;
    //! Updates weights and biases
    virtual void update(T eta) = 0;
    //! Forward propagation
    virtual void fprop(T const * in, T * out) = 0;
    //! Backpropagation
    virtual void bprop(T * in, T const * out, bool accumGradients = false) = 0;
    //! Loads the network parameter from a specified MAT-file
    virtual void load(std::string const & filename) = 0;
    //! Saves the network parameter to a specified MAT-file
    virtual void save(std::string const & filename) const = 0;
    //! Writes layer outputs to a specified MAT-file
    virtual void writeOut(std::string const & filename, T * const out = NULL) const = 0;
    //! Returns the input size
    virtual size_t sizeIn() const = 0;
    //! Returns the output size
    virtual size_t sizeOut() const = 0;
    //! Returns the trainable parameters of the network
    virtual void trainableParam(TrainableParam & param) = 0;
    //! Returns a string that describes the network
    virtual std::string toString() const = 0;
    //! Returns the number of trainable parameters of the network
    virtual size_t numTrainableParam() const = 0;
    //! Returns the number of connections of the network
    virtual size_t numConnections() const = 0;

private:
    //! Cpy-Ctr, disabled
    NeuralNet(NeuralNet const & rhs);
    //! Assignment, disabled
    NeuralNet & operator=(NeuralNet const & rhs);
};

CNNPLUS_NS_END

#endif // CNNPLUS_NEURALNET_HH
