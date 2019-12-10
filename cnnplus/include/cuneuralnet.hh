/**************************************************************************//**
 *
 * \file   cuneuralnet.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jul 06 2009
 *
 * $Id: cuneuralnet.hh 1652 2009-07-06 11:08:42Z dast $
 *
 * \brief  Header for cnnplus::CuNeuralNet.
 *
 *****************************************************************************/

#ifndef CNNPLUS_CUNEURALNET_HH
#define CNNPLUS_CUNEURALNET_HH

#include "common.hh"
#include "neuralnet.hh"
#include "error.hh"

CNNPLUS_NS_BEGIN

//! Represents a neural network, running on the GPU
template<typename T>
class CuNeuralNet : public NeuralNet<T>
{
public:
    virtual void writeOut(std::string const & filename, T * const out = NULL) const
    {
        throw NotImplementedError(
            "Function 'writeOut' not supported by networks, running on the GPU.");
    }
};

CNNPLUS_NS_END

#endif // CNNPLUS_CUNEURALNET_HH
