/**************************************************************************//**
 *
 * \file   facefinder.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jul 17 2009
 *
 * $Id: facefinder.cc 3251 2010-01-12 21:03:22Z dast $
 *
 *****************************************************************************/

/*****************************************************************************/
/* - - - - - - - - - - - - - - P R O T O T Y P E - - - - - - - - - - - - - - */
/*****************************************************************************/

#include "timer.hh"
#include "facefindernet.hh"
#include "cimg.hh"
#include "contbl.hh"
#include "error.hh"
#include "types.hh"
#include <iostream>
#include <vector>
#include <string>
#include <ippcore.h>
#include <ippi.h>
#include <ippvm.h>
#include <mat.h>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <float.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
using namespace cnnplus;

#define USE_OPENMP
#define USE_IPP

#ifdef NDEBUG
#  define IPP_SAFE_CALL(call) call
#else
#  define IPP_SAFE_CALL(call) CNNPLUS_VERIFY(call == ippStsNoErr)
#endif

#define ippsTanh_32f ippsTanh_32f_A24 // ippsTanh_32f_A11, ippsTanh_32f_A21

class ConvFaceFilter
{
    class Layer
    {
    public:
        Layer(size_t const numMapsOut, Size const & maxSizeMapsIn, Size const & maxSizeMapsOut)
            : mapsOut_(numMapsOut), maxSizeMapsIn_(maxSizeMapsIn),
            sizeMapsOut_(maxSizeMapsOut), strideMapsOut_(stride(sizeMapsOut_))
        {
            CNNPLUS_ASSERT(numMapsOut > 0);
            CNNPLUS_ASSERT(maxSizeMapsIn.area() > 0);
            CNNPLUS_ASSERT(maxSizeMapsOut.area() > 0);
            CNNPLUS_ASSERT(maxSizeMapsOut.height() <= maxSizeMapsIn.height() &&
                           maxSizeMapsOut.width()  <= maxSizeMapsIn.width());

            for (size_t i = 0; i < mapsOut_.size(); ++i)
                mapsOut_[i] = (float *) ippMalloc(sizeMapsOut_.height() * strideMapsOut_);
        }
        virtual ~Layer()
        {
            for (size_t i = 0; i < mapsOut_.size(); ++i)
                ippFree(mapsOut_[i]);
        }
        mxArray * writeOut() const
        {
            mwSize const dims[] = { sizeMapsOut_.height(), sizeMapsOut_.width(), numMapsOut() };
            mxArray * arrOut = mxCreateNumericArray(countof(dims), dims, mxDOUBLE_CLASS, mxREAL);
            if (!arrOut) throw MatlabError("Failed to create array.");

            double * pArrOut = (double *) mxGetData(arrOut);

            for (size_t j = 0; j < numMapsOut(); ++j)
            {
                for (size_t r = 0; r < sizeMapsOut_.height(); ++r)
                {
                    for (size_t c = 0; c < sizeMapsOut_.width(); ++c)
                    {
                        pArrOut[j * sizeMapsOut_.area() + r + c * sizeMapsOut_.height()]
                            = mapsOut_[j][r * strideMapsOut_/sizeof(float) + c];
                    }
                }
            }

            return arrOut;
        }

        Size          maxSizeMapsIn()          const { return maxSizeMapsIn_;  }
        float const * mapOut(size_t const num) const { return mapsOut_[num];   }
        size_t        numMapsOut()             const { return mapsOut_.size(); }
        Size          sizeMapsOut()            const { return sizeMapsOut_;    }
        size_t        strideMapsOut()          const { return strideMapsOut_;  }

    protected:
        size_t stride(Size const & size) const
        {
            return ((size.width() * sizeof(float) + 31) / 32) * 32;
        }
        bool checkType(mxArray const * arr, std::string const & type) const
        {
            if (!arr) return false;
            mxArray const * arrType = mxGetField(arr, 0, "type");
            if (!arrType) return false;
            char * pszType = mxArrayToString(arrType);
            if (!pszType) return false;
            std::string const strType(pszType);
            mxFree(pszType);
            return (strType == type);
        }
        bool checkArr(mxArray const * arr, mwSize const ndims, mwSize const * dims) const
        {
            if (!arr) return false;
            if (!mxIsDouble(arr)) return false;
            if (mxIsComplex(arr)) return false;
            if (mxGetNumberOfDimensions(arr) > ndims) return false;
            for (mwSize i = 0; i < ndims; ++i) {
                if (i < mxGetNumberOfDimensions(arr)) {
                    if (mxGetDimensions(arr)[i] != dims[i])
                        return false;
                }
                else {
                    if (dims[i] != 1)
                        return false;
                }
            }
            return true;
        }

    protected:
        std::vector<float *> mapsOut_;
        Size const           maxSizeMapsIn_;
        Size                 sizeMapsOut_;
        size_t               strideMapsOut_;
    };

    class ConvLayer : public Layer
    {
    public:
        ConvLayer(size_t const numMapsIn, size_t const numMapsOut, Size const & maxSizeMapsIn, Size const & sizeKernel)
            : Layer(numMapsOut, maxSizeMapsIn, Size(maxSizeMapsIn.height() - (sizeKernel.height() - 1),
                                                    maxSizeMapsIn.width()  - (sizeKernel.width()  - 1))),
            conTbl_(numMapsOut, numMapsIn), sizeKernel_(sizeKernel), weights_(conTbl_.size()), biases_(numMapsOut),
            tmpMaps_(numMapsOut)
        {
            CNNPLUS_ASSERT(numMapsIn > 0);
            CNNPLUS_ASSERT(sizeKernel.area() > 0);
            CNNPLUS_ASSERT(maxSizeMapsIn.height() >= sizeKernel.height() &&
                           maxSizeMapsIn.width()  >= sizeKernel.width());

            for (size_t i = 0; i < conTbl_.size(); ++i)
                weights_[i] = (float *) ippMalloc(sizeKernel_.area() * sizeof(float));
            for (size_t i = 0; i < tmpMaps_.size(); ++i)
                tmpMaps_[i] = (float *) ippMalloc(sizeMapsOut_.height() * strideMapsOut_);
        }
        virtual ~ConvLayer()
        {
            for (size_t i = 0; i < conTbl_.size(); ++i)
                ippFree(weights_[i]);
            for (size_t i = 0; i < tmpMaps_.size(); ++i)
                ippFree(tmpMaps_[i]);
        }
        Size calcSizeMapsOut(Size const & sizeMapsIn) const
        {
            CNNPLUS_ASSERT(sizeMapsIn.height() >= sizeKernel_.height() &&
                           sizeMapsIn.width()  >= sizeKernel_.width());

            return Size(sizeMapsIn.height() - (sizeKernel_.height() - 1),
                        sizeMapsIn.width()  - (sizeKernel_.width()  - 1));
        }
        void load(mxArray const * arr)
        {
            if (!arr || !mxIsStruct(arr) || !checkType(arr, "c"))
                throw MatlabError("Failed to read convolutional layer.");

            conTbl_.load(arr);

            mxArray const * arrW = mxGetField(arr, 0, "weights");
            {
                mwSize const dims[] = { sizeKernel_.height(), sizeKernel_.width(), conTbl_.cols(), conTbl_.rows() };
                if (!checkArr(arrW, countof(dims), dims))
                    throw MatlabError("Failed to read 'weights'.");
            }

            mxArray const * arrB = mxGetField(arr, 0, "biases");
            {
                mwSize const dims[] = { conTbl_.rows(), 1 };
                if (!checkArr(arrB, countof(dims), dims))
                    throw MatlabError("Failed to read 'biases'.");
            }

            double const * pArrW = (double const *) mxGetData(arrW);
            double const * pArrB = (double const *) mxGetData(arrB);

            for (size_t j = 0; j < conTbl_.rows(); ++j)
            {
                for (size_t i = 0; i < conTbl_.cols(); ++i)
                {
                    float * pKernel = weights_[j * conTbl_.cols() + i];

                    for (size_t n = sizeKernel_.area(), r = 0; r < sizeKernel_.height(); ++r)
                    {
                        for (size_t c = 0; c < sizeKernel_.width(); ++c)
                        {
                            pKernel[--n] = (float)
                                pArrW[(j * conTbl_.cols() + i) * sizeKernel_.area() + r + c * sizeKernel_.height()];
                        }
                    }
                }

                biases_[j] = (float) pArrB[j];
            }
        }
        void fprop(float const * mapIn, size_t const strideMapIn, Size const & sizeMapsIn)
        {
            CNNPLUS_ASSERT(conTbl_.cols() == 1 && conTbl_.rows() == numMapsOut());
            CNNPLUS_ASSERT(sizeMapsIn.height() <= maxSizeMapsIn_.height() &&
                           sizeMapsIn.width()  <= maxSizeMapsIn_.width());

            sizeMapsOut_   = calcSizeMapsOut(sizeMapsIn);
            strideMapsOut_ = stride(sizeMapsOut_);

            IppiSize const  dstRoiSize = { sizeMapsOut_.width(), sizeMapsOut_.height() },
                            kernelSize = { sizeKernel_.width(),  sizeKernel_.height()  };
            IppiPoint const anchor     = { kernelSize.width - 1, kernelSize.height - 1 };

            size_t const len = sizeMapsOut_.height() * strideMapsOut_/sizeof(float);

#ifdef USE_OPENMP
#  pragma omp parallel for
#endif
            for (int j = 0; j < (int) conTbl_.rows(); ++j)
            {
#ifdef USE_IPP
                IPP_SAFE_CALL(ippiFilter_32f_C1R(mapIn, strideMapIn, tmpMaps_[j], strideMapsOut_, dstRoiSize, weights_[j], kernelSize, anchor));
                IPP_SAFE_CALL(ippiAddC_32f_C1IR(biases_[j], tmpMaps_[j], strideMapsOut_, dstRoiSize));
                IPP_SAFE_CALL(ippsTanh_32f(tmpMaps_[j], mapsOut_[j], len));
#else
                for (int r = 0; r < dstRoiSize.height; ++r)
                {
                    for (int c = 0; c < dstRoiSize.width; ++c)
                    {
                        float tmp = 0;

                        for (int y = 0; y < kernelSize.height; ++y)
                        {
                            for (int x = 0; x < kernelSize.width; ++x)
                            {
                                float const a =
                                    mapIn[(r + y) * strideMapIn/sizeof(float) + (c + x)];
                                float const b =
                                    weights_[j][(kernelSize.height - y - 1) * kernelSize.width + (kernelSize.width - x - 1)];
                                tmp += a * b;
                            }
                        }

                        mapsOut_[j][r * strideMapsOut_/sizeof(float) + c] = tanhf(tmp + biases_[j]);
                    }
                }
#endif
            }
        }
        void fprop(Layer const & prevLayer)
        {
            CNNPLUS_ASSERT(prevLayer.sizeMapsOut().height() <= maxSizeMapsIn_.height() &&
                           prevLayer.sizeMapsOut().width()  <= maxSizeMapsIn_.width());

            sizeMapsOut_   = calcSizeMapsOut(prevLayer.sizeMapsOut());
            strideMapsOut_ = stride(sizeMapsOut_);

            IppiSize const  dstRoiSize = { sizeMapsOut_.width(), sizeMapsOut_.height() },
                            kernelSize = { sizeKernel_.width(),  sizeKernel_.height()  };
            IppiPoint const anchor     = { kernelSize.width - 1, kernelSize.height - 1 };

            size_t const len = sizeMapsOut_.height() * strideMapsOut_/sizeof(float);

#ifdef USE_OPENMP
#  pragma omp parallel for
#endif
            for (int j = 0; j < (int) conTbl_.rows(); ++j)
            {
#ifdef USE_IPP
                IPP_SAFE_CALL(ippiSet_32f_C1R(0, tmpMaps_[j], strideMapsOut_, dstRoiSize));

                for (size_t i = 0; i < conTbl_.cols(); ++i)
                {
                    if (!conTbl_.at(j, i))
                        continue;

                    IPP_SAFE_CALL(ippiFilter_32f_C1R(prevLayer.mapOut(i), prevLayer.strideMapsOut(), mapsOut_[j], strideMapsOut_,
                                                     dstRoiSize, weights_[j * conTbl_.cols() + i], kernelSize, anchor));
                    IPP_SAFE_CALL(ippiAdd_32f_C1IR(mapsOut_[j], strideMapsOut_, tmpMaps_[j], strideMapsOut_, dstRoiSize));
                }

                IPP_SAFE_CALL(ippiAddC_32f_C1IR(biases_[j], tmpMaps_[j], strideMapsOut_, dstRoiSize));
                IPP_SAFE_CALL(ippsTanh_32f(tmpMaps_[j], mapsOut_[j], len));
#else
                for (int r = 0; r < dstRoiSize.height; ++r)
                {
                    for (int c = 0; c < dstRoiSize.width; ++c)
                    {
                        float tmp = 0;

                        for (size_t i = 0; i < conTbl_.cols(); ++i)
                        {
                            if (!conTbl_.at(j, i))
                                continue;

                            for (int y = 0; y < kernelSize.height; ++y)
                            {
                                for (int x = 0; x < kernelSize.width; ++x)
                                {
                                    float const a =
                                        prevLayer.mapOut(i)[(r + y) * prevLayer.strideMapsOut()/sizeof(float) + (c + x)];
                                    float const b =
                                        weights_[j * conTbl_.cols() + i][(kernelSize.height - y - 1) * kernelSize.width + (kernelSize.width - x - 1)];
                                    tmp += a * b;
                                }
                            }
                        }

                        mapsOut_[j][r * strideMapsOut_/sizeof(float) + c] = tanhf(tmp + biases_[j]);
                    }
                }
#endif
            }
        }

    protected:
        ConTbl               conTbl_;
        Size const           sizeKernel_;
        std::vector<float *> weights_;
        std::vector<float>   biases_;
        std::vector<float *> tmpMaps_;
    };

    class SubLayer : public Layer
    {
    public:
        SubLayer(size_t const numMapsOut, Size const & maxSizeMapsIn)
            : Layer(numMapsOut, maxSizeMapsIn, maxSizeMapsIn / 2), weights_(numMapsOut), biases_(numMapsOut),
            tmpMaps_(numMapsOut)
        {
            CNNPLUS_ASSERT(maxSizeMapsIn.height() >= 2 && maxSizeMapsIn.height() % 2 == 0 &&
                           maxSizeMapsIn.width()  >= 2 && maxSizeMapsIn.width()  % 2 == 0);

            for (size_t i = 0; i < tmpMaps_.size(); ++i)
                tmpMaps_[i] = (float *) ippMalloc(sizeMapsOut_.height() * strideMapsOut_);
        }
        virtual ~SubLayer()
        {
            for (size_t i = 0; i < tmpMaps_.size(); ++i)
                ippFree(tmpMaps_[i]);
        }
        Size calcSizeMapsOut(Size const & sizeMapsIn) const
        {
            CNNPLUS_ASSERT(sizeMapsIn.height() >= 2 && sizeMapsIn.height() % 2 == 0 &&
                           sizeMapsIn.width()  >= 2 && sizeMapsIn.width()  % 2 == 0);

            return sizeMapsIn / 2;
        }
        void load(mxArray const * arr)
        {
            if (!arr || !mxIsStruct(arr) || !checkType(arr, "s"))
                throw MatlabError("Failed to read subsampling layer.");

            mxArray const * arrW = mxGetField(arr, 0, "weights");
            {
                mwSize const dims[] = { numMapsOut(), 1 };
                if (!checkArr(arrW, countof(dims), dims))
                    throw MatlabError("Failed to read 'weights'.");
            }

            mxArray const * arrB = mxGetField(arr, 0, "biases");
            {
                mwSize const dims[] = { numMapsOut(), 1 };
                if (!checkArr(arrB, countof(dims), dims))
                    throw MatlabError("Failed to read 'biases'.");
            }

            double const * pArrW = (double const *) mxGetData(arrW);
            double const * pArrB = (double const *) mxGetData(arrB);

            for (size_t i = 0; i < numMapsOut(); ++i)
            {
                weights_[i] = (float) pArrW[i];
                biases_[i]  = (float) pArrB[i];
            }
        }
        void fprop(Layer const & prevLayer)
        {
            CNNPLUS_ASSERT(prevLayer.sizeMapsOut().height() <= maxSizeMapsIn_.height() &&
                           prevLayer.sizeMapsOut().width()  <= maxSizeMapsIn_.width());

            sizeMapsOut_   = calcSizeMapsOut(prevLayer.sizeMapsOut());
            strideMapsOut_ = stride(sizeMapsOut_);

            IppiSize const srcSize    = { prevLayer.sizeMapsOut().width(), prevLayer.sizeMapsOut().height() },
                           dstRoiSize = { sizeMapsOut_.width(),            sizeMapsOut_.height()            };

            IppiRect const srcRoi = { 0, 0, srcSize.width, srcSize.height };

            size_t const len = sizeMapsOut_.height() * strideMapsOut_/sizeof(float);

#ifdef USE_OPENMP
#  pragma omp parallel for
#endif
            for (int j = 0; j < (int) numMapsOut(); ++j)
            {
#ifdef USE_IPP
                IPP_SAFE_CALL(ippiResize_32f_C1R(prevLayer.mapOut(j), srcSize, prevLayer.strideMapsOut(), srcRoi,
                                                 tmpMaps_[j], strideMapsOut_, dstRoiSize, 0.5, 0.5, IPPI_INTER_SUPER));
                IPP_SAFE_CALL(ippiMulC_32f_C1IR(weights_[j] * 4, tmpMaps_[j], strideMapsOut_, dstRoiSize));
                IPP_SAFE_CALL(ippiAddC_32f_C1IR(biases_[j], tmpMaps_[j], strideMapsOut_, dstRoiSize));
                IPP_SAFE_CALL(ippsTanh_32f(tmpMaps_[j], mapsOut_[j], len));
#else
                float const * in = prevLayer.mapOut(j);
                size_t const strideIn = prevLayer.strideMapsOut();
                float * out = mapsOut_[j];
                size_t const strideOut = strideMapsOut_;

                IPP_SAFE_CALL(ippiSet_32f_C1R(0, out, strideOut, dstRoiSize));

                for (int r = 0; r < dstRoiSize.height; ++r)
                {
                    for (int c = 0; c < dstRoiSize.width; ++c)
                    {
                        float const tmp =
                            in[(r * 2 + 0) * strideIn/sizeof(float) + (c * 2 + 0)] +
                            in[(r * 2 + 0) * strideIn/sizeof(float) + (c * 2 + 1)] +
                            in[(r * 2 + 1) * strideIn/sizeof(float) + (c * 2 + 0)] +
                            in[(r * 2 + 1) * strideIn/sizeof(float) + (c * 2 + 1)];

                        out[r * strideOut/sizeof(float) + c] = tanhf(tmp * weights_[j] + biases_[j]);
                    }
                }
#endif
            }
        }

    protected:
        std::vector<float>   weights_;
        std::vector<float>   biases_;
        std::vector<float *> tmpMaps_;
    };

    class FullLayer : public ConvLayer
    {
    public:
        FullLayer(size_t const numMapsIn, size_t const numMapsOut, Size const & maxSizeMapsIn, Size const & sizeKernel)
            : ConvLayer(numMapsIn, numMapsOut, maxSizeMapsIn, sizeKernel)
        {
            CNNPLUS_ASSERT(conTbl_.fullConnected());
        }
        void load(mxArray const * arr)
        {
            if (!arr || !mxIsStruct(arr) || !checkType(arr, "f"))
                throw MatlabError("Failed to read convolutional layer.");

            mxArray const * arrW = mxGetField(arr, 0, "weights");
            {
                mwSize const dims[] = { conTbl_.rows(), conTbl_.cols() * sizeKernel_.area() };
                if (!checkArr(arrW, countof(dims), dims))
                    throw MatlabError("Failed to read 'weights'.");
            }

            mxArray const * arrB = mxGetField(arr, 0, "biases");
            {
                mwSize const dims[] = { conTbl_.rows(), 1 };
                if (!checkArr(arrB, countof(dims), dims))
                    throw MatlabError("Failed to read 'biases'.");
            }

            double const * pArrW = (double const *) mxGetData(arrW);
            double const * pArrB = (double const *) mxGetData(arrB);

            for (size_t j = 0; j < conTbl_.rows(); ++j)
            {
                for (size_t i = 0; i < conTbl_.cols(); ++i)
                {
                    float * pKernel = weights_[j * conTbl_.cols() + i];

                    for (size_t n = sizeKernel_.area(), r = 0; r < sizeKernel_.height(); ++r)
                    {
                        for (size_t c = 0; c < sizeKernel_.width(); ++c)
                        {
                            pKernel[--n] = (float)
                                pArrW[(j * conTbl_.cols() + i) * sizeKernel_.area() + r + c * sizeKernel_.height()];
                        }
                    }
                }

                biases_[j] = (float) pArrB[j];
            }
        }
    };

public:
    ConvFaceFilter(Size const & maxSize)
        : c1_(1, 4, maxSize, Size(5, 5)),
          s1_(4, c1_.sizeMapsOut()),
          c2_(4, 14, s1_.sizeMapsOut(), Size(3, 3)),
          s2_(14, c2_.sizeMapsOut()),
          c3_(14, 14, s2_.sizeMapsOut(), Size(6, 6)),
          f4_(14, 1, c3_.sizeMapsOut(), Size(1, 1))
    {
        CNNPLUS_ASSERT(maxSize.height() >= 32 &&
                       maxSize.width()  >= 32);
        CNNPLUS_ASSERT(maxSize.height() <= c1_.maxSizeMapsIn().height() &&
                       maxSize.width()  <= c1_.maxSizeMapsIn().width());
        CNNPLUS_ASSERT(maxSize.height() % 4 == 0 &&
                       maxSize.width()  % 4 == 0);
    }
    void run(float const * image, size_t const stride, Size const & size)
    {
        CNNPLUS_ASSERT(size.height() >= 32 &&
                       size.width()  >= 32);
        CNNPLUS_ASSERT(size.height() <= c1_.maxSizeMapsIn().height() &&
                       size.width()  <= c1_.maxSizeMapsIn().width());
        CNNPLUS_ASSERT(size.height() % 4 == 0 &&
                       size.width()  % 4 == 0);

        c1_.fprop(image, stride, size);
        s1_.fprop(c1_);
        c2_.fprop(s1_);
        s2_.fprop(c2_);
        c3_.fprop(s2_);
        f4_.fprop(c3_);
    }
    void load(std::string const & filename)
    {
        MATFile * file = matOpen(filename.c_str(), "r");
        if (!file)
            throw MatlabError("Failed to open '" + filename + "'.");

        mxArray * arrLayer = NULL;

        try {
            if (!(arrLayer = matGetVariable(file, "layer")) || !mxIsCell(arrLayer))
                throw MatlabError("Failed to read 'layer'.");

            c1_.load(mxGetCell(arrLayer, 0));
            s1_.load(mxGetCell(arrLayer, 1));
            c2_.load(mxGetCell(arrLayer, 2));
            s2_.load(mxGetCell(arrLayer, 3));
            c3_.load(mxGetCell(arrLayer, 4));
            f4_.load(mxGetCell(arrLayer, 5));
        }
        catch (...) {
            if (arrLayer) mxDestroyArray(arrLayer);
            matClose(file);
            throw;
        }

        mxDestroyArray(arrLayer);
        matClose(file);
    }
    void writeOut(std::string const & filename) const
    {
        MATFile * file = matOpen(filename.c_str(), "w");
        if (!file)
            throw MatlabError("Failed to create '" + filename + "'.");

        mxArray * arrLayer = NULL;

        try {
            if (!(arrLayer = mxCreateCellMatrix(6, 1)))
                throw MatlabError("Failed to create array.");

            mxSetCell(arrLayer, 0, c1_.writeOut());
            mxSetCell(arrLayer, 1, s1_.writeOut());
            mxSetCell(arrLayer, 2, c2_.writeOut());
            mxSetCell(arrLayer, 3, s2_.writeOut());
            mxSetCell(arrLayer, 4, c3_.writeOut());
            mxSetCell(arrLayer, 5, f4_.writeOut());

            if (matPutVariable(file, "layer", arrLayer))
                throw MatlabError("Failed to write 'layer'.");
        }
        catch (...) {
            if (arrLayer) mxDestroyArray(arrLayer);
            matClose(file);
            throw;
        }

        mxDestroyArray(arrLayer);
        matClose(file);
    }
    Size calcSizeOut(Size const & sizeIn) const
    {
        return f4_.calcSizeMapsOut(c3_.calcSizeMapsOut(s2_.calcSizeMapsOut(
            c2_.calcSizeMapsOut(s1_.calcSizeMapsOut(c1_.calcSizeMapsOut(sizeIn))))));
    }

    Size          maxSizeIn() const { return c1_.maxSizeMapsIn(); }
    Size          sizeOut()   const { return f4_.sizeMapsOut();   }
    size_t        strideOut() const { return f4_.strideMapsOut(); }
    float const * out()       const { return f4_.mapOut(0);       }

private:
    ConvLayer  c1_;
    SubLayer   s1_;
    ConvLayer  c2_;
    SubLayer   s2_;
    ConvLayer  c3_;
    FullLayer  f4_;
};

class ConvFaceFinder
{
public:
    struct Candidate
    {
        Candidate() : y_(0), x_(0), h_(0), w_(0), o_(0)
        {}
        Candidate(float const y, float const x, float const h, float const w, float const o)
            : y_(y), x_(x), h_(h), w_(w), o_(o)
        {}
        float y() const { return y_; }
        float x() const { return x_; }
        float h() const { return h_; }
        float w() const { return w_; }
        float o() const { return o_; }

        bool operator<(Candidate const & other) const
        {
            return o_ > other.o();
        }

    private:
        float y_, x_, h_, w_, o_;
    };

    struct Cluster
    {
        Cluster() : y_(0), x_(0), h_(0), w_(0), o_(0)
        {}
        Cluster(Candidate const & c)
            : y_(c.y()), x_(c.x()), h_(c.h()), w_(c.w()), o_(c.o())
        {
            candidates_.push_back(c);
        }
        bool inside(Candidate const & c)
        {
            return
                (c.y() >= y_ - h_/4) && (c.y() <= y_ + h_/4) &&
                (c.x() >= x_ - w_/4) && (c.x() <= x_ + w_/4);
        }
        void add(Candidate const & c)
        {
            candidates_.push_back(c);

            float tmp = y_ = x_ = h_ = w_ = 0;
            std::vector<Candidate>::const_iterator
                i = candidates_.begin(), e = candidates_.end();
            for (; i != e; ++i) {
                tmp += i->o();
                y_  += i->y() * i->o();
                x_  += i->x() * i->o();
                h_  += i->h() * i->o();
                w_  += i->w() * i->o();
                o_   = std::max(o_, i->o());
            }
            y_ /= tmp;
            x_ /= tmp;
            h_ /= tmp;
            w_ /= tmp;
        }
        bool addIfInside(Candidate const & c)
        {
            if (inside(c)) {
                add(c);
                return true;
            }
            return false;
        }
        operator Candidate() const
        {
            return Candidate(y_, x_, h_, w_, o_);
        }

        float y() const { return y_; }
        float x() const { return x_; }
        float h() const { return h_; }
        float w() const { return w_; }
        float o() const { return o_; }
        size_t size() const { return candidates_.size(); }

    private:
        std::vector<Candidate> candidates_;
        float y_, x_, h_, w_, o_;
    };

    ConvFaceFinder(Size const & imageSize, float const scaleFactor = 1.2f) :
        imageSize_(imageSize), filter_(Size(imageSize.height() / 4 * 4, imageSize.width() / 4 * 4))
    {
        CNNPLUS_ASSERT(imageSize.height() >= 32 && imageSize.width() >= 32);
        CNNPLUS_ASSERT(scaleFactor > 1 && scaleFactor <= 2);

        Size sizeIn = filter_.maxSizeIn();
        while (sizeIn.height() >= 32 && sizeIn.width() >= 32)
        {
            CNNPLUS_ASSERT(sizeIn.height() % 4 == 0 && sizeIn.width() % 4 == 0);

            sizeSubImages_.push_back(sizeIn);
            strideSubImages_.push_back(0);
            subImages_.push_back(ippiMalloc_32f_C1(
                sizeSubImages_.back().width(), sizeSubImages_.back().height(), &strideSubImages_.back()));

            sizeOutImages_.push_back(filter_.calcSizeOut(sizeIn));
            strideOutImages_.push_back(0);
            outImages_.push_back(ippiMalloc_32f_C1(
                sizeOutImages_.back().width(), sizeOutImages_.back().height(), &strideOutImages_.back()));

            thresholds_.push_back(0);

            sizeIn = Size((size_t)((float) sizeIn.height() / scaleFactor) / 4 * 4,
                          (size_t)((float) sizeIn.width()  / scaleFactor) / 4 * 4);
        }

        std::reverse(sizeSubImages_.begin(), sizeSubImages_.end());
        std::reverse(strideSubImages_.begin(), strideSubImages_.end());
        std::reverse(subImages_.begin(), subImages_.end());

        std::reverse(sizeOutImages_.begin(), sizeOutImages_.end());
        std::reverse(strideOutImages_.begin(), strideOutImages_.end());
        std::reverse(outImages_.begin(), outImages_.end());

        mirrorImage_ = ippiMalloc_32f_C1(32, 32, &strideMirrorImage_);
        fineImage_ = ippiMalloc_32f_C1(44, 44, &strideFineImage_);
    }
    ~ConvFaceFinder()
    {
        for (size_t i = 0; i < subImages_.size(); ++i)
            ippiFree(subImages_[i]);
        for (size_t i = 0; i < outImages_.size(); ++i)
            ippiFree(outImages_[i]);
        ippiFree(mirrorImage_);
        ippiFree(fineImage_);
    }
#if 1
    float run(float const * image, size_t const strideImage, std::vector<Candidate> & candidates)
    {
        IppiSize const srcSize = { imageSize_.width(), imageSize_.height() };
        IppiRect const srcRoi = { 0, 0, srcSize.width, srcSize.height };

        float Tn = 0; int Kn = 0;

        for (size_t i = 0; i < subImages_.size(); ++i)
        {
            IppiSize const dstRoiSize = { sizeSubImages_[i].width(), sizeSubImages_[i].height() };
            IPP_SAFE_CALL(ippiResize_32f_C1R(
                image, srcSize, strideImage, srcRoi, subImages_[i], strideSubImages_[i], dstRoiSize,
                (double) dstRoiSize.width  / (double) srcSize.width,
                (double) dstRoiSize.height / (double) srcSize.height,
                IPPI_INTER_LINEAR));

            filter_.run(subImages_[i], strideSubImages_[i], sizeSubImages_[i]);
            CNNPLUS_ASSERT(filter_.sizeOut() == sizeOutImages_[i]);

            IppiSize const roiSize = { sizeOutImages_[i].width(), sizeOutImages_[i].height() };
            IPP_SAFE_CALL(ippiThreshold_LT_32f_C1R(filter_.out(), filter_.strideOut(),
                outImages_[i], strideOutImages_[i], roiSize, 0));

            double S = 0;
            IPP_SAFE_CALL(ippiSum_32f_C1R(outImages_[i], strideOutImages_[i], roiSize, &S, ippAlgHintNone)); // ippAlgHintFast, ippAlgHintAccurate

            int Pn = 0;
            IPP_SAFE_CALL(ippiCountInRange_32f_C1R(outImages_[i], strideOutImages_[i], roiSize, &Pn, FLT_MIN, FLT_MAX));

            if (Pn > 0) {
                Tn = (float)((Kn * Tn + S) / (Kn + Pn));
                Kn = Kn + Pn;
            }
            CNNPLUS_ASSERT(Tn >= 0);
            thresholds_[i] = Tn;
        }

        CNNPLUS_ASSERT(Tn == thresholds_.back());

        for (size_t i = 0; i < outImages_.size(); ++i)
        {
            float const scaleY = (float) imageSize_.height() / (float) sizeSubImages_[i].height();
            float const scaleX = (float) imageSize_.width()  / (float) sizeSubImages_[i].width();

            for (size_t y = 0; y < sizeOutImages_[i].height(); ++y)
            {
                for (size_t x = 0; x < sizeOutImages_[i].width(); ++x)
                {
                    float const val = outImages_[i][y * strideOutImages_[i]/sizeof(float) + x];
                    if (val >= thresholds_[i])
                    {
                        IppiSize const roiSize = { 32, 32 };
                        size_t const r = y * 4, c = x * 4;
                    #ifndef NDEBUG
                        filter_.run(subImages_[i] + r * strideSubImages_[i]/sizeof(float) + c,
                            strideSubImages_[i], Size(32, 32));
                        CNNPLUS_ASSERT(filter_.sizeOut().height() == 1 && filter_.sizeOut().width() == 1);
                        CNNPLUS_ASSERT(*filter_.out() == val);
                    #endif
                        IPP_SAFE_CALL(ippiMirror_32f_C1R(
                            subImages_[i] + r * strideSubImages_[i]/sizeof(float) + c, strideSubImages_[i],
                            mirrorImage_, strideMirrorImage_, roiSize, ippAxsVertical));
                        filter_.run(mirrorImage_, strideMirrorImage_, Size(32, 32));
                        CNNPLUS_ASSERT(filter_.sizeOut().height() == 1 && filter_.sizeOut().width() == 1);
                        float const avg = (*filter_.out() + val) / 2;
                        if (avg >= Tn) {
                            candidates.push_back(
                                Candidate(((y * 4) + 15.5f) * scaleY,
                                          ((x * 4) + 15.5f) * scaleX,
                                          32 * scaleY, 32 * scaleX, avg));
                        }
                        outImages_[i][y * strideOutImages_[i]/sizeof(float) + x] = avg;
                    }
                    else {
                        outImages_[i][y * strideOutImages_[i]/sizeof(float) + x] = 0;
                    }
                }
            }
        }

        return Tn;
    }
#endif
#if 1
    void run(float const * image, size_t const strideImage, std::vector<Candidate> & candidates, float const THR_FACE, float const THR_POS = 0)
    {
        CNNPLUS_ASSERT(THR_FACE >= 0 && THR_POS >= 0);

        IppiSize const srcSize = { imageSize_.width(), imageSize_.height() };
        IppiRect const srcRoi = { 0, 0, srcSize.width, srcSize.height };

        for (size_t i = 0; i < subImages_.size(); ++i)
        {
            float const scaleY = (float) imageSize_.height() / (float) sizeSubImages_[i].height();
            float const scaleX = (float) imageSize_.width()  / (float) sizeSubImages_[i].width();

            IppiSize const dstRoiSize = { sizeSubImages_[i].width(), sizeSubImages_[i].height() };
            IPP_SAFE_CALL(ippiResize_32f_C1R(
                image, srcSize, strideImage, srcRoi, subImages_[i], strideSubImages_[i], dstRoiSize,
                (double) 1 / scaleX, //dstRoiSize.width  / (double) srcSize.width,
                (double) 1 / scaleY, //dstRoiSize.height / (double) srcSize.height,
                IPPI_INTER_LINEAR));

            filter_.run(subImages_[i], strideSubImages_[i], sizeSubImages_[i]);
            CNNPLUS_ASSERT(filter_.sizeOut() == sizeOutImages_[i]);
            IppiSize const roiSize = { sizeOutImages_[i].width(), sizeOutImages_[i].height() };
            IPP_SAFE_CALL(ippiCopy_32f_C1R(filter_.out(), filter_.strideOut(), outImages_[i], strideOutImages_[i], roiSize));

            for (size_t y = 0; y < sizeOutImages_[i].height(); ++y)
            {
                for (size_t x = 0; x < sizeOutImages_[i].width(); ++x)
                {
                    float const val = outImages_[i][y * strideOutImages_[i]/sizeof(float) + x];
                    if (val >= THR_POS)
                    {
                        IppiSize const roiSize = { 32, 32 };
                        size_t const r = y * 4, c = x * 4;
                    #ifndef NDEBUG
                        filter_.run(subImages_[i] + r * strideSubImages_[i]/sizeof(float) + c,
                            strideSubImages_[i], Size(32, 32));
                        CNNPLUS_ASSERT(filter_.sizeOut().height() == 1 && filter_.sizeOut().width() == 1);
                        CNNPLUS_ASSERT(*filter_.out() == val);
                    #endif
                        IPP_SAFE_CALL(ippiMirror_32f_C1R(
                            subImages_[i] + r * strideSubImages_[i]/sizeof(float) + c, strideSubImages_[i],
                            mirrorImage_, strideMirrorImage_, roiSize, ippAxsVertical));
                        filter_.run(mirrorImage_, strideMirrorImage_, Size(32, 32));
                        CNNPLUS_ASSERT(filter_.sizeOut().height() == 1 && filter_.sizeOut().width() == 1);
                        float const avg = (*filter_.out() + val) / 2;
                        if (avg >= THR_FACE) {
                            candidates.push_back(
                                Candidate(((y * 4) + 15.5f) * scaleY,
                                          ((x * 4) + 15.5f) * scaleX,
                                          32 * scaleY, 32 * scaleX, avg));
                        }
                        #if 0
                        {
                            CImg<float> img(roiSize.width * 2 + 1, roiSize.height);
                            IPP_SAFE_CALL(ippiCopy_32f_C1R(
                                subImages_[i] + r * strideSubImages_[i]/sizeof(float) + c, strideSubImages_[i],
                                img.data, img.width * sizeof(float), roiSize));
                            IPP_SAFE_CALL(ippiCopy_32f_C1R(mirrorImage_, strideMirrorImage_,
                                img.data + roiSize.width + 1, img.width * sizeof(float), roiSize));
                            for (size_t n = 0; n < img.size(); ++n)
                                img.data[n] *= 255;
                            char fname[256];
                            sprintf(fname, "IMG%03d_%03d_%03d_%s.bmp", i, x, y, (avg >= T2) ? "Y" : "N");
                            img.save_bmp(fname);
                        }
                        #endif
                        outImages_[i][y * strideOutImages_[i]/sizeof(float) + x] = avg;
                    }
                    else {
                        outImages_[i][y * strideOutImages_[i]/sizeof(float) + x] = 0;
                    }
                }
            }
        }
    }
#else
    void run(float const * image, size_t const strideImage, std::vector<Candidate> & candidates, float const THR_FACE = 0)
    {
        CNNPLUS_ASSERT(THR_FACE >= 0);

        IppiSize const srcSize = { imageSize_.width(), imageSize_.height() };
        IppiRect const srcRoi = { 0, 0, srcSize.width, srcSize.height };

        for (size_t i = 0; i < subImages_.size(); ++i)
        {
            float const scaleY = (float) imageSize_.height() / (float) sizeSubImages_[i].height();
            float const scaleX = (float) imageSize_.width()  / (float) sizeSubImages_[i].width();

            IppiSize const dstRoiSize = { sizeSubImages_[i].width(), sizeSubImages_[i].height() };
            IPP_SAFE_CALL(ippiResize_32f_C1R(
                image, srcSize, strideImage, srcRoi, subImages_[i], strideSubImages_[i], dstRoiSize,
                (double) 1 / scaleX, //dstRoiSize.width  / (double) srcSize.width,
                (double) 1 / scaleY, //dstRoiSize.height / (double) srcSize.height,
                IPPI_INTER_LINEAR));

            filter_.run(subImages_[i], strideSubImages_[i], sizeSubImages_[i]);

            for (size_t y = 0; y < filter_.sizeOut().height(); ++y)
            {
                for (size_t x = 0; x < filter_.sizeOut().width(); ++x)
                {
                    float const val = filter_.out()[y * filter_.strideOut()/sizeof(float) + x];
                    if (val >= THR_FACE) {
                        candidates.push_back(
                            Candidate(((y * 4) + 15.5f) * scaleY,
                                      ((x * 4) + 15.5f) * scaleX,
                                      32 * scaleY,
                                      32 * scaleX,
                                      val));
                        #if 0
                        {
                            size_t const r = y * 4, c = x * 4;
                            IppiSize const roiSize = { 32, 32 };
                            CImg<float> img(roiSize.width, roiSize.height);
                            IPP_SAFE_CALL(ippiCopy_32f_C1R(
                                subImages_[i] + r * strideSubImages_[i]/sizeof(float) + c, strideSubImages_[i],
                                img.data, img.width * sizeof(float), roiSize));
                            for (size_t n = 0; n < img.size(); ++n)
                                img.data[n] *= 255;
                            char fname[256];
                            sprintf(fname, "IMG%03d_%03d_%03d.bmp", i, x, y);
                            img.save_bmp(fname);
                        }
                        #endif
                    }
                }
            }
        }
    }
#endif
    bool searchFace(float const * image, size_t const strideImage, Candidate & candidate, float const THR_FACE = 0)
    {
        CNNPLUS_ASSERT(THR_FACE >= 0);

        IppiSize const srcSize = { imageSize_.width(), imageSize_.height() };
        IppiRect const srcRoi = { 0, 0, srcSize.width, srcSize.height };
        std::vector<Candidate> candidates;

        for (size_t i = 0; i < subImages_.size(); ++i)
        {
            float const scaleY = (float) imageSize_.height() / (float) sizeSubImages_[i].height();
            float const scaleX = (float) imageSize_.width()  / (float) sizeSubImages_[i].width();

            IppiSize const dstRoiSize = { sizeSubImages_[i].width(), sizeSubImages_[i].height() };
            IPP_SAFE_CALL(ippiResize_32f_C1R(
                image, srcSize, strideImage, srcRoi, subImages_[i], strideSubImages_[i], dstRoiSize,
                (double) 1 / scaleX, //dstRoiSize.width  / (double) srcSize.width,
                (double) 1 / scaleY, //dstRoiSize.height / (double) srcSize.height,
                IPPI_INTER_LINEAR));

            filter_.run(subImages_[i], strideSubImages_[i], sizeSubImages_[i]);

            for (size_t y = 0; y < filter_.sizeOut().height(); ++y)
            {
                for (size_t x = 0; x < filter_.sizeOut().width(); ++x)
                {
                    float const val = filter_.out()[y * filter_.strideOut()/sizeof(float) + x];
                    if (val >= THR_FACE) {
                        candidates.push_back(
                            Candidate(((y * 4) + 15.5f) * scaleY,
                                      ((x * 4) + 15.5f) * scaleX,
                                      32 * scaleY,
                                      32 * scaleX,
                                      val));
                    }
                }
            }
        }

        cluster(candidates);
        //std::sort(candidates.begin(), candidates.end());
        if (!candidates.empty()) {
            candidate = candidates.front();
            return true;
        }
        return false;
    }
    void cluster(std::vector<Candidate> & candidates)
    {
        std::sort(candidates.begin(), candidates.end());

        std::vector<Cluster> clusters;
        while (!candidates.empty()) {
            Candidate const & c = candidates.front();
            std::vector<Cluster>::iterator
                i = clusters.begin(), e = clusters.end();
            for (; i != e; ++i) {
                if (i->addIfInside(c))
                    break;
            }
            if (i == e) {
                clusters.push_back(Cluster(c));
            }
            candidates.erase(candidates.begin());
        }

        std::vector<Cluster>::const_iterator
            i = clusters.begin(), e = clusters.end();
        for (; i != e; ++i)
            candidates.push_back(*i);
    }
    void removeOverlaps(std::vector<Candidate> & candidates)
    {
        std::sort(candidates.begin(), candidates.end());

        std::vector<Candidate> cand;
        for (size_t i = 0; i < candidates.size(); ++i)
        {
            float const r1x1 = candidates[i].x() - candidates[i].w()/2;
            float const r1x2 = candidates[i].x() + candidates[i].w()/2;
            float const r1y1 = candidates[i].y() - candidates[i].h()/2;
            float const r1y2 = candidates[i].y() + candidates[i].h()/2;

            size_t j = 0;
            for (; j < cand.size(); ++j)
            {
                float const r2x1 = cand[j].x() - cand[j].w()/2;
                float const r2x2 = cand[j].x() + cand[j].w()/2;
                float const r2y1 = cand[j].y() - cand[j].h()/2;
                float const r2y2 = cand[j].y() + cand[j].h()/2;

                if (r1x1 <= r2x2 && r2x1 <= r1x2 && r1y1 <= r2y2 && r2y1 <= r1y2)
                    break;
            }
            if (j == cand.size())
                cand.push_back(candidates[i]);
        }
        candidates = cand;
    }
    void fineSearch(float const * image, size_t const strideImage, std::vector<Candidate> & candidates, float const THR_FACE, size_t const NOK, std::vector<Candidate> * pCand = NULL)
    {
        IppiSize const srcSize = { imageSize_.width(), imageSize_.height() };
        IppiSize const dstRoiSize = { 32, 32 };

        std::vector<Cluster> clusters;

        for (std::vector<Candidate>::const_iterator c = candidates.begin(); c != candidates.end(); ++c)
        {
            std::vector<Candidate> cand;

            for (float s = 0.7f; s < 1.4f; s += .1f)
            //int const r = 2;
            //for (int s = -3; s < 4; ++s)
            {
                float const fx = 32.f / c->w() * s;
                float const fy = 32.f / c->h() * s;
                float const x = c->x() - 16.f / fx;
                float const y = c->y() - 16.f / fy;
                float const w = 32.f / fx;
                float const h = 32.f / fy;

                //float const fx = 32.f / c->w();
                //float const fy = 32.f / c->h();
                //float const x = c->x() - 16.f / fx + (s * r);
                //float const y = c->y() - 16.f / fy + (s * r);
                //float const w = 32.f / fx - (2 * s * r);
                //float const h = 32.f / fy - (2 * s * r);

                IppiRect const srcRoi = {
                    (int)(x < 0 ? x - .5f : x + .5f),
                    (int)(y < 0 ? y - .5f : y + .5f),
                    (int)(w + .5f), (int)(h + .5f)
                };
                if (srcRoi.x < 0 || srcRoi.y < 0 ||
                    srcRoi.x + srcRoi.width  > srcSize.width ||
                    srcRoi.y + srcRoi.height > srcSize.height)
                {
                    continue; // TODO
                }
                IPP_SAFE_CALL(ippiResize_32f_C1R(
                    image, srcSize, strideImage, srcRoi,
                    fineImage_, strideFineImage_, dstRoiSize,
                    (double) dstRoiSize.width / srcRoi.width,
                    (double) dstRoiSize.height / srcRoi.height,
                    IPPI_INTER_LINEAR));
                #if 0
                {
                    CImg<float> img(dstRoiSize.width, dstRoiSize.height);
                    IPP_SAFE_CALL(ippiCopy_32f_C1R(fineImage_, strideFineImage_,
                        img.data, img.width * sizeof(float), dstRoiSize));
                    for (size_t n = 0; n < img.size(); ++n)
                        img.data[n] *= 255;
                    char fname[256];
                    sprintf(fname, "IMG_%3.1f.bmp", s);
                    //sprintf(fname, "IMG_%d.bmp", s + 3);
                    img.save_bmp(fname);
                }
                #endif
                filter_.run(fineImage_, strideFineImage_, Size(32, 32));
                CNNPLUS_ASSERT(filter_.sizeOut().height() == 1 && filter_.sizeOut().width() == 1);
                float const out = *filter_.out();
                if (out >= THR_FACE) {
                    cand.push_back(Candidate(
                        srcRoi.y + 15.5f * (float) srcRoi.width / dstRoiSize.width,
                        srcRoi.x + 15.5f * (float) srcRoi.height / dstRoiSize.height,
                        32 * (float) srcRoi.width / dstRoiSize.width,
                        32 * (float) srcRoi.height / dstRoiSize.height,
                        out));
                    if (pCand)
                        pCand->push_back(cand.back());
                }
            }

            Cluster cluster;
            std::sort(cand.begin(), cand.end());
            for (std::vector<Candidate>::const_iterator i = cand.begin(); i != cand.end(); ++i)
                cluster.add(*i);
            clusters.push_back(cluster);
        }

        candidates.clear();
        for (std::vector<Cluster>::const_iterator i = clusters.begin(); i != clusters.end(); ++i)
        {
            if (i->size() >= NOK)
                candidates.push_back(*i);
        }
    }
    void load(std::string const & filename)
    {
        filter_.load(filename);
    }
    void writeOut(std::string const & filename) const
    {
        filter_.writeOut(filename);
    }

private:
    Size const           imageSize_;
    ConvFaceFilter       filter_;
    std::vector<float *> subImages_;
    std::vector<Size>    sizeSubImages_;
    std::vector<int>     strideSubImages_;
    std::vector<float *> outImages_;
    std::vector<Size>    sizeOutImages_;
    std::vector<int>     strideOutImages_;
    float *              mirrorImage_;
    float *              fineImage_;
    int                  strideMirrorImage_;
    int                  strideFineImage_;
    std::vector<float>   thresholds_;
};

void draw(CImg<float> & img, std::vector<ConvFaceFinder::Candidate> const & candidates, unsigned char const color[3])
{
    for (size_t i = 0; i < candidates.size(); ++i)
    {
#if 1
        printf("%4d [x = %6.2f, y = %6.2f, w = %6.2f, h = %6.2f, o = %10.8f] (%6.3f)\n",
            i + 1, candidates[i].x(), candidates[i].y(), candidates[i].w(), candidates[i].h(),
            candidates[i].o(), 32 / candidates[i].w());
#endif
        float const w = candidates[i].w(),
                    h = candidates[i].h();
        int const  x0 = (int)(candidates[i].x() - w/2),
                   y0 = (int)(candidates[i].y() - h/2),
                   x1 = (int)(candidates[i].x() + w/2),
                   y1 = (int)(candidates[i].y() + h/2);
        img.draw_line(x0, y0, x1, y0, color);
        img.draw_line(x1, y0, x1, y1, color);
        img.draw_line(x1, y1, x0, y1, color);
        img.draw_line(x0, y1, x0, y0, color);
        img.draw_circle((int) candidates[i].x(), (int) candidates[i].y(), 1, color);
    }
}

CvFont font;

void draw(IplImage * image, std::vector<ConvFaceFinder::Candidate> const & candidates, CvScalar color)
{
    for (size_t i = 0; i < candidates.size(); ++i)
    {
#if 0
        printf("%4d [x = %6.2f, y = %6.2f, w = %6.2f, h = %6.2f, o = %10.8f] (%6.3f)\n",
            i + 1, candidates[i].x(), candidates[i].y(), candidates[i].w(), candidates[i].h(),
            candidates[i].o(), 32 / candidates[i].w());
#endif
        float const w = candidates[i].w(),
                    h = candidates[i].h();
        int const  x0 = (int)(candidates[i].x() - w/2),
                   y0 = (int)(candidates[i].y() - h/2),
                   x1 = (int)(candidates[i].x() + w/2),
                   y1 = (int)(candidates[i].y() + h/2);
        CvPoint pt1; pt1.x = x0; pt1.y = y0;
        CvPoint pt2; pt2.x = x1; pt2.y = y1;
        CvPoint pt; pt.x = (int) candidates[i].x(); pt.y = (int) candidates[i].y();

        cvRectangle(image, pt1, pt2, color);
#if 1
        //char txt[256];
        //sprintf(txt, "%d [%4.2f]", i+1, candidates[i].o());
        //cvPutText(image, txt, cvPoint(pt1.x, pt1.y-10), &font, CV_RGB(255,255,255));

        pt1.x -= 1; pt1.y -= 1; pt2.x += 1; pt2.y += 1;
        cvRectangle(image, pt1, pt2, CV_RGB(255,255,255));
        pt1.x += 2; pt1.y += 2; pt2.x -= 2; pt2.y -= 2;
        cvRectangle(image, pt1, pt2, CV_RGB(255,255,255));
#endif
        cvCircle(image, pt, 3, color);
    }

    //char txt[256];
    //sprintf(txt, "%d face(s)", candidates.size());
    //cvPutText(image, txt, cvPoint(1, 10), &font, CV_RGB(255,255,255));
}

int T = 95, THR_FACE = 90, NOK = 4;

int main(int argc, char *argv[])
{
#ifdef USE_OPENMP
    omp_set_num_threads(omp_get_num_procs());
#endif

#if 0
    //CImg<float> img("C:\\Dokumente und Einstellungen\\dast\\Desktop\\cnnplus_build\\geeks.bmp");
    CImg<float> img("C:\\Dokumente und Einstellungen\\emp002216\\Desktop\\cnnplus_build\\test3.bmp");
    CImg<float> img2(img);
    for (size_t i = 0; i < img.size(); ++i)
        img.data[i] /= 255.0f;
#if 0
    {
        FaceFinderNet<float> net;
        net.load("E:\\Informatik-Studium\\Masterarbeit\\svn\\cnn\\tests\\facefindernet\\epoch050.mat");
        float out[1] = { 0 };
        net.fprop(img.data, out);
        net.writeOut("NET_out.mat", out);
    }
#endif

    ConvFaceFinder cff(Size(img.height, img.width));
    //cff.load("E:\\Informatik-Studium\\Masterarbeit\\svn\\cnn\\tests\\facefindernet\\epoch050.mat");
    cff.load("D:\\uni\\svn\\cnn\\tests\\facefindernet\\epoch050.mat");

    std::vector<ConvFaceFinder::Candidate> candidates, cand;

    Timer tm;
    {
        cff.run(img.data, img.width * sizeof(float), candidates, 0.99f);
    #if 0
        {
            unsigned char const green[] = { 0, 255, 0 };
            draw(img2, candidates, green);
        }
    #endif
        cff.cluster(candidates);
        cff.removeOverlaps(candidates);
        cff.fineSearch(img.data, img.width * sizeof(float), candidates, 0.95f, 4, &cand);
    #if 1
        {
            unsigned char const green[] = { 0, 255, 0 };
            draw(img2, cand, green);
        }
    #endif
        //cff.writeOut("CFF_out.mat");
    }
    std::stringstream ss;
    ss << tm.report("CFF") << " - found " << candidates.size() << " face(s)";
    std::string const title = ss.str();

    unsigned char const red[] = { 255, 0, 0 };
    draw(img2, candidates, red);
    CImgDisplay disp(img2, title.c_str());
    disp.flush();
    while (!disp.is_closed && !disp.button && !disp.released_key)
    {}
#else
    //CvCapture * capture = cvCaptureFromCAM(0);
    CvCapture * capture = cvCaptureFromFile("C:\\Dokumente und Einstellungen\\emp002216\\Desktop\\06-15group1_lg.jpg");
    //CvCapture * capture = cvCaptureFromAVI("C:\\Dokumente und Einstellungen\\emp002216\\Desktop\\cnnplus_build\\video8.mp4");
    if (!capture) return -1;
    //cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 640);
    //cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 480);
    CvVideoWriter * writer = NULL;
    ConvFaceFinder * cff = NULL;

    cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 0.8, 0.8, 0.0, 1, CV_AA);

    cvNamedWindow("CFF");
    cvCreateTrackbar("T", "CFF", &T, 100, NULL);
    cvCreateTrackbar("THR_FACE", "CFF", &THR_FACE, 100, NULL);
    cvCreateTrackbar("NOK", "CFF", &NOK, 7, NULL);

    IplImage * image = NULL;
    int strideImg = 0; float * img = NULL;

    IplImage * frame = NULL;
    //for (;;)
    {
        //int i = 0, e = 4;
        //for (; i < e; ++i) {
            frame = cvQueryFrame(capture);
        //    if (!frame) break;
        //}
        //if (i < e) break;

        if (image == NULL)
        {
            image = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, 1);
            img = ippiMalloc_32f_C1(frame->width, frame->height, &strideImg);
            cff = new ConvFaceFinder(Size(frame->height, frame->width));
            cff->load("D:\\uni\\svn\\cnn\\tests\\facefindernet\\epoch050.mat");

            double const fps = cvGetCaptureProperty(capture, CV_CAP_PROP_FPS); //25;
            int const fourcc = /*(int) cvGetCaptureProperty(capture, CV_CAP_PROP_FOURCC);*/ CV_FOURCC('P','I','M','1');
            //writer = cvCreateVideoWriter("C:\\Dokumente und Einstellungen\\emp002216\\Desktop\\cnnplus_build\\out.avi", fourcc, fps, cvSize(frame->width, frame->height), 1);
        }

        cvCvtColor(frame, image, CV_RGB2GRAY);
        for (int y = 0; y < image->height; ++y) {
            for (int x = 0; x < image->width; ++x) {
                img[y * strideImg/sizeof(float) + x] = (float) *((unsigned char*)image->imageData + y * image->widthStep + x) / 255.f;
            }
        }

        std::vector<ConvFaceFinder::Candidate> candidates, cand;
        cff->run(img, strideImg, candidates, (float) T / 100.f, 0.99f);
        cff->cluster(candidates);
        cff->removeOverlaps(candidates);
        cff->fineSearch(img, strideImg, candidates, (float) THR_FACE / 100.f, (size_t) NOK, &cand);
        draw(frame, cand, CV_RGB(0, 255, 0));
        //if (candidates.size() > 1)
        //    candidates.erase(candidates.begin() + 1, candidates.end());
        draw(frame, candidates, CV_RGB(255, 0, 0));

        if (writer) {
            for (int i = 0; i < 5; ++i)
                cvWriteToAVI(writer, frame);
        }
        cvShowImage("CFF", frame);

        //if (cvWaitKey(10) >= 0)
        //    break;
    }
    cvWaitKey();
    cvSaveImage("C:\\Dokumente und Einstellungen\\emp002216\\Desktop\\out.bmp", frame);

    if (writer)
        cvReleaseVideoWriter(&writer);
    cvReleaseCapture(&capture);
    cvDestroyWindow("CFF");
    cvReleaseImage(&image);
    ippiFree(img);
    delete cff;
#endif

    return 0;
}
