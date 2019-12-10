/**************************************************************************//**
 *
 * \file   tests.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Feb 06 2009
 *
 * $Id: tests.cc 1835 2009-07-21 08:03:24Z dast $
 *
 * \brief  Various tests, only for debugging.
 *
 *****************************************************************************/

#include "neuralnet.hh"
#include "fulllayer.hh"
#include "convlayer.hh"
#include "sublayer.hh"
#include "errorfnc.hh"
#include "simardnet.hh"
#include "jahrernet.hh"
#include "garcianet.hh"
#include "lenet5.hh"
#include "error.hh"
#include "matvecli.hh"
#include "gradienttest.hh"
#include "jacobiantest.hh"
#include "propsbench.hh"
#include "mnistsource.hh"
//#include "cimg.hh"
#include <iostream>
#include <cstdio>
#include <iomanip>

CNNPLUS_NS_BEGIN

template<typename T>
class SingleFullLayerNet : public NeuralNet<T>
{
public:
    SingleFullLayerNet(size_t const sizeIn, size_t const sizeOut)
        : layer_(Size(1, sizeIn), Size(1, sizeOut))
    {}
    virtual void forget(T sigma, bool scale) {
        matvecli::randreseed();
        layer_.forget(sigma, scale);
    }
    virtual void forget() {
        matvecli::randreseed();
        layer_.forget();
    }
    virtual void reset() { return layer_.reset(); }
    virtual void update(T eta) { return layer_.update(eta); }
    virtual void fprop(T const * in, T * out) {
        layer_.fprop(in, sizeIn(), out, sizeOut());
    }
    virtual void bprop(T * in, T const * out, bool accumGradients = false) {
        layer_.bprop(in, sizeIn(), out, sizeOut(), accumGradients);
    }
    virtual void load(std::string const & filename) {
        throw NotImplementedError("Sorry!");
    }
    virtual void save(std::string const & filename) const {
        throw NotImplementedError("Sorry!");
    }
    virtual void writeOut(std::string const & filename, T * const out = NULL) const {
        throw NotImplementedError("Sorry!");
    }
    virtual size_t sizeIn() const { return layer_.sizeIn().area(); }
    virtual size_t sizeOut() const { return layer_.sizeOut().area(); }
    virtual void trainableParam(typename NeuralNet<T>::TrainableParam & param) {
        param.resize(1);
        layer_.trainableParam(param[0]);
    }
    virtual std::string toString() const { return "SingleFullLayerNet"; }
    virtual size_t numTrainableParam() const { return layer_.numTrainableParam(); }
    virtual size_t numConnections() const { return layer_.numConnections(); }
private:
    FullLayer<T> layer_;
};

template<typename T>
class SingleConvLayerNet : public NeuralNet<T>
{
public:
    SingleConvLayerNet(Size const & sizeMapsIn, size_t const numMapsIn,
                       size_t const numMapsOut, Size const & sizeKernel,
                       size_t const stepV, size_t const stepH, double const connProp)
        : layer_(sizeMapsIn, numMapsIn, numMapsOut, sizeKernel, stepV, stepH, connProp)
    {}
    virtual void forget(T sigma, bool scale) {
        matvecli::randreseed();
        layer_.forget(sigma, scale);
    }
    virtual void forget() {
        matvecli::randreseed();
        layer_.forget();
    }
    virtual void reset() { return layer_.reset(); }
    virtual void update(T eta) { return layer_.update(eta); }
    virtual void fprop(T const * in, T * out) {
        layer_.fprop(in,  layer_.sizeIn().width(),
                     out, layer_.sizeOut().width());
    }
    virtual void bprop(T * in, T const * out, bool accumGradients = false) {
        layer_.bprop(in,  layer_.sizeIn().width(),
                     out, layer_.sizeOut().width(),
                     accumGradients);
    }
    virtual void load(std::string const & filename) {
        throw NotImplementedError("Sorry!");
    }
    virtual void save(std::string const & filename) const {
        throw NotImplementedError("Sorry!");
    }
    virtual void writeOut(std::string const & filename, T * const out = NULL) const {
        throw NotImplementedError("Sorry!");
    }
    virtual size_t sizeIn() const { return layer_.sizeIn().area(); }
    virtual size_t sizeOut() const { return layer_.sizeOut().area(); }
    virtual void trainableParam(typename NeuralNet<T>::TrainableParam & param) {
        param.resize(1);
        layer_.trainableParam(param[0]);
    }
    virtual std::string toString() const { return "SingleConvLayerNet"; }
    virtual size_t numTrainableParam() const { return layer_.numTrainableParam(); }
    virtual size_t numConnections() const { return layer_.numConnections(); }
private:
    ConvLayer<T> layer_;
};

template<typename T>
class SingleSubLayerNet : public NeuralNet<T>
{
public:
    SingleSubLayerNet(Size const & sizeMapsIn,
                      size_t const numMaps,
                      Size const & sizeSample)
        : layer_(sizeMapsIn, numMaps, sizeSample)
    {}
    virtual void forget(T sigma, bool scale) {
        matvecli::randreseed();
        layer_.forget(sigma, scale);
    }
    virtual void forget() {
        matvecli::randreseed();
        layer_.forget();
    }
    virtual void reset() { return layer_.reset(); }
    virtual void update(T eta) { return layer_.update(eta); }
    virtual void fprop(T const * in, T * out) {
        layer_.fprop(in,  layer_.sizeIn().width(),
                     out, layer_.sizeOut().width());
    }
    virtual void bprop(T * in, T const * out, bool accumGradients = false) {
        layer_.bprop(in,  layer_.sizeIn().width(),
                     out, layer_.sizeOut().width(),
                     accumGradients);
    }
    virtual void load(std::string const & filename) {
        throw NotImplementedError("Sorry!");
    }
    virtual void save(std::string const & filename) const {
        throw NotImplementedError("Sorry!");
    }
    virtual void writeOut(std::string const & filename, T * const out = NULL) const {
        throw NotImplementedError("Sorry!");
    }
    virtual size_t sizeIn() const { return layer_.sizeIn().area(); }
    virtual size_t sizeOut() const { return layer_.sizeOut().area(); }
    virtual void trainableParam(typename NeuralNet<T>::TrainableParam & param) {
        param.resize(1);
        layer_.trainableParam(param[0]);
    }
    virtual std::string toString() const { return "SingleSubLayerNet"; }
    virtual size_t numTrainableParam() const { return layer_.numTrainableParam(); }
    virtual size_t numConnections() const { return layer_.numConnections(); }
private:
    SubLayer<T> layer_;
};

template<typename T>
void mul_test()
{
    T const A[]  = {   1,   2,   3,   4,  0,
                       5,   6,   7,   8,  0,
                       9,  10,  11,  12,  0  };

    T const At[] = {   1,   5,   9,   0,  0,
                       2,   6,  10,   0,  0,
                       3,   7,  11,   0,  0,
                       4,   8,  12,   0,  0  };

    T const B[]  = {  13,  14,  15,   0,  0,
                      16,  17,  18,   0,  0,
                      19,  20,  21,   0,  0,
                      22,  23,  24,   0,  0  };

    T const Bt[] = {  13,  16,  19,  22,  0,
                      14,  17,  20,  23,  0,
                      15,  18,  21,  24,  0  };

    T const D[]  = { 190, 200, 210,   0,  0,
                     470, 496, 522,   0,  0,
                     750, 792, 834,   0,  0  };

    T C[3 * 5] = { 0 };

    matvecli::mulmm<T,'n','n'>(A, 5, 3, 4, B, 5, 4, 3, C, 5);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            CNNPLUS_ASSERT(C[r * 5 + c] == D[r * 5 + c]);
        }
    }

    matvecli::mulmm<T,'n','t'>(A, 5, 3, 4, Bt, 5, 3, 4, C, 5);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            CNNPLUS_ASSERT(C[r * 5 + c] == D[r * 5 + c]);
        }
    }

    matvecli::mulmm<T,'t','n'>(At, 5, 4, 3, B, 5, 4, 3, C, 5);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            CNNPLUS_ASSERT(C[r * 5 + c] == D[r * 5 + c]);
        }
    }

    matvecli::mulmm<T,'t','t'>(At, 5, 4, 3, Bt, 5, 3, 4, C, 5);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            CNNPLUS_ASSERT(C[r * 5 + c] == D[r * 5 + c]);
        }
    }
}

template<typename T>
void gemm_test()
{
    T const A[]  = {   1,   2,   3,   4,  0,
                       5,   6,   7,   8,  0,
                       9,  10,  11,  12,  0  };

    T const At[] = {   1,   5,   9,   0,  0,
                       2,   6,  10,   0,  0,
                       3,   7,  11,   0,  0,
                       4,   8,  12,   0,  0  };

    T const B[]  = {  13,  14,  15,   0,  0,
                      16,  17,  18,   0,  0,
                      19,  20,  21,   0,  0,
                      22,  23,  24,   0,  0  };

    T const Bt[] = {  13,  16,  19,  22,  0,
                      14,  17,  20,  23,  0,
                      15,  18,  21,  24,  0  };

    T const D[]  = { 190, 200, 210,   0,  0,
                     470, 496, 522,   0,  0,
                     750, 792, 834,   0,  0  };

    T C[3 * 5] = { 0 };

    matvecli::zerom<T>(C, 5, 3, 3);

    matvecli::gemm<T,'n','n'>(A, 5, 3, 4, B, 5, 4, 3, C, 5);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            CNNPLUS_ASSERT(C[r * 5 + c] == D[r * 5 + c]);
        }
    }

    matvecli::zerom<T>(C, 5, 3, 3);

    matvecli::gemm<T,'n','t'>(A, 5, 3, 4, Bt, 5, 3, 4, C, 5);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            CNNPLUS_ASSERT(C[r * 5 + c] == D[r * 5 + c]);
        }
    }

    matvecli::zerom<T>(C, 5, 3, 3);

    matvecli::gemm<T,'t','n'>(At, 5, 4, 3, B, 5, 4, 3, C, 5);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            CNNPLUS_ASSERT(C[r * 5 + c] == D[r * 5 + c]);
        }
    }

    matvecli::zerom<T>(C, 5, 3, 3);

    matvecli::gemm<T,'t','t'>(At, 5, 4, 3, Bt, 5, 3, 4, C, 5);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            CNNPLUS_ASSERT(C[r * 5 + c] == D[r * 5 + c]);
        }
    }
}

template<typename T>
void gemv_test()
{
    T const A[]  = {   1,   2,   3,   4,  0,
                       5,   6,   7,   8,  0,
                       9,  10,  11,  12,  0  };

    T const At[] = {   1,   5,   9,   0,  0,
                       2,   6,  10,   0,  0,
                       3,   7,  11,   0,  0,
                       4,   8,  12,   0,  0  };

    T const x[]  = {  13,  14,  15,  16,  0  };

    T const z[]  = { 150, 382, 614 };

    T y[3] = { 0 };

    matvecli::zerov<T>(y, 3);

    matvecli::gemv<T,'n'>(A, 5, 3, 4, x, 4, y);
    for (int i = 0; i < 3; ++i) {
        CNNPLUS_ASSERT(y[i] == z[i]);
    }

    matvecli::zerov<T>(y, 3);

    matvecli::gemv<T,'t'>(At, 5, 4, 3, x, 4, y);
    for (int i = 0; i < 3; ++i) {
        CNNPLUS_ASSERT(y[i] == z[i]);
    }
}

template<typename T>
void ger_test()
{
    T const x[] = {   1,   2,   3,   4  };
    T const y[] = {   5,   6,   7       };
    T const C[] = {   5,   6,   7,   0,
                     10,  12,  14,   0,
                     15,  18,  21,   0,
                     20,  24,  28,   0
                  };

    T A[4 * 4] = { 0 };

    matvecli::ger<T>(x, 4, y, 3, A, 4);
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 3; ++c) {
            CNNPLUS_ASSERT(A[r * 4 + c] == C[r * 4 + c]);
        }
    }
}

template<typename T>
void axpy_test()
{
    T const x[] = {   1,   2,   3,   4  };
    T y[]       = {   5,   6,   7,   8  };
    T const z[] = {   6,   8,  10,  12  };

    matvecli::axpy<T>(x, 4, y);
    for (int i = 0; i < 4; ++i) {
        CNNPLUS_ASSERT(y[i] == z[i]);
    }
}

template<typename T>
void matvecli_test()
{
    mul_test<T>();
    gemm_test<T>();
    gemv_test<T>();
    ger_test<T>();
    axpy_test<T>();
}

template<typename T>
void test_and_bench(size_t const numMapsL1,
                    size_t const numMapsL2,
                    size_t const sizeL3,
                    Size const & sizeImgIn)
{
    SimardNet<T> net(sizeImgIn, numMapsL1, numMapsL2, sizeL3);

    std::cout << numMapsL1 << "," << numMapsL2 << "," << sizeL3
        << "," << 10 << "," << sizeImgIn.toString() << ": ";

    GradientTest<T> test1;
    std::cout << "gradient " << (test1.run(net) ? "OK" : "FAILED")
        << " [err = " << test1.err() << "], ";

    JacobianTest<T> test2;
    std::cout << "jacobian " << (test2.run(net) ? "OK" : "FAILED")
        << " [err = " << test2.err() << "], ";

    std::cout << PropsBench().run(net) << " sec" << std::endl;
}

void run_test()
{
    matvecli_test<float>();
    matvecli_test<double>();

    test_and_bench<float>(5, 50, 100, Size(29, 29));
    test_and_bench<double>(5, 50, 100, Size(29, 29));
}

void run_layer_test()
{
#if 0
    for (size_t i = 1; i <= 10; ++i) {
        //SimardNet<double> net1(Size(29, 29), 5 * i, 50 * i, 100 * i);
        JahrerNet<double> net2(20 * i, 80 * i, 1, 350 * i, 1);
        std::cout << i << ": " //<< PropsBench().run(net1) << "\t"
            << PropsBench().run(net2) << std::endl;
    }
    return;
#endif
    //SingleFullLayerNet<double> net(100, 10);
    //SingleConvLayerNet<double> net(Size(29, 29), 5, 10, Size(5, 5), 1, 1, 0.5);
    //SingleSubLayerNet<double> net(Size(28, 28), 8, Size(2, 2));
    //SimardNet<double> net(Size(29, 29), 5, 50, 100);
    //JahrerNet<double> net;
    //GarciaNet<double> net;
    LeNet5<double> net;
#if 1
    JacobianTest<double> test2;
    std::cout << "jacobian " << (test2.run(net) ? "OK" : "FAILED")
        << " [err = " << test2.err() << "], ";

    GradientTest<double> test1;
    std::cout << "gradient " << (test1.run(net) ? "OK" : "FAILED")
        << " [err = " << test1.err() << "], ";
#endif
    std::cout << PropsBench().run(net) << " sec" << std::endl;
}

//! Swap endian order of a type \a T
/*!
\par Example:
\code
unsigned int a = 0xabcd;
a = mathli::endianSwap(a); // = 0xdcba
\endcode
*/
template<typename T> inline T& endianSwap(T& a)
{
    if (sizeof(a) != 1)
    {
        unsigned char *pStart = reinterpret_cast<unsigned char*>(&a),
            *pEnd = pStart + sizeof(a);
        for (size_t i = 0; i < sizeof(a) / 2; ++i)
        {
            unsigned char const tmp = *pStart;
            *(pStart++) = *(--pEnd);
            *pEnd = tmp;
        }
    }
    return a;
}

/*
void mnist_distort()
{
    std::string const files[11][2] = {
        "train-images.idx3-ubyte",       "train-labels.idx1-ubyte",
        "train-images-idx3-ubyte-0.mat", "train-labels-idx1-ubyte-0.mat",
        "train-images-idx3-ubyte-1.mat", "train-labels-idx1-ubyte-1.mat",
        "train-images-idx3-ubyte-2.mat", "train-labels-idx1-ubyte-2.mat",
        "train-images-idx3-ubyte-3.mat", "train-labels-idx1-ubyte-3.mat",
        "train-images-idx3-ubyte-4.mat", "train-labels-idx1-ubyte-4.mat",
        "train-images-idx3-ubyte-5.mat", "train-labels-idx1-ubyte-5.mat",
        "train-images-idx3-ubyte-6.mat", "train-labels-idx1-ubyte-6.mat",
        "train-images-idx3-ubyte-7.mat", "train-labels-idx1-ubyte-7.mat",
        "train-images-idx3-ubyte-8.mat", "train-labels-idx1-ubyte-8.mat",
        "train-images-idx3-ubyte-9.mat", "train-labels-idx1-ubyte-9.mat",
    };

    FILE * fimg = fopen("train-images-distort.idx3-ubyte", "wb");
    FILE * flbl = fopen("train-labels-distort.idx1-ubyte", "wb");

    {
        unsigned int magicNumber = 0x00000803;
        endianSwap(magicNumber);
        fwrite(&magicNumber, sizeof(magicNumber), 1, fimg);

        unsigned int dims[] = { 660000, 28, 28 };
        for (int i = 0; i < countof(dims); ++i)
            endianSwap(dims[i]);
        fwrite(dims, sizeof(dims[0]), countof(dims), fimg);
    }

    {
        unsigned int magicNumber = 0x00000801;
        endianSwap(magicNumber);
        fwrite(&magicNumber, sizeof(magicNumber), 1, flbl);

        unsigned int dims[] = { 660000 };
        for (int i = 0; i < countof(dims); ++i)
            endianSwap(dims[i]);
        fwrite(dims, sizeof(dims[0]), countof(dims), flbl);
    }

    int n = 0;

    for (; n < 1; ++n)
    {
        MnistFileSource<unsigned char> mnist(28, 28);
        mnist.open(files[n][0], files[n][1]);

        for (int i = 0; i < mnist.size(); mnist.next(), ++i)
        {
            unsigned char tmp[28 * 28] = {0};
            unsigned char lbl = (unsigned char) mnist.fprop(tmp);

            fwrite(tmp,  sizeof(tmp), 1, fimg);
            fwrite(&lbl, sizeof(lbl), 1, flbl);
        }

        mnist.close();

        printf("file %d/%d\n", n + 1, 11);
        fflush(stdout);
    }

    for (; n < 11; ++n)
    {
        FILE * fimg2 = fopen(files[n][0].c_str(), "rb");
        FILE * flbl2 = fopen(files[n][1].c_str(), "rb");

        {
            unsigned int t = 0;
            fread(&t, sizeof(t), 1, fimg2);
            //endianSwap(t);
            CNNPLUS_ASSERT(t == 0x1e3d4c55);

            unsigned int ndims = 0;
            fread(&ndims, sizeof(ndims), 1, fimg2);
            //endianSwap(ndims);
            CNNPLUS_ASSERT(ndims == 3);

            unsigned int dim = 0;
            fread(&dim, sizeof(dim), 1, fimg2);
            //endianSwap(dim);
            CNNPLUS_ASSERT(dim == 60000);

            fread(&dim, sizeof(dim), 1, fimg2);
            //endianSwap(dim);
            CNNPLUS_ASSERT(dim == 28);

            fread(&dim, sizeof(dim), 1, fimg2);
            //endianSwap(dim);
            CNNPLUS_ASSERT(dim == 28);
        }

        {
            unsigned int t = 0;
            fread(&t, sizeof(t), 1, flbl2);
            //endianSwap(t);
            CNNPLUS_ASSERT(t == 0x1e3d4c55);

            unsigned int ndims = 0;
            fread(&ndims, sizeof(ndims), 1, flbl2);
            //endianSwap(ndims);
            CNNPLUS_ASSERT(ndims == 1);

            unsigned int dim = 0;
            fread(&dim, sizeof(dim), 1, flbl2);
            //endianSwap(dim);
            CNNPLUS_ASSERT(dim == 60000);

            fread(&dim, sizeof(dim), 1, flbl2);
            //endianSwap(dim);
            CNNPLUS_ASSERT(dim == 1);

            fread(&dim, sizeof(dim), 1, flbl2);
            //endianSwap(dim);
            CNNPLUS_ASSERT(dim == 1);
        }

        for (int i = 0; i < 60000; ++i)
        {
            unsigned char tmp[28 * 28] = {0};
            unsigned char lbl = 0;

            fread(tmp,  sizeof(tmp), 1, fimg2);
            fread(&lbl, sizeof(lbl), 1, flbl2);

            fwrite(tmp,  sizeof(tmp), 1, fimg);
            fwrite(&lbl, sizeof(lbl), 1, flbl);
        }

        fclose(fimg2);
        fclose(flbl2);

        printf("file %d/%d\n", n + 1, 11);
        fflush(stdout);
    }

    fclose(fimg);
    fclose(flbl);

#if 0
    {
        MnistFileSource<unsigned char> mnist(28, 28);
        mnist.open("train-images-idx3-ubyte-distort",
                   "train-labels-idx1-ubyte-distort");

        for (int i = 1; i <= mnist.size(); mnist.next(), ++i)
        {
            unsigned char tmp[28 * 28] = {0};
            int const lbl = mnist.fprop(tmp);

            std::stringstream ss;
            ss << std::setfill('0') << std::setw(6)
                << i << "-" << lbl << ".bmp";
            std::cout << lbl << ": " << ss.str() << std::endl;
            fflush(stdout);
            CImg<unsigned char> image(tmp, 28, 28);
            image.save_bmp(ss.str().c_str());
        }

        mnist.close();
    }
#endif
}
*/
/*
void disp_digits()
{
    MnistFileSource<unsigned char> mnist(29, 29);
    mnist.open("E:\\Informatik-Studium\\Masterarbeit\\svn\\mnist\\data\\t10k-images.idx3-ubyte",
               "E:\\Informatik-Studium\\Masterarbeit\\svn\\mnist\\data\\t10k-labels.idx1-ubyte");

    unsigned char * pData =
        static_cast<unsigned char*>(malloc(29 * 29 * 10000));
    for (int r = 0; r < 100; ++r) {
        for (int c = 0; c < 100; ++c) {
            unsigned char tmp[29 * 29];
            int const label = mnist.fprop(tmp);
            for (int i = 0; i < 29; ++i)
                memcpy(pData + r * 29 * 29 * 100 + c * 29 + i * 29 * 100,
                       tmp + i * 29, 29);
            mnist.next();
        }
    }

    mnist.close();

    CImg<unsigned char> image(pData, 100 * 29, 100 * 29);
    image.save_bmp("digits.bmp");
    CImgDisplay disp(image, "t10k-images");
    disp.flush();
    while (!disp.is_closed && !disp.button)
    {}

    free(pData);
}
*/

#if 0
    MnistMemSource<double> mnist(28, 28);
    mnist.load(argv[1], argv[2]);
    double tmp[28 * 28];
    {
        int hist[10] = {0};
        for (int i = 1; i <= mnist.size(); mnist.next(), ++i)
            ++hist[mnist.fprop(tmp)];
        for (int i = 0; i < 10; ++i)
            printf("%d\n", hist[i]);
        printf("\n");
    }
    {
        SubsetDataSrc<double> subMnist(mnist, 0, 50000);
        int hist[10] = {0};
        for (int i = 1; i <= subMnist.size(); subMnist.next(), ++i)
            ++hist[subMnist.fprop(tmp)];
        for (int i = 0; i < 10; ++i)
            printf("%d\n", hist[i]);
        printf("\n");
    }
    {
        SubsetDataSrc<double> subMnist(mnist, 50000, 10000);
        int hist[10] = {0};
        for (int i = 1; i <= subMnist.size(); subMnist.next(), ++i)
            ++hist[subMnist.fprop(tmp)];
        for (int i = 0; i < 10; ++i)
            printf("%d\n", hist[i]);
        printf("\n");
    }
    return 0;
#endif
#if 0
    MnistMemSource<float> mnist(28, 28);
    mnist.load(argv[1], argv[2]);
    ElastDistDataSrc<float> distMnist(mnist, 1, 20);
    distMnist.shift(mnist.size());
    DataSrcView view(10);
    view.run(mnist, distMnist);
    return 0;
#endif
#if 0
    MnistMemSource<float> mnist1(28, 28);
    MnistMemSource<float> mnist2(28, 28);
    mnist1.load(argv[1], argv[2]);
    mnist2.load(argv[1], argv[2]);

    ElastDistDataSrc<float> distMnist1(mnist1);
    ElastDistDataSrc<float> distMnist2(mnist2);

    distMnist1.shuffle();
    distMnist2.shuffle();

    DataSrcView view(10);
    view.run(distMnist1, distMnist2);
    return 0;
#endif
#if 0
    CbclFaceDataSrc<double> faces(20, 20);
    faces.load("D:\\uni\\svn\\cbcl\\faces\\svm.train.normgrey");
    faces.shuffle();

    DataSrcView view(10, 1000);
    view.run(faces);
    return 0;
#endif
/*
    CbclFaceDataSrc<float> facesTrn(20, 20);
    CbclFaceDataSrc<float> facesTst(20, 20);
    facesTrn.load("D:\\uni\\svn\\cbcl\\faces\\svm.train.normgrey");
    facesTst.load("D:\\uni\\svn\\cbcl\\faces\\svm.test.normgrey");

    GarciaNet<float> net;
    net.forget();
*/

CNNPLUS_NS_END
