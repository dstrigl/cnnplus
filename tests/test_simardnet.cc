/**************************************************************************//**
 *
 * \file   test_simardnet.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jun 30 2009
 *
 * $Id: test_simardnet.cc 1750 2009-07-15 09:34:21Z dast $
 *
 *****************************************************************************/

#include "simardnet.hh"
#include "jacobiantest.hh"
#include "gradienttest.hh"
#ifdef CNNPLUS_USE_CUDA
#include "cusimardnet.hh"
#include "cugradienttest.hh"
#endif // CNNPLUS_USE_CUDA
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/CompilerOutputter.h>
#include <iostream>
using namespace cnnplus;

class SimardNetTest : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(SimardNetTest);
    CPPUNIT_TEST(testJacobianD);
    CPPUNIT_TEST(testGradientD);
    CPPUNIT_TEST(testJacobianF);
    CPPUNIT_TEST(testGradientF);
#ifdef CNNPLUS_USE_CUDA
    CPPUNIT_TEST(testCuJacobianF);
    CPPUNIT_TEST(testCuGradientF);
#endif // CNNPLUS_USE_CUDA
    CPPUNIT_TEST_SUITE_END();

public:
    SimardNetTest();
    void testJacobianD();
    void testGradientD();
    void testJacobianF();
    void testGradientF();
#ifdef CNNPLUS_USE_CUDA
    void testCuJacobianF();
    void testCuGradientF();
#endif // CNNPLUS_USE_CUDA

private:
    SimardNet<double> netD_;
    SimardNet<float>  netF_;
#ifdef CNNPLUS_USE_CUDA
    CuSimardNet<float> cuNetF_;
#endif // CNNPLUS_USE_CUDA
};

SimardNetTest::SimardNetTest()
    : netD_(), netF_()
#ifdef CNNPLUS_USE_CUDA
    , cuNetF_()
#endif // CNNPLUS_USE_CUDA
{}

void SimardNetTest::testJacobianD()
{
    JacobianTest<double> test;
    bool const ok = test.run(netD_);
    std::cout << "JacobianTest<double>: " << test.err() << std::endl;
    CPPUNIT_ASSERT(ok);
}

void SimardNetTest::testGradientD()
{
    GradientTest<double> test;
    {
        bool const ok = test.run(netD_);
        std::cout << "GradientTest<double>: " << test.err() << std::endl;
        CPPUNIT_ASSERT(ok);
    }
    {
        bool const ok = test.run(netD_, true);
        std::cout << "GradientTest<double>: " << test.err() << std::endl;
        CPPUNIT_ASSERT(ok);
    }
}

void SimardNetTest::testJacobianF()
{
    JacobianTest<float> test;
    bool const ok = test.run(netF_);
    std::cout << "JacobianTest<float>: " << test.err() << std::endl;
    CPPUNIT_ASSERT(ok);
}

void SimardNetTest::testGradientF()
{
    GradientTest<float> test;
    {
        bool const ok = test.run(netF_);
        std::cout << "GradientTest<float>: " << test.err() << std::endl;
        CPPUNIT_ASSERT(ok);
    }
    {
        bool const ok = test.run(netF_, true);
        std::cout << "GradientTest<float>: " << test.err() << std::endl;
        CPPUNIT_ASSERT(ok);
    }
}

#ifdef CNNPLUS_USE_CUDA
void SimardNetTest::testCuJacobianF()
{
    JacobianTest<float> test;
    bool const ok = test.run(cuNetF_);
    std::cout << "[CUDA] JacobianTest<float>: " << test.err() << std::endl;
    CPPUNIT_ASSERT(ok);
}

void SimardNetTest::testCuGradientF()
{
    CuGradientTest<float> test;
    {
        bool const ok = test.run(cuNetF_);
        std::cout << "[CUDA] GradientTest<float>: " << test.err() << std::endl;
        CPPUNIT_ASSERT(ok);
    }
    {
        bool const ok = test.run(cuNetF_, true);
        std::cout << "[CUDA] GradientTest<float>: " << test.err() << std::endl;
        CPPUNIT_ASSERT(ok);
    }
}
#endif // CNNPLUS_USE_CUDA

int test_simardnet(int, char*[])
{
    CppUnit::TextTestRunner runner;
    runner.setOutputter(new CppUnit::CompilerOutputter(&runner.result(), std::cerr));
    runner.addTest(SimardNetTest::suite());
    return runner.run() ? 0 : 1;
}