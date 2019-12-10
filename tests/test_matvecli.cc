/**************************************************************************//**
 *
 * \file   test_matvecli.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Jun 01 2009
 *
 * $Id: test_matvecli.cc 1396 2009-06-04 06:33:11Z dast $
 *
 *****************************************************************************/

#include "matvecli.hh"
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/CompilerOutputter.h>
using namespace cnnplus;

class MatvecliTest : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(MatvecliTest);
    // TODO
    CPPUNIT_TEST_SUITE_END();
};

int test_matvecli(int, char*[])
{
    CppUnit::TextTestRunner runner;
    runner.setOutputter(new CppUnit::CompilerOutputter(&runner.result(), std::cerr));
    runner.addTest(MatvecliTest::suite());
    return runner.run() ? 0 : 1;
}
