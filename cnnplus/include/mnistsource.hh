/**************************************************************************//**
 *
 * \file   mnistsource.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Dec 07 2008
 *
 * $Id: mnistsource.hh 1827 2009-07-20 18:17:58Z dast $
 *
 * \brief  Header for cnnplus::MnistFileSource and cnnplus::MnistMemSource.
 *
 *****************************************************************************/

#ifndef CNNPLUS_MNISTSOURCE_HH
#define CNNPLUS_MNISTSOURCE_HH

#include "common.hh"
#include "datasource.hh"
#include <fstream>
#include <vector>

CNNPLUS_NS_BEGIN

//! Base class for MNIST data source
template<typename T>
class MnistDataSource : public DataSource<T>
{
public:
    //! Ctr
    /*! \param imgSize desired size of the images (patterns)
     */
    MnistDataSource(Size const & imgSize, T dataRangeMin, T dataRangeMax);

    virtual void shuffle();
    virtual void shift(int dist);
    virtual void reset();
    virtual size_t sizeOut() const;
    virtual Size sizePattern() const;
    virtual int idx(int pos = -1) const;

protected:
    //! Reads header of image and label file
    void readHeaders(std::ifstream & fileImage, std::ifstream & fileLabel);
    //! Sets the desired image size
    void setRealImageSize(size_t const height, size_t const width) {
        realImageSize_.set(height, width);
        yStart_ = (desImageSize_.height() - realImageSize_.height()) / 2;
        xStart_ = (desImageSize_.width()  - realImageSize_.width() ) / 2;
        yEnd_   = yStart_ + realImageSize_.height();
        xEnd_   = xStart_ + realImageSize_.width();
    }

protected:
    std::string imageFileName_;
    std::string labelFileName_;
    Size const desImageSize_;
    Size realImageSize_;
    T const scale_, shift_;
    size_t yStart_, yEnd_;
    size_t xStart_, xEnd_;
    std::vector<int> idx_;
};

//! Data source for the MNIST database of handwritten digits
/*! \see http://yann.lecun.com/exdb/mnist/
 */
template<typename T>
class MnistFileSource : public MnistDataSource<T>
{
public:
    //! Ctr
    /*! \param imgSize desired size of the images (patterns)
     */
    MnistFileSource(Size const & imgSize = Size(28, 28),
        T dataRangeMin = -1, T dataRangeMax = 1);
    //! Dtr
    virtual ~MnistFileSource() { close(); }

    //! Opens the image and label file
    void open(std::string const & imageFile, std::string const & labelFile);
    //! Closes the image and label file
    void close() throw();

    virtual int fprop(T * out);
    virtual int seek(int pos);
    virtual std::string toString() const;

private:
    std::ifstream fileImage_, fileLabel_;
    std::streampos imageBegin_, labelBegin_;
};

//! Data source for the MNIST database of handwritten digits
/*! \see http://yann.lecun.com/exdb/mnist/
 */
template<typename T>
class MnistMemSource : public MnistDataSource<T>
{
public:
    //! Ctr
    /*! \param imgSize desired size of the images (patterns)
     */
    MnistMemSource(Size const & imgSize = Size(28, 28),
        T dataRangeMin = -1, T dataRangeMax = 1);
    //! Dtr
    virtual ~MnistMemSource() { unload(); }

    //! Loads the image and label file
    void load(std::string const & imageFile, std::string const & labelFile);
    //! Unloads the image and label file
    void unload() throw();

    virtual int fprop(T * out);
    virtual int seek(int pos);
    virtual std::string toString() const;

private:
    unsigned char * images_;
    unsigned char * labels_;
};

CNNPLUS_NS_END

#endif // CNNPLUS_MNISTSOURCE_HH
